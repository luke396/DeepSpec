from collections.abc import Mapping
import fcntl
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import openai
import torch
from safetensors.torch import load_file

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.target_cache_dataset import ConversationCollator


_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUEST_TIMEOUT_SECONDS = 120
_LOCK_TIMEOUT_SECONDS = 10


def _validate_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("vllm_endpoint must be a non-empty string")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("vllm_endpoint must be an HTTP(S) base URL")
    return value


def _wait_for_hidden_file(path: Path) -> None:
    lock_path = Path(f"{path}.lock")
    if not lock_path.exists():
        return

    descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for hidden-state lock: {lock_path}"
                    ) from None
                time.sleep(0.1)
    finally:
        os.close(descriptor)
    lock_path.unlink()


class LiveHiddenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data_path,
        tokenizer,
        chat_template: str,
        max_length: int,
        min_loss_tokens: int,
        vllm_endpoint: str,
        vllm_model: str,
        hidden_states_path,
        target_model_name_or_path: str,
        target_revision: str,
        target_layer_ids,
        hidden_size: int,
        final_norm_weight: torch.Tensor,
        final_norm_eps: float,
        expected_num_samples: int | None,
    ):
        data_path = Path(data_path)
        if not data_path.is_file():
            raise ValueError(f"train JSONL path is not a file: {data_path}")
        if (
            not isinstance(target_model_name_or_path, str)
            or not target_model_name_or_path
        ):
            raise ValueError("target_model_name_or_path must be non-empty")
        if not isinstance(vllm_model, str) or not vllm_model:
            raise ValueError("vllm_model must be non-empty")
        if not isinstance(target_revision, str) or not _GIT_REVISION_RE.fullmatch(
            target_revision
        ):
            raise ValueError("target_revision must be a lowercase 40-character git SHA")

        self.max_length = int(max_length)
        min_loss_tokens = int(min_loss_tokens)
        self.hidden_size = int(hidden_size)
        self.target_layer_ids = [int(value) for value in target_layer_ids]
        if (
            self.max_length <= 0
            or min_loss_tokens <= 0
            or self.hidden_size <= 0
            or not self.target_layer_ids
        ):
            raise ValueError(
                "max_length, min_loss_tokens, hidden_size, and target layers "
                "must be positive"
            )
        if self.target_layer_ids != sorted(set(self.target_layer_ids)):
            raise ValueError("target_layer_ids must be sorted and unique")
        if not math.isfinite(float(final_norm_eps)) or float(final_norm_eps) <= 0:
            raise ValueError("final RMSNorm epsilon must be positive and finite")
        if (
            not isinstance(final_norm_weight, torch.Tensor)
            or final_norm_weight.dtype != torch.bfloat16
            or final_norm_weight.shape != (self.hidden_size,)
            or not torch.isfinite(final_norm_weight).all()
        ):
            raise ValueError("final RMSNorm weight must be finite BF16")

        hidden_states_path = Path(hidden_states_path)
        if not hidden_states_path.is_dir():
            raise ValueError(
                "hidden_states_path must be an existing shared directory: "
                f"{hidden_states_path}"
            )

        self.vllm_endpoint = _validate_endpoint(vllm_endpoint)
        self.vllm_model = vllm_model
        self.hidden_states_path = hidden_states_path.resolve(strict=True)
        self.target_model_name_or_path = target_model_name_or_path
        self.final_norm_weight = final_norm_weight.detach().cpu().clone()
        self.final_norm_eps = float(final_norm_eps)
        self._client = None
        self._token_collator = ConversationCollator(
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=self.max_length,
            min_loss_tokens=min_loss_tokens,
        )
        self.dataset = JsonLineDataset([str(data_path)])
        if expected_num_samples is not None and len(self.dataset) != int(
            expected_num_samples
        ):
            raise ValueError(
                "live dataset sample count mismatch: "
                f"{len(self.dataset)} != {int(expected_num_samples)}"
            )
        self.manifest = {
            "target_model_name_or_path": target_model_name_or_path,
            "target_revision": target_revision,
            "target_layer_ids": list(self.target_layer_ids),
            "hidden_size": self.hidden_size,
        }

    def __len__(self):
        return len(self.dataset)

    def _setup_client(self):
        client = openai.OpenAI(
            base_url=self.vllm_endpoint,
            api_key="EMPTY",
            max_retries=3,
        )
        models = client.models.list().data
        model_ids = [model.id for model in models]
        if self.vllm_model not in model_ids:
            raise ValueError(
                "vLLM model identity mismatch: "
                f"{self.vllm_model!r} not in {model_ids!r}"
            )
        self._client = client

    def _resolve_hidden_file(self, raw_path) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("vLLM response has no hidden_states_path")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise ValueError("vLLM returned a non-absolute hidden-state path")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.hidden_states_path)
        except ValueError:
            raise ValueError(
                "vLLM returned a path outside configured hidden-state directory"
            ) from None
        return resolved

    def _load_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._client is None:
            self._setup_client()
        token_ids = input_ids.tolist()
        response = self._client.completions.create(
            model=self.vllm_model,
            prompt=token_ids,
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        kv_transfer_params = getattr(response, "kv_transfer_params", None)
        if not isinstance(kv_transfer_params, Mapping):
            raise ValueError("vLLM response has no kv_transfer_params")
        hidden_file = self._resolve_hidden_file(
            kv_transfer_params.get("hidden_states_path")
        )

        try:
            if not response.choices:
                raise ValueError("vLLM response has no completion choice")
            response_token_ids = getattr(response.choices[0], "prompt_token_ids", None)
            if response_token_ids != token_ids:
                raise ValueError("vLLM response token IDs mismatch")
            _wait_for_hidden_file(hidden_file)
            resolved_file = hidden_file.resolve(strict=True)
            if resolved_file != hidden_file or not resolved_file.is_file():
                raise ValueError("vLLM hidden-state path is not a regular file")
            tensors = load_file(hidden_file)
            if set(tensors) != {"token_ids", "hidden_states"}:
                raise ValueError(
                    f"vLLM hidden-state file fields mismatch: {sorted(tensors)}"
                )
            file_token_ids = tensors["token_ids"]
            hidden_states = tensors["hidden_states"]
            if file_token_ids.ndim != 1 or file_token_ids.tolist() != token_ids:
                raise ValueError("vLLM hidden-state file token IDs mismatch")
            expected_shape = (
                len(token_ids),
                len(self.target_layer_ids) + 1,
                self.hidden_size,
            )
            if tuple(hidden_states.shape) != expected_shape:
                raise ValueError(
                    "vLLM hidden-state shape mismatch: "
                    f"{tuple(hidden_states.shape)} != {expected_shape}"
                )
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("vLLM hidden-state dtype must be BF16")
            if not torch.isfinite(hidden_states).all():
                raise ValueError("vLLM hidden-state finite check failed")
            target_hidden = hidden_states[:, :-1].flatten(1)
            target_last_pre_norm = hidden_states[:, -1]
        finally:
            hidden_file.unlink(missing_ok=True)
        return target_hidden, target_last_pre_norm

    def _apply_final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized *= torch.rsqrt(variance + self.final_norm_eps)
        return (normalized.to(hidden_states.dtype) * self.final_norm_weight).to(
            torch.bfloat16
        )

    def __getitem__(self, index):
        batch = self._token_collator([self.dataset[index]])
        if batch is None:
            raise ValueError(
                f"training sample {index} has fewer than the required loss tokens"
            )
        input_ids = batch["input_ids"][0]
        loss_mask = batch["loss_mask"][0]
        target_hidden, target_last_pre_norm = self._load_hidden_states(
            input_ids=input_ids
        )
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden,
            "target_last_hidden_states": self._apply_final_norm(target_last_pre_norm),
        }
