import fcntl
import os
import time
from pathlib import Path

import openai
import torch
from safetensors.torch import load_file

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.target_cache_dataset import ConversationCollator


_REQUEST_TIMEOUT_SECONDS = 120
_LOCK_TIMEOUT_SECONDS = 10


def resolve_loss_temperature(row) -> float:
    """Resolve one canonical regen row's per-sample loss temperature.

    Reads `sampling.temperature` as preserved through fold/regeneration.
    Greedy requests (T == 0) and rows without a usable value fall back to
    1.0 — the frozen experiment policy: temperature-alignment has no
    first-order effect on argmax-verified traffic, and 1.0 reproduces the
    historical loss for unannotated data. Validation lives here, on the
    CPU data path, so the loss seam can trust the batch tensor.
    """
    sampling = row.get("sampling")
    if not isinstance(sampling, dict):
        return 1.0
    value = sampling.get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    value = float(value)
    if not 0.0 < value < float("inf"):
        return 1.0
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
        target_layer_ids,
        hidden_size: int,
        final_norm_weight: torch.Tensor,
        final_norm_eps: float,
        expected_num_samples: int | None,
    ):
        data_path = Path(data_path)
        self.max_length = int(max_length)
        min_loss_tokens = int(min_loss_tokens)
        self.hidden_size = int(hidden_size)
        self.target_layer_ids = [int(value) for value in target_layer_ids]

        self.vllm_endpoint = vllm_endpoint
        self.vllm_model = vllm_model
        self.hidden_states_path = Path(hidden_states_path).resolve(strict=True)
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
        resolved = Path(raw_path).resolve(strict=False)
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
        hidden_file = self._resolve_hidden_file(
            response.kv_transfer_params["hidden_states_path"]
        )

        try:
            _wait_for_hidden_file(hidden_file)
            tensors = load_file(hidden_file)
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
        row = self.dataset[index]
        batch = self._token_collator([row])
        if batch is None:
            raise RuntimeError(
                "prepared live hidden data contract changed for sample "
                f"{index}"
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
            "loss_temperature": torch.tensor(
                resolve_loss_temperature(row), dtype=torch.float32
            ),
        }
