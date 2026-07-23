import math
import re
from pathlib import Path

import torch
from speculators.train.data import ArrowDataset


_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class LiveHiddenAdapter(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        datapath,
        max_length: int,
        vllm_endpoint: str,
        target_model_name_or_path: str,
        target_revision: str,
        target_layer_ids,
        hidden_size: int,
        final_norm_weight: torch.Tensor,
        final_norm_eps: float,
        expected_num_samples: int | None,
    ):
        if not isinstance(vllm_endpoint, str) or not vllm_endpoint:
            raise ValueError("vllm_endpoint must be a non-empty string")
        if not _GIT_REVISION_RE.fullmatch(target_revision):
            raise ValueError("target_revision must be a lowercase 40-character git SHA")
        if int(max_length) <= 0 or int(hidden_size) <= 0:
            raise ValueError("max_length and hidden_size must be positive")
        if not math.isfinite(float(final_norm_eps)) or float(final_norm_eps) <= 0:
            raise ValueError("final RMSNorm epsilon must be positive and finite")
        if (
            final_norm_weight.dtype != torch.bfloat16
            or final_norm_weight.shape != (int(hidden_size),)
            or not torch.isfinite(final_norm_weight).all()
        ):
            raise ValueError("final RMSNorm weight must be finite BF16")

        self.max_length = int(max_length)
        self.hidden_size = int(hidden_size)
        self.target_layer_ids = [int(value) for value in target_layer_ids]
        self.final_norm_weight = final_norm_weight.detach().cpu().clone()
        self.final_norm_eps = float(final_norm_eps)
        cache_path = Path(datapath) / ".deepspec-live-cache-disabled"
        if cache_path.exists():
            raise ValueError(f"live hidden cache path must not exist: {cache_path}")
        self.dataset = ArrowDataset(
            max_len=self.max_length,
            datapath=datapath,
            hidden_states_path=cache_path,
            vllm_endpoint=vllm_endpoint,
            on_missing="generate",
            on_generate="delete",
        )
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

    def __getitem__(self, index):
        sample = self.dataset[index]
        if sample is None:
            raise RuntimeError(
                f"live hidden provider returned no sample for index {index}"
            )

        input_ids = sample["input_ids"]
        loss_mask = sample["loss_mask"]
        target_hidden = sample["hidden_states"]
        target_last_pre_norm = sample["verifier_last_hidden_states"]
        tensors = (input_ids, loss_mask, target_hidden, target_last_pre_norm)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("live hidden provider fields must be tensors")
        if input_ids.ndim != 1:
            raise ValueError("live hidden input_ids must be one-dimensional")
        seq_len = input_ids.shape[0]
        expected_hidden_shape = (seq_len, len(self.target_layer_ids) * self.hidden_size)
        if (
            seq_len == 0
            or seq_len > self.max_length
            or loss_mask.shape != input_ids.shape
        ):
            raise ValueError("live hidden token or loss-mask shape mismatch")
        if target_hidden.shape != expected_hidden_shape:
            raise ValueError(
                "live hidden shape mismatch: "
                f"{tuple(target_hidden.shape)} != {expected_hidden_shape}"
            )
        if target_last_pre_norm.shape != (seq_len, self.hidden_size):
            raise ValueError("live final hidden shape mismatch")
        if (
            target_hidden.dtype != torch.bfloat16
            or target_last_pre_norm.dtype != torch.bfloat16
        ):
            raise ValueError("live hidden dtype must be BF16")
        if not all(
            torch.isfinite(value).all()
            for value in (target_hidden, target_last_pre_norm)
        ):
            raise ValueError("live hidden finite check failed")

        target_last = target_last_pre_norm.float()
        variance = target_last.square().mean(dim=-1, keepdim=True)
        target_last *= torch.rsqrt(variance + self.final_norm_eps)
        target_last = (
            target_last.to(target_last_pre_norm.dtype) * self.final_norm_weight
        ).to(torch.bfloat16)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden,
            "target_last_hidden_states": target_last,
        }
