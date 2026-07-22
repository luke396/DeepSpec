"""Task-specific live-hidden DSpark trainer and exact-eight parity gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist

from deepspec.data.live_hidden_dataset import (
    LiveHiddenCollator,
    LiveHiddenDataset,
    build_stock_vllm_requester,
)
from deepspec.data.target_cache_dataset import (
    CacheCollator,
    CacheDataset,
    validate_train_cache,
)
from deepspec.data.cuda_prefetcher import move_batch_to_device
from deepspec.modeling.dspark.loss import _collect_local_terms, compute_dspark_loss
from deepspec.trainer.ckpt_manager import discover_latest_checkpoint
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from deepspec.utils import print_on_global_main


def _required_string(value, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_live_source_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"source manifest has empty row {line_number}")
            row = json.loads(line)
            if not isinstance(row, Mapping) or set(row) != {
                "source_row_id",
                "input_ids",
                "loss_mask",
            }:
                raise ValueError("source manifest row must use exact fields")
            input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
            loss_mask = torch.tensor(row["loss_mask"], dtype=torch.uint8)
            if input_ids.ndim != 1 or not input_ids.numel():
                raise ValueError("source manifest input_ids must be a non-empty vector")
            if loss_mask.shape != input_ids.shape:
                raise ValueError("source manifest loss_mask must match input_ids")
            rows.append(
                {
                    "source_row_id": row["source_row_id"],
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }
            )
    if not rows:
        raise ValueError("source manifest must not be empty")
    return rows


def load_producer_manifests(path: str | Path) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("producer manifests file must be a non-empty JSON list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("each producer manifest must be a JSON object")
    return value


def _symmetric_relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = 2.0 * torch.abs(left.float() - right.float())
    denominator = torch.abs(left.float()) + torch.abs(right.float()) + 1e-12
    return float((numerator / denominator).item())


def _local_loss_components(outputs, *, loss_decay_gamma: float, l1_loss_alpha: float):
    terms, has_confidence = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=loss_decay_gamma,
        l1_loss_alpha=l1_loss_alpha,
    )
    zero = terms["ce_loss_num"].new_zeros(())
    components = {
        "ce": terms["ce_loss_num"] / (terms["ce_loss_den"] + 1e-6),
        "l1": zero,
        "confidence": zero,
    }
    if terms["l1_loss_den"].item() > 0:
        components["l1"] = terms["l1_loss_num"] / (terms["l1_loss_den"] + 1e-6)
    if has_confidence:
        components["confidence"] = terms["confidence_loss_num"] / (
            terms["confidence_loss_den"] + 1e-6
        )
    return components


class LiveQwen3DSparkTrainer(Qwen3DSparkTrainer):
    data_collator_cls = LiveHiddenCollator

    def __init__(self, local_rank, args):
        if discover_latest_checkpoint(args.logging.checkpoint_dir) is not None:
            raise ValueError(
                "formal live training must be one uninterrupted fresh run; "
                "use the generic config for the separate native-resume gate"
            )
        self._target_final_norm_weight = None
        self._target_final_norm_eps = None
        self._parity_dataset = None
        self._parity_count = 0
        self._expected_replica_ids = None
        super().__init__(local_rank, args)

    def capture_target_model(self, target_model):
        target_backbone = getattr(target_model, "model", None)
        target_norm = getattr(target_backbone, "norm", None)
        weight = getattr(target_norm, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
            raise ValueError("target model does not expose Qwen3 final RMSNorm weight")
        eps = getattr(target_norm, "variance_epsilon", None)
        if eps is None:
            eps = getattr(target_norm, "eps", None)
        if eps is None or not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError(
                "target model does not expose a valid final RMSNorm epsilon"
            )
        self._target_final_norm_weight = (
            weight.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        )
        self._target_final_norm_eps = float(eps)

    def build_train_dataset(self):
        data_args = self.args.data
        source_rows = load_live_source_manifest(
            _required_string(
                data_args.source_manifest_path,
                field="data.source_manifest_path",
            )
        )
        producer_manifests = load_producer_manifests(
            _required_string(
                data_args.producer_manifests_path,
                field="data.producer_manifests_path",
            )
        )
        self._expected_replica_ids = [
            manifest["producer_replica_id"] for manifest in producer_manifests
        ]
        if (
            self._target_final_norm_weight is None
            or self._target_final_norm_eps is None
        ):
            raise ValueError(
                "target final RMSNorm was not captured before dataset build"
            )
        dataset = LiveHiddenDataset(
            source_dataset=source_rows,
            request_hidden_states=build_stock_vllm_requester(
                endpoint=_required_string(
                    data_args.vllm_endpoint,
                    field="data.vllm_endpoint",
                ),
                model=_required_string(data_args.vllm_model, field="data.vllm_model"),
            ),
            producer_manifests=producer_manifests,
            ledger_dir=_required_string(data_args.ledger_dir, field="data.ledger_dir"),
            target_model_name_or_path=self.args.model.target_model_name_or_path,
            target_revision=self.args.model.target_revision,
            target_layer_ids=self.draft_model.target_layer_ids,
            final_norm_weight=self._target_final_norm_weight,
            final_norm_eps=self._target_final_norm_eps,
            request_timeout_s=float(data_args.request_timeout_s),
            max_retries=int(data_args.max_retries),
            selection_policy_digest=_required_string(
                data_args.selection_policy_digest,
                field="data.selection_policy_digest",
            ),
            num_epochs=int(self.args.train.num_train_epochs),
            run_id=_required_string(data_args.run_id, field="data.run_id"),
            consumer_boot_id=_required_string(
                data_args.consumer_boot_id,
                field="data.consumer_boot_id",
            ),
        )
        parity_cache_path = data_args.parity_cache_path
        if parity_cache_path is not None:
            self._parity_dataset = CacheDataset(cache_dir=parity_cache_path)
            if len(self._parity_dataset) != len(dataset):
                raise ValueError(
                    "parity cache and live source have different row counts"
                )
            validate_train_cache(
                train_dataset=self._parity_dataset,
                draft_model=self.draft_model,
                target_model_name_or_path=self.args.model.target_model_name_or_path,
            )
        return dataset

    def _assert_hidden_parity(self, *, live_batch, oracle_batch):
        cosine_min = float(self.args.data.hidden_cosine_min)
        relative_rmse_max = float(self.args.data.hidden_relative_rmse_max)
        hidden_size = int(self.draft_model.config.hidden_size)
        num_layers = len(self.draft_model.target_layer_ids)
        batch_size = live_batch["input_ids"].shape[0]
        for batch_index in range(batch_size):
            seq_len = int(live_batch["attention_mask"][batch_index].sum().item())
            planes = (
                (
                    live_batch["target_hidden_states"][batch_index, :seq_len].view(
                        seq_len, num_layers, hidden_size
                    ),
                    oracle_batch["target_hidden_states"][batch_index, :seq_len].view(
                        seq_len, num_layers, hidden_size
                    ),
                    num_layers,
                ),
                (
                    live_batch["target_last_hidden_states"][batch_index, :seq_len].view(
                        seq_len, 1, hidden_size
                    ),
                    oracle_batch["target_last_hidden_states"][
                        batch_index, :seq_len
                    ].view(seq_len, 1, hidden_size),
                    1,
                ),
            )
            for live_planes, oracle_planes, plane_count in planes:
                for plane_index in range(plane_count):
                    live = live_planes[:, plane_index].float().flatten()
                    oracle = oracle_planes[:, plane_index].float().flatten()
                    oracle_rms = torch.sqrt(torch.mean(oracle.square()))
                    relative_rmse = torch.sqrt(torch.mean((live - oracle).square())) / (
                        oracle_rms + 1e-12
                    )
                    if torch.equal(live, oracle):
                        cosine = 1.0
                    else:
                        cosine = float(
                            torch.nn.functional.cosine_similarity(
                                live.unsqueeze(0), oracle.unsqueeze(0)
                            ).item()
                        )
                    if (
                        cosine < cosine_min
                        or float(relative_rmse.item()) > relative_rmse_max
                    ):
                        raise ValueError(
                            "HF/stock-vLLM hidden parity threshold failed: "
                            f"cosine={cosine}, relative_rmse={relative_rmse.item()}"
                        )

    def _run_parity_gate(self, *, batch, live_outputs):
        if self._parity_dataset is None:
            return
        source_indices = batch["_live_source_indices"].detach().cpu().tolist()
        oracle_batch = CacheCollator()(
            [self._parity_dataset[int(index)] for index in source_indices]
        )
        oracle_batch = move_batch_to_device(oracle_batch, self.device)
        for field in ("input_ids", "loss_mask", "attention_mask"):
            if not torch.equal(batch[field], oracle_batch[field]):
                raise ValueError(f"live/oracle {field} does not match")
        self._assert_hidden_parity(live_batch=batch, oracle_batch=oracle_batch)
        with torch.no_grad():
            oracle_outputs = self.model(
                input_ids=oracle_batch["input_ids"],
                target_hidden_states=oracle_batch["target_hidden_states"],
                loss_mask=oracle_batch["loss_mask"],
                target_last_hidden_states=oracle_batch["target_last_hidden_states"],
            )
        live_components = _local_loss_components(
            live_outputs,
            loss_decay_gamma=float(self.args.model.loss_decay_gamma),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
        )
        oracle_components = _local_loss_components(
            oracle_outputs,
            loss_decay_gamma=float(self.args.model.loss_decay_gamma),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
        )
        limit = float(self.args.data.loss_symmetric_relative_error_max)
        for name in live_components:
            error = _symmetric_relative_error(
                live_components[name], oracle_components[name]
            )
            if error > limit:
                raise ValueError(
                    f"live/oracle {name} loss symmetric relative error {error} > {limit}"
                )
        if not torch.equal(
            live_outputs.draft_logits.argmax(dim=-1),
            oracle_outputs.draft_logits.argmax(dim=-1),
        ):
            raise ValueError("live/oracle greedy draft token ids do not match")
        self._parity_count += len(source_indices)

    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        self._run_parity_gate(batch=batch, live_outputs=outputs)
        return compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )

    def train(self):
        super().train()
        self.train_dataset.assert_training_complete(
            expected_replica_ids=self._expected_replica_ids
        )
        if self._parity_dataset is not None:
            count = torch.tensor(
                self._parity_count, device=self.device, dtype=torch.long
            )
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            if int(count.item()) != len(self.train_dataset):
                raise ValueError(
                    "G1 parity did not cover each exact source row once: "
                    f"{count.item()} != {len(self.train_dataset)}"
                )
            print_on_global_main(
                f"G1 live parity passed for {int(count.item())} exact source rows."
            )
