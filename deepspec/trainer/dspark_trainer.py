import math

import torch

from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def capture_target_model(self, target_model):
        if self.args.data.get("live_data_path") is None:
            return
        if getattr(target_model.config, "model_type", None) != "qwen3":
            raise ValueError("live hidden training currently supports only Qwen3")
        norm = target_model.model.norm
        eps = getattr(norm, "variance_epsilon", None)
        if eps is None:
            eps = getattr(norm, "eps", None)
        if eps is None or not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError("target model has no valid final RMSNorm epsilon")
        self._target_final_norm_weight = (
            norm.weight.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        )
        self._target_final_norm_eps = float(eps)

    def build_train_dataset(self):
        data_args = self.args.data
        live_data_path = data_args.get("live_data_path")
        if live_data_path is None:
            return super().build_train_dataset()
        if data_args.get("target_cache_path") is not None:
            raise ValueError(
                "live_data_path and target_cache_path are mutually exclusive"
            )
        from deepspec.data.live_hidden_adapter import LiveHiddenAdapter

        return LiveHiddenAdapter(
            datapath=live_data_path,
            max_length=int(data_args.max_length),
            vllm_endpoint=data_args.vllm_endpoint,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
            target_revision=self.args.model.target_revision,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=int(self.draft_model.config.hidden_size),
            final_norm_weight=self._target_final_norm_weight,
            final_norm_eps=self._target_final_norm_eps,
            expected_num_samples=data_args.expected_num_samples,
        )

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)
