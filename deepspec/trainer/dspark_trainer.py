import torch

from deepspec.data import CacheCollator
from deepspec.data.live_hidden_prefilter import (
    load_prepared_live_hidden_metadata,
    validate_prepared_live_hidden_data,
)
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
        if self.args.data.get("train_jsonl_path") is None:
            return
        norm = target_model.model.norm
        self._target_final_norm_weight = (
            norm.weight.detach().to(device="cpu", dtype=torch.bfloat16).clone()
        )
        self._target_final_norm_eps = float(norm.variance_epsilon)

    def build_train_dataset(self):
        data_args = self.args.data
        train_jsonl_path = data_args.get("train_jsonl_path")
        if train_jsonl_path is None:
            return super().build_train_dataset()
        if data_args.get("target_cache_path") is not None:
            raise ValueError(
                "train_jsonl_path and target_cache_path are mutually exclusive"
            )
        train_manifest_path = data_args.get("train_manifest_path")
        if train_manifest_path is None:
            raise ValueError(
                "live hidden training requires data.train_manifest_path"
            )
        from deepspec.data.live_hidden_dataset import LiveHiddenDataset

        if self.global_rank == 0:
            prepared = validate_prepared_live_hidden_data(
                manifest_path=train_manifest_path,
                filtered_path=train_jsonl_path,
                chat_template=data_args.chat_template,
                max_length=int(data_args.max_length),
                min_loss_tokens=int(data_args.min_loss_tokens),
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                target_revision=self.args.model.get("target_revision"),
            )
            print(
                "Prepared live hidden data: "
                f"{prepared.accepted_samples}/{prepared.source_samples} accepted, "
                f"{prepared.rejected_samples} rejected -> {prepared.filtered_path}",
                flush=True,
            )
        torch.distributed.barrier()
        prepared = load_prepared_live_hidden_metadata(
            manifest_path=train_manifest_path,
            filtered_path=train_jsonl_path,
        )

        return LiveHiddenDataset(
            data_path=train_jsonl_path,
            tokenizer=self.tokenizer,
            chat_template=data_args.chat_template,
            max_length=int(data_args.max_length),
            min_loss_tokens=int(data_args.min_loss_tokens),
            vllm_endpoint=data_args.vllm_endpoint,
            vllm_model=data_args.vllm_model,
            hidden_states_path=data_args.hidden_states_path,
            target_layer_ids=self.draft_model.target_layer_ids,
            hidden_size=int(self.draft_model.config.hidden_size),
            final_norm_weight=self._target_final_norm_weight,
            final_norm_eps=self._target_final_norm_eps,
            expected_num_samples=prepared.accepted_samples,
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
