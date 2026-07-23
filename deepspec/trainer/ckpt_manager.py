import os
import random
import re
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_EXTERNAL_DRAFT_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_target_layers",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
    "rms_norm_eps",
    "rope_parameters",
    "max_position_embeddings",
    "layer_types",
    "sliding_window",
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "num_anchors",
    "tie_word_embeddings",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "markov_rank",
    "markov_head_type",
)


def discover_latest_checkpoint(checkpoint_dir):
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)


def select_draft_initialization(
    *,
    resume_checkpoint_dir: str | None,
    init_name_or_path: str | None,
    init_revision: str | None,
) -> str:
    if resume_checkpoint_dir is not None:
        return "resume"
    has_path = isinstance(init_name_or_path, str) and bool(init_name_or_path)
    has_revision = isinstance(init_revision, str) and bool(init_revision)
    if has_path != has_revision:
        raise ValueError(
            "external draft init requires both init_draft_name_or_path and "
            "init_draft_revision"
        )
    if has_revision and not _GIT_REVISION_RE.fullmatch(init_revision):
        raise ValueError(
            "external draft init revision must be a lowercase 40-character git SHA"
        )
    return "external" if has_path else "scratch"


def _validate_external_draft(*, expected_model, loaded_model) -> None:
    for field in _EXTERNAL_DRAFT_CONFIG_FIELDS:
        expected = getattr(expected_model.config, field, None)
        loaded = getattr(loaded_model.config, field, None)
        if loaded != expected:
            raise ValueError(
                f"external draft config.{field} mismatch: {loaded!r} != {expected!r}"
            )


def load_external_draft_model(
    *,
    init_name_or_path: str,
    init_revision: str,
    draft_model,
    device,
    precision_dtype,
):
    if not _GIT_REVISION_RE.fullmatch(init_revision):
        raise ValueError(
            "external draft init revision must be a lowercase 40-character git SHA"
        )
    loaded_model, loading_info = type(draft_model).from_pretrained(
        init_name_or_path,
        revision=init_revision,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
        output_loading_info=True,
    )
    for field in (
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    ):
        if loading_info.get(field):
            raise ValueError(
                f"external draft loader reported {field}: {loading_info[field]}"
            )
    resolved_revision = getattr(loaded_model.config, "_commit_hash", None)
    if resolved_revision is not None and resolved_revision != init_revision:
        raise ValueError(
            "external draft resolved revision mismatch: "
            f"{resolved_revision} != {init_revision}"
        )
    _validate_external_draft(
        expected_model=draft_model,
        loaded_model=loaded_model,
    )
    loaded_model = loaded_model.to(device=device, dtype=precision_dtype)
    loaded_model.initialize_embeddings_and_head(
        embed_tokens=draft_model.embed_tokens,
        lm_head=draft_model.lm_head,
        freeze=True,
    )
    return loaded_model


def save_train_config(*, train_config, checkpoint_dir: str) -> str:
    dest_path = os.path.join(checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    if not is_global_main_process():
        return dest_path

    ensure_dir(checkpoint_dir)
    shutil.copy(train_config._origin_config_path, dest_path)
    opts = train_config._origin_opts
    if opts:
        with open(dest_path, "a", encoding="utf-8") as handle:
            handle.write("\n\n# --opts overrides applied at save time\n")
            for opt in opts:
                handle.write(_render_opt_assignment(opt) + "\n")
    return dest_path


def _render_opt_assignment(opt: str) -> str:
    key, raw_value = opt.split("=", 1)
    head, *rest = key.split(".")
    accessors = "".join(f"[{part!r}]" for part in rest)
    value = yaml.safe_load(raw_value)
    return f"{head}{accessors} = {value!r}"


@dataclass(frozen=True)
class TrainingResumeState:
    # next_micro_step is the single source of truth for training progress;
    # global_step and current_epoch are derived from it together with
    # gradient_accumulation_steps / micro_batches_per_epoch.
    next_micro_step: int


def load_resume_draft_model(
    *,
    resume_checkpoint_dir: str,
    draft_model,
    device,
    precision_dtype,
    global_rank: int,
):
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)
    resumed_model = type(draft_model).from_pretrained(
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
    )
    resumed_model = resumed_model.to(device=device, dtype=precision_dtype)
    resumed_model.set_embedding_head_trainable(False)
    return resumed_model


def load_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
) -> TrainingResumeState:
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    next_micro_step = int(checkpoint["next_micro_step"])
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps."
    )

    saved_rank = int(checkpoint["global_rank"])
    assert saved_rank == int(global_rank)
    
    saved_world_size = int(checkpoint["world_size"])
    assert saved_world_size == int(world_size)
    
    saved_local_batch_size = int(checkpoint["local_batch_size"])
    assert saved_local_batch_size == int(local_batch_size)

    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])

    global_step = next_micro_step // gradient_accumulation_steps
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    print_on_global_main(
        (
            "AUTO-RESUME from "
            f"{resume_checkpoint_dir}, next_micro_step={next_micro_step}, "
            "to force fresh run change exp_name or remove step_latest"
        )
    )
    print_on_local_main(
        f"Resumed from {resume_checkpoint_dir}: "
        f"next_micro_step={next_micro_step}, global_step={global_step}, "
        f"epoch={current_epoch}"
    )
    return TrainingResumeState(next_micro_step=next_micro_step)


def save_checkpoint(
    *,
    model,
    draft_model,
    optimizer,
    checkpoint_dir_root: str,
    train_config,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    if is_global_main_process():
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=checkpoint_dir,
    )
    training_state = _serialize_training_state(
        optimizer=optimizer,
        next_micro_step=next_micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_rank=global_rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
    )
    torch.save(
        training_state,
        _rank_training_state_path(checkpoint_dir, global_rank),
    )
    dist.barrier()
    if is_global_main_process():
        safe_symlink(
            checkpoint_dir,
            os.path.join(checkpoint_dir_root, "step_latest"),
        )
        print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
    dist.barrier()
    return checkpoint_dir


def _rank_training_state_path(checkpoint_dir: str, global_rank: int) -> str:
    return os.path.join(
        checkpoint_dir,
        f"training_state.rank{int(global_rank)}.pt",
    )


def _serialize_training_state(
    *,
    optimizer,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
):
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    return {
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def _full_model_state_dict(model):
    assert isinstance(model, FSDP), "training model must be wrapped in FSDP"
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config,
    ):
        return model.state_dict()


def _save_model_checkpoint(*, model, draft_model, checkpoint_dir: str):
    state_dict = _full_model_state_dict(model)
    if is_global_main_process():
        draft_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key
            if normalized_key.startswith("_orig_mod."):
                normalized_key = normalized_key[len("_orig_mod.") :]
            draft_state_dict[normalized_key] = value
        assert draft_state_dict, "Failed to extract draft model state_dict from checkpoint."
        draft_model.save_pretrained(checkpoint_dir, state_dict=draft_state_dict)
