import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "deepspec_ckpt_manager_under_test",
    Path(__file__).parents[1] / "deepspec/trainer/ckpt_manager.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_external_draft_model = _MODULE.load_external_draft_model
select_draft_initialization = _MODULE.select_draft_initialization


REVISION = "a" * 40


def _loading_info(**overrides):
    info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    info.update(overrides)
    return info


def _config(*, revision=REVISION):
    return SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_target_layers=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        max_position_embeddings=32768,
        layer_types=["full_attention", "full_attention"],
        sliding_window=None,
        block_size=7,
        target_layer_ids=[0, 1],
        mask_token_id=31,
        num_anchors=4,
        tie_word_embeddings=False,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        markov_rank=2,
        markov_head_type="vanilla",
        _attn_implementation="flex_attention",
        _commit_hash=revision,
    )


class FakeDraftModel:
    loaded_model = None
    loading_info = None

    def __init__(self, *, config=None):
        self.config = config or _config()
        self.embedding_head_trainable = True
        self.embed_tokens = object()
        self.lm_head = object()
        self.initialized_from = None
        self.to_args = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert path == "published-draft"
        assert kwargs["revision"] == REVISION
        assert kwargs["output_loading_info"] is True
        return cls.loaded_model, cls.loading_info

    def to(self, *, device, dtype):
        self.to_args = (device, dtype)
        return self

    def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze):
        self.initialized_from = (embed_tokens, lm_head)
        self.embedding_head_trainable = not freeze


def _load(expected):
    return load_external_draft_model(
        init_name_or_path="published-draft",
        init_revision=REVISION,
        draft_model=expected,
        device=torch.device("cpu"),
        precision_dtype=torch.bfloat16,
    )


def test_initialization_precedence_and_pair_validation():
    assert (
        select_draft_initialization(
            resume_checkpoint_dir="/checkpoints/step_1",
            init_name_or_path="published-draft",
            init_revision=REVISION,
        )
        == "resume"
    )
    assert (
        select_draft_initialization(
            resume_checkpoint_dir=None,
            init_name_or_path="published-draft",
            init_revision=REVISION,
        )
        == "external"
    )
    assert (
        select_draft_initialization(
            resume_checkpoint_dir=None,
            init_name_or_path=None,
            init_revision=None,
        )
        == "scratch"
    )
    with pytest.raises(ValueError, match="requires both"):
        select_draft_initialization(
            resume_checkpoint_dir=None,
            init_name_or_path="published-draft",
            init_revision=None,
        )


def test_external_init_loads_only_exact_compatible_weights():
    expected = FakeDraftModel()
    loaded = FakeDraftModel()
    FakeDraftModel.loaded_model = loaded
    FakeDraftModel.loading_info = _loading_info()
    result = _load(expected)

    assert result is loaded
    assert result.embedding_head_trainable is False
    assert result.initialized_from == (expected.embed_tokens, expected.lm_head)
    assert result.to_args == (torch.device("cpu"), torch.bfloat16)


def test_external_init_rejects_incomplete_weights():
    expected = FakeDraftModel()
    FakeDraftModel.loaded_model = FakeDraftModel()
    FakeDraftModel.loading_info = _loading_info(missing_keys=["draft.layers.0.weight"])

    with pytest.raises(ValueError, match="missing_keys"):
        _load(expected)


def test_external_init_rejects_incompatible_config():
    expected = FakeDraftModel()
    loaded = FakeDraftModel()
    loaded.config.block_size = 8
    FakeDraftModel.loaded_model = loaded
    FakeDraftModel.loading_info = _loading_info()

    with pytest.raises(ValueError, match=r"config\.block_size mismatch"):
        _load(expected)
