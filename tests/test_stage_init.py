import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_source_module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ckpt_manager = _load_source_module(
    "task79_ckpt_manager", "deepspec/trainer/ckpt_manager.py"
)
load_external_draft_model = ckpt_manager.load_external_draft_model
select_draft_initialization = ckpt_manager.select_draft_initialization


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _load_base_trainer_module(monkeypatch):
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)
    monkeypatch.setitem(sys.modules, "deepspec.trainer.ckpt_manager", ckpt_manager)
    return _load_source_module(
        "task79_base_trainer", "deepspec/trainer/base_trainer.py"
    )


def _config(**overrides):
    values = {
        "model_type": "qwen3",
        "architectures": ["Qwen3DSparkModel"],
        "vocab_size": 151936,
        "hidden_size": 8,
        "num_target_layers": 36,
        "num_hidden_layers": 5,
        "block_size": 7,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
        "num_anchors": 512,
        "tie_word_embeddings": False,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "_attn_implementation": "flex_attention",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeModel:
    loaded = None
    loading_info = None
    call = None

    def __init__(self, config=None, state=None):
        self.config = config or _config()
        self._state = state or {
            "layers.0.weight": torch.zeros(2, 3),
            "confidence_head.weight": torch.zeros(1, 2),
        }
        self.embedding_head_trainable = True

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.call = (path, kwargs)
        return cls.loaded, cls.loading_info or {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, *, strict):
        assert strict is True
        self._state = state

    def to(self, **_kwargs):
        return self

    def set_embedding_head_trainable(self, value):
        self.embedding_head_trainable = value


@pytest.mark.parametrize(
    ("resume", "path", "revision", "expected"),
    [
        ("/stage/step_50", "/published", "a" * 40, "resume"),
        ("/stage/step_50", "/published", None, "resume"),
        (None, "/published", "a" * 40, "external"),
        (None, None, None, "scratch"),
    ],
)
def test_initialization_priority_is_resume_then_external_then_scratch(
    resume, path, revision, expected
):
    assert (
        select_draft_initialization(
            resume_checkpoint_dir=resume,
            init_name_or_path=path,
            init_revision=revision,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("path", "revision"),
    [("/published", None), (None, "a" * 40), ("/published", "main")],
)
def test_external_init_requires_an_exact_paired_revision(path, revision):
    with pytest.raises(ValueError, match="external draft init"):
        select_draft_initialization(
            resume_checkpoint_dir=None,
            init_name_or_path=path,
            init_revision=revision,
        )


def test_external_init_loads_only_strict_matching_weights():
    expected = _FakeModel()
    loaded = _FakeModel()
    loaded.config._commit_hash = "a" * 40
    _FakeModel.loaded = loaded
    _FakeModel.loading_info = None

    result = load_external_draft_model(
        init_name_or_path="deepseek-ai/dspark_qwen3_8b_block7",
        init_revision="a" * 40,
        draft_model=expected,
        device="cpu",
        precision_dtype=torch.bfloat16,
    )

    assert result is expected
    assert not expected.embedding_head_trainable
    assert _FakeModel.call[0] == "deepseek-ai/dspark_qwen3_8b_block7"
    assert _FakeModel.call[1]["revision"] == "a" * 40
    assert _FakeModel.call[1]["output_loading_info"] is True


@pytest.mark.parametrize(
    ("loaded_config", "message"),
    [
        (_config(block_size=8), "block_size"),
        (_config(target_layer_ids=[1, 9, 17, 25, 34]), "target_layer_ids"),
        (_config(enable_confidence_head=False), "enable_confidence_head"),
    ],
)
def test_external_init_rejects_research_object_drift(loaded_config, message):
    expected = _FakeModel()
    _FakeModel.loaded = _FakeModel(config=loaded_config)
    _FakeModel.loaded.config._commit_hash = "a" * 40
    _FakeModel.loading_info = None

    with pytest.raises(ValueError, match=message):
        load_external_draft_model(
            init_name_or_path="published",
            init_revision="a" * 40,
            draft_model=expected,
            device="cpu",
            precision_dtype=torch.bfloat16,
        )


def test_external_init_rejects_loader_and_state_shape_drift():
    expected = _FakeModel()
    _FakeModel.loaded = _FakeModel(
        state={
            "layers.0.weight": torch.zeros(4, 3),
            "confidence_head.weight": torch.zeros(1, 2),
        }
    )
    _FakeModel.loaded.config._commit_hash = "a" * 40
    _FakeModel.loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    with pytest.raises(ValueError, match="shape"):
        load_external_draft_model(
            init_name_or_path="published",
            init_revision="a" * 40,
            draft_model=expected,
            device="cpu",
            precision_dtype=torch.bfloat16,
        )

    _FakeModel.loaded = _FakeModel()
    _FakeModel.loaded.config._commit_hash = "a" * 40
    _FakeModel.loading_info = {
        "missing_keys": ["markov_head.weight"],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    with pytest.raises(ValueError, match="missing_keys"):
        load_external_draft_model(
            init_name_or_path="published",
            init_revision="a" * 40,
            draft_model=expected,
            device="cpu",
            precision_dtype=torch.bfloat16,
        )


def test_external_init_requires_loader_revision_readback():
    expected = _FakeModel()
    _FakeModel.loaded = _FakeModel()
    _FakeModel.loading_info = None

    with pytest.raises(ValueError, match="resolved revision"):
        load_external_draft_model(
            init_name_or_path="published",
            init_revision="a" * 40,
            draft_model=expected,
            device="cpu",
            precision_dtype=torch.bfloat16,
        )


def test_default_dataset_seam_preserves_offline_cache(monkeypatch):
    base_trainer = _load_base_trainer_module(monkeypatch)

    expected = object()
    calls = []
    monkeypatch.setattr(
        base_trainer,
        "CacheDataset",
        lambda *, cache_dir: calls.append(("build", cache_dir)) or expected,
    )
    trainer = base_trainer.BaseTrainer.__new__(base_trainer.BaseTrainer)
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(target_cache_path="/offline/cache"),
        model=SimpleNamespace(target_model_name_or_path="target"),
    )
    trainer.draft_model = object()

    trainer.train_dataset = trainer.build_train_dataset()

    assert trainer.train_dataset is expected
    assert calls[0] == ("build", "/offline/cache")


def test_target_revision_and_capture_hook_are_owned_by_model_build(monkeypatch):
    base_trainer = _load_base_trainer_module(monkeypatch)
    calls = []

    class Draft:
        def to(self, **kwargs):
            calls.append(("draft-to", kwargs))
            return self

        def initialize_embeddings_and_head(self, **kwargs):
            calls.append(("initialize", kwargs))

    class Target:
        def to(self, **kwargs):
            calls.append(("target-to", kwargs))
            return self

        def eval(self):
            calls.append(("target-eval", {}))
            return self

        def get_input_embeddings(self):
            return "embeddings"

        def get_output_embeddings(self):
            return "head"

    monkeypatch.setattr(
        base_trainer.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("tokenizer", path, kwargs)) or "tokenizer",
    )
    monkeypatch.setattr(
        base_trainer.AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("config", path, kwargs)) or "config",
    )
    monkeypatch.setattr(
        base_trainer.AutoModelForCausalLM,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("target", path, kwargs)) or Target(),
    )

    class Trainer(base_trainer.BaseTrainer):
        def _build_draft_model(self, *, target_config, model_args):
            calls.append(("draft", target_config, model_args))
            return Draft()

        def capture_target_model(self, target_model):
            calls.append(("capture", target_model))

    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        model=_AttrDict(
            target_model_name_or_path="Qwen/Qwen3-8B",
            target_revision="b" * 40,
        )
    )
    trainer.device = "cpu"
    trainer.precision_dtype = torch.bfloat16

    draft_model, tokenizer = trainer.build_models()

    assert isinstance(draft_model, Draft)
    assert tokenizer == "tokenizer"
    for kind in ("tokenizer", "config", "target"):
        call = next(item for item in calls if item[0] == kind)
        assert call[1] == "Qwen/Qwen3-8B"
        assert call[2]["revision"] == "b" * 40
    assert any(item[0] == "capture" for item in calls)


def test_trainer_records_live_consumption_only_after_backward(monkeypatch):
    base_trainer = _load_base_trainer_module(monkeypatch)
    calls = []
    trainer = base_trainer.BaseTrainer.__new__(base_trainer.BaseTrainer)
    trainer.next_micro_step = 4
    trainer.gradient_accumulation_steps = 2
    trainer.train_dataset = SimpleNamespace(
        record_batch_consumed=lambda **kwargs: calls.append(kwargs)
    )

    trainer.record_batch_consumed(
        {
            "_live_source_indices": torch.tensor([3, 7]),
            "_live_epoch_indices": torch.tensor([1, 1]),
            "_live_source_row_ids": torch.tensor([[1] * 32, [2] * 32]),
            "_live_visit_ids": torch.tensor([[3] * 32, [4] * 32]),
        }
    )

    assert calls == [
        {
            "source_indices": [3, 7],
            "epoch_indices": [1, 1],
            "source_row_ids": ["01" * 32, "02" * 32],
            "visit_ids": ["03" * 32, "04" * 32],
            "global_step": 2,
            "micro_step": 4,
        }
    ]
    source = inspect.getsource(base_trainer.BaseTrainer.train)
    assert source.index("loss.backward()") < source.index("self.next_micro_step += 1")
    assert source.index("self.next_micro_step += 1") < source.index(
        "self.record_batch_consumed(batch)"
    )


@pytest.mark.parametrize(
    ("resume_path", "init_path", "expected_mode", "expected_micro_step"),
    [
        ("/checkpoint/step_latest", "/published", "resume", 4),
        (None, "/published", "external", 0),
        (None, None, "scratch", 0),
    ],
)
def test_trainer_initialization_keeps_resume_state_and_external_state_fresh(
    monkeypatch,
    resume_path,
    init_path,
    expected_mode,
    expected_micro_step,
):
    base_trainer = _load_base_trainer_module(monkeypatch)
    events = []
    monkeypatch.setattr(base_trainer, "init_dist", lambda _rank: ("cpu", 0, 1))
    monkeypatch.setattr(base_trainer, "is_global_main_process", lambda: False)
    monkeypatch.setattr(base_trainer, "print_on_local_main", lambda _message: None)
    monkeypatch.setattr(
        base_trainer,
        "SuspendController",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(base_trainer.training_logger, "init", lambda **_kwargs: None)
    monkeypatch.setattr(
        base_trainer,
        "discover_latest_checkpoint",
        lambda _path: resume_path,
    )
    monkeypatch.setattr(
        base_trainer,
        "load_resume_draft_model",
        lambda **_kwargs: events.append("resume-weights") or object(),
    )
    monkeypatch.setattr(
        base_trainer,
        "load_external_draft_model",
        lambda **_kwargs: events.append("external-weights") or object(),
    )
    monkeypatch.setattr(
        base_trainer,
        "load_training_state",
        lambda **_kwargs: (
            events.append("resume-state") or SimpleNamespace(next_micro_step=4)
        ),
    )

    class Optimizer:
        def __init__(self, *_args, **_kwargs):
            events.append("optimizer")

    monkeypatch.setattr(base_trainer, "BF16Optimizer", Optimizer)
    monkeypatch.setattr(
        base_trainer.BaseTrainer,
        "build_models",
        lambda self: (object(), object()),
    )
    monkeypatch.setattr(
        base_trainer.BaseTrainer,
        "_wrap_with_fsdp",
        lambda self, model: events.append("fsdp") or model,
    )
    monkeypatch.setattr(
        base_trainer.BaseTrainer,
        "build_train_dataset",
        lambda self: [object()],
    )
    monkeypatch.setattr(base_trainer, "validate_train_cache", lambda **_kwargs: None)
    monkeypatch.setattr(base_trainer.BaseTrainer, "info_board", lambda self: None)
    args = _AttrDict(
        model=_AttrDict(
            target_model_name_or_path="target",
            init_draft_name_or_path=init_path,
            init_draft_revision="a" * 40 if init_path else None,
        ),
        train=_AttrDict(
            precision="bf16",
            torch_compile=False,
            local_batch_size=1,
            global_batch_size=1,
            num_train_epochs=1,
            max_train_steps=1,
            lr=1e-4,
            warmup_ratio=0.0,
            weight_decay=0.0,
        ),
        logging=_AttrDict(
            checkpoint_dir="/checkpoint",
            logging_steps=1,
            tensorboard_dir=None,
        ),
    )

    trainer = base_trainer.BaseTrainer(0, args)

    assert trainer.next_micro_step == expected_micro_step
    if expected_mode == "resume":
        assert events == ["resume-weights", "fsdp", "optimizer", "resume-state"]
    elif expected_mode == "external":
        assert events == ["external-weights", "fsdp", "optimizer"]
    else:
        assert events == ["fsdp", "optimizer"]
