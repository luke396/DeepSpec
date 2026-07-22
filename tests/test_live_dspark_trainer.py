import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load_source_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def live_trainer_module(monkeypatch):
    class FakeQwen3DSparkTrainer:
        def train(self):
            self.parent_train_called = True

    trainer_module = types.ModuleType("deepspec.trainer.dspark_trainer")
    trainer_module.Qwen3DSparkTrainer = FakeQwen3DSparkTrainer
    monkeypatch.setitem(sys.modules, "deepspec.trainer.dspark_trainer", trainer_module)
    trainer_package = types.ModuleType("deepspec.trainer")
    trainer_package.__path__ = []
    trainer_package.Qwen3DSparkTrainer = FakeQwen3DSparkTrainer
    monkeypatch.setitem(sys.modules, "deepspec.trainer", trainer_package)

    loss_module = types.ModuleType("deepspec.modeling.dspark.loss")
    loss_module._collect_local_terms = lambda **_kwargs: ({}, False)
    loss_module.compute_dspark_loss = lambda **_kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "deepspec.modeling.dspark.loss", loss_module)

    checkpoint_module = types.ModuleType("deepspec.trainer.ckpt_manager")
    checkpoint_module.discover_latest_checkpoint = lambda _path: None
    monkeypatch.setitem(sys.modules, "deepspec.trainer.ckpt_manager", checkpoint_module)

    utils_module = types.ModuleType("deepspec.utils")
    utils_module.__path__ = []
    utils_module.print_on_global_main = lambda _message: None
    monkeypatch.setitem(sys.modules, "deepspec.utils", utils_module)
    constant_module = types.ModuleType("deepspec.utils.constant")
    constant_module.BASE_CKPT_DIR = "/checkpoints"
    constant_module.BASE_TB_DIR = "/tensorboard"
    constant_module.QWEN_3_8B = "Qwen/Qwen3-8B"
    monkeypatch.setitem(sys.modules, "deepspec.utils.constant", constant_module)

    return _load_source_module(
        "deepspec.trainer.live_dspark_trainer",
        "deepspec/trainer/live_dspark_trainer.py",
    )


def test_generic_config_stays_scratch_and_live_config_owns_exact_caller(
    monkeypatch, live_trainer_module
):
    generic = _load_source_module(
        "task94_generic_config", "config/dspark/dspark_qwen3_8b.py"
    )
    fresh = _load_source_module(
        "task94_fresh_config", "config/dspark/dspark_qwen3_8b_fresh.py"
    )
    live = _load_source_module(
        "task94_live_config", "config/dspark/dspark_qwen3_8b_live.py"
    )

    assert "init_draft_name_or_path" not in generic.model
    assert "init_draft_revision" not in generic.model
    assert fresh.model["init_draft_name_or_path"] == (
        "deepseek-ai/dspark_qwen3_8b_block7"
    )
    assert fresh.model["init_draft_revision"] == (
        "03326e5043815da1f81b109078b2889737c26017"
    )
    assert fresh.train["trainer_cls"] is live_trainer_module.Qwen3DSparkTrainer
    assert fresh.train["max_train_steps"] == 1
    assert live.model["init_draft_name_or_path"] == (
        "deepseek-ai/dspark_qwen3_8b_block7"
    )
    assert live.model["init_draft_revision"] == (
        "03326e5043815da1f81b109078b2889737c26017"
    )
    assert live.train["trainer_cls"] is live_trainer_module.LiveQwen3DSparkTrainer
    assert live.train["max_train_steps"] == 150
    assert live.logging["checkpointing_steps"] == 50
    assert {
        "source_manifest_path",
        "producer_manifests_path",
        "ledger_dir",
        "vllm_endpoint",
        "selection_policy_digest",
        "run_id",
        "consumer_boot_id",
        "parity_cache_path",
    } <= set(live.data)


def test_tracked_live_caller_loads_exact_manifests_and_assembles_dataset(
    monkeypatch, tmp_path: Path, live_trainer_module
):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "source_row_id": "a" * 64,
                "input_ids": [11, 12],
                "loss_mask": [0, 1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    producer_path = tmp_path / "producers.json"
    producer_path.write_text(
        json.dumps([{"producer_replica_id": "replica-0"}]), encoding="utf-8"
    )
    captured = {}
    monkeypatch.setattr(
        live_trainer_module,
        "build_stock_vllm_requester",
        lambda **kwargs: captured.setdefault("requester", kwargs) or object(),
    )

    class Dataset:
        def __init__(self, **kwargs):
            captured["dataset"] = kwargs

    monkeypatch.setattr(live_trainer_module, "LiveHiddenDataset", Dataset)
    trainer = live_trainer_module.LiveQwen3DSparkTrainer.__new__(
        live_trainer_module.LiveQwen3DSparkTrainer
    )
    trainer._target_final_norm_weight = torch.ones(4, dtype=torch.bfloat16)
    trainer._target_final_norm_eps = 1e-6
    trainer._parity_dataset = None
    trainer.draft_model = SimpleNamespace(target_layer_ids=[1, 9])
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            target_model_name_or_path="Qwen/Qwen3-8B",
            target_revision="b" * 40,
        ),
        train=SimpleNamespace(num_train_epochs=3),
        data=SimpleNamespace(
            source_manifest_path=str(source_path),
            producer_manifests_path=str(producer_path),
            ledger_dir=str(tmp_path / "ledger"),
            vllm_endpoint="http://127.0.0.1:8000/v1",
            vllm_model="Qwen/Qwen3-8B",
            request_timeout_s=60.0,
            max_retries=2,
            selection_policy_digest="c" * 64,
            run_id="run-0",
            consumer_boot_id="boot-0",
            parity_cache_path=None,
        ),
    )

    result = trainer.build_train_dataset()

    assert isinstance(result, Dataset)
    assert captured["requester"] == {
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": "Qwen/Qwen3-8B",
    }
    assert captured["dataset"]["source_dataset"][0]["source_row_id"] == "a" * 64
    assert captured["dataset"]["producer_manifests"] == [
        {"producer_replica_id": "replica-0"}
    ]
    assert captured["dataset"]["num_epochs"] == 3
    assert trainer._expected_replica_ids == ["replica-0"]


def test_live_trainer_captures_target_norm_and_runs_exact_parity_gate(
    monkeypatch, live_trainer_module
):
    trainer = live_trainer_module.LiveQwen3DSparkTrainer.__new__(
        live_trainer_module.LiveQwen3DSparkTrainer
    )
    weight = torch.tensor([1.0, 2.0])
    trainer.capture_target_model(
        SimpleNamespace(
            model=SimpleNamespace(
                norm=SimpleNamespace(weight=weight, variance_epsilon=1e-6)
            )
        )
    )
    assert trainer._target_final_norm_weight.dtype == torch.bfloat16
    assert trainer._target_final_norm_weight.device.type == "cpu"
    assert trainer._target_final_norm_eps == 1e-6

    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
    last_hidden = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.bfloat16)
    draft_logits = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    live_outputs = SimpleNamespace(draft_logits=draft_logits)
    oracle_outputs = SimpleNamespace(draft_logits=draft_logits.clone())
    model_calls = []
    trainer.model = lambda **kwargs: model_calls.append(kwargs) or oracle_outputs
    trainer.device = torch.device("cpu")
    trainer.draft_model = SimpleNamespace(
        target_layer_ids=[1], config=SimpleNamespace(hidden_size=2)
    )
    trainer.args = SimpleNamespace(
        data=SimpleNamespace(
            hidden_cosine_min=0.999,
            hidden_relative_rmse_max=0.01,
            loss_symmetric_relative_error_max=0.01,
        ),
        model=SimpleNamespace(loss_decay_gamma=4.0, l1_loss_alpha=0.9),
    )
    trainer._parity_dataset = [
        {
            "input_ids": torch.tensor([11, 12]),
            "loss_mask": torch.tensor([0, 1], dtype=torch.uint8),
            "target_hidden_states": hidden[0],
            "target_last_hidden_states": last_hidden[0],
        }
    ]
    trainer._parity_count = 0
    monkeypatch.setattr(
        live_trainer_module,
        "_local_loss_components",
        lambda *_args, **_kwargs: {
            "ce": torch.tensor(1.0),
            "l1": torch.tensor(2.0),
            "confidence": torch.tensor(3.0),
        },
    )
    batch = {
        "input_ids": torch.tensor([[11, 12]]),
        "loss_mask": torch.tensor([[0, 1]], dtype=torch.uint8),
        "attention_mask": torch.tensor([[1, 1]]),
        "target_hidden_states": hidden,
        "target_last_hidden_states": last_hidden,
        "_live_source_indices": torch.tensor([0]),
    }

    trainer._run_parity_gate(batch=batch, live_outputs=live_outputs)

    assert trainer._parity_count == 1
    assert len(model_calls) == 1
    assert set(model_calls[0]) == {
        "input_ids",
        "target_hidden_states",
        "loss_mask",
        "target_last_hidden_states",
    }


def test_live_train_always_calls_final_reader(live_trainer_module):
    calls = []
    trainer = live_trainer_module.LiveQwen3DSparkTrainer.__new__(
        live_trainer_module.LiveQwen3DSparkTrainer
    )
    trainer._parity_dataset = None
    trainer._expected_replica_ids = ["replica-0", "replica-1"]
    trainer.train_dataset = SimpleNamespace(
        assert_training_complete=lambda **kwargs: calls.append(kwargs)
    )

    trainer.train()

    assert trainer.parent_train_called is True
    assert calls == [{"expected_replica_ids": ["replica-0", "replica-1"]}]


def test_live_source_manifest_rejects_shape_or_field_drift(
    tmp_path: Path, live_trainer_module
):
    path = tmp_path / "source.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_row_id": "a" * 64,
                "input_ids": [1, 2],
                "loss_mask": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loss_mask"):
        live_trainer_module.load_live_source_manifest(path)

    path.write_text(
        json.dumps(
            {
                "source_row_id": "a" * 64,
                "input_ids": [1],
                "loss_mask": [1],
                "unexpected": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact fields"):
        live_trainer_module.load_live_source_manifest(path)
