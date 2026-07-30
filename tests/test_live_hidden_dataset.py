import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

import deepspec.data.live_hidden_dataset as live_module
import deepspec.data.jsonl_dataset as jsonl_module
from deepspec.data.live_hidden_dataset import LiveHiddenDataset
from deepspec.data.live_hidden_prefilter import (
    map_live_hidden_index,
    prepare_live_hidden_prefilter,
)
from deepspec.utils.config import ConfigNode


NORM_WEIGHT = torch.arange(1, 9, dtype=torch.bfloat16) / 8


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )

    @staticmethod
    def _encode(text, max_length=None):
        token_ids = [index + 1 for index in range(len(text))]
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def __call__(
        self,
        text,
        *,
        max_length,
        truncation,
        return_tensors,
        add_special_tokens,
    ):
        assert truncation is True
        assert return_tensors == "pt"
        assert add_special_tokens is False
        token_ids = self._encode(text, max_length=max_length)
        return SimpleNamespace(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
        )

    def encode(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
        max_length,
    ):
        assert add_special_tokens is False
        assert truncation is True
        return self._encode(text, max_length=max_length)


class FakeCompletions:
    def __init__(self, hidden_dir: Path):
        self.hidden_dir = hidden_dir
        self.calls = []
        self.mode = "valid"
        self.model_id = "Qwen/Qwen3-8B"
        self.last_hidden_states = None
        self.last_path = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        token_ids = list(kwargs["prompt"])
        response_token_ids = list(token_ids)
        if self.mode == "token_mismatch":
            response_token_ids[-1] += 1

        num_layers = 2 if self.mode == "shape" else 3
        hidden_states = torch.arange(
            len(token_ids) * num_layers * 8,
            dtype=torch.bfloat16,
        ).reshape(len(token_ids), num_layers, 8)
        if self.mode == "finite":
            hidden_states[0, 0, 0] = float("nan")
        self.last_hidden_states = hidden_states.clone()

        output_dir = (
            self.hidden_dir.parent / "foreign"
            if self.mode == "foreign_path"
            else self.hidden_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hs_request.safetensors"
        save_file(
            {
                "token_ids": torch.tensor(response_token_ids, dtype=torch.long),
                "hidden_states": hidden_states,
            },
            output_path,
        )
        self.last_path = output_path
        return SimpleNamespace(
            choices=[SimpleNamespace(prompt_token_ids=response_token_ids)],
            kv_transfer_params={"hidden_states_path": str(output_path)},
        )


class FakeOpenAI:
    def __init__(self, completions):
        self.completions = completions
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=completions.model_id)]
            )
        )


def _write_dataset(path: Path, count=1):
    rows = [
        {
            "id": f"row-{index}",
            "conversations": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_dspark_trainer(monkeypatch):
    stubs = {
        "deepspec.modeling.dspark.gemma4": {"Gemma4DSparkModel": object},
        "deepspec.modeling.dspark.gemma4.config": {
            "build_draft_config": lambda **_: None
        },
        "deepspec.modeling.dspark.loss": {"compute_dspark_loss": lambda **_: None},
        "deepspec.modeling.dspark.qwen3": {"Qwen3DSparkModel": object},
        "deepspec.modeling.dspark.qwen3.config": {
            "build_draft_config": lambda **_: None
        },
        "deepspec.trainer.base_trainer": {"BaseTrainer": object},
    }
    for module_name, attributes in stubs.items():
        module = types.ModuleType(module_name)
        for name, value in attributes.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    spec = importlib.util.spec_from_file_location(
        "deepspec_dspark_trainer_under_test",
        Path(__file__).parents[1] / "deepspec/trainer/dspark_trainer.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3DSparkTrainer


def _dataset(monkeypatch, tmp_path, *, expected_num_samples=1):
    monkeypatch.setattr(jsonl_module, "CACHE_DIR", str(tmp_path / "index-cache"))
    data_path = tmp_path / "regen.canonical.jsonl"
    _write_dataset(data_path)
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    completions = FakeCompletions(hidden_dir)
    fake_client = FakeOpenAI(completions)
    monkeypatch.setattr(live_module.openai, "OpenAI", lambda **_: fake_client)
    dataset = LiveHiddenDataset(
        data_path=data_path,
        tokenizer=FakeTokenizer(),
        chat_template="qwen",
        max_length=32768,
        min_loss_tokens=1,
        vllm_endpoint="http://hidden-router:8000/v1",
        vllm_model="Qwen/Qwen3-8B",
        hidden_states_path=hidden_dir,
        target_layer_ids=[1, 9],
        hidden_size=8,
        final_norm_weight=NORM_WEIGHT,
        final_norm_eps=1e-6,
        expected_num_samples=expected_num_samples,
    )
    return dataset, completions


def test_direct_jsonl_tokenization_hidden_request_and_qwen_final_norm(
    monkeypatch, tmp_path
):
    dataset, completions = _dataset(monkeypatch, tmp_path)

    result = dataset[0]

    assert len(dataset) == 1
    assert set(result) == {
        "input_ids",
        "loss_mask",
        "target_hidden_states",
        "target_last_hidden_states",
    }
    assert completions.calls == [
        {
            "model": "Qwen/Qwen3-8B",
            "prompt": result["input_ids"].tolist(),
            "max_tokens": 1,
            "extra_body": {"return_token_ids": True},
            "timeout": 120,
        }
    ]
    assert result["loss_mask"].sum().item() >= 1
    assert torch.equal(
        result["target_hidden_states"],
        completions.last_hidden_states[:, :-1].flatten(1),
    )

    norm = Qwen3RMSNorm(8, eps=1e-6).to(dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(NORM_WEIGHT)
    assert torch.equal(
        result["target_last_hidden_states"],
        norm(completions.last_hidden_states[:, -1]),
    )
    assert completions.last_path is not None
    assert not completions.last_path.exists()


def test_sample_count_mismatch_fails_before_training(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="sample count mismatch"):
        _dataset(monkeypatch, tmp_path, expected_num_samples=2)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("token_mismatch", "token IDs mismatch"),
        ("shape", "shape mismatch"),
        ("finite", "finite check failed"),
        ("foreign_path", "outside configured hidden-state directory"),
    ],
)
def test_invalid_hidden_response_fails_training(
    monkeypatch, tmp_path, failure, message
):
    dataset, completions = _dataset(monkeypatch, tmp_path)
    completions.mode = failure

    with pytest.raises((RuntimeError, ValueError), match=message):
        dataset[0]
    if failure != "foreign_path":
        assert completions.last_path is not None
        assert not completions.last_path.exists()


def test_vllm_model_identity_mismatch_fails_before_hidden_request(
    monkeypatch, tmp_path
):
    dataset, completions = _dataset(monkeypatch, tmp_path)
    completions.model_id = "different-model"

    with pytest.raises(ValueError, match="model identity mismatch"):
        dataset[0]
    assert completions.calls == []


def test_live_prefilter_excludes_rejections_without_copying_conversations(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(jsonl_module, "CACHE_DIR", str(tmp_path / "index-cache"))
    data_path = tmp_path / "regen.canonical.jsonl"
    rows = [
        {
            "id": "accepted",
            "conversations": [
                {"role": "user", "content": "short question"},
                {"role": "assistant", "content": "usable answer"},
            ],
        },
        {
            "id": "reject-no-answer",
            "conversations": [
                {"role": "user", "content": "private user-only content"}
            ],
        },
        {
            "id": "reject-truncated",
            "conversations": [
                {"role": "user", "content": "private long prompt " + "x" * 300},
                {"role": "assistant", "content": "answer after truncation"},
            ],
        },
    ]
    _write_rows(data_path, rows)

    prefilter = prepare_live_hidden_prefilter(
        source_path=data_path,
        output_dir=tmp_path / "prefilter",
        tokenizer=FakeTokenizer(),
        chat_template="qwen",
        max_length=200,
        min_loss_tokens=1,
        expected_num_samples=3,
        target_model_name_or_path="Qwen/Qwen3-8B",
        target_revision="b" * 40,
        replace_existing=True,
    )

    assert prefilter.source_samples == 3
    assert prefilter.accepted_samples == 1
    assert prefilter.rejected_samples == 2
    assert prefilter.rejected_indices == (1, 2)
    assert {path.name for path in (tmp_path / "prefilter").iterdir()} == {
        "manifest.json",
        "rejected.jsonl",
    }
    rejected = [
        json.loads(line)
        for line in prefilter.rejected_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["id"] for item in rejected] == [
        "reject-no-answer",
        "reject-truncated",
    ]
    assert all(
        set(item)
        <= {
            "source_index",
            "id",
            "reason",
            "sequence_tokens",
            "loss_tokens",
            "error_type",
        }
        for item in rejected
    )
    rejected_text = prefilter.rejected_path.read_text(encoding="utf-8")
    assert "private user-only content" not in rejected_text
    assert "private long prompt" not in rejected_text

    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    completions = FakeCompletions(hidden_dir)
    fake_client = FakeOpenAI(completions)
    monkeypatch.setattr(live_module.openai, "OpenAI", lambda **_: fake_client)
    dataset = LiveHiddenDataset(
        data_path=data_path,
        tokenizer=FakeTokenizer(),
        chat_template="qwen",
        max_length=200,
        min_loss_tokens=1,
        vllm_endpoint="http://hidden-router:8000/v1",
        vllm_model="Qwen/Qwen3-8B",
        hidden_states_path=hidden_dir,
        target_layer_ids=[1, 9],
        hidden_size=8,
        final_norm_weight=NORM_WEIGHT,
        final_norm_eps=1e-6,
        expected_num_samples=3,
        rejected_indices=prefilter.rejected_indices,
    )

    assert len(dataset) == 1
    assert dataset._source_index(0) == 0
    assert dataset[0]["loss_mask"].sum().item() >= 1
    assert len(completions.calls) == 1


@pytest.mark.parametrize(
    ("index", "expected_source_index"),
    [(0, 1), (1, 3), (2, 4), (3, 6)],
)
def test_live_prefilter_maps_logical_indices_around_rejections(
    index, expected_source_index
):
    assert (
        map_live_hidden_index(
            index,
            source_samples=7,
            rejected_indices=(0, 2, 5),
        )
        == expected_source_index
    )


def test_live_prefilter_rejects_resume_artifact_drift(tmp_path):
    data_path = tmp_path / "regen.canonical.jsonl"
    _write_dataset(data_path)
    output_dir = tmp_path / "prefilter"
    kwargs = dict(
        source_path=data_path,
        output_dir=output_dir,
        tokenizer=FakeTokenizer(),
        chat_template="qwen",
        max_length=32768,
        min_loss_tokens=1,
        expected_num_samples=1,
        target_model_name_or_path="Qwen/Qwen3-8B",
        target_revision="b" * 40,
    )
    prepare_live_hidden_prefilter(replace_existing=True, **kwargs)

    _write_rows(
        data_path,
        [
            {
                "id": "changed",
                "conversations": [
                    {"role": "user", "content": "changed question"},
                    {"role": "assistant", "content": "changed answer"},
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="no longer matches"):
        prepare_live_hidden_prefilter(replace_existing=False, **kwargs)


def test_live_prefilter_requires_existing_artifacts_on_resume(tmp_path):
    data_path = tmp_path / "regen.canonical.jsonl"
    _write_dataset(data_path)

    with pytest.raises(ValueError, match="requires its existing"):
        prepare_live_hidden_prefilter(
            source_path=data_path,
            output_dir=tmp_path / "missing-prefilter",
            tokenizer=FakeTokenizer(),
            chat_template="qwen",
            max_length=32768,
            min_loss_tokens=1,
            expected_num_samples=1,
            target_model_name_or_path="Qwen/Qwen3-8B",
            target_revision="b" * 40,
            replace_existing=False,
        )


def test_dspark_trainer_prefilters_before_building_training_schedule(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(jsonl_module, "CACHE_DIR", str(tmp_path / "index-cache"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    trainer_cls = _load_dspark_trainer(monkeypatch)
    data_path = tmp_path / "regen.canonical.jsonl"
    _write_rows(
        data_path,
        [
            {
                "id": "accepted",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            {
                "id": "rejected",
                "conversations": [
                    {"role": "user", "content": "no assistant response"}
                ],
            },
        ],
    )
    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()

    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(
        model=ConfigNode(
            {
                "target_model_name_or_path": "Qwen/Qwen3-8B",
                "target_revision": "b" * 40,
            }
        ),
        data=ConfigNode(
            {
                "target_cache_path": None,
                "train_jsonl_path": str(data_path),
                "vllm_endpoint": "http://hidden-router:8000/v1",
                "vllm_model": "Qwen/Qwen3-8B",
                "hidden_states_path": str(hidden_dir),
                "expected_num_samples": 2,
                "chat_template": "qwen",
                "max_length": 32768,
                "min_loss_tokens": 1,
            }
        ),
    )
    trainer.tokenizer = FakeTokenizer()
    trainer.checkpoint_dir_root = str(tmp_path / "checkpoints")
    trainer.global_rank = 0
    trainer.resume_checkpoint_dir = None
    trainer.draft_model = SimpleNamespace(
        target_layer_ids=[1, 9],
        config=SimpleNamespace(hidden_size=8),
    )
    trainer._target_final_norm_weight = NORM_WEIGHT
    trainer._target_final_norm_eps = 1e-6

    dataset = trainer.build_train_dataset()

    assert len(dataset.dataset) == 2
    assert len(dataset) == 1
    assert dataset.rejected_indices == (1,)
