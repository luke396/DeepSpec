import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

import deepspec.data.live_hidden_adapter as live_module
from deepspec.data.live_hidden_adapter import LiveHiddenAdapter


REVISION = "b" * 40
NORM_WEIGHT = torch.arange(1, 9, dtype=torch.bfloat16) / 8


def _sample():
    return {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1], dtype=torch.uint8),
        "hidden_states": torch.arange(48, dtype=torch.bfloat16).reshape(3, 16),
        "verifier_last_hidden_states": torch.arange(24, dtype=torch.bfloat16).reshape(
            3, 8
        ),
        "lengths": torch.tensor([3]),
        "position_ids": torch.arange(3),
    }


class FakeArrowDataset:
    init_kwargs = None
    next_sample = None
    size = 1600

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def __len__(self):
        return type(self).size

    def __getitem__(self, index):
        assert index == 0
        return type(self).next_sample


def _adapter(monkeypatch, tmp_path, *, size=1600):
    monkeypatch.setattr(live_module, "ArrowDataset", FakeArrowDataset)
    FakeArrowDataset.next_sample = _sample()
    FakeArrowDataset.size = size
    return LiveHiddenAdapter(
        datapath=tmp_path,
        max_length=32768,
        vllm_endpoint="http://hidden-router:8000/v1",
        target_model_name_or_path="Qwen/Qwen3-8B",
        target_revision=REVISION,
        target_layer_ids=[1, 9],
        hidden_size=8,
        final_norm_weight=NORM_WEIGHT,
        final_norm_eps=1e-6,
        expected_num_samples=1600,
    )


def test_single_endpoint_mapping_and_qwen_final_norm(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)

    assert len(adapter) == 1600
    assert FakeArrowDataset.init_kwargs["vllm_endpoint"] == (
        "http://hidden-router:8000/v1"
    )
    assert "model" not in FakeArrowDataset.init_kwargs
    assert FakeArrowDataset.init_kwargs["on_missing"] == "generate"
    assert FakeArrowDataset.init_kwargs["on_generate"] == "delete"

    result = adapter[0]
    assert set(result) == {
        "input_ids",
        "loss_mask",
        "target_hidden_states",
        "target_last_hidden_states",
    }
    assert torch.equal(result["target_hidden_states"], _sample()["hidden_states"])

    norm = Qwen3RMSNorm(8, eps=1e-6).to(dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(NORM_WEIGHT)
    assert torch.equal(
        result["target_last_hidden_states"],
        norm(_sample()["verifier_last_hidden_states"]),
    )


def test_missing_hidden_response_fails_training(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch, tmp_path)
    FakeArrowDataset.next_sample = None

    with pytest.raises(RuntimeError, match="returned no sample"):
        adapter[0]


def test_sample_count_mismatch_fails_before_training(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="sample count mismatch"):
        _adapter(monkeypatch, tmp_path, size=1599)


@pytest.mark.parametrize("failure", ["shape", "finite"])
def test_invalid_hidden_response_fails_training(monkeypatch, tmp_path, failure):
    adapter = _adapter(monkeypatch, tmp_path)
    sample = _sample()
    if failure == "shape":
        sample["hidden_states"] = torch.zeros(3, 8, dtype=torch.bfloat16)
    else:
        sample["verifier_last_hidden_states"][0, 0] = float("nan")
    FakeArrowDataset.next_sample = sample

    with pytest.raises(ValueError, match=failure):
        adapter[0]
