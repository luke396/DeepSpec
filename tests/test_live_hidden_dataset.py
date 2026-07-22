import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from deepspec.data.live_hidden_dataset import (
    LiveHiddenCollator,
    LiveHiddenDataset,
    ProvenanceLedger,
    build_stock_vllm_requester,
    build_visit_id,
    qwen3_final_rms_norm,
)
from deepspec.data.target_cache_dataset import validate_train_cache
from deepspec.utils.distributed import StatelessResumableDistributedSampler


SELECTION_DIGEST = "9" * 64


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_id(index: int) -> str:
    return _sha({"source": index})


def _request_id(visit_id: str) -> str:
    return _sha({"kind": "live-hidden-request", "visit_id": visit_id})


def _producer_manifest(root: Path, *, replica_id: str = "replica-0") -> dict:
    layer_manifest = {
        "target_layer_ids": [1, 9],
        "final_plane": "post-final-block-pre-final-rmsnorm",
        "hidden_size": 4,
        "dtype": "bfloat16",
    }
    identity = {
        "vllm_revision": "b" * 40,
        "container_image": "vllm@sha256:" + "c" * 64,
        "target_weights_digest": "d" * 64,
        "target_config_digest": "e" * 64,
        "dtype": "bfloat16",
        "kernel": "stock-vllm",
        "schema_version": 1,
    }
    return {
        "producer_replica_id": replica_id,
        "producer_boot_id": f"{replica_id}-boot-0",
        "shared_storage_path": str(root),
        "target_revision": "a" * 40,
        "layer_manifest": layer_manifest,
        "layer_manifest_digest": _sha(layer_manifest),
        "producer_identity": identity,
        "producer_identity_digest": _sha(identity),
    }


def _write_shared(path: Path, *, tokens=None, hidden=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokens is None:
        tokens = torch.tensor([11, 12], dtype=torch.int64)
    if hidden is None:
        hidden = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    save_file({"token_ids": tokens, "hidden_states": hidden}, path)
    return path


def _source(index: int = 0) -> dict:
    return {
        "source_row_id": _source_id(index),
        "input_ids": torch.tensor([11, 12], dtype=torch.long),
        "loss_mask": torch.tensor([0, 1], dtype=torch.long),
    }


def _dataset(
    tmp_path: Path,
    request_fn,
    *,
    manifests=None,
    max_retries=1,
    consumer_boot_id="consumer-boot-0",
    source=None,
    num_epochs=3,
):
    producer_root = tmp_path / "producer-0"
    return LiveHiddenDataset(
        source_dataset=source or [_source()],
        request_hidden_states=request_fn,
        producer_manifests=manifests or [_producer_manifest(producer_root)],
        ledger_dir=tmp_path / "ledger",
        target_model_name_or_path="Qwen/Qwen3-8B",
        target_revision="a" * 40,
        target_layer_ids=[1, 9],
        final_norm_weight=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
        final_norm_eps=1e-6,
        request_timeout_s=0.1,
        max_retries=max_retries,
        selection_policy_digest=SELECTION_DIGEST,
        num_epochs=num_epochs,
        run_id="formal-run-0",
        consumer_boot_id=consumer_boot_id,
    )


def test_live_adapter_delivers_then_trainer_consumes_after_backward(tmp_path: Path):
    shared = _write_shared(tmp_path / "producer-0" / "response.safetensors")
    original = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    dataset = _dataset(tmp_path, lambda _request: shared, num_epochs=1)

    validate_train_cache(
        train_dataset=dataset,
        draft_model=SimpleNamespace(
            target_layer_ids=[1, 9], config=SimpleNamespace(hidden_size=4)
        ),
        target_model_name_or_path="Qwen/Qwen3-8B",
    )
    item = dataset[(0, 0)]

    assert torch.equal(item["input_ids"], torch.tensor([11, 12]))
    assert torch.equal(item["loss_mask"], torch.tensor([0, 1]))
    assert torch.equal(item["target_hidden_states"], original[:, :2].flatten(1))
    expected_last = qwen3_final_rms_norm(
        original[:, 2],
        weight=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
        eps=1e-6,
    )
    assert torch.equal(item["target_last_hidden_states"], expected_last)
    assert not shared.exists()
    assert (tmp_path / "ledger" / "delivered.jsonl").exists()
    assert not (tmp_path / "ledger" / "consumed.jsonl").exists()

    batch = LiveHiddenCollator()([item])
    assert batch["_live_source_indices"].tolist() == [0]
    assert batch["_live_epoch_indices"].tolist() == [0]
    source_row_id = _source_id(0)
    visit_id = build_visit_id(
        selection_policy_digest=SELECTION_DIGEST,
        source_row_id=source_row_id,
        epoch_index=0,
    )
    assert bytes(batch["_live_source_row_ids"][0].tolist()).hex() == source_row_id
    assert bytes(batch["_live_visit_ids"][0].tolist()).hex() == visit_id
    dataset.record_batch_consumed(
        source_indices=[0],
        epoch_indices=[0],
        source_row_ids=[source_row_id],
        visit_ids=[visit_id],
        global_step=1,
        micro_step=1,
    )
    dataset.assert_training_complete(expected_replica_ids=["replica-0"])

    attempt = json.loads((tmp_path / "ledger" / "attempts.jsonl").read_text())
    assert set(attempt) == {
        "request_id",
        "row_id",
        "token_digest",
        "attempt_id",
        "producer_replica_id",
        "producer_boot_id",
        "producer_identity_digest",
        "target_revision",
        "layer_manifest_digest",
        "hidden_payload_digest",
    }


def test_prefetched_item_is_delivered_but_not_consumed(tmp_path: Path):
    first = _write_shared(tmp_path / "producer-0" / "first.safetensors")
    dataset = _dataset(tmp_path, lambda _request: first, num_epochs=1)

    dataset[(0, 0)]

    assert len((tmp_path / "ledger" / "delivered.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "ledger" / "consumed.jsonl").exists()
    with pytest.raises(ValueError, match="consumed=1"):
        dataset.assert_training_complete()


def test_final_rms_norm_matches_pinned_qwen3_operation_order():
    torch.manual_seed(0)
    hidden = (torch.randn(64, 32) * 10).to(torch.bfloat16)
    weight = torch.randn(32).to(torch.bfloat16)
    hidden_fp32 = hidden.to(torch.float32)
    expected = weight * (
        hidden_fp32 * torch.rsqrt(hidden_fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
    ).to(hidden.dtype)

    actual = qwen3_final_rms_norm(hidden, weight=weight, eps=1e-6)

    assert torch.equal(actual, expected)


def _ledger(tmp_path: Path, *, boot="consumer-boot-0") -> ProvenanceLedger:
    return ProvenanceLedger(
        tmp_path / "ledger",
        run_id="formal-run-0",
        consumer_boot_id=boot,
        selection_policy_digest=SELECTION_DIGEST,
    )


def _attempt(*, request_id: str, row_id: str, attempt_id: str) -> dict:
    return {
        "request_id": request_id,
        "row_id": row_id,
        "token_digest": "a" * 64,
        "attempt_id": attempt_id,
        "producer_replica_id": "replica-0",
        "producer_boot_id": "boot-0",
        "producer_identity_digest": "b" * 64,
        "target_revision": "c" * 40,
        "layer_manifest_digest": "d" * 64,
        "hidden_payload_digest": "e" * 64,
    }


def test_ledger_folds_only_identical_response_attempts(tmp_path: Path):
    ledger = _ledger(tmp_path)
    source_row_id = _source_id(0)
    row_id = build_visit_id(
        selection_policy_digest=SELECTION_DIGEST,
        source_row_id=source_row_id,
        epoch_index=0,
    )
    request_id = _request_id(row_id)
    ledger.record_request(
        request_id=request_id,
        row_id=row_id,
        source_row_id=source_row_id,
        epoch_index=0,
        token_digest="a" * 64,
    )
    first = ledger.record_response(
        _attempt(request_id=request_id, row_id=row_id, attempt_id="attempt-1")
    )
    second = ledger.record_response(
        _attempt(request_id=request_id, row_id=row_id, attempt_id="attempt-2")
    )

    assert first == second == "attempt-1"
    assert len((tmp_path / "ledger" / "attempts.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ledger.record_response(
            {
                **_attempt(
                    request_id=request_id, row_id=row_id, attempt_id="attempt-3"
                ),
                "hidden_payload_digest": "f" * 64,
            }
        )


@pytest.mark.parametrize(
    ("tokens", "hidden", "message"),
    [
        (torch.tensor([11, 13]), None, "token"),
        (None, torch.zeros(2, 2, 4, dtype=torch.bfloat16), "layer"),
        (
            None,
            torch.tensor(
                [
                    [[0.0, 1.0, 2.0, 3.0]] * 3,
                    [[0.0, 1.0, float("nan"), 3.0]] * 3,
                ],
                dtype=torch.bfloat16,
            ),
            "finite",
        ),
    ],
)
def test_live_adapter_records_then_rejects_payload_drift(
    tmp_path: Path, tokens, hidden, message
):
    shared = _write_shared(
        tmp_path / "producer-0" / "response.safetensors",
        tokens=tokens,
        hidden=hidden,
    )
    dataset = _dataset(tmp_path, lambda _request: shared)

    with pytest.raises(ValueError, match=message):
        dataset[(0, 0)]
    assert not shared.exists()
    assert len((tmp_path / "ledger" / "attempts.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "ledger" / "accepted-attempts.jsonl").exists()


def test_live_adapter_rejects_unknown_boot_and_retry_exhaustion(tmp_path: Path):
    unknown = _write_shared(tmp_path / "unknown" / "response.safetensors")
    dataset = _dataset(tmp_path / "unknown-case", lambda _request: unknown)
    with pytest.raises(ValueError, match="registered producer boot"):
        dataset[(0, 0)]

    calls = 0

    def unavailable(_request):
        nonlocal calls
        calls += 1
        raise TimeoutError("producer unavailable")

    dataset = _dataset(tmp_path / "retry-case", unavailable, max_retries=2)
    with pytest.raises(TimeoutError, match="producer unavailable"):
        dataset[(0, 0)]
    assert calls == 3


def test_live_adapter_retries_one_visit_with_deterministic_attempt_ids(tmp_path: Path):
    shared = _write_shared(tmp_path / "producer-0" / "response.safetensors")
    requests = []

    def flaky(request):
        requests.append(request)
        if len(requests) == 1:
            raise TimeoutError("transient")
        return shared

    dataset = _dataset(tmp_path, flaky, max_retries=1, num_epochs=1)
    dataset[(0, 0)]

    assert requests[0]["request_id"] == requests[1]["request_id"]
    assert requests[0]["row_id"] == requests[1]["row_id"]
    assert requests[0]["attempt_id"] != requests[1]["attempt_id"]
    assert all(len(request["attempt_id"]) == 64 for request in requests)


def test_producer_catalog_requires_same_identity_and_exact_layer_manifest(
    tmp_path: Path,
):
    first = _producer_manifest(tmp_path / "producer-0", replica_id="replica-0")
    second = _producer_manifest(tmp_path / "producer-1", replica_id="replica-1")
    second["producer_identity"]["kernel"] = "drifted"
    second["producer_identity_digest"] = _sha(second["producer_identity"])
    with pytest.raises(ValueError, match="same producer identity"):
        _dataset(tmp_path, lambda _request: None, manifests=[first, second])

    second = _producer_manifest(tmp_path / "producer-1", replica_id="replica-1")
    second["layer_manifest"]["target_layer_ids"] = [1, 10]
    second["layer_manifest_digest"] = _sha(second["layer_manifest"])
    with pytest.raises(ValueError, match="layer manifest"):
        _dataset(tmp_path, lambda _request: None, manifests=[first, second])

    invalid = _producer_manifest(tmp_path / "producer-invalid")
    invalid["producer_identity"]["container_image"] = "vllm:latest"
    invalid["producer_identity_digest"] = _sha(invalid["producer_identity"])
    with pytest.raises(ValueError, match="immutable digest"):
        _dataset(tmp_path, lambda _request: None, manifests=[invalid])


def test_stock_vllm_requester_is_lazy_and_recreates_client_after_pid_change(
    monkeypatch, tmp_path
):
    calls = []
    pid = 100
    response = types.SimpleNamespace(
        kv_transfer_params={
            "hidden_states_path": str(tmp_path / "response.safetensors")
        }
    )

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    class OpenAI:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.completions = Completions()

    module = types.ModuleType("openai")
    module.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setattr("deepspec.data.live_hidden_dataset.os.getpid", lambda: pid)
    requester = build_stock_vllm_requester(
        endpoint="http://127.0.0.1:8000/v1", model="Qwen/Qwen3-8B"
    )
    assert calls == []

    path = requester(
        {"request_id": "request-0", "input_ids": [11, 12], "timeout_s": 3.0}
    )

    assert path == str(tmp_path / "response.safetensors")
    assert calls[0] == (
        "client",
        {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
            "max_retries": 0,
        },
    )
    assert calls[1] == {
        "model": "Qwen/Qwen3-8B",
        "prompt": [11, 12],
        "max_tokens": 1,
        "extra_body": {"return_token_ids": True},
        "extra_headers": {"x-request-id": "request-0"},
        "timeout": 3.0,
    }
    requester({"request_id": "request-1", "input_ids": [11, 12], "timeout_s": 3.0})
    assert sum(call[0] == "client" for call in calls if isinstance(call, tuple)) == 1

    pid = 101
    requester({"request_id": "request-2", "input_ids": [11, 12], "timeout_s": 3.0})
    assert sum(call[0] == "client" for call in calls if isinstance(call, tuple)) == 2


def test_sampler_preserves_offline_int_indices_and_live_epoch_tuple_resume():
    offline = list(range(6))
    offline_sampler = StatelessResumableDistributedSampler(
        dataset=offline,
        num_replicas=2,
        rank=0,
        total_size=6,
        start_global_offset_samples=0,
        num_samples=5,
    )
    assert all(isinstance(index, int) for index in offline_sampler)

    class Live(list):
        requires_epoch_index = True

    live = Live(range(6))
    full = list(
        StatelessResumableDistributedSampler(
            dataset=live,
            num_replicas=2,
            rank=0,
            total_size=6,
            start_global_offset_samples=0,
            num_samples=7,
        )
    )
    resumed = list(
        StatelessResumableDistributedSampler(
            dataset=live,
            num_replicas=2,
            rank=0,
            total_size=6,
            start_global_offset_samples=2,
            num_samples=5,
        )
    )
    assert resumed == full[2:7]
    assert [epoch for _, epoch in full] == [0, 0, 0, 1, 1, 1, 2]


def test_mixed_consumer_boot_or_restart_is_rejected(tmp_path: Path):
    _dataset(tmp_path, lambda _request: None, consumer_boot_id="boot-a")

    with pytest.raises(ValueError, match="mixed consumer boot or restarted run"):
        _dataset(tmp_path, lambda _request: None, consumer_boot_id="boot-b")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_final_reader_proves_exact_three_epoch_1600_source_coverage(tmp_path: Path):
    ledger = _ledger(tmp_path)
    source_ids = [_source_id(index) for index in range(1600)]
    requests = []
    attempts = []
    accepted = []
    delivered = []
    consumed = []
    for epoch_index in range(3):
        for source_row_id in source_ids:
            row_id = build_visit_id(
                selection_policy_digest=SELECTION_DIGEST,
                source_row_id=source_row_id,
                epoch_index=epoch_index,
            )
            request_id = _request_id(row_id)
            attempt_id = _sha({"attempt": row_id})
            request = {
                "request_id": request_id,
                "row_id": row_id,
                "source_row_id": source_row_id,
                "epoch_index": epoch_index,
                "token_digest": "a" * 64,
                "run_id": "formal-run-0",
                "consumer_boot_id": "consumer-boot-0",
            }
            accepted_row = {
                **_attempt(request_id=request_id, row_id=row_id, attempt_id=attempt_id),
                "accepted_attempt_id": attempt_id,
            }
            del accepted_row["attempt_id"]
            delivery = {
                **{
                    key: value
                    for key, value in request.items()
                    if key != "token_digest"
                },
                "accepted_attempt_id": attempt_id,
            }
            requests.append(request)
            attempts.append(
                _attempt(request_id=request_id, row_id=row_id, attempt_id=attempt_id)
            )
            accepted.append(accepted_row)
            delivered.append(delivery)
            consumed.append({**delivery, "global_step": 1, "micro_step": 1})
    _write_jsonl(ledger.requests_path, requests)
    _write_jsonl(ledger.attempts_path, attempts)
    _write_jsonl(ledger.accepted_path, accepted)
    _write_jsonl(ledger.delivered_path, delivered)
    _write_jsonl(ledger.consumed_path, consumed)

    ledger.assert_training_complete(
        source_ids, num_epochs=3, expected_replica_ids=["replica-0"]
    )

    consumed[-1]["consumer_boot_id"] = "restarted-boot"
    _write_jsonl(ledger.consumed_path, consumed)
    with pytest.raises(ValueError, match="run/boot"):
        ledger.assert_training_complete(source_ids, num_epochs=3)
