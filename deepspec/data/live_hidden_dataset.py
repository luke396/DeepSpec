"""Live stock-vLLM shared-file adapter for DeepSpec's existing batch ABI."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file

from deepspec.data.target_cache_dataset import CacheCollator


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ATTEMPT_FIELDS = {
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
_RUN_FIELDS = {"run_id", "consumer_boot_id", "selection_policy_digest"}
_REQUEST_FIELDS = {
    "request_id",
    "row_id",
    "source_row_id",
    "epoch_index",
    "token_digest",
    "run_id",
    "consumer_boot_id",
}
_DELIVERED_FIELDS = {
    "request_id",
    "row_id",
    "source_row_id",
    "epoch_index",
    "accepted_attempt_id",
    "run_id",
    "consumer_boot_id",
}
_CONSUMED_FIELDS = _DELIVERED_FIELDS | {"global_step", "micro_step"}
_ACCEPTED_FIELDS = (_ATTEMPT_FIELDS - {"attempt_id"}) | {"accepted_attempt_id"}
_PRODUCER_FIELDS = {
    "producer_replica_id",
    "producer_boot_id",
    "shared_storage_path",
    "target_revision",
    "layer_manifest",
    "layer_manifest_digest",
    "producer_identity",
    "producer_identity_digest",
}
_LAYER_MANIFEST_FIELDS = {
    "target_layer_ids",
    "final_plane",
    "hidden_size",
    "dtype",
}
_PRODUCER_IDENTITY_FIELDS = {
    "vllm_revision",
    "container_image",
    "target_weights_digest",
    "target_config_digest",
    "dtype",
    "kernel",
    "schema_version",
}
_IMAGE_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_digest(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_digest(value, *, field: str) -> str:
    value = _required_string(value, field=field)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path.name} contains an empty row at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} row {line_number} must be a JSON object")
            rows.append(value)
    return rows


def _append_jsonl(path: Path, value: Mapping) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Mapping) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_visit_id(
    *, selection_policy_digest: str, source_row_id: str, epoch_index: int
) -> str:
    selection_policy_digest = _require_digest(
        selection_policy_digest, field="selection_policy_digest"
    )
    source_row_id = _require_digest(source_row_id, field="source_row_id")
    if isinstance(epoch_index, bool) or int(epoch_index) < 0:
        raise ValueError("epoch_index must be a non-negative integer")
    return _json_digest(
        {
            "selection_policy_digest": selection_policy_digest,
            "source_row_id": source_row_id,
            "epoch_index": int(epoch_index),
        }
    )


def _build_request_id(visit_id: str) -> str:
    return _json_digest({"kind": "live-hidden-request", "visit_id": visit_id})


def _build_attempt_id(request_id: str, attempt_number: int) -> str:
    return _json_digest(
        {
            "kind": "live-hidden-response-attempt",
            "request_id": request_id,
            "attempt_number": int(attempt_number),
        }
    )


def _digest_tensor(value: str) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)


class ProvenanceLedger:
    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        consumer_boot_id: str,
        selection_policy_digest: str,
    ):
        self.directory = Path(directory).resolve(strict=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = _required_string(run_id, field="run_id")
        self.consumer_boot_id = _required_string(
            consumer_boot_id, field="consumer_boot_id"
        )
        self.selection_policy_digest = _require_digest(
            selection_policy_digest, field="selection_policy_digest"
        )
        self.run_path = self.directory / "run.json"
        self.requests_path = self.directory / "requests.jsonl"
        self.attempts_path = self.directory / "attempts.jsonl"
        self.accepted_path = self.directory / "accepted-attempts.jsonl"
        self.delivered_path = self.directory / "delivered.jsonl"
        self.consumed_path = self.directory / "consumed.jsonl"
        self.lock_path = self.directory / ".ledger.lock"
        self._bind_run()

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _bind_run(self) -> None:
        run = {
            "run_id": self.run_id,
            "consumer_boot_id": self.consumer_boot_id,
            "selection_policy_digest": self.selection_policy_digest,
        }
        with self._locked():
            if self.run_path.exists():
                existing = json.loads(self.run_path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict) or set(existing) != _RUN_FIELDS:
                    raise ValueError(
                        "training ledger run identity must use exact fields"
                    )
                if existing != run:
                    raise ValueError(
                        "training ledger contains a mixed consumer boot or restarted run"
                    )
            else:
                _atomic_json(self.run_path, run)

    def _check_run_fields(self, row: Mapping) -> None:
        if (
            row.get("run_id") != self.run_id
            or row.get("consumer_boot_id") != self.consumer_boot_id
        ):
            raise ValueError("training ledger row does not match the bound run/boot")

    def record_request(
        self,
        *,
        request_id: str,
        row_id: str,
        source_row_id: str,
        epoch_index: int,
        token_digest: str,
    ) -> None:
        if isinstance(epoch_index, bool) or int(epoch_index) < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        request = {
            "request_id": _required_string(request_id, field="request_id"),
            "row_id": _required_string(row_id, field="row_id"),
            "source_row_id": _require_digest(source_row_id, field="source_row_id"),
            "epoch_index": int(epoch_index),
            "token_digest": _require_digest(token_digest, field="token_digest"),
            "run_id": self.run_id,
            "consumer_boot_id": self.consumer_boot_id,
        }
        with self._locked():
            rows = _read_jsonl(self.requests_path)
            if any(row.get("request_id") == request_id for row in rows):
                raise ValueError(
                    f"logical request {request_id!r} was requested more than once"
                )
            if any(row.get("row_id") == row_id for row in rows):
                raise ValueError(f"frozen row {row_id!r} was requested more than once")
            _append_jsonl(self.requests_path, request)

    def _normalize_attempt(self, attempt: Mapping) -> dict:
        if set(attempt) != _ATTEMPT_FIELDS:
            raise ValueError("response attempt must use exact provenance fields")
        normalized = {
            name: _required_string(attempt[name], field=name)
            for name in _ATTEMPT_FIELDS
        }
        for field in (
            "token_digest",
            "producer_identity_digest",
            "layer_manifest_digest",
            "hidden_payload_digest",
        ):
            _require_digest(normalized[field], field=field)
        if not _GIT_REVISION_RE.fullmatch(normalized["target_revision"]):
            raise ValueError("target_revision must be a lowercase 40-character git SHA")
        return normalized

    def record_attempt(self, attempt: Mapping) -> None:
        normalized = self._normalize_attempt(attempt)
        with self._locked():
            requests = _read_jsonl(self.requests_path)
            matching_requests = [
                row
                for row in requests
                if row.get("request_id") == normalized["request_id"]
                and row.get("row_id") == normalized["row_id"]
            ]
            if len(matching_requests) != 1:
                raise ValueError("response attempt has no matching logical request")
            self._check_run_fields(matching_requests[0])
            if matching_requests[0].get("token_digest") != normalized["token_digest"]:
                raise ValueError("response attempt token digest does not match request")
            if any(
                row.get("attempt_id") == normalized["attempt_id"]
                for row in _read_jsonl(self.attempts_path)
            ):
                raise ValueError(
                    f"duplicate response attempt_id {normalized['attempt_id']!r}"
                )
            _append_jsonl(self.attempts_path, normalized)

    def accept_attempt(self, *, request_id: str, attempt_id: str) -> str:
        request_id = _required_string(request_id, field="request_id")
        attempt_id = _required_string(attempt_id, field="attempt_id")
        with self._locked():
            attempts = [
                row
                for row in _read_jsonl(self.attempts_path)
                if row.get("request_id") == request_id
                and row.get("attempt_id") == attempt_id
            ]
            if len(attempts) != 1:
                raise ValueError("accepted pointer must reference one recorded attempt")
            normalized = attempts[0]
            accepted_rows = [
                row
                for row in _read_jsonl(self.accepted_path)
                if row.get("request_id") == normalized["request_id"]
            ]
            if len(accepted_rows) > 1:
                raise ValueError(
                    "logical request has multiple canonical accepted attempts"
                )
            if accepted_rows:
                accepted = accepted_rows[0]
                comparable_fields = _ATTEMPT_FIELDS - {"attempt_id"}
                if any(
                    accepted.get(field) != normalized[field]
                    for field in comparable_fields
                ):
                    raise ValueError(
                        "conflicting duplicate response for logical request"
                    )
                return str(accepted["accepted_attempt_id"])

            accepted = {
                **normalized,
                "accepted_attempt_id": normalized["attempt_id"],
            }
            del accepted["attempt_id"]
            _append_jsonl(self.accepted_path, accepted)
            return normalized["attempt_id"]

    def record_response(self, attempt: Mapping) -> str:
        self.record_attempt(attempt)
        return self.accept_attempt(
            request_id=str(attempt["request_id"]),
            attempt_id=str(attempt["attempt_id"]),
        )

    def _canonical_delivery(
        self,
        *,
        request_id: str,
        row_id: str,
        source_row_id: str,
        epoch_index: int,
    ) -> dict:
        request_id = _required_string(request_id, field="request_id")
        row_id = _required_string(row_id, field="row_id")
        source_row_id = _require_digest(source_row_id, field="source_row_id")
        if isinstance(epoch_index, bool) or int(epoch_index) < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        with self._locked():
            accepted = [
                row
                for row in _read_jsonl(self.accepted_path)
                if row.get("request_id") == request_id
            ]
            requests = [
                row
                for row in _read_jsonl(self.requests_path)
                if row.get("request_id") == request_id
            ]
        if len(accepted) != 1 or len(requests) != 1:
            raise ValueError("delivery does not reference one accepted logical request")
        request = requests[0]
        expected = {
            "row_id": row_id,
            "source_row_id": source_row_id,
            "epoch_index": int(epoch_index),
        }
        if any(request.get(field) != value for field, value in expected.items()):
            raise ValueError("delivery identity does not match the logical request")
        if accepted[0].get("row_id") != row_id:
            raise ValueError(
                "delivery does not reference the canonical accepted attempt"
            )
        self._check_run_fields(request)
        return {
            "request_id": request_id,
            **expected,
            "accepted_attempt_id": _required_string(
                accepted[0].get("accepted_attempt_id"),
                field="accepted_attempt_id",
            ),
            "run_id": self.run_id,
            "consumer_boot_id": self.consumer_boot_id,
        }

    def record_delivered(
        self,
        *,
        request_id: str,
        row_id: str,
        source_row_id: str,
        epoch_index: int,
    ) -> None:
        delivered = self._canonical_delivery(
            request_id=request_id,
            row_id=row_id,
            source_row_id=source_row_id,
            epoch_index=epoch_index,
        )
        with self._locked():
            delivered_rows = _read_jsonl(self.delivered_path)
            if any(row.get("row_id") == row_id for row in delivered_rows):
                raise ValueError("frozen visit was delivered more than once")
            _append_jsonl(self.delivered_path, delivered)

    def record_consumed(
        self,
        *,
        request_id: str,
        row_id: str,
        source_row_id: str,
        epoch_index: int,
        global_step: int,
        micro_step: int,
    ) -> None:
        if any(
            isinstance(value, bool) or int(value) < 0
            for value in (global_step, micro_step)
        ):
            raise ValueError("global_step and micro_step must be non-negative integers")
        consumed = {
            **self._canonical_delivery(
                request_id=request_id,
                row_id=row_id,
                source_row_id=source_row_id,
                epoch_index=epoch_index,
            ),
            "global_step": int(global_step),
            "micro_step": int(micro_step),
        }
        with self._locked():
            delivered = [
                row
                for row in _read_jsonl(self.delivered_path)
                if row.get("row_id") == row_id
            ]
            if len(delivered) != 1 or delivered[0] != {
                field: consumed[field] for field in _DELIVERED_FIELDS
            }:
                raise ValueError("consumption does not reference one delivered visit")
            consumed_rows = _read_jsonl(self.consumed_path)
            if any(row.get("row_id") == row_id for row in consumed_rows):
                raise ValueError("frozen visit was consumed more than once")
            _append_jsonl(self.consumed_path, consumed)

    def assert_training_complete(
        self,
        source_row_ids: Sequence[str],
        *,
        num_epochs: int,
        expected_replica_ids: Sequence[str] | None = None,
    ) -> None:
        sources = [
            _require_digest(value, field="source_row_id") for value in source_row_ids
        ]
        if len(sources) != len(set(sources)):
            raise ValueError("expected source row ids must be unique")
        if isinstance(num_epochs, bool) or int(num_epochs) <= 0:
            raise ValueError("num_epochs must be a positive integer")
        expected = {
            build_visit_id(
                selection_policy_digest=self.selection_policy_digest,
                source_row_id=source_row_id,
                epoch_index=epoch_index,
            ): (source_row_id, epoch_index)
            for epoch_index in range(int(num_epochs))
            for source_row_id in sources
        }
        with self._locked():
            request_rows = _read_jsonl(self.requests_path)
            attempt_rows = _read_jsonl(self.attempts_path)
            accepted_rows = _read_jsonl(self.accepted_path)
            delivered_rows = _read_jsonl(self.delivered_path)
            consumed_rows = _read_jsonl(self.consumed_path)
        expected_requests = {row_id: _build_request_id(row_id) for row_id in expected}
        expected_fields = {
            "requested": _REQUEST_FIELDS,
            "accepted": _ACCEPTED_FIELDS,
            "delivered": _DELIVERED_FIELDS,
            "consumed": _CONSUMED_FIELDS,
        }
        if any(set(row) != _ACCEPTED_FIELDS for row in accepted_rows):
            raise ValueError("accepted ledger row must use exact fields")
        normalized_attempts = [self._normalize_attempt(row) for row in attempt_rows]
        attempt_ids = [row["attempt_id"] for row in normalized_attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("response attempt ids must be unique")
        requests_by_id = {row["request_id"]: row for row in request_rows}
        for attempt in normalized_attempts:
            request = requests_by_id.get(attempt["request_id"])
            if (
                request is None
                or request.get("row_id") != attempt["row_id"]
                or request.get("token_digest") != attempt["token_digest"]
            ):
                raise ValueError("response attempt does not match a logical request")
        attempts_by_id = {row["attempt_id"]: row for row in normalized_attempts}
        for accepted in accepted_rows:
            attempt = attempts_by_id.get(accepted["accepted_attempt_id"])
            if attempt is None or any(
                accepted.get(field) != attempt[field]
                for field in _ATTEMPT_FIELDS - {"attempt_id"}
            ):
                raise ValueError(
                    "canonical accepted pointer does not match one response attempt"
                )
        accepted_attempts = {
            row["row_id"]: row["accepted_attempt_id"] for row in accepted_rows
        }
        for name, rows in (
            ("requested", request_rows),
            ("accepted", accepted_rows),
            ("delivered", delivered_rows),
            ("consumed", consumed_rows),
        ):
            row_ids = [row.get("row_id") for row in rows]
            if set(row_ids) != set(expected) or len(row_ids) != len(set(row_ids)):
                raise ValueError(
                    "training ledger does not prove "
                    f"requested=accepted=delivered=consumed=1: {name}"
                )
            for row in rows:
                if set(row) != expected_fields[name]:
                    raise ValueError(f"{name} ledger row must use exact fields")
                source_row_id, epoch_index = expected[str(row["row_id"])]
                if row.get("request_id") != expected_requests[str(row["row_id"])]:
                    raise ValueError(f"{name} request identity does not match visit")
                if name != "accepted" and (
                    row.get("source_row_id") != source_row_id
                    or row.get("epoch_index") != epoch_index
                ):
                    raise ValueError(f"{name} visit identity does not match policy")
                if name != "accepted":
                    self._check_run_fields(row)
                if name in {"delivered", "consumed"} and row.get(
                    "accepted_attempt_id"
                ) != accepted_attempts.get(str(row["row_id"])):
                    raise ValueError(
                        f"{name} does not reference the canonical accepted attempt"
                    )
                if name == "consumed" and any(
                    isinstance(row[field], bool) or int(row[field]) < 0
                    for field in ("global_step", "micro_step")
                ):
                    raise ValueError("consumed step metadata must be non-negative")
        if expected_replica_ids is not None:
            observed = {str(row.get("producer_replica_id")) for row in accepted_rows}
            if observed != set(expected_replica_ids):
                raise ValueError(
                    "accepted attempts do not cover the exact producer replicas"
                )


class LiveHiddenCollator(CacheCollator):
    def __call__(self, features: list[dict]):
        batch = super().__call__(features)
        batch["_live_source_indices"] = torch.tensor(
            [int(feature["_live_source_index"]) for feature in features],
            dtype=torch.long,
        )
        batch["_live_epoch_indices"] = torch.tensor(
            [int(feature["_live_epoch_index"]) for feature in features],
            dtype=torch.long,
        )
        batch["_live_source_row_ids"] = torch.stack(
            [feature["_live_source_row_id"] for feature in features]
        )
        batch["_live_visit_ids"] = torch.stack(
            [feature["_live_visit_id"] for feature in features]
        )
        return batch


class _ProducerCatalog:
    def __init__(
        self,
        manifests: Sequence[Mapping],
        *,
        target_revision: str,
        target_layer_ids: Sequence[int],
    ):
        if not manifests:
            raise ValueError("at least one producer boot manifest is required")
        self.producers = []
        expected_identity_digest = None
        expected_layer_digest = None
        roots = set()
        replica_ids = set()
        boot_ids = set()
        for manifest in manifests:
            if set(manifest) != _PRODUCER_FIELDS:
                raise ValueError("producer boot manifest must use exact fields")
            identity = manifest["producer_identity"]
            if (
                not isinstance(identity, Mapping)
                or set(identity) != _PRODUCER_IDENTITY_FIELDS
            ):
                raise ValueError("producer_identity must use exact fields")
            identity_digest = _json_digest(identity)
            if identity_digest != manifest["producer_identity_digest"]:
                raise ValueError("producer identity digest does not match manifest")
            if expected_identity_digest is None:
                expected_identity_digest = identity_digest
            elif identity_digest != expected_identity_digest:
                raise ValueError(
                    "all producer boots must use the same producer identity"
                )
            if not _GIT_REVISION_RE.fullmatch(str(identity["vllm_revision"])):
                raise ValueError("producer vLLM revision must be an exact git SHA")
            if not _IMAGE_DIGEST_RE.fullmatch(str(identity["container_image"])):
                raise ValueError(
                    "producer container image must use an immutable digest"
                )
            if (
                identity["dtype"] != "bfloat16"
                or identity["kernel"] != "stock-vllm"
                or identity["schema_version"] != 1
            ):
                raise ValueError("producer identity runtime contract is unsupported")
            _require_digest(
                identity["target_weights_digest"], field="target_weights_digest"
            )
            _require_digest(
                identity["target_config_digest"], field="target_config_digest"
            )

            layer = manifest["layer_manifest"]
            if not isinstance(layer, Mapping) or set(layer) != _LAYER_MANIFEST_FIELDS:
                raise ValueError("layer manifest must use exact fields")
            layer_digest = _json_digest(layer)
            if layer_digest != manifest["layer_manifest_digest"]:
                raise ValueError("layer manifest digest does not match manifest")
            if [int(value) for value in layer["target_layer_ids"]] != [
                int(value) for value in target_layer_ids
            ]:
                raise ValueError(
                    "producer layer manifest does not match target layer ids"
                )
            if layer["final_plane"] != "post-final-block-pre-final-rmsnorm":
                raise ValueError("producer layer manifest final plane is unsupported")
            if layer["dtype"] != "bfloat16" or int(layer["hidden_size"]) <= 0:
                raise ValueError(
                    "producer layer manifest dtype/hidden size is unsupported"
                )
            if expected_layer_digest is None:
                expected_layer_digest = layer_digest
            elif layer_digest != expected_layer_digest:
                raise ValueError("all producer boots must use the same layer manifest")

            if manifest["target_revision"] != target_revision:
                raise ValueError("producer target revision does not match live adapter")
            replica_id = _required_string(
                manifest["producer_replica_id"], field="producer_replica_id"
            )
            boot_id = _required_string(
                manifest["producer_boot_id"], field="producer_boot_id"
            )
            root = Path(
                _required_string(
                    manifest["shared_storage_path"], field="shared_storage_path"
                )
            ).resolve(strict=False)
            if root in roots or replica_id in replica_ids or boot_id in boot_ids:
                raise ValueError(
                    "producer roots, replica ids, and boot ids must be unique"
                )
            roots.add(root)
            replica_ids.add(replica_id)
            boot_ids.add(boot_id)
            self.producers.append(
                {
                    "root": root,
                    "producer_replica_id": replica_id,
                    "producer_boot_id": boot_id,
                    "producer_identity_digest": identity_digest,
                    "target_revision": target_revision,
                    "layer_manifest_digest": layer_digest,
                    "hidden_size": int(layer["hidden_size"]),
                }
            )

    def resolve(self, path: Path) -> dict:
        if path.is_symlink():
            raise ValueError("shared hidden response must not be a symlink")
        resolved = path.resolve(strict=True)
        matches = [
            producer
            for producer in self.producers
            if resolved.is_relative_to(producer["root"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "shared hidden response does not belong to one registered producer boot"
            )
        if not resolved.is_file():
            raise ValueError(
                "shared hidden response must be a regular non-symlink file"
            )
        return matches[0]


def qwen3_final_rms_norm(
    hidden_states: torch.Tensor,
    *,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or weight.ndim != 1:
        raise ValueError("Qwen3 final RMSNorm expects [tokens, hidden] and [hidden]")
    if hidden_states.shape[-1] != weight.shape[0]:
        raise ValueError("Qwen3 final RMSNorm weight does not match hidden size")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("Qwen3 final RMSNorm epsilon must be positive and finite")
    input_dtype = hidden_states.dtype
    normalized = hidden_states.to(torch.float32)
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + float(eps))
    return weight.to(hidden_states.device) * normalized.to(input_dtype)


class LiveHiddenDataset(torch.utils.data.Dataset):
    requires_epoch_index = True

    def __init__(
        self,
        *,
        source_dataset,
        request_hidden_states: Callable[[dict], str | Path],
        producer_manifests: Sequence[Mapping],
        ledger_dir: str | Path,
        target_model_name_or_path: str,
        target_revision: str,
        target_layer_ids: Sequence[int],
        final_norm_weight: torch.Tensor,
        final_norm_eps: float,
        request_timeout_s: float,
        max_retries: int,
        selection_policy_digest: str,
        num_epochs: int,
        run_id: str,
        consumer_boot_id: str,
    ):
        if not _GIT_REVISION_RE.fullmatch(target_revision):
            raise ValueError("target revision must be a lowercase 40-character git SHA")
        if not target_layer_ids:
            raise ValueError("target layer ids must not be empty")
        if not math.isfinite(float(request_timeout_s)) or float(request_timeout_s) <= 0:
            raise ValueError("request timeout must be positive and finite")
        if isinstance(max_retries, bool) or int(max_retries) < 0:
            raise ValueError("max retries must be a non-negative integer")
        self.source_dataset = source_dataset
        self.request_hidden_states = request_hidden_states
        self.selection_policy_digest = _require_digest(
            selection_policy_digest, field="selection_policy_digest"
        )
        if isinstance(num_epochs, bool) or int(num_epochs) <= 0:
            raise ValueError("num_epochs must be a positive integer")
        self.num_epochs = int(num_epochs)
        self.run_id = _required_string(run_id, field="run_id")
        self.consumer_boot_id = _required_string(
            consumer_boot_id, field="consumer_boot_id"
        )
        self.ledger = ProvenanceLedger(
            ledger_dir,
            run_id=self.run_id,
            consumer_boot_id=self.consumer_boot_id,
            selection_policy_digest=self.selection_policy_digest,
        )
        self.target_model_name_or_path = _required_string(
            target_model_name_or_path,
            field="target_model_name_or_path",
        )
        self.target_revision = target_revision
        self.target_layer_ids = [int(value) for value in target_layer_ids]
        self.final_norm_eps = float(final_norm_eps)
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = int(max_retries)
        self.catalog = _ProducerCatalog(
            producer_manifests,
            target_revision=target_revision,
            target_layer_ids=target_layer_ids,
        )
        if (
            not isinstance(final_norm_weight, torch.Tensor)
            or final_norm_weight.ndim != 1
            or final_norm_weight.dtype != torch.bfloat16
            or final_norm_weight.numel() != self.catalog.producers[0]["hidden_size"]
            or not torch.isfinite(final_norm_weight).all()
        ):
            raise ValueError(
                "Qwen3 final RMSNorm weight must be finite bfloat16 with manifest hidden size"
            )
        self.final_norm_weight = final_norm_weight.detach().to(device="cpu")
        self.manifest = {
            "target_model_name_or_path": self.target_model_name_or_path,
            "target_layer_ids": list(self.target_layer_ids),
            "hidden_size": self.catalog.producers[0]["hidden_size"],
        }

    def __len__(self):
        return len(self.source_dataset)

    def _identify_response(self, path: Path):
        producer = self.catalog.resolve(path)
        lock_path = Path(f"{path}.lock")
        deadline = time.monotonic() + self.request_timeout_s
        while lock_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for shared hidden response lock: {lock_path}"
                )
            time.sleep(0.01)
        payload_digest = _file_digest(path)
        return producer, payload_digest

    def _load_response(
        self,
        path: Path,
        *,
        expected_tokens: torch.Tensor,
        hidden_size: int,
    ):
        try:
            tensors = load_file(path, device="cpu")
        finally:
            path.unlink()
        if set(tensors) != {"token_ids", "hidden_states"}:
            raise ValueError("shared hidden response must contain exact tensor fields")
        token_ids = tensors["token_ids"]
        hidden = tensors["hidden_states"]
        if token_ids.dtype not in (torch.int32, torch.int64) or not torch.equal(
            token_ids.to(torch.long), expected_tokens.to(torch.long)
        ):
            raise ValueError(
                "shared hidden response token ids do not match request token ids"
            )
        expected_layers = len(self.target_layer_ids) + 1
        if hidden.ndim != 3 or hidden.shape != (
            expected_tokens.numel(),
            expected_layers,
            hidden_size,
        ):
            raise ValueError(
                "shared hidden response layer/shape does not match layer manifest"
            )
        if hidden.dtype != torch.bfloat16:
            raise ValueError(
                "shared hidden response dtype does not match layer manifest"
            )
        if not torch.isfinite(hidden).all():
            raise ValueError("shared hidden response tensors must be finite")
        target_hidden_states = hidden[:, :-1].flatten(1)
        target_last_hidden_states = qwen3_final_rms_norm(
            hidden[:, -1],
            weight=self.final_norm_weight,
            eps=self.final_norm_eps,
        )
        return target_hidden_states, target_last_hidden_states

    def _visit_identity(
        self, source_index: int, epoch_index: int
    ) -> tuple[str, str, str]:
        if not 0 <= int(source_index) < len(self.source_dataset):
            raise IndexError(source_index)
        if not 0 <= int(epoch_index) < self.num_epochs:
            raise IndexError(epoch_index)
        source = self.source_dataset[int(source_index)]
        if not isinstance(source, Mapping) or set(source) != {
            "source_row_id",
            "input_ids",
            "loss_mask",
        }:
            raise ValueError("live hidden source row must use exact source fields")
        source_row_id = _require_digest(source["source_row_id"], field="source_row_id")
        visit_id = build_visit_id(
            selection_policy_digest=self.selection_policy_digest,
            source_row_id=source_row_id,
            epoch_index=int(epoch_index),
        )
        return source_row_id, visit_id, _build_request_id(visit_id)

    def __getitem__(self, index):
        if not (
            isinstance(index, tuple)
            and len(index) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in index
            )
        ):
            raise TypeError(
                "live hidden dataset index must be (source_index, epoch_index)"
            )
        source_index, epoch_index = index
        source = self.source_dataset[source_index]
        if not isinstance(source, Mapping) or set(source) != {
            "source_row_id",
            "input_ids",
            "loss_mask",
        }:
            raise ValueError("live hidden source row must use exact source fields")
        source_row_id, row_id, request_id = self._visit_identity(
            source_index, epoch_index
        )
        input_ids = source["input_ids"]
        loss_mask = source["loss_mask"]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            raise ValueError("live hidden source input_ids must be a 1-D tensor")
        if (
            not isinstance(loss_mask, torch.Tensor)
            or loss_mask.shape != input_ids.shape
        ):
            raise ValueError("live hidden source loss_mask must match input_ids")
        token_digest = _json_digest([int(token) for token in input_ids.tolist()])
        self.ledger.record_request(
            request_id=request_id,
            row_id=row_id,
            source_row_id=source_row_id,
            epoch_index=epoch_index,
            token_digest=token_digest,
        )
        last_error = None
        response_path = None
        attempt_id = None
        for attempt_number in range(1, self.max_retries + 2):
            attempt_id = _build_attempt_id(request_id, attempt_number)
            request = {
                "request_id": request_id,
                "row_id": row_id,
                "token_digest": token_digest,
                "attempt_id": attempt_id,
                "input_ids": [int(token) for token in input_ids.tolist()],
                "timeout_s": self.request_timeout_s,
            }
            try:
                response_path = Path(self.request_hidden_states(request))
                break
            except Exception as error:
                last_error = error
        if response_path is None:
            assert last_error is not None
            raise last_error
        assert attempt_id is not None
        producer, payload_digest = self._identify_response(response_path)
        attempt = {
            "request_id": request_id,
            "row_id": row_id,
            "token_digest": token_digest,
            "attempt_id": attempt_id,
            "producer_replica_id": producer["producer_replica_id"],
            "producer_boot_id": producer["producer_boot_id"],
            "producer_identity_digest": producer["producer_identity_digest"],
            "target_revision": producer["target_revision"],
            "layer_manifest_digest": producer["layer_manifest_digest"],
            "hidden_payload_digest": payload_digest,
        }
        try:
            self.ledger.record_attempt(attempt)
        except Exception:
            response_path.unlink(missing_ok=True)
            raise
        target_hidden_states, target_last_hidden_states = self._load_response(
            response_path,
            expected_tokens=input_ids,
            hidden_size=producer["hidden_size"],
        )
        canonical_attempt_id = self.ledger.accept_attempt(
            request_id=request_id,
            attempt_id=attempt_id,
        )
        self.ledger.record_delivered(
            request_id=request_id,
            row_id=row_id,
            source_row_id=source_row_id,
            epoch_index=epoch_index,
        )
        assert canonical_attempt_id == attempt_id
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "target_hidden_states": target_hidden_states,
            "target_last_hidden_states": target_last_hidden_states,
            "_live_source_index": source_index,
            "_live_epoch_index": epoch_index,
            "_live_source_row_id": _digest_tensor(source_row_id),
            "_live_visit_id": _digest_tensor(row_id),
        }

    def record_batch_consumed(
        self,
        *,
        source_indices: Sequence[int],
        epoch_indices: Sequence[int],
        source_row_ids: Sequence[str],
        visit_ids: Sequence[str],
        global_step: int,
        micro_step: int,
    ) -> None:
        if (
            len(
                {
                    len(source_indices),
                    len(epoch_indices),
                    len(source_row_ids),
                    len(visit_ids),
                }
            )
            != 1
        ):
            raise ValueError("live batch identity metadata lengths differ")
        for source_index, epoch_index, carried_source_id, carried_visit_id in zip(
            source_indices, epoch_indices, source_row_ids, visit_ids, strict=True
        ):
            source_row_id, row_id, request_id = self._visit_identity(
                int(source_index), int(epoch_index)
            )
            if carried_source_id != source_row_id or carried_visit_id != row_id:
                raise ValueError(
                    "live batch identity drifted before trainer consumption"
                )
            self.ledger.record_consumed(
                request_id=request_id,
                row_id=row_id,
                source_row_id=source_row_id,
                epoch_index=int(epoch_index),
                global_step=global_step,
                micro_step=micro_step,
            )

    def assert_training_complete(
        self, *, expected_replica_ids: Sequence[str] | None = None
    ) -> None:
        source_row_ids = [
            _require_digest(row["source_row_id"], field="source_row_id")
            for row in self.source_dataset
        ]
        self.ledger.assert_training_complete(
            source_row_ids,
            num_epochs=self.num_epochs,
            expected_replica_ids=expected_replica_ids,
        )


class _StockVllmRequester:
    def __init__(self, *, endpoint: str, model: str):
        self.endpoint = _required_string(endpoint, field="vLLM endpoint")
        self.model = _required_string(model, field="vLLM model")
        self._pid = None
        self._client = None

    def _client_for_process(self):
        pid = os.getpid()
        if self._client is None or self._pid != pid:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.endpoint,
                api_key="EMPTY",
                max_retries=0,
            )
            self._pid = pid
        return self._client

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pid"] = None
        state["_client"] = None
        return state

    def __call__(self, request: dict) -> str:
        response = self._client_for_process().completions.create(
            model=self.model,
            prompt=request["input_ids"],
            max_tokens=1,
            extra_body={"return_token_ids": True},
            extra_headers={"x-request-id": request["request_id"]},
            timeout=request["timeout_s"],
        )
        transfer = getattr(response, "kv_transfer_params", None)
        if not isinstance(transfer, Mapping):
            raise ValueError(
                "stock vLLM response is missing shared hidden transfer metadata"
            )
        path = transfer.get("hidden_states_path")
        return _required_string(path, field="stock vLLM hidden_states_path")


def build_stock_vllm_requester(*, endpoint: str, model: str):
    return _StockVllmRequester(endpoint=endpoint, model=model)
