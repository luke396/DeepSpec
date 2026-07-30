"""Prepare and validate train-ready JSONL for live hidden-state training."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from deepspec.data.parser import preprocess_record


LIVE_HIDDEN_DATA_VERSION = 1
REJECTED_FILE_NAME = "rejected.jsonl"
MANIFEST_FILE_NAME = "manifest.json"


@dataclass(frozen=True)
class PreparedLiveHiddenData:
    manifest_path: Path
    filtered_path: Path
    rejected_path: Path
    source_samples: int
    accepted_samples: int
    rejected_samples: int
    rejected_indices: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_and_count_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            count += 1
    return digest.hexdigest(), count


def _json_line(payload) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_identity(tokenizer):
    tokenizer_type = type(tokenizer)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_revision = (
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
    )
    identity = {
        "class": f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}",
        "name_or_path": (
            str(tokenizer.name_or_path)
            if getattr(tokenizer, "name_or_path", None) is not None
            else None
        ),
        "resolved_revision": (
            str(resolved_revision) if resolved_revision is not None else None
        ),
    }

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None:
        identity["chat_template_sha256"] = _sha256_json(chat_template)

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        identity["backend_sha256"] = hashlib.sha256(
            backend.to_str().encode("utf-8")
        ).hexdigest()
    elif hasattr(tokenizer, "get_vocab"):
        identity["vocab_sha256"] = _sha256_json(tokenizer.get_vocab())
    return identity


def _atomic_json_dump(payload, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _record_id(record):
    if not isinstance(record, dict):
        return None
    value = record.get("id")
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    return None


def _rejection(
    *,
    source_index: int,
    record_id,
    reason: str,
    sequence_tokens: int | None = None,
    loss_tokens: int | None = None,
    error_type: str | None = None,
):
    payload = {
        "source_index": int(source_index),
        "reason": reason,
    }
    if record_id is not None:
        payload["id"] = record_id
    if sequence_tokens is not None:
        payload["sequence_tokens"] = int(sequence_tokens)
    if loss_tokens is not None:
        payload["loss_tokens"] = int(loss_tokens)
    if error_type is not None:
        payload["error_type"] = error_type
    return payload


def _training_contract(
    *,
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
    target_model_name_or_path: str,
    target_revision: str | None,
):
    return {
        "target_model_name_or_path": str(target_model_name_or_path),
        "target_revision": (
            str(target_revision) if target_revision is not None else None
        ),
        "tokenizer": _tokenizer_identity(tokenizer),
        "chat_template": str(chat_template),
        "max_length": int(max_length),
        "min_loss_tokens": int(min_loss_tokens),
    }


def prepare_live_hidden_data(
    *,
    source_path,
    filtered_path,
    artifact_dir,
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
    expected_num_samples: int | None,
    target_model_name_or_path: str,
    target_revision: str | None,
    replace_existing: bool = False,
) -> PreparedLiveHiddenData:
    source_path = Path(source_path).resolve(strict=True)
    filtered_path = Path(filtered_path).resolve(strict=False)
    artifact_dir = Path(artifact_dir).resolve(strict=False)
    if source_path == filtered_path:
        raise ValueError("source and filtered JSONL paths must be different")

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = artifact_dir / REJECTED_FILE_NAME
    manifest_path = artifact_dir / MANIFEST_FILE_NAME
    output_paths = (filtered_path, rejected_path, manifest_path)
    if not replace_existing and any(path.exists() for path in output_paths):
        raise FileExistsError("prepared live hidden output already exists")

    filtered_tmp_path = filtered_path.with_name(
        f".{filtered_path.name}.tmp-{os.getpid()}"
    )
    rejected_tmp_path = artifact_dir / f".{REJECTED_FILE_NAME}.tmp-{os.getpid()}"
    source_digest = hashlib.sha256()
    filtered_digest = hashlib.sha256()
    rejected_digest = hashlib.sha256()
    source_samples = 0
    rejected_samples = 0

    try:
        with (
            source_path.open("rb") as source_handle,
            filtered_tmp_path.open("wb") as filtered_handle,
            rejected_tmp_path.open("wb") as rejected_handle,
        ):
            for source_index, raw_line in enumerate(source_handle):
                source_samples += 1
                source_digest.update(raw_line)
                record = None
                record_id = None
                rejection = None

                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    record_id = _record_id(record)
                except UnicodeDecodeError as exc:
                    rejection = _rejection(
                        source_index=source_index,
                        record_id=None,
                        reason="invalid_utf8",
                        error_type=type(exc).__name__,
                    )
                except json.JSONDecodeError as exc:
                    rejection = _rejection(
                        source_index=source_index,
                        record_id=None,
                        reason="invalid_json",
                        error_type=type(exc).__name__,
                    )

                if rejection is None:
                    try:
                        processed = preprocess_record(
                            record=record,
                            tokenizer=tokenizer,
                            chat_template=chat_template,
                            max_length=int(max_length),
                        )
                    except Exception as exc:
                        rejection = _rejection(
                            source_index=source_index,
                            record_id=record_id,
                            reason="preprocess_error",
                            error_type=type(exc).__name__,
                        )
                    else:
                        sequence_tokens = int(processed["input_ids"].shape[0])
                        loss_tokens = int(processed["loss_mask"].sum().item())
                        if loss_tokens < int(min_loss_tokens):
                            rejection = _rejection(
                                source_index=source_index,
                                record_id=record_id,
                                reason="insufficient_loss_tokens",
                                sequence_tokens=sequence_tokens,
                                loss_tokens=loss_tokens,
                            )

                if rejection is None:
                    filtered_handle.write(raw_line)
                    filtered_digest.update(raw_line)
                else:
                    encoded = _json_line(rejection)
                    rejected_handle.write(encoded)
                    rejected_digest.update(encoded)
                    rejected_samples += 1

            for handle in (filtered_handle, rejected_handle):
                handle.flush()
                os.fsync(handle.fileno())

        if expected_num_samples is not None and source_samples != int(
            expected_num_samples
        ):
            raise ValueError(
                "live dataset sample count mismatch: "
                f"{source_samples} != {int(expected_num_samples)}"
            )

        accepted_samples = source_samples - rejected_samples
        if accepted_samples <= 0:
            raise ValueError("live hidden preparation found no trainable samples")

        manifest = {
            "version": LIVE_HIDDEN_DATA_VERSION,
            "source_file": source_path.name,
            "source_sha256": source_digest.hexdigest(),
            "source_samples": source_samples,
            "filtered_file": filtered_path.name,
            "filtered_sha256": filtered_digest.hexdigest(),
            "accepted_samples": accepted_samples,
            "rejected_file": REJECTED_FILE_NAME,
            "rejected_sha256": rejected_digest.hexdigest(),
            "rejected_samples": rejected_samples,
            **_training_contract(
                tokenizer=tokenizer,
                chat_template=chat_template,
                max_length=max_length,
                min_loss_tokens=min_loss_tokens,
                target_model_name_or_path=target_model_name_or_path,
                target_revision=target_revision,
            ),
        }

        os.replace(filtered_tmp_path, filtered_path)
        os.replace(rejected_tmp_path, rejected_path)
        _atomic_json_dump(manifest, manifest_path)
        return validate_prepared_live_hidden_data(
            manifest_path=manifest_path,
            filtered_path=filtered_path,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=max_length,
            min_loss_tokens=min_loss_tokens,
            target_model_name_or_path=target_model_name_or_path,
            target_revision=target_revision,
        )
    finally:
        filtered_tmp_path.unlink(missing_ok=True)
        rejected_tmp_path.unlink(missing_ok=True)


def _load_rejected_indices(
    *, rejected_path: Path, source_samples: int
) -> tuple[int, ...]:
    rejected_indices = []
    with rejected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            rejected_indices.append(int(entry["source_index"]))
    if (
        rejected_indices != sorted(set(rejected_indices))
        or any(index < 0 or index >= source_samples for index in rejected_indices)
    ):
        raise ValueError("invalid live hidden rejected index")
    return tuple(rejected_indices)


def load_prepared_live_hidden_metadata(
    *, manifest_path, filtered_path
) -> PreparedLiveHiddenData:
    manifest_path = Path(manifest_path).resolve(strict=True)
    filtered_path = Path(filtered_path).resolve(strict=True)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("version", -1)) != LIVE_HIDDEN_DATA_VERSION:
        raise ValueError("unsupported prepared live hidden data version")

    rejected_path = manifest_path.parent / manifest["rejected_file"]
    source_samples = int(manifest["source_samples"])
    accepted_samples = int(manifest["accepted_samples"])
    rejected_samples = int(manifest["rejected_samples"])
    rejected_indices = _load_rejected_indices(
        rejected_path=rejected_path,
        source_samples=source_samples,
    )
    if (
        len(rejected_indices) != rejected_samples
        or accepted_samples != source_samples - rejected_samples
    ):
        raise ValueError("invalid prepared live hidden sample counts")

    return PreparedLiveHiddenData(
        manifest_path=manifest_path,
        filtered_path=filtered_path,
        rejected_path=rejected_path,
        source_samples=source_samples,
        accepted_samples=accepted_samples,
        rejected_samples=rejected_samples,
        rejected_indices=rejected_indices,
    )


def validate_prepared_live_hidden_data(
    *,
    manifest_path,
    filtered_path,
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
    target_model_name_or_path: str,
    target_revision: str | None,
) -> PreparedLiveHiddenData:
    prepared = load_prepared_live_hidden_metadata(
        manifest_path=manifest_path,
        filtered_path=filtered_path,
    )
    with prepared.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_contract = _training_contract(
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
        min_loss_tokens=min_loss_tokens,
        target_model_name_or_path=target_model_name_or_path,
        target_revision=target_revision,
    )
    for field, expected in expected_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"prepared live hidden {field} mismatch: "
                f"{manifest.get(field)!r} != {expected!r}"
            )

    filtered_sha256, filtered_samples = _sha256_file_and_count_lines(
        prepared.filtered_path
    )
    if filtered_sha256 != manifest["filtered_sha256"]:
        raise ValueError("prepared live hidden JSONL checksum mismatch")
    if filtered_samples != prepared.accepted_samples:
        raise ValueError("prepared live hidden JSONL sample count mismatch")
    if _sha256_file(prepared.rejected_path) != manifest["rejected_sha256"]:
        raise ValueError("prepared live hidden rejected index checksum mismatch")
    return prepared
