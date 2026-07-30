"""Tokenizer-only prefilter artifacts for live hidden-state training."""

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from deepspec.data.parser import preprocess_record


LIVE_HIDDEN_PREFILTER_VERSION = 1
REJECTED_FILE_NAME = "rejected.jsonl"
MANIFEST_FILE_NAME = "manifest.json"


@dataclass(frozen=True)
class LiveHiddenPrefilter:
    manifest_path: Path
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
        "id": record_id,
        "reason": reason,
    }
    if sequence_tokens is not None:
        payload["sequence_tokens"] = int(sequence_tokens)
    if loss_tokens is not None:
        payload["loss_tokens"] = int(loss_tokens)
    if error_type is not None:
        payload["error_type"] = error_type
    return payload


def _existing_prefilter_matches(
    *, manifest_path: Path, rejected_path: Path, expected_manifest
) -> bool:
    if not manifest_path.is_file() or not rejected_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest == expected_manifest
        and _sha256_file(rejected_path) == expected_manifest["rejected_sha256"]
    )


def prepare_live_hidden_prefilter(
    *,
    source_path,
    output_dir,
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
    expected_num_samples: int | None,
    target_model_name_or_path: str,
    target_revision: str | None,
    replace_existing: bool,
) -> LiveHiddenPrefilter:
    source_path = Path(source_path).resolve(strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = output_dir / REJECTED_FILE_NAME
    manifest_path = output_dir / MANIFEST_FILE_NAME
    rejected_tmp_path = output_dir / f".{REJECTED_FILE_NAME}.tmp-{os.getpid()}"

    if not replace_existing and (
        not manifest_path.is_file() or not rejected_path.is_file()
    ):
        raise ValueError(
            "resumed live hidden training requires its existing prefilter artifacts"
        )

    source_digest = hashlib.sha256()
    rejected_digest = hashlib.sha256()
    source_samples = 0
    rejected_samples = 0

    try:
        with (
            source_path.open("rb") as source_handle,
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
                    if isinstance(record, dict):
                        record_id = record.get("id")
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

                if rejection is not None:
                    encoded = _json_line(rejection)
                    rejected_handle.write(encoded)
                    rejected_digest.update(encoded)
                    rejected_samples += 1

            rejected_handle.flush()
            os.fsync(rejected_handle.fileno())

        if expected_num_samples is not None and source_samples != int(
            expected_num_samples
        ):
            raise ValueError(
                "live dataset sample count mismatch: "
                f"{source_samples} != {int(expected_num_samples)}"
            )

        accepted_samples = source_samples - rejected_samples
        if accepted_samples <= 0:
            raise ValueError("live hidden prefilter found no trainable samples")

        manifest = {
            "version": LIVE_HIDDEN_PREFILTER_VERSION,
            "source_jsonl_path": str(source_path),
            "source_sha256": source_digest.hexdigest(),
            "source_samples": source_samples,
            "accepted_samples": accepted_samples,
            "rejected_samples": rejected_samples,
            "rejected_file": REJECTED_FILE_NAME,
            "rejected_sha256": rejected_digest.hexdigest(),
            "target_model_name_or_path": str(target_model_name_or_path),
            "target_revision": target_revision,
            "tokenizer": _tokenizer_identity(tokenizer),
            "chat_template": str(chat_template),
            "max_length": int(max_length),
            "min_loss_tokens": int(min_loss_tokens),
        }

        if _existing_prefilter_matches(
            manifest_path=manifest_path,
            rejected_path=rejected_path,
            expected_manifest=manifest,
        ):
            return load_live_hidden_prefilter(output_dir)

        if manifest_path.exists() and not replace_existing:
            raise ValueError(
                "live hidden prefilter no longer matches the resumed training run"
            )

        os.replace(rejected_tmp_path, rejected_path)
        _atomic_json_dump(manifest, manifest_path)
        return load_live_hidden_prefilter(output_dir)
    finally:
        rejected_tmp_path.unlink(missing_ok=True)


def load_live_hidden_prefilter(output_dir) -> LiveHiddenPrefilter:
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILE_NAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if int(manifest.get("version", -1)) != LIVE_HIDDEN_PREFILTER_VERSION:
        raise ValueError("unsupported live hidden prefilter version")

    rejected_path = output_dir / manifest["rejected_file"]
    if _sha256_file(rejected_path) != manifest["rejected_sha256"]:
        raise ValueError("live hidden rejected index checksum mismatch")

    rejected_indices = []
    with rejected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            rejected_indices.append(int(entry["source_index"]))

    source_samples = int(manifest["source_samples"])
    rejected_samples = int(manifest["rejected_samples"])
    accepted_samples = int(manifest["accepted_samples"])
    if (
        rejected_indices != sorted(set(rejected_indices))
        or len(rejected_indices) != rejected_samples
        or accepted_samples != source_samples - rejected_samples
        or any(index < 0 or index >= source_samples for index in rejected_indices)
    ):
        raise ValueError("invalid live hidden rejected index")

    return LiveHiddenPrefilter(
        manifest_path=manifest_path,
        rejected_path=rejected_path,
        source_samples=source_samples,
        accepted_samples=accepted_samples,
        rejected_samples=rejected_samples,
        rejected_indices=tuple(rejected_indices),
    )


def map_live_hidden_index(
    index: int,
    *,
    source_samples: int,
    rejected_indices: tuple[int, ...],
) -> int:
    accepted_samples = int(source_samples) - len(rejected_indices)
    if not 0 <= int(index) < accepted_samples:
        raise IndexError(index)
    if not rejected_indices:
        return int(index)

    target_count = int(index) + 1
    low = 0
    high = int(source_samples) - 1
    while low < high:
        middle = (low + high) // 2
        accepted_through_middle = middle + 1 - bisect_right(
            rejected_indices, middle
        )
        if accepted_through_middle >= target_count:
            high = middle
        else:
            low = middle + 1
    return low
