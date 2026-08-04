"""Per-request sampling resolution for conversation regeneration.

Covers the pure seams only: sidecar resolution (`resolve_row_sampling`)
and request assembly (`build_query_kwargs`). Server calls and the
threaded main loop are exercised by SpecLoop smoke runs, not here.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "generate_train_data.py"
_SPEC = importlib.util.spec_from_file_location("generate_train_data", _SCRIPT)
generate_train_data = importlib.util.module_from_spec(_SPEC)
sys.modules["generate_train_data"] = generate_train_data
_SPEC.loader.exec_module(generate_train_data)


def _args(**overrides):
    values = {
        "model": "target",
        "temperature": 0.7,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "repetition_penalty": None,
        "max_tokens": 4096,
        "sampling_mode": "per-request",
        "enable_thinking": False,
        "disable_thinking": True,
        "is_gpt_oss": False,
        "is_reasoning_model": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _fresh_counts():
    return {key: 0 for key in generate_train_data.SAMPLING_VALIDATORS}


def test_valid_sidecar_fields_become_overrides():
    counts = _fresh_counts()
    sample = {
        "sampling": {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "frequency_penalty": 0.7,
            "presence_penalty": -0.25,
            "stop": ["END"],
        }
    }
    overrides = generate_train_data.resolve_row_sampling(sample, counts)
    assert overrides == sample["sampling"]
    assert sum(counts.values()) == 0


def test_absent_fields_are_not_overridden_and_not_counted():
    counts = _fresh_counts()
    overrides = generate_train_data.resolve_row_sampling(
        {"sampling": {"temperature": 0.5}}, counts
    )
    assert overrides == {"temperature": 0.5}
    assert sum(counts.values()) == 0

    assert generate_train_data.resolve_row_sampling({}, counts) == {}
    assert generate_train_data.resolve_row_sampling({"sampling": "bogus"}, counts) == {}
    assert sum(counts.values()) == 0


def test_invalid_fields_fall_back_per_field_and_count():
    counts = _fresh_counts()
    sample = {
        "sampling": {
            "temperature": -0.5,
            "top_p": 0,
            "top_k": 0,
            "frequency_penalty": 3.5,
            "presence_penalty": "high",
            "stop": ["ok", ""],
        }
    }
    overrides = generate_train_data.resolve_row_sampling(sample, counts)
    assert overrides == {}
    assert counts == {
        "temperature": 1,
        "top_p": 1,
        "top_k": 1,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "stop": 1,
    }

    counts = _fresh_counts()
    mixed = {"sampling": {"temperature": 0.6, "top_k": True}}
    assert generate_train_data.resolve_row_sampling(mixed, counts) == {"temperature": 0.6}
    assert counts["top_k"] == 1


def test_above_one_temperature_from_traffic_is_honored():
    # ~3% of internal traffic samples above T=1 (up to ~1.33); the sidecar
    # range is wider than the CLI's [0, 1] on purpose.
    counts = _fresh_counts()
    overrides = generate_train_data.resolve_row_sampling(
        {"sampling": {"temperature": 1.33}}, counts
    )
    assert overrides == {"temperature": 1.33}
    assert sum(counts.values()) == 0


def test_disabled_top_k_passes_through():
    counts = _fresh_counts()
    overrides = generate_train_data.resolve_row_sampling(
        {"sampling": {"top_k": -1}}, counts
    )
    assert overrides == {"top_k": -1}
    assert sum(counts.values()) == 0


def test_overrides_win_over_cli_globals_in_query_kwargs():
    args = _args(top_p=0.8, top_k=20)
    messages = [{"role": "user", "content": "hi"}]
    kwargs = generate_train_data.build_query_kwargs(
        args,
        messages,
        sampling={
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 40,
            "frequency_penalty": 0.7,
            "presence_penalty": -0.25,
            "stop": ["END"],
        },
    )
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.95
    assert kwargs["frequency_penalty"] == 0.7
    assert kwargs["presence_penalty"] == -0.25
    assert kwargs["stop"] == ["END"]
    assert kwargs["extra_body"]["top_k"] == 40
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_partial_overrides_keep_cli_globals_for_the_rest():
    args = _args(top_p=0.8, top_k=20)
    messages = [{"role": "user", "content": "hi"}]
    kwargs = generate_train_data.build_query_kwargs(
        args, messages, sampling={"temperature": 0.5}
    )
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.8
    assert kwargs["extra_body"]["top_k"] == 20
    assert "frequency_penalty" not in kwargs
    assert "stop" not in kwargs


def test_global_mode_kwargs_are_unchanged_by_construction():
    args = _args(sampling_mode="global")
    messages = [{"role": "user", "content": "hi"}]
    assert generate_train_data.build_query_kwargs(args, messages) == (
        generate_train_data.build_query_kwargs(args, messages, sampling=None)
    )


def test_output_row_preserves_sampling_sidecar(monkeypatch):
    # call_sglang mutates the row in place, so whatever rode in on the input
    # line (fold sidecar, sampling) survives into the output artifact.
    class _Message:
        content = "regen"
        reasoning_content = None

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(generate_train_data, "OpenAI", _Client)
    sample = {
        "conversations": [{"role": "user", "content": "hi"}],
        "sampling": {"temperature": 0.5},
        "fold": {"source_line": 3},
    }
    result = generate_train_data.call_sglang(
        _args(), "127.0.0.1:8000", sample, sampling={"temperature": 0.5}
    )
    assert result["status"] == "success"
    assert result["sampling"] == {"temperature": 0.5}
    assert result["fold"] == {"source_line": 3}
    assert json.loads(json.dumps(result))["sampling"]["temperature"] == 0.5


def test_cli_exposes_explicit_sampling_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_train_data.py",
            "--model",
            "target",
            "--server-address",
            "127.0.0.1:8000",
            "--input-file-path",
            "in.jsonl",
            "--output-file-path",
            "out.jsonl",
            "--sampling-mode",
            "per-request",
        ],
    )
    args = generate_train_data.parse_args()
    assert args.sampling_mode == "per-request"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_train_data.py",
            "--model",
            "target",
            "--server-address",
            "127.0.0.1:8000",
            "--input-file-path",
            "in.jsonl",
            "--output-file-path",
            "out.jsonl",
        ],
    )
    assert generate_train_data.parse_args().sampling_mode == "global"

    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_train_data.py", "--sampling-mode", "sometimes"],
    )
    with pytest.raises(SystemExit):
        generate_train_data.parse_args()
