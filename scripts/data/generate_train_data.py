import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate JSONL conversations through OpenAI-compatible sglang servers."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-address", nargs="+", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--sampling-mode",
        choices=("global", "per-request"),
        default="global",
        help=(
            "global: every request uses the CLI sampling values. per-request: "
            "each input row's `sampling` object overrides them field by field; "
            "absent fields keep the CLI values, present-but-invalid fields "
            "fall back to them and are counted."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")

    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if args.top_p is not None and not 0.0 <= args.top_p <= 1.0:
        raise ValueError("top-p must be between 0.0 and 1.0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if args.min_p is not None and not 0.0 <= args.min_p <= 1.0:
        raise ValueError("min-p must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be greater than 0")


def get_random_reasoning_effort():
    return random.choices(["low", "medium", "high"], weights=[4, 4, 2], k=1)[0]


def _valid_temperature(value):
    # Matches the SpecLoop expand convention (any finite T >= 0), NOT the
    # [0, 1] range validate_args enforces on the CLI value: sidecar values
    # were already accepted by serving (traffic carries T up to ~1.33), and
    # clamping them here would silently distort the per-request policy.
    return isinstance(value, (int, float)) and 0.0 <= float(value) < float("inf")


def _valid_top_p(value):
    return isinstance(value, (int, float)) and 0.0 < float(value) <= 1.0


def _valid_top_k(value):
    # -1 disables top_k on the serving side; SpecLoop fold rows carry it raw.
    return isinstance(value, int) and (value == -1 or value > 0)


def _valid_penalty(value):
    return isinstance(value, (int, float)) and -2.0 <= float(value) <= 2.0


def _valid_stop(value):
    # Empty lists resolve to no override so downstream can trust that a
    # resolved `stop` is always meaningful.
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


# Per-request `sampling` fields honored from the input row, with their
# validators. This is the same allowlist SpecLoop fold preserves.
SAMPLING_VALIDATORS = {
    "temperature": _valid_temperature,
    "top_p": _valid_top_p,
    "top_k": _valid_top_k,
    "frequency_penalty": _valid_penalty,
    "presence_penalty": _valid_penalty,
    "stop": _valid_stop,
}


def resolve_row_sampling(sample, fallback_counts):
    """Extract one row's usable `sampling` overrides in per-request mode.

    Field-by-field: present-and-valid values are returned as overrides;
    absent fields are simply not overridden (the CLI globals apply);
    present-but-invalid values are dropped and counted, falling back to
    the CLI globals the same way. Never fatal: a raw traffic sidecar is
    allowed to carry garbage, and regeneration must keep going.
    """
    sidecar = sample.get("sampling")
    if not isinstance(sidecar, dict):
        return {}
    overrides = {}
    for key, is_valid in SAMPLING_VALIDATORS.items():
        if key not in sidecar:
            continue
        if is_valid(sidecar[key]):
            overrides[key] = sidecar[key]
        else:
            fallback_counts[key] += 1
    return overrides


def compute_context_length(conversations):
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    length += len(part["text"].split())
    return length


def build_query_kwargs(args, messages, max_tokens=None, sampling=None):
    query_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty

    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k

    # Per-request overrides land after the CLI globals so a row's own
    # sampling policy wins field by field. resolve_row_sampling is the sole
    # producer and only emits validated, meaningful values, so they are
    # trusted verbatim here.
    if sampling:
        for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"):
            if key in sampling:
                query_kwargs[key] = sampling[key]
        if "top_k" in sampling:
            extra_body["top_k"] = sampling["top_k"]

    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.enable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    if args.disable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if extra_body:
        query_kwargs["extra_body"] = extra_body

    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def error_sample(sample, message):
    sample["status"] = "error"
    sample["error"] = message
    return sample


def call_sglang(args, server_address, sample, max_tokens=None, sampling=None):
    conversations = sample.get("conversations")
    if not conversations:
        return error_sample(sample, "Missing conversations")
    if conversations[0].get("role") == "assistant":
        return error_sample(sample, "Data starts with an assistant message")

    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    regenerated = []

    for message in conversations:
        role = message.get("role")
        if role == "system":
            regenerated.append(message)
            continue
        if role == "assistant":
            continue
        if role != "user":
            return error_sample(sample, f"Invalid message role: {role}")

        regenerated.append(message)
        try:
            response = client.chat.completions.create(
                **build_query_kwargs(
                    args, regenerated, max_tokens=max_tokens, sampling=sampling
                )
            )
        except Exception as exc:
            return error_sample(sample, str(exc))

        response_message = {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
        if args.is_reasoning_model:
            response_message["thinking"] = response.choices[0].message.reasoning_content
        regenerated.append(response_message)

    sample["conversations"] = regenerated
    sample["status"] = "success"
    return sample


def count_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def find_resume_offset(output_path, error_path):
    if not os.path.exists(output_path):
        return 0, 0, 0

    success_count = count_lines(output_path)
    error_count = count_lines(error_path) if os.path.exists(error_path) else 0
    return success_count + error_count, success_count, error_count


def validate_server(args, server_address, probe):
    start_time = time.perf_counter()
    try:
        result = call_sglang(args, server_address, dict(probe), max_tokens=1)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    elapsed = time.perf_counter() - start_time
    return server_address, result, elapsed


def validate_servers(args):
    probe = {"conversations": [{"role": "user", "content": "Hello"}]}
    server_results = {}
    server_count = len(args.server_address)

    print(f"Validating {server_count} sglang servers in parallel...", flush=True)
    with ThreadPoolExecutor(max_workers=server_count) as executor:
        future_to_server = {
            executor.submit(
                validate_server,
                args,
                server_address,
                probe,
            ): server_address
            for server_address in args.server_address
        }

        for completed_count, future in enumerate(as_completed(future_to_server), 1):
            server_address = future_to_server[future]
            try:
                _, result, elapsed = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
                elapsed = 0.0

            if result.get("status") == "success":
                server_results[server_address] = True
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"ok server {server_address} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            else:
                server_results[server_address] = False
                print(
                    f"[validate {completed_count}/{server_count}] "
                    f"skip server {server_address} elapsed={elapsed:.2f}s: "
                    f"{result.get('error')}",
                    flush=True,
                )

    valid_servers = [
        server_address
        for server_address in args.server_address
        if server_results[server_address]
    ]
    invalid_servers = [
        server_address
        for server_address in args.server_address
        if not server_results[server_address]
    ]

    print(f"Available servers ({len(valid_servers)}/{server_count}): {valid_servers}")
    print(
        f"Unavailable servers ({len(invalid_servers)}/{server_count}): "
        f"{invalid_servers}"
    )
    if not valid_servers:
        raise RuntimeError("No available sglang server")
    return valid_servers


def write_finished_result(
    future,
    output_handle,
    error_handle,
    stats,
):
    sample = future.result()
    if sample["status"] == "error":
        error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["errors"] += 1
        return

    context_length = compute_context_length(sample.get("conversations", []))
    stats["context_sum"] += context_length
    stats["context_min"] = (
        context_length
        if stats["context_min"] is None
        else min(stats["context_min"], context_length)
    )
    stats["context_max"] = max(stats["context_max"], context_length)
    stats["success"] += 1
    output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_config(args):
    print("Configuration:")
    print(f"  model: {args.model}")
    print(f"  servers: {args.server_address}")
    print(f"  input: {args.input_file_path}")
    print(f"  output: {args.output_file_path}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  sampling_mode: {args.sampling_mode}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  min_p: {args.min_p}")
    print(f"  resume: {args.resume}")


def main():
    args = parse_args()
    validate_args(args)
    print_config(args)

    total_lines = count_lines(args.input_file_path)
    error_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    skip_lines, existing_success, existing_errors = (
        find_resume_offset(args.output_file_path, error_path)
        if args.resume
        else (0, 0, 0)
    )
    if skip_lines >= total_lines:
        print(f"All {total_lines} samples are already processed.")
        return

    if args.resume and skip_lines > 0:
        print(
            "Resume mode: "
            f"{existing_success} success, {existing_errors} errors, skip {skip_lines}"
        )

    valid_servers = validate_servers(args)
    print(f"Using servers: {valid_servers}")

    file_mode = "a" if args.resume and skip_lines > 0 else "w"
    stats = {
        "success": 0,
        "errors": 0,
        "context_sum": 0,
        "context_min": None,
        "context_max": 0,
    }
    fallback_counts = {key: 0 for key in SAMPLING_VALIDATORS}
    rows_with_sampling = 0
    queues = {server_address: [] for server_address in valid_servers}
    next_server_index = 0
    submitted_count = 0

    with (
        open(args.input_file_path, "r", encoding="utf-8") as input_handle,
        open(args.output_file_path, file_mode, encoding="utf-8") as output_handle,
        open(error_path, file_mode, encoding="utf-8") as error_handle,
        ThreadPoolExecutor(max_workers=args.concurrency * len(valid_servers)) as executor,
    ):
        for _ in range(skip_lines):
            next(input_handle, None)

        progress_total = (
            total_lines
            if args.num_samples is None
            else min(total_lines, skip_lines + args.num_samples)
        )
        progress = tqdm(total=progress_total, initial=skip_lines, desc="Processing")
        for line in input_handle:
            if args.num_samples is not None and submitted_count >= args.num_samples:
                break

            sample = json.loads(line)
            sampling = None
            if args.sampling_mode == "per-request":
                if isinstance(sample.get("sampling"), dict):
                    rows_with_sampling += 1
                sampling = resolve_row_sampling(sample, fallback_counts)
            server_address = valid_servers[next_server_index]
            next_server_index = (next_server_index + 1) % len(valid_servers)

            while len(queues[server_address]) >= args.concurrency:
                wrote_result = False
                for future in list(queues[server_address]):
                    if future.done():
                        write_finished_result(
                            future, output_handle, error_handle, stats
                        )
                        queues[server_address].remove(future)
                        wrote_result = True
                        break
                if not wrote_result:
                    time.sleep(0.05)

            future = executor.submit(
                call_sglang, args, server_address, sample, None, sampling
            )
            queues[server_address].append(future)
            submitted_count += 1
            progress.update(1)

        for server_address in valid_servers:
            for future in queues[server_address]:
                write_finished_result(future, output_handle, error_handle, stats)
        progress.close()

    print("Processing completed.")
    print(f"  success: {stats['success']}")
    print(f"  errors: {stats['errors']}")
    if args.sampling_mode == "per-request":
        print(f"  rows_with_sampling: {rows_with_sampling}")
        fallback_total = sum(fallback_counts.values())
        print(f"  sampling_fallbacks: {fallback_total}")
        for key, count in fallback_counts.items():
            if count:
                print(f"    {key}: {count}")
        manifest_path = args.output_file_path.replace(".jsonl", "_sampling_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
            json.dump(
                {
                    "sampling_mode": args.sampling_mode,
                    "submitted": submitted_count,
                    "rows_with_sampling": rows_with_sampling,
                    "sampling_fallbacks": fallback_counts,
                    "cli_fallback_values": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                    },
                },
                manifest_handle,
                ensure_ascii=False,
                indent=2,
            )
            manifest_handle.write("\n")
        print(f"  sampling_manifest: {manifest_path}")
    if stats["success"] > 0:
        avg_context = stats["context_sum"] / stats["success"]
        print(f"  context_min: {stats['context_min']}")
        print(f"  context_max: {stats['context_max']}")
        print(f"  context_avg: {avg_context:.2f}")


if __name__ == "__main__":
    main()
