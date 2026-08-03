import argparse
import json
import os

from transformers import AutoTokenizer

from deepspec.data.live_hidden_data import prepare_live_hidden_data


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize train-ready JSONL for DeepSpec live hidden-state training."
        )
    )
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--target-model-name-or-path", required=True)
    parser.add_argument("--target-revision")
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--min-loss-tokens", required=True, type=int)
    parser.add_argument("--expected-num-samples", type=int)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.min_loss_tokens <= 0:
        parser.error("--min-loss-tokens must be positive")
    if args.expected_num_samples is not None and args.expected_num_samples <= 0:
        parser.error("--expected-num-samples must be positive")
    return args


def main():
    args = parse_args()
    revision_kwargs = {}
    if args.target_revision is not None and not os.path.isdir(
        args.target_model_name_or_path
    ):
        revision_kwargs["revision"] = args.target_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_name_or_path,
        **revision_kwargs,
    )
    prepared = prepare_live_hidden_data(
        source_path=args.input_file_path,
        filtered_path=args.output_file_path,
        artifact_dir=args.artifact_dir,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        min_loss_tokens=args.min_loss_tokens,
        expected_num_samples=args.expected_num_samples,
        target_model_name_or_path=args.target_model_name_or_path,
        target_revision=args.target_revision,
        replace_existing=args.replace_existing,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(prepared.manifest_path),
                "filtered_path": str(prepared.filtered_path),
                "rejected_path": str(prepared.rejected_path),
                "source_samples": prepared.source_samples,
                "accepted_samples": prepared.accepted_samples,
                "rejected_samples": prepared.rejected_samples,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
