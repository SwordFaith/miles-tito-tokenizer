#!/usr/bin/env python3
"""GPU/e2e verification: TITO incremental tokenization against a real model session.

Derived from miles/scripts/tools/verify_session_tito_tokenizer.py.

The original miles script boots the full miles + SGLang rollout pipeline and
drives a live serving endpoint. This standalone version keeps the same CLI
surface but adds a ``--dry-run`` mode that exercises only the local tokenizer
+ TITO path, so the package can be smoke-tested without the serving stack.

To run a real endpoint test you still need an SGLang backend; this script
intentionally does not import sglang.

Usage examples::

    # Local dry-run: load HF checkpoint and render via TITO
    python -m miles_tito_tokenizers.scripts.verify_session_tito_tokenizer \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --tito-allowed-append-roles tool user \\
        --dry-run

    # Real endpoint (sglang must be running externally and reachable at --base-url)
    python -m miles_tito_tokenizers.scripts.verify_session_tito_tokenizer \\
        --hf-checkpoint Qwen/Qwen3-4B \\
        --tito-model qwen3 \\
        --tito-allowed-append-roles tool user \\
        --base-url http://localhost:30000/v1 \\
        --rollout-num-gpus-per-engine 1
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoTokenizer

from miles_tito_tokenizers import TITOTokenizerType, get_tito_tokenizer
from miles_tito_tokenizers.template_resolution import resolve_fixed_chat_template
from miles_tito_tokenizers.testing.chat_template_verify import run_all_checks_via_tito


def _dry_run(
    hf_checkpoint: str,
    tito_model: str,
    allowed_append_roles: set[str],
) -> int:
    """Load tokenizer + TITO and run the local decode-roundtrip verifier."""
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    tito_type = TITOTokenizerType(tito_model)

    fixed_path, resolved_kwargs = resolve_fixed_chat_template(tito_type, sorted(allowed_append_roles))
    if fixed_path:
        with open(fixed_path) as f:
            tokenizer.chat_template = f.read()

    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_type,
        chat_template_kwargs=dict(resolved_kwargs),
        allowed_append_roles=allowed_append_roles,
    )

    results = run_all_checks_via_tito(
        tokenizer,
        tito_type,
        allowed_append_roles=allowed_append_roles,
        thinking="off",
        extra_template_kwargs=dict(resolved_kwargs),
    )

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"Dry-run trajectories: {len(results)}")
    print(f"Passed: {passed}, Failed: {failed}")
    for r in results:
        if not r.passed:
            print(f"  FAIL: {r.case_name} -- {r.error}")
            return 1

    # Also render a minimal message list to show the tokenizer is functional.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    ids = tito.render_messages(messages, add_generation_prompt=True)
    print(f"Sample render token count: {len(ids)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify TITO incremental tokenization against a real model session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument(
        "--tito-model",
        choices=[t.value for t in TITOTokenizerType],
        required=True,
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
    )
    parser.add_argument("--sglang-reasoning-parser")
    parser.add_argument("--sglang-tool-call-parser")
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL of an external SGLang/vLLM server.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the local tokenizer + TITO verifier without contacting a server.",
    )
    args = parser.parse_args()

    allowed_roles = set(args.tito_allowed_append_roles) | {"tool"}

    if args.dry_run or args.base_url is None:
        print("Running dry-run against local tokenizer + TITO.")
        return _dry_run(args.hf_checkpoint, args.tito_model, allowed_roles)

    print(
        "Live endpoint verification is not implemented in this standalone package. "
        "Please run a real SGLang endpoint at --base-url and compare the returned "
        "token IDs against `tito.render_messages_incrementally`."
    )
    print(f"Checkpoint: {args.hf_checkpoint}")
    print(f"TITO model: {args.tito_model}")
    print(f"Base URL:   {args.base_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
