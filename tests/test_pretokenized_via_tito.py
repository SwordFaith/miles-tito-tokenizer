"""Unit tests for ``verify_append_only_via_tito_instance`` / ``run_all_checks_via_tito``.

Derived from miles/tests/fast/utils/chat_template_utils/test_pretokenized_via_tito.py.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from miles_tito_tokenizers import (
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
)
from miles_tito_tokenizers.testing.chat_template_verify import (
    run_all_checks_via_tito,
    verify_append_only_via_tito_instance,
)
from miles_tito_tokenizers.testing.mock_trajectories import SingleToolTrajectory
from miles_tito_tokenizers.tokenizer import Qwen3TITOTokenizer


def _setup_tokenizer_with_registered_template(
    model_id: str, tito_type: TITOTokenizerType
):
    """Mirror what production wiring does at startup."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(tito_type, ["tool"])
    if fixed_path:
        with open(fixed_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer, dict(extra_kwargs)


# ---------------------------------------------------------------------------
# (1) PASS on registered families × role surfaces
# ---------------------------------------------------------------------------

_PASS_PARAMS = [
    (
        TITOTokenizerType.QWEN3,
        "Qwen/Qwen3-4B",
        ["tool"],
    ),
    (
        TITOTokenizerType.QWEN3,
        "Qwen/Qwen3-4B",
        ["tool", "user"],
    ),
]


@pytest.mark.parametrize("family,model_id,roles", _PASS_PARAMS)
def test_via_tito_pass_on_registered_families(family, model_id, roles):
    """Registered TITO families round-trip cleanly via decode-roundtrip."""
    tokenizer, extra_kwargs = _setup_tokenizer_with_registered_template(model_id, family)
    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=family,
        chat_template_kwargs=extra_kwargs,
        allowed_append_roles=roles,
    )

    for n in SingleToolTrajectory.PRETOKENIZE_POSITIONS:
        result = verify_append_only_via_tito_instance(
            tito,
            tokenizer,
            SingleToolTrajectory.MESSAGES,
            pretokenized_num_message=n,
            tools=SingleToolTrajectory.TOOLS,
            case_name=f"{family.value}_n{n}",
            **extra_kwargs,
        )
        assert result.passed, result.error


# ---------------------------------------------------------------------------
# (2) FAIL on a test-local buggy subclass
# ---------------------------------------------------------------------------


class _BuggyQwen3TITOTokenizer(Qwen3TITOTokenizer):
    """Test-only Qwen3 variant with the ``\\n`` boundary insertion deleted."""

    def merge_tokens(
        self,
        old_messages,
        new_messages,
        pretokenized_token_ids,
        tools=None,
    ):
        incremental = self.tokenize_additional_non_assistant(
            old_messages, new_messages, tools
        )
        return list(pretokenized_token_ids) + incremental


def test_via_tito_fail_on_buggy_qwen3_subclass():
    """A buggy ``merge_tokens`` produces a junction-level diff that the verifier surfaces."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(
        TITOTokenizerType.QWEN3, ["tool"]
    )
    with open(fixed_path) as f:
        tokenizer.chat_template = f.read()

    tito = _BuggyQwen3TITOTokenizer(
        tokenizer,
        chat_template_kwargs=extra_kwargs,
        allowed_append_roles=["tool"],
    )

    result = verify_append_only_via_tito_instance(
        tito,
        tokenizer,
        SingleToolTrajectory.MESSAGES,
        pretokenized_num_message=3,
        tools=SingleToolTrajectory.TOOLS,
        **extra_kwargs,
    )
    assert not result.passed
    assert "Decode-roundtrip mismatch" in (result.error or "")


# ---------------------------------------------------------------------------
# (3) run_all_checks_via_tito smoke test
# ---------------------------------------------------------------------------


def test_run_all_checks_via_tito_smoke():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(
        TITOTokenizerType.QWEN3, ["tool"]
    )
    with open(fixed_path) as f:
        tokenizer.chat_template = f.read()

    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.QWEN3,
        allowed_append_roles={"tool"},
        thinking="off",
        extra_template_kwargs=extra_kwargs,
    )
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    assert passed, "expected at least one passing case"
    assert not failed, f"unexpected failures: {failed[:3]}"
