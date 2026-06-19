"""Verify that a chat template satisfies the append-only invariant.

Derived from miles.utils.test_utils.chat_template_verify.
Reused under the Apache License 2.0 of the original miles project.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles_tito_tokenizers.template_jinja import apply_chat_template_from_str

if TYPE_CHECKING:
    from miles_tito_tokenizers.tokenizer import TITOTokenizer
    from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType


@dataclass
class VerifyResult:
    """Result of a single append-only verification case."""

    case_name: str
    passed: bool = False
    error: str | None = None


def simulate_pretokenized_path(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Simulate the pretokenized incremental path at text level."""
    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tools,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )

    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    """Standard path: render all messages with generation prompt."""
    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )


def assert_pretokenized_equals_standard(
    chat_template, messages, pretokenized_num_message, tools=None, **kwargs
):
    """Assert pretokenized incremental path produces same text as standard full render."""
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(
        chat_template, messages, pretokenized_num_message, tools=tools, **kwargs
    )
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


def verify_append_only(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Check that the template satisfies the append-only invariant."""
    try:
        standard = get_standard_result(
            chat_template, deepcopy(messages), tools=tools, **template_kwargs
        )
        pretokenized = simulate_pretokenized_path(
            chat_template,
            deepcopy(messages),
            pretokenized_num_message,
            tools=tools,
            **template_kwargs,
        )
        if pretokenized != standard:
            return VerifyResult(
                case_name=case_name,
                passed=False,
                error=f"Pretokenized (N={pretokenized_num_message}) != standard",
            )
        return VerifyResult(case_name=case_name, passed=True)
    except ValueError as e:
        return VerifyResult(case_name=case_name, passed=False, error=str(e))
    except Exception as e:
        return VerifyResult(
            case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Built-in test cases
# ---------------------------------------------------------------------------

from miles_tito_tokenizers.testing.mock_trajectories import (  # noqa: E402
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiRoleSequenceTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiTurnNoToolTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)


def _short_name(cls: type) -> str:
    name = cls.__name__.replace("Trajectory", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_TRAJECTORIES: list[type] = [
    SingleToolTrajectory,
    MultiTurnTrajectory,
    MultiToolSingleTurnTrajectory,
    ParallelToolsTrajectory,
    LongChainTrajectory,
    MultiUserToolChainTrajectory,
    RetrySystemTrajectory,
    IntermediateSystemTrajectory,
    SimpleNoToolTrajectory,
    MultiTurnNoToolTrajectory,
    SingleToolThinkingTrajectory,
    MultiTurnThinkingTrajectory,
    LongChainThinkingTrajectory,
    MultiUserTurnThinkingTrajectory,
    IntermediateSystemThinkingTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiRoleSequenceTrajectory,
]


@dataclass(frozen=True)
class CaseSpec:
    """One verify case with classification metadata copied from its trajectory."""

    case_name: str
    messages: list[dict]
    tools: list[dict] | None
    pretokenized_num_message: int
    append_roles: frozenset[str]
    is_thinking: bool


def _expand(traj_cls: type) -> list[CaseSpec]:
    """Expand one trajectory into one CaseSpec per PRETOKENIZE_POSITIONS value."""
    name = _short_name(traj_cls)
    return [
        CaseSpec(
            case_name=f"{name}_n{n}",
            messages=list(traj_cls.MESSAGES),
            tools=traj_cls.TOOLS,
            pretokenized_num_message=n,
            append_roles=traj_cls.APPEND_ROLES,
            is_thinking=traj_cls.IS_THINKING,
        )
        for n in traj_cls.PRETOKENIZE_POSITIONS
    ]


ALL_CASES: list[CaseSpec] = [c for t in _TRAJECTORIES for c in _expand(t)]

THINKING_MODES: tuple[str, ...] = ("off", "on", "both")


def select_cases(
    *,
    allowed_append_roles: set[str],
    is_thinking: bool | None,
) -> list[CaseSpec]:
    """Select trajectory cases by append-role surface and (optionally) thinking flag."""
    out = []
    for case in ALL_CASES:
        if not case.append_roles.issubset(allowed_append_roles):
            continue
        if is_thinking is not None and case.is_thinking != is_thinking:
            continue
        out.append(case)
    return out


def enable_thinking_variants(thinking: str) -> list[dict]:
    """Return the list of ``enable_thinking`` kwarg variants to apply per case."""
    if thinking == "off":
        return [{"enable_thinking": False}]
    if thinking == "on":
        return [{"enable_thinking": True}]
    if thinking == "both":
        return [{"enable_thinking": True}, {"enable_thinking": False}]
    raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")


def format_case_id(case: CaseSpec, kwargs: dict) -> str:
    """Human-readable label for a ``(case, template_kwargs)`` tuple."""
    parts = []
    if case.is_thinking:
        parts.append("thinking")
    if "enable_thinking" in kwargs:
        parts.append(f"eth={kwargs['enable_thinking']}")
    if "clear_thinking" in kwargs:
        parts.append(f"ct={kwargs['clear_thinking']}")
    if "preserve_thinking" in kwargs:
        parts.append(f"pt={kwargs['preserve_thinking']}")
    return f"{case.case_name}-{'-'.join(parts)}"


@dataclass
class CoverageReport:
    """Coverage of cases across ``(is_thinking, append_roles \\ {tool})``."""

    covered: list[tuple[bool, tuple[str, ...]]]
    missing: list[tuple[bool, tuple[str, ...]]]


def check_coverage() -> CoverageReport:
    """Enumerate ``thinking × append-role-subset`` combinations and report gaps."""
    combos: set[tuple[bool, tuple[str, ...]]] = set()
    for case in ALL_CASES:
        roles = tuple(sorted(case.append_roles - {"tool"}))
        combos.add((case.is_thinking, roles))

    covered: list[tuple[bool, tuple[str, ...]]] = []
    missing: list[tuple[bool, tuple[str, ...]]] = []

    for is_thinking in (False, True):
        for roles in [(), ("user",), ("system",), ("user", "system")]:
            key = (is_thinking, roles)
            if key in combos:
                covered.append(key)
            else:
                missing.append(key)

    return CoverageReport(covered=covered, missing=missing)


def run_all_checks(
    chat_template: str,
    *,
    allowed_append_roles: set[str],
    thinking: str,
    extra_template_kwargs: dict[str, Any] | None = None,
) -> list[VerifyResult]:
    """Run verification cases filtered by *allowed_append_roles* and *thinking*."""
    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(
        allowed_append_roles=allowed_append_roles, is_thinking=is_thinking_filter
    )
    kwarg_variants = enable_thinking_variants(thinking)
    base_kwargs = dict(extra_template_kwargs or {})

    results: list[VerifyResult] = []
    for case in selected:
        for kwargs in kwarg_variants:
            merged = {**base_kwargs, **kwargs}
            case_name = format_case_id(case, merged)
            results.append(
                verify_append_only(
                    chat_template,
                    case.messages,
                    case.pretokenized_num_message,
                    tools=case.tools,
                    case_name=case_name,
                    **merged,
                )
            )
    return results


# ---------------------------------------------------------------------------
# TITO-instance verification: decode-roundtrip equality
# ---------------------------------------------------------------------------


def verify_append_only_via_tito_instance(
    tito: "TITOTokenizer",
    tokenizer: Any,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Decode-roundtrip verify with a pre-built TITO instance."""
    try:
        n = pretokenized_num_message
        m = n
        while m < len(messages) and messages[m].get("role") != "assistant":
            m += 1
        if m == n:
            return VerifyResult(
                case_name=case_name,
                passed=False,
                error=(
                    f"Empty appendix at N={n}: messages[{n}] is assistant. "
                    "PRETOKENIZE_POSITIONS must land at a post-assistant boundary."
                ),
            )

        prefix_msgs = deepcopy(messages[:n])
        full_msgs = deepcopy(messages[:m])

        prefix_text = tito.render_messages(
            prefix_msgs,
            tools=tools,
            add_generation_prompt=False,
        )
        full_text = tito.render_messages(
            full_msgs,
            tools=tools,
            add_generation_prompt=True,
        )

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        trailing = tito.trailing_token_ids
        while prefix_ids and prefix_ids[-1] in trailing:
            prefix_ids = prefix_ids[:-1]
        merged_ids = tito.merge_tokens(prefix_msgs, full_msgs, prefix_ids, tools=tools)
        merged_text = tokenizer.decode(merged_ids)

        if merged_text == full_text:
            return VerifyResult(case_name=case_name, passed=True)

        common_len = min(len(merged_text), len(full_text))
        diff_idx = next(
            (i for i in range(common_len) if merged_text[i] != full_text[i]),
            common_len,
        )
        ctx_start = max(0, diff_idx - 60)
        ctx_end = diff_idx + 60
        return VerifyResult(
            case_name=case_name,
            passed=False,
            error=(
                f"Decode-roundtrip mismatch (N={pretokenized_num_message}) at char {diff_idx}\n"
                f"  expected: ...{full_text[ctx_start:ctx_end]!r}...\n"
                f"  actual:   ...{merged_text[ctx_start:ctx_end]!r}..."
            ),
        )
    except Exception as e:
        return VerifyResult(
            case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}"
        )


def verify_append_only_via_tito(
    tokenizer: Any,
    tito_model: "TITOTokenizerType | str",
    allowed_append_roles: list[str],
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    """Decode-roundtrip verify, building TITO from the registered family."""
    from miles_tito_tokenizers import get_tito_tokenizer

    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        chat_template_kwargs=dict(template_kwargs),
        allowed_append_roles=list(allowed_append_roles),
    )
    return verify_append_only_via_tito_instance(
        tito,
        tokenizer,
        messages,
        pretokenized_num_message,
        tools=tools,
        case_name=case_name,
        **template_kwargs,
    )


def run_all_checks_via_tito(
    tokenizer: Any,
    tito_model: "TITOTokenizerType | str",
    *,
    allowed_append_roles: set[str],
    thinking: str,
    extra_template_kwargs: dict[str, Any] | None = None,
) -> list[VerifyResult]:
    """Run verification cases through TITO + tokenizer."""
    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(
        allowed_append_roles=allowed_append_roles, is_thinking=is_thinking_filter
    )
    kwarg_variants = enable_thinking_variants(thinking)
    base_kwargs = dict(extra_template_kwargs or {})

    results: list[VerifyResult] = []
    for case in selected:
        # TITO incremental requires a non-empty non-assistant appendix at the
        # boundary. Trajectories that end at the assistant turn (e.g. plain
        # [sys, user, assistant]) have no appendix to verify and are silently
        # skipped here -- the string-based primitive still covers them at the
        # text-prefix layer.
        msgs = case.messages
        n = case.pretokenized_num_message
        if n >= len(msgs) or msgs[n].get("role") == "assistant":
            continue
        for kwargs in kwarg_variants:
            merged = {**base_kwargs, **kwargs}
            case_name = format_case_id(case, merged)
            results.append(
                verify_append_only_via_tito(
                    tokenizer,
                    tito_model,
                    list(allowed_append_roles),
                    case.messages,
                    case.pretokenized_num_message,
                    tools=case.tools,
                    case_name=case_name,
                    **merged,
                )
            )
    return results
