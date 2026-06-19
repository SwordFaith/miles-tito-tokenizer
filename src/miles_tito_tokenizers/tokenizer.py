"""TITO tokenizer — incremental tokenization for pretokenized prefix reuse.

Derived from miles.utils.chat_template_utils.tito_tokenizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miles_tito_tokenizers.comparator import TokenSeqComparator
from miles_tito_tokenizers.message_utils import assert_messages_append_only_with_allowed_role
from miles_tito_tokenizers.template import apply_chat_template

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

_VALID_ROLES = frozenset({"tool", "user", "system"})

_DUMMY_SYSTEM: dict[str, Any] = {"role": "system", "content": "dummy system"}


@dataclass(frozen=True)
class FixedTemplateRow:
    """A ``(roles, template, extra_kwargs)`` row owned by a TITO tokenizer family."""

    allowed_roles: frozenset[str]
    template: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dummy assistant message with tool_calls matching *tool_responses*."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id") or f"call0000{i}",
                "type": "function",
                "function": {
                    "name": resp.get("name") or "dummy_func",
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }


class TITOTokenizer:
    """Incremental tokenization and prefix merging for appended non-assistant turns."""

    max_trim_tokens: int = 0
    trailing_token_ids: frozenset[int] = frozenset()

    SUPPORTED_TEMPLATES: tuple[FixedTemplateRow, ...] = ()

    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        special_token_ids: set[int] | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._assistant_start_str = assistant_start_str
        self.allowed_append_roles: list[str] = (
            allowed_append_roles if allowed_append_roles is not None else ["tool"]
        )
        self.special_token_ids: set[int] = special_token_ids

    def create_comparator(self) -> TokenSeqComparator:
        """Create a :class:`TokenSeqComparator` configured for this tokenizer."""
        return TokenSeqComparator(
            self.tokenizer,
            assistant_start_str=self._assistant_start_str,
            special_token_ids=self.special_token_ids,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
    ) -> str | list[int]:
        return apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _split_appended_segments(
        self, appended_messages: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        segments: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(appended_messages):
            role = appended_messages[i]["role"]
            if role == "tool":
                j = i + 1
                while j < len(appended_messages) and appended_messages[j]["role"] == "tool":
                    j += 1
                segments.append(appended_messages[i:j])
                i = j
                continue
            if role in {"user", "system"}:
                segments.append([appended_messages[i]])
                i += 1
                continue
            raise ValueError(f"unsupported appended role for TITO segmentation: {role}")

        return segments

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render base vs base+appended and return tokenized suffix."""
        text_without = self.render_messages(base_messages, add_generation_prompt=False, tools=tools)
        text_with = self.render_messages(
            base_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if not text_with.startswith(text_without):
            roles = [msg["role"] for msg in appended_messages] if appended_messages else ["generation_prompt"]
            raise ValueError(f"rendered suffix diff failed for {roles}")
        return self._encode_text(text_with[len(text_without) :])

    def _tokenize_tool_segment(
        self,
        appended_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM, _build_dummy_assistant(appended_messages)],
            appended_messages,
            tools=tools,
        )

    def _tokenize_user_and_system_segment(
        self,
        appended_message: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return self._tokenize_rendered_suffix(
            [_DUMMY_SYSTEM],
            [appended_message],
            tools=tools,
        )

    def tokenize_additional_non_assistant(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Compute incremental token IDs for non-assistant messages appended after the prefix."""
        assert_messages_append_only_with_allowed_role(
            old_messages, new_messages, self.allowed_append_roles
        )
        appended_messages = new_messages[len(old_messages) :]
        incremental: list[int] = []

        for segment in self._split_appended_segments(appended_messages):
            role = segment[0]["role"]
            if role == "tool":
                incremental.extend(self._tokenize_tool_segment(segment, tools))
            elif role == "user" or role == "system":
                incremental.extend(self._tokenize_user_and_system_segment(segment[0], tools))
            else:
                raise ValueError(f"unsupported appended role for TITO tokenization: {role}")

        return incremental + self._tokenize_rendered_suffix(
            new_messages,
            [],
            tools=tools,
            add_generation_prompt=True,
        )

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Merge pretokenized prefix with incremental tokens."""
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        return list(pretokenized_token_ids) + incremental


class Qwen3TITOTokenizer(TITOTokenizer):
    """Qwen3 variant: handles missing newline at the boundary."""

    reasoning_parser = "qwen3"
    tool_call_parser = "qwen25"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )

    _default_assistant_start_str: str = "<|im_start|>assistant"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
        return prefix + incremental


class Qwen35TITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3.5 — same boundary behavior as Qwen3, distinct fixed template."""

    tool_call_parser = "qwen3_coder"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3.5_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3.5_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )


class QwenNextTITOTokenizer(Qwen3TITOTokenizer):
    """Qwen3-Thinking-2507 / Qwen3-Next-Thinking — same boundary behavior as Qwen3."""

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template="qwen3_thinking_2507_and_next_fixed.jinja",
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="qwen3_thinking_2507_and_next_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )


class GLM47TITOTokenizer(TITOTokenizer):
    """GLM 4.7 variant: handles ambiguous boundary tokens in ``merge_tokens``."""

    reasoning_parser = "glm45"
    tool_call_parser = "glm47"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"clear_thinking": False},
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user", "system"}),
            template=None,
            extra_kwargs={"clear_thinking": False},
        ),
    )

    max_trim_tokens: int = 1
    _default_assistant_start_str: str = "<|assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )
        self._observation_id: int = tokenizer.convert_tokens_to_ids("<|observation|>")
        self._user_id: int = tokenizer.convert_tokens_to_ids("<|user|>")
        self._ambiguous_boundary_ids: set[int] = {self._observation_id, self._user_id}
        self.trailing_token_ids = frozenset(self._ambiguous_boundary_ids)

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
        return prefix + incremental


class Nemotron3TITOTokenizer(Qwen3TITOTokenizer):
    """NVIDIA Nemotron 3 family: ``<|im_end|>\n`` message boundaries."""

    reasoning_parser = "nemotron_3"
    tool_call_parser = "qwen3_coder"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"truncate_history_thinking": False},
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user", "system"}),
            template=None,
            extra_kwargs={"truncate_history_thinking": False},
        ),
    )

    _default_assistant_start_str: str = "<|im_start|>assistant\n"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )


def _kimi_segment_special_token_ids(tokenizer: Any) -> set[int]:
    """Kimi specials minus ``<|im_middle|>`` (intra-turn separator, not boundary)."""
    return TokenSeqComparator.collect_special_ids(tokenizer) - {
        tokenizer.convert_tokens_to_ids("<|im_middle|>")
    }


class Kimi25TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.5: ``<|im_end|>`` boundary (no trailing newline)."""

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="kimi_k25_fixed.jinja",
            extra_kwargs={"preserve_thinking": True},
        ),
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
            allowed_append_roles=allowed_append_roles,
        )


class Kimi26TITOTokenizer(TITOTokenizer):
    """Moonshot Kimi K2.6: same boundary as K2.5 + native ``preserve_thinking`` kwarg."""

    reasoning_parser = "kimi_k2"
    tool_call_parser = "kimi_k2_raw_id"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template=None,
            extra_kwargs={"preserve_thinking": True},
        ),
    )

    _default_assistant_start_str: str = "<|im_assistant|>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            special_token_ids=_kimi_segment_special_token_ids(tokenizer),
            allowed_append_roles=allowed_append_roles,
        )


class MinimaxM25TITOTokenizer(TITOTokenizer):
    """MiniMax-M2.5 family: bespoke tag set."""

    reasoning_parser = "minimax-append-think"
    tool_call_parser = "minimax-m2"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="minimax_m25_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )

    _default_assistant_start_str: str = "]~b]ai"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs,
            assistant_start_str or self._default_assistant_start_str,
            allowed_append_roles=allowed_append_roles,
        )
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        assert len(nl_ids) == 1, f"Expected single newline token, got {nl_ids}"
        self._newline_id: int = nl_ids[0]
        self._eos_id: int = tokenizer.convert_tokens_to_ids("[e~[")
        self.trailing_token_ids = frozenset({self._newline_id})

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
        prefix = list(pretokenized_token_ids)
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
        return prefix + incremental


class MinimaxM27TITOTokenizer(MinimaxM25TITOTokenizer):
    """MiniMax-M2.7 family: identical to M2.5, only fixed jinja differs."""

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
        FixedTemplateRow(
            allowed_roles=frozenset({"tool", "user"}),
            template="minimax_m27_fixed.jinja",
            extra_kwargs={"clear_thinking": False},
        ),
    )


class DeepSeekV32TITOTokenizer(TITOTokenizer):
    """DeepSeek V3.2 — official encoder via vendored ``encoding_dsv32``."""

    reasoning_parser = "deepseek-v3"
    tool_call_parser = "deepseekv32"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
            allowed_append_roles=allowed_append_roles,
        )


class DeepSeekV4TITOTokenizer(TITOTokenizer):
    """DeepSeek V4 — official encoder via vendored ``encoding_dsv4``."""

    reasoning_parser = "deepseek-v4"
    tool_call_parser = "deepseekv4"

    SUPPORTED_TEMPLATES = (
        FixedTemplateRow(
            allowed_roles=frozenset({"tool"}),
            template=None,
        ),
    )

    _DEFAULT_ASSISTANT_START = "<｜Assistant｜>"

    def __init__(
        self,
        tokenizer: Any,
        chat_template_kwargs: dict[str, Any] | None = None,
        assistant_start_str: str | None = None,
        allowed_append_roles: list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            assistant_start_str=assistant_start_str or self._DEFAULT_ASSISTANT_START,
            special_token_ids={
                tokenizer.convert_tokens_to_ids("<｜User｜>"),
                tokenizer.convert_tokens_to_ids("<｜Assistant｜>"),
            },
            allowed_append_roles=allowed_append_roles,
        )
