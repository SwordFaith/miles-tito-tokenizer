"""Tokenizer type enum and factory.

Derived from miles.utils.chat_template_utils.tito_tokenizer.
"""

from __future__ import annotations

from typing import Any

from miles_tito_tokenizers._compat import StrEnum
from miles_tito_tokenizers.template_resolution import (
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)
from miles_tito_tokenizers.tokenizer import (
    DeepSeekV32TITOTokenizer,
    DeepSeekV4TITOTokenizer,
    GLM47TITOTokenizer,
    Kimi25TITOTokenizer,
    Kimi26TITOTokenizer,
    MinimaxM25TITOTokenizer,
    MinimaxM27TITOTokenizer,
    Nemotron3TITOTokenizer,
    Qwen35TITOTokenizer,
    Qwen3TITOTokenizer,
    QwenNextTITOTokenizer,
    TITOTokenizer,
)


class TITOTokenizerType(StrEnum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    QWENNEXT = "qwennext"
    GLM47 = "glm47"
    NEMOTRON3 = "nemotron3"
    KIMI25 = "kimi25"
    KIMI26 = "kimi26"
    MINIMAX_M25 = "minimax_m25"
    MINIMAX_M27 = "minimax_m27"
    DEEPSEEKV32 = "deepseekv32"
    DEEPSEEKV4 = "deepseekv4"

    @classmethod
    def get_tokenizer_class(cls, t: "TITOTokenizerType") -> type[TITOTokenizer]:
        """Resolve the concrete ``TITOTokenizer`` subclass for *t*."""
        match t:
            case cls.DEFAULT:
                return TITOTokenizer
            case cls.QWEN3:
                return Qwen3TITOTokenizer
            case cls.QWEN35:
                return Qwen35TITOTokenizer
            case cls.QWENNEXT:
                return QwenNextTITOTokenizer
            case cls.GLM47:
                return GLM47TITOTokenizer
            case cls.NEMOTRON3:
                return Nemotron3TITOTokenizer
            case cls.KIMI25:
                return Kimi25TITOTokenizer
            case cls.KIMI26:
                return Kimi26TITOTokenizer
            case cls.MINIMAX_M25:
                return MinimaxM25TITOTokenizer
            case cls.MINIMAX_M27:
                return MinimaxM27TITOTokenizer
            case cls.DEEPSEEKV32:
                return DeepSeekV32TITOTokenizer
            case cls.DEEPSEEKV4:
                return DeepSeekV4TITOTokenizer
            case _:
                raise ValueError(f"Unknown TITOTokenizerType: {t!r}")


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    chat_template_kwargs: dict[str, Any] | None = None,
    assistant_start_str: str | None = None,
    allowed_append_roles: list[str] | None = None,
) -> TITOTokenizer:
    """Create a ``TITOTokenizer`` instance.

    When ``tokenizer_type`` is not ``default`` and the selected family has
    ``SUPPORTED_TEMPLATES``, this function automatically resolves the smallest
    matching fixed template and merges its ``extra_kwargs`` into
    ``chat_template_kwargs``.  Explicit user kwargs always win on conflict.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must not be None")
    if isinstance(tokenizer_type, str):
        tokenizer_type = TITOTokenizerType(tokenizer_type)

    cls = TITOTokenizerType.get_tokenizer_class(tokenizer_type)

    if tokenizer_type != TITOTokenizerType.DEFAULT and cls.SUPPORTED_TEMPLATES:
        _, resolved_kwargs = resolve_fixed_chat_template(
            cls,
            set(allowed_append_roles or ["tool"]),
        )
        resolved_kwargs.update(chat_template_kwargs or {})
        chat_template_kwargs = resolved_kwargs

    kwargs: dict[str, Any] = {"chat_template_kwargs": chat_template_kwargs}
    if assistant_start_str is not None:
        kwargs["assistant_start_str"] = assistant_start_str
    if allowed_append_roles is not None:
        kwargs["allowed_append_roles"] = allowed_append_roles

    return cls(tokenizer, **kwargs)
