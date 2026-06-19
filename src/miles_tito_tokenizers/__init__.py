"""Standalone TITO (Token-In-Token-Out) tokenizers.

This package is derived from the ``miles`` project by radixark/miles
contributors and reused under the Apache License 2.0.
"""

from __future__ import annotations

from miles_tito_tokenizers.comparator import Mismatch, MismatchType, TokenSeqComparator
from miles_tito_tokenizers.encoders import deepseek_v32, deepseek_v4
from miles_tito_tokenizers.message_utils import (
    assert_messages_append_only_with_allowed_role,
    message_matches,
    normalize_tool_arguments,
)
from miles_tito_tokenizers.template import apply_chat_template, render_messages
from miles_tito_tokenizers.template_jinja import (
    apply_chat_template_from_str,
    load_hf_chat_template,
)
from miles_tito_tokenizers.template_resolution import (
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)
from miles_tito_tokenizers.tokenizer import TEMPLATE_DIR, TITOTokenizer
from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType, get_tito_tokenizer
from miles_tito_tokenizers.tool_utils import extract_tool_dicts

__all__ = [
    "apply_chat_template",
    "apply_chat_template_from_str",
    "assert_messages_append_only_with_allowed_role",
    "deepseek_v32",
    "deepseek_v4",
    "extract_tool_dicts",
    "get_tito_tokenizer",
    "load_hf_chat_template",
    "message_matches",
    "Mismatch",
    "MismatchType",
    "normalize_tool_arguments",
    "render_messages",
    "TEMPLATE_DIR",
    "resolve_fixed_chat_template",
    "resolve_reasoning_and_tool_call_parser",
    "TITOTokenizer",
    "TITOTokenizerType",
    "TokenSeqComparator",
]
