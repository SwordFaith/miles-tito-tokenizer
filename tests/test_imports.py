"""Smoke tests for package imports and basic factory behavior."""

from __future__ import annotations

import pytest

from miles_tito_tokenizers import TITOTokenizer, TITOTokenizerType, get_tito_tokenizer
from miles_tito_tokenizers.encoders import deepseek_v32, deepseek_v4
from miles_tito_tokenizers.message_utils import (
    assert_messages_append_only_with_allowed_role,
    message_matches,
)
from miles_tito_tokenizers.template_jinja import apply_chat_template_from_str


def test_all_tokenizer_types_importable() -> None:
    assert list(TITOTokenizerType)


def test_get_tito_tokenizer_default() -> None:
    class FakeTokenizer:
        name_or_path = "dummy"

    tito = get_tito_tokenizer(FakeTokenizer())
    assert isinstance(tito, TITOTokenizer)


def test_qwen3_factory_requires_tokenizer_with_encode() -> None:
    class FakeTokenizer:
        name_or_path = "Qwen/Qwen3-4B"

        def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
            return [1]

        def convert_tokens_to_ids(self, token: str) -> int:
            return 2

    tito = get_tito_tokenizer(
        FakeTokenizer(), tokenizer_type="qwen3", allowed_append_roles=["tool", "user"]
    )
    assert isinstance(tito, TITOTokenizer)


def test_deepseek_encoders_vendored() -> None:
    assert callable(deepseek_v32.encode_messages)
    assert callable(deepseek_v4.encode_messages)
    assert callable(deepseek_v32.is_deepseek_v32)
    assert callable(deepseek_v4.is_deepseek_v4)


def test_message_matches() -> None:
    assert message_matches({"role": "user", "content": "hi"}, {"role": "user", "content": "hi"})
    assert not message_matches(
        {"role": "user", "content": "hi"}, {"role": "user", "content": "hello"}
    )


def test_assert_append_only_ok() -> None:
    old = [{"role": "user", "content": "hi"}]
    new = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "result"},
    ]
    assert_messages_append_only_with_allowed_role(old, new, ["tool"])


def test_assert_append_only_fails_bad_role() -> None:
    old = [{"role": "user", "content": "hi"}]
    new = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "inject"},
    ]
    with pytest.raises(ValueError):
        assert_messages_append_only_with_allowed_role(old, new, ["tool"])


def test_apply_chat_template_from_str_basic() -> None:
    template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    messages = [{"role": "user", "content": "hello"}]
    rendered = apply_chat_template_from_str(template, messages)
    assert "user: hello" in rendered
