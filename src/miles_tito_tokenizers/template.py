"""Core chat template operations: load from HuggingFace and render from string.

Derived from miles.utils.chat_template_utils.template.
"""

from __future__ import annotations

from typing import Any

from jinja2 import TemplateError

from miles_tito_tokenizers.encoders import deepseek_v32, deepseek_v4
from miles_tito_tokenizers.message_utils import normalize_tool_arguments
from miles_tito_tokenizers.tool_utils import extract_tool_dicts


def apply_chat_template(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs: Any,
) -> str | list[int]:
    """Apply chat template via HF tokenizer in SGLang style.

    Passes ``return_dict=False`` to match SGLang's ``serving_chat.py``,
    ensuring the result is ``str`` (tokenize=False) or ``list[int]``
    (tokenize=True), not a ``BatchEncoding`` or ``dict``.
    """
    tools = extract_tool_dicts(tools)

    if deepseek_v32.is_deepseek_v32(tokenizer):
        rendered = deepseek_v32.render_messages(
            normalize_tool_arguments(messages, "json"), tools=tools, **kwargs
        )
        return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered

    if deepseek_v4.is_deepseek_v4(tokenizer):
        rendered = deepseek_v4.render_messages(
            normalize_tool_arguments(messages, "json"), tools=tools, **kwargs
        )
        return tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered

    messages = normalize_tool_arguments(messages, "dict")
    render_kwargs = dict(add_generation_prompt=add_generation_prompt, **kwargs)

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, tools=tools, return_dict=False, **render_kwargs
        )
    except TemplateError as e:
        if tools is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    tools=[t["function"] if "function" in t else t for t in tools],
                    return_dict=False,
                    **render_kwargs,
                )
            except TemplateError as te:
                raise ValueError(
                    f"Chat template rendering failed (tool format fallback): {te}"
                ) from te
        raise ValueError(f"Chat template rendering failed: {e}") from e


def render_messages(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str | list[int]:
    """Convenience wrapper over ``apply_chat_template`` matching ``TITOTokenizer``."""
    return apply_chat_template(
        messages,
        tokenizer=tokenizer,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **kwargs,
    )
