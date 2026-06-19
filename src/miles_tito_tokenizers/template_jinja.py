"""Jinja chat-template helpers.

Derived from miles.utils.chat_template_utils.template.
"""

from __future__ import annotations
from typing import Any

from transformers.utils.chat_template_utils import render_jinja_template

from miles_tito_tokenizers.message_utils import normalize_tool_arguments
from miles_tito_tokenizers.tool_utils import extract_tool_dicts


def load_hf_chat_template(model_id: str) -> str:
    """Load an original chat template from HuggingFace (cached locally).

    Falls back to the tokenizer's ``chat_template`` attribute if the file is
    not present.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.chat_template is not None:
        return tokenizer.chat_template
    raise ValueError(f"No chat_template found for {model_id}")


def apply_chat_template_from_str(
    chat_template: str,
    messages: list[dict],
    *,
    add_generation_prompt: bool = True,
    tools: list[dict[str, Any]] | None = None,
    **kwargs,
) -> str:
    """Render a Jinja2 chat template string (tokenize=False, no tokenizer needed).

    Calls HF transformers' ``render_jinja_template`` directly — the same
    function that ``tokenizer.apply_chat_template`` uses internally. Both
    SGLang and our ``apply_chat_template`` go through that same HF code path.

    Applies tool argument parsing / canonicalization and tool-format fallback.
    """

    def _render(tool_defs: list[dict[str, Any]] | None) -> str:
        rendered, _ = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tools=tool_defs,
            **kwargs,
        )
        return rendered[0]

    messages = normalize_tool_arguments(messages, "dict")
    tool_defs = extract_tool_dicts(tools)
    try:
        return _render(tool_defs)
    except Exception as e:
        if tool_defs is not None:
            try:
                return _render([t["function"] if "function" in t else t for t in tool_defs])
            except Exception as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e


def _find_template_variables(template_str: str) -> set[str]:
    try:
        from jinja2 import Environment, meta
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jinja2 is required for chat-template rendering") from exc
    env = Environment(autoescape=False)
    ast = env.parse(template_str)
    return meta.find_undeclared_variables(ast)


def template_requires_tools(template_str: str) -> bool:
    """Return True if the template references ``tools``."""
    return "tools" in _find_template_variables(template_str)
