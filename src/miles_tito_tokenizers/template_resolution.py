"""Template resolution helpers.

Derived from miles.utils.chat_template_utils.tito_tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from miles_tito_tokenizers.tokenizer import FixedTemplateRow, TITOTokenizer

if TYPE_CHECKING:
    from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType


TEMPLATE_DIR = Path(__file__).parent / "templates"
_VALID_ROLES = frozenset({"tool", "user", "system"})


def _resolve_tokenizer_class(
    tokenizer_type: "TITOTokenizerType | str | type[TITOTokenizer]",
) -> type[TITOTokenizer]:
    if isinstance(tokenizer_type, type) and issubclass(tokenizer_type, TITOTokenizer):
        return tokenizer_type
    from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType

    if isinstance(tokenizer_type, str):
        tokenizer_type = TITOTokenizerType(tokenizer_type)
    return TITOTokenizerType.get_tokenizer_class(tokenizer_type)


def resolve_fixed_chat_template(
    tito_model: "tuple[FixedTemplateRow, ...] | TITOTokenizerType | str | type[TITOTokenizer] | None" = None,
    allowed_append_roles: Iterable[str] | None = None,
    *,
    supported_templates: "tuple[FixedTemplateRow, ...] | TITOTokenizerType | str | type[TITOTokenizer] | None" = None,
) -> tuple[str | None, dict[str, Any]]:
    """Smallest-superset lookup over a family's ``SUPPORTED_TEMPLATES``.

    ``tito_model`` may be:
      - a ``TITOTokenizerType`` value,
      - a model-family string (e.g. ``"qwen3"``),
      - a ``TITOTokenizer`` subclass, or
      - a tuple of ``FixedTemplateRow`` (used directly).

    For backward compatibility with older mtt callers, ``supported_templates``
    is accepted as a keyword alias for ``tito_model``.

    Returns ``(template_path, extra_kwargs)``:

    - ``template_path``: absolute path to a bundled ``.jinja`` file, or ``None``
      when the matched row registers HF-native (kwargs-only fix).
    - ``extra_kwargs``: kwargs the caller should merge into
      ``apply_chat_template``.
    """
    if tito_model is None and supported_templates is None:
        raise TypeError("resolve_fixed_chat_template() requires either tito_model or supported_templates")
    rows_input = supported_templates if supported_templates is not None else tito_model

    if isinstance(rows_input, tuple):
        rows = rows_input
        tito_model_label = "custom"
    elif isinstance(rows_input, type) and issubclass(rows_input, TITOTokenizer):
        rows = rows_input.SUPPORTED_TEMPLATES
        tito_model_label = rows_input.__name__
    else:
        from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType

        if isinstance(rows_input, str):
            rows_input = TITOTokenizerType(rows_input)
        rows = rows_input.get_tokenizer_class(rows_input).SUPPORTED_TEMPLATES
        tito_model_label = rows_input.value

    requested = frozenset(allowed_append_roles or {"tool"})
    invalid = requested - _VALID_ROLES
    if invalid:
        raise ValueError(
            f"Unknown roles in allowed_append_roles: {sorted(invalid)}. "
            f"Supported: {sorted(_VALID_ROLES)}."
        )

    candidates = [row for row in rows if requested.issubset(row.allowed_roles)]
    if not candidates:
        raise ValueError(
            f"No SUPPORTED_TEMPLATES row registered for tito_model={tito_model_label} "
            f"with allowed_append_roles={sorted(requested)}. Register a row in "
            f"{tito_model_label}.SUPPORTED_TEMPLATES (template=None for HF-native models)."
        )

    # Pick the most specific superset. Ties surface registration mistakes
    # immediately rather than depending on iteration order.
    min_size = min(len(row.allowed_roles) for row in candidates)
    minimal = [row for row in candidates if len(row.allowed_roles) == min_size]
    if len(minimal) > 1:
        raise ValueError(
            f"Ambiguous fixed-template registration for tito_model={tito_model_label}, "
            f"requested_roles={sorted(requested)}: multiple equally-minimal supersets "
            f"{[sorted(row.allowed_roles) for row in minimal]}. Register a stricter row to disambiguate."
        )
    row = minimal[0]

    path = str(TEMPLATE_DIR / row.template) if row.template else None
    return path, dict(row.extra_kwargs)


def resolve_reasoning_and_tool_call_parser(
    tito_model: "TITOTokenizerType | str",
    user_reasoning_parser: str | None = None,
    user_tool_call_parser: str | None = None,
    *,
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve sglang parser values for a tokenizer family.

    Falls back to class attributes on the TITO tokenizer subclass. Does not
    import sglang.  ``reasoning_parser`` / ``tool_call_parser`` are accepted as
    keyword aliases for ``user_reasoning_parser`` / ``user_tool_call_parser``.
    """
    user_reasoning_parser = user_reasoning_parser or reasoning_parser
    user_tool_call_parser = user_tool_call_parser or tool_call_parser
    cls = _resolve_tokenizer_class(tito_model)

    def _resolve_one(field: str, bound: str | None, user: str | None) -> str | None:
        if user is None:
            return bound
        if bound is None:
            return user
        if user != bound:
            raise ValueError(
                f"--{field.replace('_', '-')}: {user!r} disagrees with the {field} "
                f"registered for tito_model={getattr(tito_model, 'value', tito_model)!r}: {bound!r}. "
                "The parser is bound on the TITO subclass; either pass the bound value or omit the flag."
            )
        return user

    return (
        _resolve_one("reasoning_parser", cls.reasoning_parser, user_reasoning_parser),
        _resolve_one("tool_call_parser", cls.tool_call_parser, user_tool_call_parser),
    )
