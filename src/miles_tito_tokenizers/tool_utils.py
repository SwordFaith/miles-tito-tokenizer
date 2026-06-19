"""Tool utilities.

Derived from miles.utils.chat_template_utils.template.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class _FunctionSchema(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class _ToolModel(BaseModel):
    type: str = "function"
    function: _FunctionSchema


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Canonicalize tools via Pydantic, returning full tool model dumps."""
    if tools is None:
        return None
    return [_ToolModel.model_validate(tool).model_dump() for tool in tools]
