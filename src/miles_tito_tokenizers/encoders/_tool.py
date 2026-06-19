"""Minimal OpenAI-format Tool model compatible with SGLang's protocol.Tool.

This replaces the upstream ``sglang.srt.entrypoints.openai.protocol.Tool`` import
so that the vendored DeepSeek encoders do not depend on the SGLang package.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class _FunctionSchema(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: _FunctionSchema

    model_config = {"extra": "allow"}
