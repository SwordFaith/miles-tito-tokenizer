"""Vendored DeepSeek encoders.

These encoders are vendored from SGLang under the Apache License 2.0.
They are pure-Python string templates and do not depend on the SGLang runtime.
"""

from __future__ import annotations

from miles_tito_tokenizers.encoders.deepseek_v32 import render_messages as render_deepseek_v32
from miles_tito_tokenizers.encoders.deepseek_v4 import render_messages as render_deepseek_v4

__all__ = ["render_deepseek_v32", "render_deepseek_v4"]
