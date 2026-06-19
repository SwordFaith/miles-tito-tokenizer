"""Compatibility helpers."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from backports.strenum import StrEnum  # type: ignore[no-redef]

__all__ = ["StrEnum"]
