"""Message utilities for append-only validation.

Derived from miles.utils.chat_template_utils.template.
"""

from __future__ import annotations

import copy
from typing import Any


_TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")


def _normalize_value(value: Any) -> Any:
    """Normalize falsy sentinels that produce identical Jinja2 output.

    None, "" and [] are all falsy in Jinja2 and render the same way,
    but client libraries may interchange them.
    """
    if value is None or value == "" or value == []:
        return None
    return value


def message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare only the fields that affect chat-template tokenization."""
    for key in _TEMPLATE_RELEVANT_KEYS:
        if _normalize_value(stored.get(key)) != _normalize_value(new.get(key)):
            return False
    return True


_DEFAULT_APPEND_ROLES: list[str] = ["tool"]


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: list[str] | None = None,
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*.

    The stored prefix must match exactly (compared by template-relevant keys),
    and any appended messages must have a role in *allowed_append_roles*
    (default: ``{'tool'}``).
    """
    if allowed_append_roles is None:
        allowed_append_roles = _DEFAULT_APPEND_ROLES

    if not stored_messages:
        return

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not message_matches(stored_msg, new_messages[i]):
            diffs = {
                key: {
                    "stored": repr(stored_msg.get(key))[:200],
                    "new": repr(new_messages[i].get(key))[:200],
                }
                for key in _TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    for j, msg in enumerate(new_messages[len(stored_messages) :]):
        if msg.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={allowed_append_roles}"
            )


def normalize_tool_arguments(messages: list[dict], format: str) -> list[dict]:
    """Deep-copy *messages*, normalize assistant ``content: None`` -> "", and coerce
    tool_call arguments to the requested string format ("dict" or "json").
    """
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("content") is None:
            msg["content"] = ""
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            import json

            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments")
                if format == "json" and isinstance(args, dict):
                    tc["function"]["arguments"] = json.dumps(args)
                elif format == "dict" and isinstance(args, str):
                    tc["function"]["arguments"] = json.loads(args)
    return normalized
