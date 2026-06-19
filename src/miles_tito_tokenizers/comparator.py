"""TokenSeqComparator: segment token IDs by special-token boundaries and compare sequences.

Derived from miles.utils.chat_template_utils.token_seq_comparator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Segment:
    """A contiguous run of token IDs — either a special token or a content segment."""

    token_ids: list[int] = field(default_factory=list)
    is_special: bool = False


class MismatchType(Enum):
    # Segment count or structure (special/content pattern) differs between
    # expected and actual.  When this happens, segments can't be aligned so
    # no per-segment comparison is possible.
    SPECIAL_TOKEN_COUNT = "special_token_count"

    # A special-token segment has the same position in both sequences but
    # contains a different token ID.
    SPECIAL_TOKEN_TYPE = "special_token_type"

    # Non-assistant content (user, system, tool, etc.) differs.  This indicates
    # a bug in the TITO algorithm — these regions should match exactly.
    NON_ASSISTANT_TEXT = "non_assistant_text"

    # Assistant content differs.  Expected and non-severe: assistant tokens
    # are inherited directly from the pretokenized prefix across turns,
    # so they may not match the chat template's canonical tokenization.
    ASSISTANT_TEXT = "assistant_text"


@dataclass
class Mismatch:
    """A single difference found between two token sequences."""

    type: MismatchType
    segment_index: int = -1
    expected_text: str | None = None
    actual_text: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "segment_index": self.segment_index,
            "expected_text": self.expected_text,
            "actual_text": self.actual_text,
            "detail": self.detail,
        }


class TokenSeqComparator:
    """Segment token sequences by special tokens and compare them.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    special_token_ids : set[int] | None
        Token IDs that mark segment boundaries.  Default None auto-detects
        them (see :meth:`collect_special_ids`).  Pass an explicit set to
        override — needed when the tokenizer flags a token ``special=True``
        even though it lives inside a role's turn (e.g. Kimi's
        ``<|im_middle|>`` between role name and body) and so must not
        split the segment.
    assistant_start_str : str
        Decoded text prefix identifying assistant content segments, e.g.
        ``"<|im_start|>assistant"`` (Qwen3) or ``"<|assistant|>"`` (GLM).
        Used to classify content mismatches as assistant vs non-assistant.
    trim_trailing_ids : set[int] | None
        Token IDs to strip from both sequence tails before comparison
        (see :func:`_trim_trailing`).  Stored as a default; callers of
        :meth:`compare_sequences` may supply additional IDs that are
        **unioned** with this set.
    """

    def __init__(
        self,
        tokenizer,
        assistant_start_str: str | None = None,
        special_token_ids: set[int] | None = None,
        trim_trailing_ids: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        if special_token_ids is not None:
            self._special_ids = set(special_token_ids)
        else:
            self._special_ids = self.collect_special_ids(tokenizer)
        self._assistant_start_str = assistant_start_str
        self._trim_trailing_ids: set[int] | None = (
            set(trim_trailing_ids) if trim_trailing_ids else None
        )
    @property
    def special_token_ids(self) -> set[int]:
        """Set of token IDs treated as segment boundaries."""
        return set(self._special_ids)

    @property
    def assistant_start_str(self) -> str | None:
        """Prefix used to identify assistant content segments."""
        return self._assistant_start_str

    @property
    def trim_trailing_ids(self) -> set[int] | None:
        """Default trailing token IDs stripped before comparison."""
        return self._trim_trailing_ids

    @staticmethod
    def collect_special_ids(tokenizer) -> set[int]:
        """Collect token IDs with ``special=True`` from the tokenizer."""
        ids = set(getattr(tokenizer, "all_special_ids", []))
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if decoder:
            ids |= {k for k, v in decoder.items() if v.special}
        return ids

    def segment_by_special_tokens(self, token_ids: list[int]) -> list[Segment]:
        """Split *token_ids* into segments at special-token boundaries."""
        if not token_ids:
            return []

        segments: list[Segment] = []
        current: list[int] = []
        for tid in token_ids:
            if tid in self._special_ids:
                if current:
                    segments.append(Segment(token_ids=current))
                    current = []
                segments.append(Segment(token_ids=[tid], is_special=True))
            else:
                current.append(tid)
        if current:
            segments.append(Segment(token_ids=current))
        return segments

    def compare_sequences(
        self,
        expected_ids: list[int],
        actual_ids: list[int],
        trim_trailing_ids: set[int] | None = None,
    ) -> list[Mismatch]:
        """Compare two token-ID sequences and return mismatches."""
        expected_ids = list(expected_ids)
        actual_ids = list(actual_ids)
        trim = self._trim_trailing_ids or set()
        if trim_trailing_ids:
            trim = trim | trim_trailing_ids
        if trim:
            expected_ids = _trim_trailing(expected_ids, trim)
            actual_ids = _trim_trailing(actual_ids, trim)

        exp_segs = self.segment_by_special_tokens(expected_ids)
        act_segs = self.segment_by_special_tokens(actual_ids)

        structural = self._check_segment_structure(exp_segs, act_segs)
        if structural:
            return [structural]

        mismatches: list[Mismatch] = []
        for idx, (exp, act) in enumerate(zip(exp_segs, act_segs, strict=True)):
            is_assistant_content = self._is_assistant_content(
                exp_segs, idx
            ) and self._is_assistant_content(act_segs, idx)
            m = self._compare_single_segment(
                idx, exp, act, is_assistant_content=is_assistant_content
            )
            if m is not None:
                mismatches.append(m)
        return mismatches

    def _check_segment_structure(
        self,
        exp_segs: list[Segment],
        act_segs: list[Segment],
    ) -> Mismatch | None:
        """Pre-check segment count and special/content pattern."""
        if len(exp_segs) != len(act_segs):
            detail = (
                f"segment count differs: expected {len(exp_segs)}, got {len(act_segs)}"
            )
        elif [s.is_special for s in exp_segs] != [s.is_special for s in act_segs]:
            detail = "segment structure (special/content pattern) differs"
        else:
            return None
        return Mismatch(
            type=MismatchType.SPECIAL_TOKEN_COUNT,
            segment_index=-1,
            expected_text=self._describe_structure(exp_segs),
            actual_text=self._describe_structure(act_segs),
            detail=detail,
        )

    def _compare_single_segment(
        self,
        idx: int,
        exp: Segment,
        act: Segment,
        *,
        is_assistant_content: bool,
    ) -> Mismatch | None:
        """Compare a single aligned segment pair."""
        if exp.is_special:
            if exp.token_ids != act.token_ids:
                return Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_TYPE,
                    segment_index=idx,
                    expected_text=self._decode(exp.token_ids),
                    actual_text=self._decode(act.token_ids),
                )
            return None

        exp_text = self._decode(exp.token_ids)
        act_text = self._decode(act.token_ids)
        if exp_text == act_text:
            return None

        return Mismatch(
            type=MismatchType.ASSISTANT_TEXT
            if is_assistant_content
            else MismatchType.NON_ASSISTANT_TEXT,
            segment_index=idx,
            expected_text=exp_text,
            actual_text=act_text,
        )

    def _is_assistant_content(self, segments: list[Segment], idx: int) -> bool:
        """Check if the content segment at *idx* belongs to an assistant message."""
        if self._assistant_start_str is None:
            return False
        if segments[idx].is_special:
            return False
        if idx == 0:
            return False
        prev = segments[idx - 1]
        if not prev.is_special:
            return False
        special_text = self._decode(prev.token_ids)
        content_prefix = self._decode(segments[idx].token_ids[:20])
        return (special_text + content_prefix).startswith(self._assistant_start_str)

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _describe_structure(self, segments: list[Segment]) -> str:
        return " ".join(
            f"[{self._decode(s.token_ids)}]"
            if s.is_special
            else f"({len(s.token_ids)} tokens)"
            for s in segments
        )


def _trim_trailing(ids: list[int], to_remove: set[int]) -> list[int]:
    """Strip trailing token IDs that belong to *to_remove*."""
    if not ids:
        return ids
    end = len(ids)
    while end > 0 and ids[end - 1] in to_remove:
        end -= 1
    return ids[:end]
