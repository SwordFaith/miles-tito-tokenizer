"""Tests for TITOTokenizer: merge_tokens boundary logic, incremental tokenization, and factory.

Derived from miles/tests/fast/utils/chat_template_utils/test_tito_tokenizer.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from miles_tito_tokenizers import (
    MismatchType,
    apply_chat_template,
    resolve_fixed_chat_template,
)
from miles_tito_tokenizers.tokenizer import (
    GLM47TITOTokenizer,
    Qwen3TITOTokenizer,
    Qwen35TITOTokenizer,
    QwenNextTITOTokenizer,
    TITOTokenizer,
    _build_dummy_assistant,
)
from miles_tito_tokenizers.tokenizer_type import TITOTokenizerType, get_tito_tokenizer
from miles_tito_tokenizers.testing.mock_trajectories import (
    IntermediateSystemTrajectory,
    LongChainTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)

# ---------------------------------------------------------------------------
# Tokenizer cache
# ---------------------------------------------------------------------------

_TOK_CACHE: dict[tuple[str, str | None], AutoTokenizer] = {}


def _get_tokenizer(
    model_id: str, tito_type: TITOTokenizerType | None = None
) -> AutoTokenizer:
    chat_template_path = (
        resolve_fixed_chat_template(tito_type, ["tool"])[0]
        if tito_type is not None
        else None
    )
    cache_key = (model_id, chat_template_path)
    if cache_key not in _TOK_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if chat_template_path:
            with open(chat_template_path) as f:
                tok.chat_template = f.read()
        _TOK_CACHE[cache_key] = tok
    return _TOK_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TITO_MODELS: dict[str, tuple[str, type[TITOTokenizer], TITOTokenizerType]] = {
    "qwen3": ("Qwen/Qwen3-4B", Qwen3TITOTokenizer, TITOTokenizerType.QWEN3),
}

# GLM-4.7-Flash is only available if the user has access; keep it optional.
try:
    AutoTokenizer.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    _TITO_MODELS["glm47"] = (
        "zai-org/GLM-4.7-Flash",
        GLM47TITOTokenizer,
        TITOTokenizerType.GLM47,
    )
except Exception:
    pass

_ALLOWED_APPEND_ROLES = ["tool", "user", "system"]


@pytest.fixture(params=list(_TITO_MODELS.keys()))
def tito(request) -> TITOTokenizer:
    model_id, cls, tito_type = _TITO_MODELS[request.param]
    return cls(_get_tokenizer(model_id, tito_type), allowed_append_roles=_ALLOWED_APPEND_ROLES)


@pytest.fixture
def qwen3_tito() -> Qwen3TITOTokenizer:
    return Qwen3TITOTokenizer(
        _get_tokenizer("Qwen/Qwen3-4B", TITOTokenizerType.QWEN3),
        allowed_append_roles=_ALLOWED_APPEND_ROLES,
    )


@pytest.fixture
def glm47_tito() -> GLM47TITOTokenizer | None:
    if "glm47" not in _TITO_MODELS:
        pytest.skip("GLM-4.7-Flash tokenizer not available")
    return GLM47TITOTokenizer(
        _get_tokenizer("zai-org/GLM-4.7-Flash", TITOTokenizerType.GLM47),
        allowed_append_roles=_ALLOWED_APPEND_ROLES,
    )


@pytest.fixture
def default_tito() -> TITOTokenizer:
    return TITOTokenizer(
        _get_tokenizer("Qwen/Qwen3-4B"), allowed_append_roles=_ALLOWED_APPEND_ROLES
    )


# ---------------------------------------------------------------------------
# Trajectory parametrization
# ---------------------------------------------------------------------------


def _find_tito_splits(traj_cls) -> list[int]:
    """Find TITO split positions from message structure."""
    msgs = traj_cls.MESSAGES
    splits = []
    for i, msg in enumerate(msgs):
        if (
            msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and i + 1 < len(msgs)
            and msgs[i + 1].get("role") in ("tool", "system")
        ):
            splits.append(i + 1)
    return splits


def _split_at(traj_cls, pos: int):
    """Split trajectory at *pos* into ``(old_msgs, new_msgs, tools)``."""
    msgs = traj_cls.MESSAGES
    end = pos
    while end < len(msgs) and msgs[end].get("role") != "assistant":
        end += 1
    return msgs[:pos], msgs[:end], traj_cls.TOOLS


_TOOL_TRAJECTORIES = [
    SingleToolTrajectory,
    MultiTurnTrajectory,
    MultiToolSingleTurnTrajectory,
    ParallelToolsTrajectory,
    LongChainTrajectory,
    RetrySystemTrajectory,
    IntermediateSystemTrajectory,
    SingleToolThinkingTrajectory,
]

_TRAJ_CASES = [
    pytest.param(traj_cls, pos, id=f"{traj_cls.__name__}-N{pos}")
    for traj_cls in _TOOL_TRAJECTORIES
    for pos in _find_tito_splits(traj_cls)
]


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_qwen3(self, qwen3_tito: Qwen3TITOTokenizer):
        assert qwen3_tito._assistant_start_str == "<|im_start|>assistant"
        assert qwen3_tito._newline_id in qwen3_tito.trailing_token_ids

    def test_glm47(self, glm47_tito: GLM47TITOTokenizer):
        if glm47_tito is None:
            pytest.skip("GLM tokenizer not available")
        assert glm47_tito._assistant_start_str == "<|assistant|>"
        assert glm47_tito._observation_id in glm47_tito.trailing_token_ids
        assert glm47_tito._user_id in glm47_tito.trailing_token_ids
        assert glm47_tito.max_trim_tokens == 1

    def test_default(self, default_tito: TITOTokenizer):
        assert default_tito._assistant_start_str is None
        assert default_tito.trailing_token_ids == frozenset()

    def test_comparator_inherits_trailing_ids(self, qwen3_tito: Qwen3TITOTokenizer):
        comp = qwen3_tito.create_comparator()
        assert comp._trim_trailing_ids == set(qwen3_tito.trailing_token_ids)


# ---------------------------------------------------------------------------
# TestMergeTokensBoundary
# ---------------------------------------------------------------------------

_BND_OLD, _BND_NEW, _BND_TOOLS = _split_at(SingleToolTrajectory, 3)


class TestMergeTokensBoundary:
    def test_qwen3_inserts_newline_after_im_end(self, qwen3_tito: Qwen3TITOTokenizer):
        incremental = qwen3_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        im_end = qwen3_tito._im_end_id
        nl = qwen3_tito._newline_id

        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, im_end], _BND_TOOLS
        )
        assert result == [100, 200, im_end, nl] + incremental

    def test_qwen3_no_newline_otherwise(self, qwen3_tito: Qwen3TITOTokenizer):
        incremental = qwen3_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = qwen3_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS
        )
        assert result == [100, 200, 300] + incremental

    def test_glm47_strips_observation(self, glm47_tito: GLM47TITOTokenizer):
        if glm47_tito is None:
            pytest.skip("GLM tokenizer not available")
        incremental = glm47_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD,
            _BND_NEW,
            [100, 200, glm47_tito._observation_id],
            _BND_TOOLS,
        )
        assert result == [100, 200] + incremental

    def test_glm47_strips_user(self, glm47_tito: GLM47TITOTokenizer):
        if glm47_tito is None:
            pytest.skip("GLM tokenizer not available")
        incremental = glm47_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, glm47_tito._user_id], _BND_TOOLS
        )
        assert result == [100, 200] + incremental

    def test_glm47_no_strip_otherwise(self, glm47_tito: GLM47TITOTokenizer):
        if glm47_tito is None:
            pytest.skip("GLM tokenizer not available")
        incremental = glm47_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = glm47_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS
        )
        assert result == [100, 200, 300] + incremental

    def test_default_concatenates(self, default_tito: TITOTokenizer):
        incremental = default_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = default_tito.merge_tokens(
            _BND_OLD, _BND_NEW, [100, 200, 300], _BND_TOOLS
        )
        assert result == [100, 200, 300] + incremental

    def test_empty_prefix(self, qwen3_tito: Qwen3TITOTokenizer):
        incremental = qwen3_tito.tokenize_additional_non_assistant(
            _BND_OLD, _BND_NEW, _BND_TOOLS
        )
        result = qwen3_tito.merge_tokens(_BND_OLD, _BND_NEW, [], _BND_TOOLS)
        assert result == incremental


# ---------------------------------------------------------------------------
# TestTokenizeAdditional
# ---------------------------------------------------------------------------


class TestTokenizeAdditional:
    @pytest.mark.parametrize("traj_cls, pos", _TRAJ_CASES)
    def test_produces_nonempty_incremental(self, tito: TITOTokenizer, traj_cls, pos):
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        incremental = tito.tokenize_additional_non_assistant(old_msgs, new_msgs, tools)
        assert len(incremental) > 0

    def test_contiguous_tool_segment_is_tokenized_together(
        self, qwen3_tito: Qwen3TITOTokenizer
    ):
        old_msgs, new_msgs, tools = _split_at(MultiToolSingleTurnTrajectory, 3)
        appended = new_msgs[len(old_msgs) :]

        segments = qwen3_tito._split_appended_segments(appended)
        assert len(segments) == 1
        assert [msg["role"] for msg in segments[0]] == ["tool", "tool"]

        incremental = qwen3_tito.tokenize_additional_non_assistant(
            old_msgs, new_msgs, tools
        )
        decoded = qwen3_tito.tokenizer.decode(incremental)
        assert MultiToolSingleTurnTrajectory.MESSAGES[3]["content"] in decoded
        assert MultiToolSingleTurnTrajectory.MESSAGES[4]["content"] in decoded

    def test_user_and_system_segments_are_singletons(self, default_tito: TITOTokenizer):
        appended = [
            {"role": "system", "content": "Use JSON."},
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
            {"role": "tool", "tool_call_id": "call_2", "content": '{"ok": false}'},
            {"role": "user", "content": "Try again"},
        ]

        segments = default_tito._split_appended_segments(appended)
        assert [[msg["role"] for msg in segment] for segment in segments] == [
            ["system"],
            ["user"],
            ["tool", "tool"],
            ["user"],
        ]

    def test_generation_prompt_is_appended_once_for_full_suffix(
        self, qwen3_tito: Qwen3TITOTokenizer
    ):
        old_msgs = list(SingleToolThinkingTrajectory.MESSAGES[:3])
        new_msgs = old_msgs + [
            SingleToolThinkingTrajectory.MESSAGES[3],
            {"role": "user", "content": "Now check Shanghai too."},
        ]
        tools = SingleToolThinkingTrajectory.TOOLS

        incremental = qwen3_tito.tokenize_additional_non_assistant(
            old_msgs, new_msgs, tools
        )
        decoded = qwen3_tito.tokenizer.decode(incremental)
        assert decoded.count(qwen3_tito._assistant_start_str) == 1
        assert decoded.endswith(
            qwen3_tito.tokenizer.decode(
                qwen3_tito._tokenize_rendered_suffix(
                    new_msgs, [], tools=tools, add_generation_prompt=True
                )
            )
        )

    def test_qwen3_tool_dummy_assistant_preserves_reasoning_shape(self):
        thinking_template_path = (
            Path(__file__).resolve().parents[1]
            / "src/miles_tito_tokenizers/templates/qwen3_thinking_2507_and_next_fixed.jinja"
        )
        if not thinking_template_path.exists():
            pytest.skip("thinking template not found")
        try:
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"thinking model not available: {e}")
        with open(thinking_template_path) as f:
            tok.chat_template = f.read()
        thinking_tito = Qwen3TITOTokenizer(
            tok,
            allowed_append_roles=_ALLOWED_APPEND_ROLES,
        )
        tool_messages = [SingleToolThinkingTrajectory.MESSAGES[3]]
        dummy_assistant = _build_dummy_assistant(tool_messages)
        rendered = thinking_tito.render_messages(
            [{"role": "system", "content": "dummy system"}, dummy_assistant],
            add_generation_prompt=False,
            tools=SingleToolThinkingTrajectory.TOOLS,
        )

        assert dummy_assistant["reasoning_content"] == " "
        assert rendered.endswith(
            '<|im_start|>assistant\n<tool_call>\n{"name": "dummy_func", "arguments": {}}\n</tool_call><|im_end|>\n'
        )

    @pytest.mark.parametrize(
        "traj_cls, pos",
        [
            pytest.param(SingleToolTrajectory, 3, id="single-tool"),
            pytest.param(RetrySystemTrajectory, 3, id="tool-plus-system"),
            pytest.param(IntermediateSystemTrajectory, 3, id="intermediate-system"),
        ],
    )
    def test_qwen3_merge_preserves_non_assistant_structure(
        self, qwen3_tito: Qwen3TITOTokenizer, traj_cls, pos
    ):
        old_msgs, new_msgs, tools = _split_at(traj_cls, pos)
        pretokenized = apply_chat_template(
            old_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
        )
        merged = qwen3_tito.merge_tokens(old_msgs, new_msgs, pretokenized, tools)
        expected = apply_chat_template(
            new_msgs,
            tokenizer=qwen3_tito.tokenizer,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
        mismatches = qwen3_tito.create_comparator().compare_sequences(expected, merged)
        assert all(m.type == MismatchType.ASSISTANT_TEXT for m in mismatches)

    def test_rejects_prefix_mutation(self, qwen3_tito: Qwen3TITOTokenizer):
        old_msgs, new_msgs, _ = _split_at(SingleToolTrajectory, 3)
        mutated_old = [{"role": "user", "content": "CHANGED"}] + list(old_msgs[1:])
        mutated_new = mutated_old + list(new_msgs[len(old_msgs) :])
        with pytest.raises(ValueError, match="mismatch"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, mutated_new)

    def test_rejects_fewer_messages(self, qwen3_tito: Qwen3TITOTokenizer):
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        with pytest.raises(ValueError, match="fewer"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, old_msgs[:1])

    def test_rejects_assistant_append(self, qwen3_tito: Qwen3TITOTokenizer):
        old_msgs = SingleToolTrajectory.MESSAGES[:3]
        bad_new = list(old_msgs) + [{"role": "assistant", "content": "hi"}]
        with pytest.raises(ValueError, match="role"):
            qwen3_tito.tokenize_additional_non_assistant(old_msgs, bad_new)


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.parametrize(
        "type_str, model_id, cls",
        [
            ("qwen3", "Qwen/Qwen3-4B", Qwen3TITOTokenizer),
            ("qwen35", "Qwen/Qwen3-4B", Qwen35TITOTokenizer),
            ("qwennext", "Qwen/Qwen3-4B", QwenNextTITOTokenizer),
            ("default", "Qwen/Qwen3-4B", TITOTokenizer),
        ],
    )
    def test_creates_correct_type(self, type_str, model_id, cls):
        tito = get_tito_tokenizer(_get_tokenizer(model_id), tokenizer_type=type_str)
        assert isinstance(tito, cls)

    def test_enum_input(self):
        tito = get_tito_tokenizer(
            _get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=TITOTokenizerType.QWEN3
        )
        assert isinstance(tito, Qwen3TITOTokenizer)

    @pytest.mark.parametrize(
        "type_str, cls",
        [("qwen35", Qwen35TITOTokenizer), ("qwennext", QwenNextTITOTokenizer)],
    )
    def test_qwen_variant_inherits_qwen3_boundary_logic(self, type_str, cls):
        tito = get_tito_tokenizer(
            _get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type=type_str
        )
        assert isinstance(tito, cls)
        assert isinstance(tito, Qwen3TITOTokenizer)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_tito_tokenizer(
                _get_tokenizer("Qwen/Qwen3-4B"), tokenizer_type="nonexistent"
            )

    def test_none_tokenizer_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            get_tito_tokenizer(None)


# ---------------------------------------------------------------------------
# TestParserBinding
# ---------------------------------------------------------------------------


class TestParserBinding:
    @pytest.mark.parametrize(
        "tito_model, expected_reasoning, expected_tool_call",
        [
            (TITOTokenizerType.QWEN3, "qwen3", "qwen25"),
            (TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder"),
            (TITOTokenizerType.QWENNEXT, "qwen3", "qwen25"),
            (TITOTokenizerType.GLM47, "glm45", "glm47"),
            (TITOTokenizerType.NEMOTRON3, "nemotron_3", "qwen3_coder"),
            (TITOTokenizerType.KIMI25, None, None),
            (TITOTokenizerType.KIMI26, "kimi_k2", "kimi_k2_raw_id"),
            (TITOTokenizerType.MINIMAX_M25, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.MINIMAX_M27, "minimax-append-think", "minimax-m2"),
            (TITOTokenizerType.DEEPSEEKV32, "deepseek-v3", "deepseekv32"),
            (TITOTokenizerType.DEEPSEEKV4, "deepseek-v4", "deepseekv4"),
            (TITOTokenizerType.DEFAULT, None, None),
        ],
    )
    def test_subclass_binding(self, tito_model, expected_reasoning, expected_tool_call):
        cls = TITOTokenizerType.get_tokenizer_class(tito_model)
        assert cls.reasoning_parser == expected_reasoning
        assert cls.tool_call_parser == expected_tool_call

    def test_resolve_returns_binding_when_user_omits(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.QWEN3
        ) == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.QWEN35
        ) == ("qwen3", "qwen3_coder")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.GLM47
        ) == ("glm45", "glm47")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.DEEPSEEKV4
        ) == ("deepseek-v4", "deepseekv4")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.DEFAULT
        ) == (None, None)

    def test_resolve_accepts_matching_user_value(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            "qwen3", "qwen3", "qwen25"
        ) == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.QWEN35, "qwen3", "qwen3_coder"
        ) == ("qwen3", "qwen3_coder")

    def test_resolve_raises_on_reasoning_mismatch(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="reasoning_parser"):
            resolve_reasoning_and_tool_call_parser(
                TITOTokenizerType.QWEN3, reasoning_parser="glm45"
            )

    def test_resolve_raises_on_tool_call_mismatch(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        with pytest.raises(ValueError, match="tool_call_parser"):
            resolve_reasoning_and_tool_call_parser(
                TITOTokenizerType.QWEN3, tool_call_parser="glm47"
            )

    def test_resolve_accepts_user_value_when_family_unbound(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.DEFAULT, "custom_reasoning", "custom_tool_call"
        ) == ("custom_reasoning", "custom_tool_call")

    def test_resolve_partial_user_input(self):
        from miles_tito_tokenizers import resolve_reasoning_and_tool_call_parser

        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.QWEN3, reasoning_parser="qwen3"
        ) == ("qwen3", "qwen25")
        assert resolve_reasoning_and_tool_call_parser(
            TITOTokenizerType.GLM47, tool_call_parser="glm47"
        ) == ("glm45", "glm47")
