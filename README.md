# miles-tito-tokenizers

Standalone TITO (Token-In-Token-Out) tokenizers extracted from the
[`miles`](https://github.com/radixark/miles) project.

This package lets you reuse miles' chat-template / incremental-tokenization logic
for multi-turn agentic rollout without importing the full miles training
framework.

## Attribution

- The TITO tokenizer design and implementation are derived from the
  `miles` project by the `radixark/miles` contributors.
- DeepSeek V3.2 and V4 encoders are vendored from
  [SGLang](https://github.com/sgl-project/sglang) under the Apache License 2.0.
- See [`ATTRIBUTION.md`](./ATTRIBUTION.md) and [`LICENSE`](./LICENSE) for
  details.

## Install

```bash
pip install miles-tito-tokenizers
```

No `sglang` installation is required — DeepSeek encoders are vendored as pure
Python.

## Quick start

```python
from transformers import AutoTokenizer
from miles_tito_tokenizers import get_tito_tokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

tito = get_tito_tokenizer(
    tokenizer,
    tokenizer_type="qwen3",
    allowed_append_roles=["tool", "user"],
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

ids = tito.render_messages(messages, add_generation_prompt=True)
print(ids)

```

For DeepSeek models:

```python
tito = get_tito_tokenizer(
    tokenizer,
    tokenizer_type="deepseekv32",
    allowed_append_roles=["tool"],
)
```

## Migrating from `transformers`

`miles_tito_tokenizers.apply_chat_template` mirrors
`tokenizer.apply_chat_template(..., return_dict=False)`, which is the format
used by SGLang and miles:

| What you want | `transformers` call | `miles_tito_tokenizers` call |
|---|---|---|
| Rendered text | `tokenizer.apply_chat_template(messages, tokenize=False)` | `mtt.apply_chat_template(messages, tokenizer=tokenizer)` |
| Token IDs | `tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)` | `mtt.apply_chat_template(messages, tokenizer=tokenizer, tokenize=True)` |
| BatchEncoding | `tokenizer.apply_chat_template(messages)` | not supported by design |

`apply_chat_template` takes `tokenizer` as a keyword-only argument so the
intent is explicit.

## Supported model families

| `tokenizer_type` | Notes |
|---|---|
| `qwen3` | Qwen3 |
| `qwen35` | Qwen3.5 |
| `qwennext` | Qwen3-Thinking-2507 / Qwen3-Next-Thinking |
| `glm47` | GLM 4.7 |
| `nemotron3` | NVIDIA Nemotron 3 |
| `kimi25` | Moonshot Kimi K2.5 |
| `kimi26` | Moonshot Kimi K2.6 |
| `minimax_m25` | MiniMax-M2.5 |
| `minimax_m27` | MiniMax-M2.7 |
| `deepseekv32` | DeepSeek V3.2 |
| `deepseekv4` | DeepSeek V4 |
| `default` | HF-native chat template, no TITO merge logic |

## Validation scripts

Two CLI helpers from miles are included:

```bash
# CPU / fast: verify that rendered token sequences are append-only.
python -m miles_tito_tokenizers.scripts.verify_chat_template \
    --model Qwen/Qwen3-4B \
    --tito-model qwen3 \
    --tito-allowed-append-roles tool user

# GPU / e2e: verify against a real SGLang-served checkpoint.
python -m miles_tito_tokenizers.scripts.verify_session_tito_tokenizer \
    --hf-checkpoint Qwen/Qwen3-4B \
    --tito-model qwen3 \
    --tito-allowed-append-roles tool user \
    --sglang-reasoning-parser <rp> \
    --sglang-tool-call-parser <tcp> \
    --rollout-num-gpus-per-engine 1
```

## Relationship to miles

`miles-tito-tokenizers` is an **independent, personal-fan package** that extracts
a well-defined slice of miles. It is not affiliated with or endorsed by the
miles team. If you are building inside miles itself, use the original
`miles.utils.chat_template_utils` module.

## License

- Package code derived from miles: see [`LICENSE`](./LICENSE) for the miles
  project license.
- Vendored DeepSeek encoders from SGLang: see
  [`LICENSE-SGLANG-DEEPSEEKV32.md`](./LICENSE-SGLANG-DEEPSEEKV32.md) and
  [`LICENSE-SGLANG-DEEPSEEKV4.md`](./LICENSE-SGLANG-DEEPSEEKV4.md).

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md).
