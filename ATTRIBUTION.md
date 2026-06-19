# Attribution

`miles-tito-tokenizers` is a standalone extraction of the TITO (Token-In-Token-Out)
tokenizer logic from the [`miles`](https://github.com/radixark/miles) project.

## Original project

- **Name:** miles
- **Repository:** https://github.com/radixark/miles
- **Copyright:** 2025 Zhipu AI
- **License:** Apache License 2.0

The tokenizer design, chat-template utilities, and validation scripts are
derived from `miles`. This package re-uses and repackages that logic so it can
be consumed without the full miles training stack.

## Vendored code

### DeepSeek V3.2 encoder

- **Source:** SGLang `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`
- **Repository:** https://github.com/sgl-project/sglang
- **Upstream note:** `# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py`
- **License:** Apache License 2.0
- **Copyright:** 2023-2024 SGLang Team

### DeepSeek V4 encoder

- **Source:** SGLang `python/sglang/srt/entrypoints/openai/encoding_dsv4.py`
- **Repository:** https://github.com/sgl-project/sglang
- **Upstream note:** `# Adapted from the DeepSeek-V4 release reference implementation.`
- **License:** Apache License 2.0
- **Copyright:** 2023-2024 SGLang Team

These encoders are vendored as pure Python so that users do not need to install
the full SGLang serving framework.

## Disclaimer

This package is an independent community effort and is **not affiliated with or
endorsed by** the miles team or Zhipu AI. If you are contributing to miles
itself, please use the upstream `miles.utils.chat_template_utils` module.
