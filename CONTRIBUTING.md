# Contributing to miles-tito-tokenizers

Thank you for your interest in improving `miles-tito-tokenizers`.

## Scope

This package is a thin extraction of the TITO tokenizer logic from
[miles](https://github.com/radixark/miles). Keep the scope narrow:

- Bug fixes and tests for tokenizer behavior.
- New TITO-supported model families.
- Documentation and packaging improvements.

Avoid adding training-framework features, rollout serving code, or features that
belong in miles itself.

## Vendored code

Two files are vendored from SGLang:

- `src/miles_tito_tokenizers/encoders/deepseek_v32.py`
- `src/miles_tito_tokenizers/encoders/deepseek_v4.py`

When syncing with upstream SGLang:

1. Copy the latest `encoding_dsv32.py` / `encoding_dsv4.py`.
2. Keep the original Apache-2.0 header.
3. Add a vendoring comment at the top with the upstream commit hash and date.
4. Update `Tool` references to use the package-local `Tool` model if SGLang's
   interface has changed.
5. Run `pytest tests/test_deepseek_v32.py tests/test_deepseek_v4.py`.

Do not introduce dependencies on the rest of SGLang.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Code style

- Follow PEP 8.
- Keep imports sorted.
- Match the existing naming conventions from miles where possible.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0, the same license used by the original miles project.
