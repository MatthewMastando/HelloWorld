# HelloWorld — trading research workspace

Monorepo for a trading research app. Phase A is a deterministic
technical-analysis engine in Python (`trw.ta`): it consumes validated
`CandleFrame`s and emits typed `Feature`s and `FeatureEvent`s (fair value
gaps, liquidity pools/sweeps, breaks of structure, order blocks, RSI
divergence, volume profile) that are stable across replay.

## Layout

- `packages/py/trw` — the `trw` Python package (src layout); `trw.ta` is the TA engine.
- `tests/` — pytest suite; `tests/ta` covers primitives and detectors.
- `scripts/check_licenses.py` — dependency license allow-list check.

## Getting started

```sh
make setup   # uv sync: python 3.12, deps, dev tools
make lint    # ruff check + ruff format --check + mypy --strict + license check
make test    # pytest -q
```
