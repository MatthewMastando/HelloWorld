"""Incremental-replay invariant: prefix runs must reproduce the full-run event
history exactly — future bars may never add, remove, or change past events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from trw.ta.engine import run_all
from trw.ta.types import CandleFrame, InstrumentKind, InstrumentSpec, SnapshotRef

SPEC = InstrumentSpec(
    symbol="M6EZ6",
    kind=InstrumentKind.FUTURE_CONTRACT,
    venue="CME",
    currency="USD",
    tick_size=Decimal("0.25"),
    tick_value=Decimal("0.25"),
    point_multiplier=Decimal("1"),
    root="M6E",
)
SNAP = SnapshotRef(
    snapshot_id="snap-replay",
    data_revision="rev-replay",
    source="fixture",
    feed_tier="synthetic",
    as_of=datetime(2026, 1, 5, 20, 0, tzinfo=UTC),
    synthetic=True,
)


def make_walk(n: int = 400, seed: int = 42) -> CandleFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.6, n)
    ts = [datetime(2026, 1, 5, 14, 30, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(50, 500, n),
            "complete": [True] * n,
        }
    )
    return CandleFrame(df, timeframe="5m")


def prefix(frame: CandleFrame, n: int) -> CandleFrame:
    return CandleFrame(frame.df.iloc[:n].copy(), timeframe=frame.timeframe, session=frame.session)


def test_replay_prefix_events_match_full_run() -> None:
    frame = make_walk()
    full = run_all(frame, SPEC, SNAP)
    full_events = {(e.feature_id, e.kind, e.observed_at_index) for e in full.events}

    for n in range(60, 400, 7):
        res = run_all(prefix(frame, n), SPEC, SNAP)
        got = {(e.feature_id, e.kind, e.observed_at_index) for e in res.events}
        expected = {t for t in full_events if t[2] <= n - 1}
        assert got == expected, f"prefix n={n} diverges: missing={expected - got} extra={got - expected}"


def test_no_event_beyond_last_complete_and_origin_before_confirm() -> None:
    frame = make_walk()
    res = run_all(frame, SPEC, SNAP)
    last = frame.last_complete_index()
    assert all(e.observed_at_index <= last for e in res.events)
    assert all(f.confirmed_index >= f.origin_index for f in res.features)
