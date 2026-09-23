"""Hand-calculated detector tests. These define the contract; do not adjust to fit code."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from trw.ta.detectors import (
    detect_bos,
    detect_fvg,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_rsi_divergence,
)
from trw.ta.params import BOSParams, FVGParams, OrderBlockParams, RSIDivergenceParams, SweepParams
from trw.ta.primitives import atr_wilder, swing_pivots
from trw.ta.types import (
    Direction,
    FeatureState,
    FeatureType,
    InstrumentKind,
    InstrumentSpec,
    InsufficientData,
    SnapshotRef,
)

from .test_primitives import make_frame

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
    snapshot_id="snap-test",
    data_revision="rev-test",
    source="fixture",
    feed_tier="synthetic",
    as_of=datetime(2026, 1, 5, 20, 0, tzinfo=UTC),
    synthetic=True,
)


def flat(n: int, px: float = 100.0) -> tuple[list[float], list[float], list[float], list[float]]:
    return [px] * n, [px + 0.5] * n, [px - 0.5] * n, [px] * n


# --------------------------------------------------------------------------- FVG


def test_bullish_fvg_zone_confirmed_at_c_close() -> None:
    close, high, low, open_ = flat(20)
    # A = bar 10 (high 100.5), B = bar 11 big up, C = bar 12 with low 102.0 > high(A)=100.5
    open_[11], close[11], high[11], low[11] = 100.0, 103.0, 103.5, 99.75
    open_[12], close[12], high[12], low[12] = 103.0, 103.5, 104.0, 102.0
    for i in range(13, 20):
        open_[i], close[i], high[i], low[i] = 103.5, 103.5, 104.0, 103.0
    f = make_frame(close, high, low, open_)
    res = detect_fvg(f, SPEC, SNAP, FVGParams())
    assert len(res.features) == 1
    fvg = res.features[0]
    assert fvg.feature_type == FeatureType.FVG
    assert fvg.direction == Direction.BULLISH
    assert fvg.origin_index == 10 and fvg.confirmed_index == 12
    assert fvg.levels["top"] == Decimal("102.00") and fvg.levels["bottom"] == Decimal("100.50")
    assert fvg.levels["midpoint"] == Decimal("101.25")
    assert fvg.state == FeatureState.CONFIRMED
    assert res.events[0].kind == "confirmed" and res.events[0].observed_at_index == 12
    # displacement tag stored separately, not a gate
    assert "displacement" in fvg.payload


def test_bearish_fvg() -> None:
    close, high, low, open_ = flat(20)
    open_[11], close[11], high[11], low[11] = 100.0, 97.0, 100.25, 96.5
    open_[12], close[12], high[12], low[12] = 97.0, 96.5, 98.0, 96.0  # high(C)=98 < low(A)=99.5
    for i in range(13, 20):
        open_[i], close[i], high[i], low[i] = 96.5, 96.5, 97.0, 96.0
    f = make_frame(close, high, low, open_)
    res = detect_fvg(f, SPEC, SNAP, FVGParams())
    assert len(res.features) == 1
    fvg = res.features[0]
    assert fvg.direction == Direction.BEARISH
    assert fvg.levels["top"] == Decimal("99.50") and fvg.levels["bottom"] == Decimal("98.00")


def test_fvg_requires_minimum_one_tick() -> None:
    close, high, low, open_ = flat(20)
    open_[11], close[11], high[11], low[11] = 100.0, 103.0, 103.5, 99.75
    # low(C) == high(A) exactly -> zero-width, not an FVG
    open_[12], close[12], high[12], low[12] = 103.0, 103.5, 104.0, 100.5
    for i in range(13, 20):
        open_[i], close[i], high[i], low[i] = 103.5, 103.5, 104.0, 103.0
    f = make_frame(close, high, low, open_)
    assert detect_fvg(f, SPEC, SNAP, FVGParams()).features == []


def test_fvg_state_tracking_touch_fill_invalidation() -> None:
    close, high, low, open_ = flat(24)
    open_[11], close[11], high[11], low[11] = 100.0, 103.0, 103.5, 99.75
    open_[12], close[12], high[12], low[12] = 103.0, 103.5, 104.0, 102.0  # zone [100.5, 102.0], mid 101.25
    for i in range(13, 24):
        open_[i], close[i], high[i], low[i] = 103.5, 103.5, 104.0, 103.0
    # bar 14 touches zone (low 101.75, closes above)
    open_[14], close[14], high[14], low[14] = 103.5, 103.0, 103.5, 101.75
    # bar 16 trades through midpoint (low 101.0) closes inside zone
    open_[16], close[16], high[16], low[16] = 103.0, 101.5, 103.0, 101.0
    # bar 18 closes below bottom -> invalidated (close-through)
    open_[18], close[18], high[18], low[18] = 101.5, 100.0, 101.5, 99.75
    f = make_frame(close, high, low, open_)
    res = detect_fvg(f, SPEC, SNAP, FVGParams())
    assert len(res.features) == 1
    kinds = [(e.kind, e.observed_at_index) for e in res.events]
    assert ("confirmed", 12) in kinds
    assert ("touched", 14) in kinds
    assert ("midpoint_touched", 16) in kinds
    assert ("invalidated", 18) in kinds
    assert res.features[0].state == FeatureState.INVALIDATED
    # events are in observation order
    assert [e.observed_at_index for e in res.events] == sorted(e.observed_at_index for e in res.events)


def test_fvg_not_confirmed_on_incomplete_bar() -> None:
    close, high, low, open_ = flat(13)
    open_[11], close[11], high[11], low[11] = 100.0, 103.0, 103.5, 99.75
    open_[12], close[12], high[12], low[12] = 103.0, 103.5, 104.0, 102.0
    f = make_frame(close, high, low, open_)
    f.df.loc[12, "complete"] = False
    assert detect_fvg(f, SPEC, SNAP, FVGParams()).features == []


def test_fvg_rejects_insufficient_data() -> None:
    close, high, low, open_ = flat(2)
    f = make_frame(close, high, low, open_)
    with pytest.raises(InsufficientData):
        detect_fvg(f, SPEC, SNAP, FVGParams())


# --------------------------------------------------------------------------- Sweeps


def test_buy_side_sweep_of_confirmed_swing_high() -> None:
    # swing high 105 at bar 5 (known at 8). Bar 12 trades to 105.5 (>=1 tick above), closes 104 (below) -> sweep
    #        0    1    2    3    4      5    6    7    8    9    10   11     12   13
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 101, 101, 101, 105.5, 101, 101]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[11] = 104.0
    low[11] = 103.0
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_liquidity_sweeps(f, SPEC, SNAP, piv, SweepParams())
    sweeps = [x for x in res.features if x.feature_type == FeatureType.LIQUIDITY_SWEEP]
    assert len(sweeps) == 1
    s = sweeps[0]
    assert s.direction == Direction.BEARISH  # buy-side sweep -> bearish signal
    assert s.levels["level"] == Decimal("105.00")
    assert s.levels["excursion"] == Decimal("0.50")
    assert s.confirmed_index == 11
    assert s.payload["side"] == "buy_side"


def test_wick_above_without_close_below_is_not_a_sweep() -> None:
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 101, 101, 101, 105.5, 101, 101]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[11] = 105.25  # closes above the level
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_liquidity_sweeps(f, SPEC, SNAP, piv, SweepParams())
    assert [x for x in res.features if x.feature_type == FeatureType.LIQUIDITY_SWEEP] == []


def test_sweep_cannot_use_pivot_before_it_is_known() -> None:
    # pivot high at bar 4 is known at bar 7; bar 6 pokes above and closes below -> NOT a sweep (level unknown yet)
    high = [101, 102, 103, 104, 105.0, 104, 105.5, 102, 101, 101, 101]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[6] = 104.0
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_liquidity_sweeps(f, SPEC, SNAP, piv, SweepParams())
    assert [x for x in res.features if x.feature_type == FeatureType.LIQUIDITY_SWEEP] == []


def test_equal_highs_form_liquidity_pool() -> None:
    # two highs at 105.0 and 105.25 (span 1 tick <= 2), 5 bars apart (>=3) -> pool; then swept at bar 16
    #        0    1    2    3    4      5    6    7    8    9      10   11   12   13   14   15   16     17   18   19
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 103, 105.25, 104, 103, 102, 101, 101, 101, 105.75, 101, 101, 101]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[16] = 104.0
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_liquidity_sweeps(f, SPEC, SNAP, piv, SweepParams())
    pools = [x for x in res.features if x.feature_type == FeatureType.LIQUIDITY_POOL]
    assert len(pools) == 1
    assert pools[0].levels["high"] == Decimal("105.25") and pools[0].levels["low"] == Decimal("105.00")
    assert pools[0].confirmed_index == 12  # second touch (bar 9) known at 9+3
    sweeps = [x for x in res.features if x.feature_type == FeatureType.LIQUIDITY_SWEEP]
    assert any(s.payload["target_kind"] == "pool" and s.confirmed_index == 16 for s in sweeps)


# --------------------------------------------------------------------------- BOS


def test_bos_close_beyond_confirmed_swing_high() -> None:
    # swing high 105 at 4 (known at 7). Bar 10 closes 105.25 (>=1 tick beyond) -> BOS bullish, swing consumed
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 101, 102, 105.5, 106, 106]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[10] = 105.25
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_bos(f, SPEC, SNAP, piv, BOSParams())
    assert len(res.features) == 1
    b = res.features[0]
    assert b.direction == Direction.BULLISH
    assert b.levels["broken_level"] == Decimal("105.00")
    assert b.confirmed_index == 10
    assert b.payload["classification"] in {"continuation", "possible_change", "unknown"}
    assert b.payload["classification"] == "unknown"  # no prior HH/HL or LH/LL structure yet
    assert any(e.kind == "consumed" and e.observed_at_index == 10 for e in res.events)


def test_bos_wick_only_does_not_count() -> None:
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 101, 102, 105.5, 103, 103]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[10] = 104.75
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    assert detect_bos(f, SPEC, SNAP, piv, BOSParams()).features == []


def test_bos_consumed_swing_not_broken_twice() -> None:
    high = [101, 102, 103, 104, 105.0, 104, 103, 102, 101, 102, 105.5, 103, 103, 105.5, 106, 106]
    low = [h - 2 for h in high]
    close = [h - 1 for h in high]
    close[10] = 105.25
    close[13] = 105.25
    open_ = [c for c in close]
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    res = detect_bos(f, SPEC, SNAP, piv, BOSParams())
    assert [b.confirmed_index for b in res.features] == [10]


# --------------------------------------------------------------------------- Order blocks


def test_bullish_order_block_selected_from_last_bearish_candle() -> None:
    n = 30
    close, high, low, open_ = flat(n)
    # swing high 102 at bar 18 (known 21)
    high[18], close[18], open_[18] = 102.0, 101.5, 101.0
    # bars 22..23: last opposite-colour (bearish) candle at 23 with zone [99.0, 100.75]
    open_[22], close[22], high[22], low[22] = 100.0, 100.5, 100.75, 99.75
    open_[23], close[23], high[23], low[23] = 100.5, 99.25, 100.75, 99.0  # bearish, body 1.25
    # bar 24 doji (ignored)
    open_[24], close[24], high[24], low[24] = 99.5, 99.5, 100.0, 99.0
    # bar 25 break candle: closes 103.0 above 102, body 3.5 >= ATR, ratio high
    open_[25], close[25], high[25], low[25] = 99.5, 103.0, 103.25, 99.25
    for i in range(26, n):
        open_[i], close[i], high[i], low[i] = 103.0, 103.0, 103.5, 102.5
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    bos = detect_bos(f, SPEC, SNAP, piv, BOSParams())
    assert [b.confirmed_index for b in bos.features] == [25]
    res = detect_order_blocks(f, SPEC, SNAP, bos, atr, OrderBlockParams())
    assert len(res.features) == 1
    ob = res.features[0]
    assert ob.direction == Direction.BULLISH
    assert ob.origin_index == 23 and ob.confirmed_index == 25
    assert ob.levels["top"] == Decimal("100.75") and ob.levels["bottom"] == Decimal("99.00")
    assert ob.levels["body_top"] == Decimal("100.50") and ob.levels["body_bottom"] == Decimal("99.25")


def test_order_block_requires_displacement_on_break_candle() -> None:
    n = 30
    close, high, low, open_ = flat(n)
    high[18], close[18], open_[18] = 102.0, 101.5, 101.0
    open_[23], close[23], high[23], low[23] = 100.5, 99.25, 100.75, 99.0
    # break candle: closes 102.25 (BOS) but body 0.5 < ATR (~1.0) -> no OB
    open_[25], close[25], high[25], low[25] = 101.75, 102.25, 102.5, 101.5
    for i in range(26, n):
        open_[i], close[i], high[i], low[i] = 102.25, 102.25, 102.5, 102.0
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    bos = detect_bos(f, SPEC, SNAP, piv, BOSParams())
    assert [b.confirmed_index for b in bos.features] == [25]
    res = detect_order_blocks(f, SPEC, SNAP, bos, atr, OrderBlockParams())
    assert res.features == []


def test_bullish_order_block_invalidates_on_close_below_low() -> None:
    n = 32
    close, high, low, open_ = flat(n)
    high[18], close[18], open_[18] = 102.0, 101.5, 101.0
    open_[23], close[23], high[23], low[23] = 100.5, 99.25, 100.75, 99.0
    open_[25], close[25], high[25], low[25] = 99.5, 103.0, 103.25, 99.25
    for i in range(26, n):
        open_[i], close[i], high[i], low[i] = 103.0, 103.0, 103.5, 102.5
    open_[28], close[28], high[28], low[28] = 103.0, 100.25, 103.0, 100.0  # revisit
    open_[30], close[30], high[30], low[30] = 100.25, 98.5, 100.5, 98.0  # close below 99.0
    f = make_frame(close, high, low, open_)
    piv = swing_pivots(f, 3, 3)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    bos = detect_bos(f, SPEC, SNAP, piv, BOSParams())
    res = detect_order_blocks(f, SPEC, SNAP, bos, atr, OrderBlockParams())
    ob = res.features[0]
    kinds = [(e.kind, e.observed_at_index) for e in res.events]
    assert ("touched", 28) in kinds
    assert ("invalidated", 30) in kinds
    assert ob.state == FeatureState.INVALIDATED


# --------------------------------------------------------------------------- RSI divergence


def _closes_with_two_lows(second_low_higher_rsi: bool) -> list[float]:
    """Build a series with confirmed price lows at two pivots; RSI is computed by the detector."""
    rng = np.random.default_rng(3)
    base = 100 + np.cumsum(rng.normal(0, 0.2, 80))
    close = base.copy()
    # first low around bar 40, second (lower) low around bar 60
    close[36:45] = np.array([99, 98, 97, 96.5, 96.0, 96.5, 97, 98, 99]) + 0
    close[56:65] = np.array([98.5, 98, 97, 96.2, 95.5, 96.2, 97, 98, 98.5]) + 0
    if second_low_higher_rsi:
        # sharp fall into first low (low RSI), gentle drift into second (higher RSI)
        close[30:36] = [103, 102.5, 102, 101.5, 101, 100]
        close[45:56] = [99.2] * 11
    else:
        # gentle into first, violent into second -> no bullish divergence
        close[30:36] = [99.5, 99.4, 99.3, 99.2, 99.1, 99.0]
        close[45:56] = [104, 104.5, 105, 104.5, 104, 103, 102, 101, 100, 99.5, 99]
    return list(close)


def test_bullish_rsi_divergence_emitted_at_second_pivot_confirmation() -> None:
    close = _closes_with_two_lows(second_low_higher_rsi=True)
    high = [c + 0.3 for c in close]
    low = [c - 0.3 for c in close]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, 3, 3)
    res = detect_rsi_divergence(f, SPEC, SNAP, piv, RSIDivergenceParams())
    bull = [d for d in res.features if d.direction == Direction.BULLISH]
    assert len(bull) >= 1
    d = bull[-1]
    p1, p2 = int(d.payload["pivot1_index"]), int(d.payload["pivot2_index"])
    assert low[p2] < low[p1]
    assert float(d.payload["rsi2"]) > float(d.payload["rsi1"]) + 2.0
    assert d.confirmed_index == p2 + 3
    assert 5 <= p2 - p1 <= 60


def test_no_bullish_divergence_when_rsi_also_lower() -> None:
    close = _closes_with_two_lows(second_low_higher_rsi=False)
    high = [c + 0.3 for c in close]
    low = [c - 0.3 for c in close]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, 3, 3)
    res = detect_rsi_divergence(f, SPEC, SNAP, piv, RSIDivergenceParams())
    assert [d for d in res.features if d.direction == Direction.BULLISH] == []


def test_rsi_divergence_requires_min_rsi_and_price_differences() -> None:
    p = RSIDivergenceParams(min_rsi_diff=2.0, min_separation=5, max_separation=60)
    assert p.min_price_ticks == 1
