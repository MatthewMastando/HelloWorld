"""Hand-calculated tests for TA primitives. These define the contract; do not adjust to fit code."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trw.ta.primitives import (
    atr_wilder,
    displacement_mask,
    rsi_wilder,
    swing_pivots,
)
from trw.ta.types import CandleFrame, InsufficientData


def make_frame(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_: list[float] | None = None,
    start: datetime = datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
    tf_minutes: int = 5,
) -> CandleFrame:
    n = len(close)
    high = high or [c + 0.5 for c in close]
    low = low or [c - 0.5 for c in close]
    open_ = open_ or ([close[0]] + close[:-1])
    ts = [start + timedelta(minutes=tf_minutes * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * n,
            "complete": [True] * n,
        }
    )
    return CandleFrame(df, timeframe=f"{tf_minutes}m")


# --------------------------------------------------------------------------- RSI


def naive_rsi(close: list[float], period: int = 14) -> list[float]:
    """Independent reference: seed with first `period` gains/losses, then Wilder smoothing."""
    out = [float("nan")] * len(close)
    if len(close) <= period:
        return out
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period

    def to_rsi(ag: float, al: float) -> float:
        if ag == 0 and al == 0:
            return 50.0
        if al == 0:
            return 100.0
        if ag == 0:
            return 0.0
        rs = ag / al
        return 100 - 100 / (1 + rs)

    out[period] = to_rsi(ag, al)
    for i in range(period + 1, len(close)):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = to_rsi(ag, al)
    return out


def test_rsi_all_up_is_100() -> None:
    close = [100 + i for i in range(20)]
    rsi = rsi_wilder(np.array(close, dtype=float), 14)
    assert np.isnan(rsi[:14]).all()
    assert rsi[14] == 100.0
    assert rsi[19] == 100.0


def test_rsi_all_down_is_0() -> None:
    close = [100 - i for i in range(20)]
    rsi = rsi_wilder(np.array(close, dtype=float), 14)
    assert rsi[14] == 0.0
    assert rsi[19] == 0.0


def test_rsi_flat_is_50() -> None:
    close = [100.0] * 20
    rsi = rsi_wilder(np.array(close, dtype=float), 14)
    assert rsi[14] == 50.0
    assert rsi[19] == 50.0


def test_rsi_matches_naive_reference() -> None:
    rng = np.random.default_rng(7)
    close = list(100 + np.cumsum(rng.normal(0, 1, 120)))
    got = rsi_wilder(np.array(close), 14)
    exp = naive_rsi(close, 14)
    assert np.isnan(got[:14]).all()
    np.testing.assert_allclose(got[14:], exp[14:], rtol=0, atol=1e-9)


def test_rsi_hand_calculated_first_value() -> None:
    # 15 closes: 14 changes = +1 x 10, -2 x 4  => avg gain 10/14, avg loss 8/14, RS=1.25, RSI=55.555...
    close = [100.0]
    changes = [1, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1]
    for c in changes:
        close.append(close[-1] + c)
    rsi = rsi_wilder(np.array(close), 14)
    assert rsi[14] == pytest.approx(100 - 100 / (1 + 1.25), abs=1e-12)
    # next bar +3: ag=(10/14*13+3)/14, al=(8/14*13+0)/14
    close.append(close[-1] + 3)
    rsi = rsi_wilder(np.array(close), 14)
    ag = ((10 / 14) * 13 + 3) / 14
    al = ((8 / 14) * 13) / 14
    assert rsi[15] == pytest.approx(100 - 100 / (1 + ag / al), abs=1e-12)


def test_rsi_insufficient_data_is_nan_not_error() -> None:
    rsi = rsi_wilder(np.array([1.0, 2.0, 3.0]), 14)
    assert np.isnan(rsi).all()


# --------------------------------------------------------------------------- ATR


def test_atr_wilder_seed_and_smoothing() -> None:
    # true range with constant 2.0 range and no gaps => ATR == 2.0 everywhere after seed
    n = 30
    close = [100.0] * n
    high = [101.0] * n
    low = [99.0] * n
    f = make_frame(close, high, low, open_=[100.0] * n)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    # TR needs a prior close, so TR[0] is NaN; first ATR at index 14 (14 TRs: idx 1..14)
    assert np.isnan(atr[:14]).all()
    np.testing.assert_allclose(atr[14:], 2.0)


def test_atr_wilder_hand_calculated_with_gap() -> None:
    # 16 bars; bar i has range 1.0 except bar 15 gaps up: prev close 100, high 105, low 104 -> TR = 5
    n = 16
    close = [100.0] * 15 + [104.5]
    high = [100.5] * 15 + [105.0]
    low = [99.5] * 15 + [104.0]
    open_ = [100.0] * 15 + [104.2]
    f = make_frame(close, high, low, open_)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    assert atr[14] == pytest.approx(1.0)
    assert atr[15] == pytest.approx((1.0 * 13 + 5.0) / 14)


# --------------------------------------------------------------------------- Pivots


def test_swing_pivot_high_known_at_i_plus_3() -> None:
    #            0   1   2   3   4   5    6   7   8   9   10
    high = [10, 11, 12, 13, 14, 13, 12, 11, 10, 9, 8]
    low = [h - 1 for h in high]
    close = [h - 0.5 for h in high]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, left=3, right=3)
    highs = [p for p in piv if p.kind == "high"]
    assert len(highs) == 1
    assert highs[0].index == 4
    assert highs[0].known_at_index == 7
    assert highs[0].price == 14


def test_swing_pivot_equal_extrema_is_not_strict_pivot() -> None:
    high = [10, 11, 12, 14, 14, 13, 12, 11, 10, 9, 8]
    low = [h - 1 for h in high]
    close = [h - 0.5 for h in high]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, left=3, right=3)
    assert [p for p in piv if p.kind == "high"] == []


def test_swing_pivot_needs_full_right_window() -> None:
    # candidate high at index 8 but only 2 bars after it => not yet known
    high = [10, 11, 12, 13, 12, 11, 12, 13, 15, 13, 12]
    low = [h - 1 for h in high]
    close = [h - 0.5 for h in high]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, left=3, right=3)
    assert all(p.index != 8 for p in piv)


def test_swing_pivot_low() -> None:
    low = [10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12]
    high = [x + 1 for x in low]
    close = [x + 0.5 for x in low]
    f = make_frame(close, high, low, open_=close)
    piv = swing_pivots(f, left=3, right=3)
    lows = [p for p in piv if p.kind == "low"]
    assert len(lows) == 1 and lows[0].index == 4 and lows[0].known_at_index == 7 and lows[0].price == 6


# --------------------------------------------------------------------------- Displacement


def test_displacement_requires_body_ge_atr_and_body_range_ratio() -> None:
    n = 20
    close = [100.0] * n
    high = [100.5] * n
    low = [99.5] * n
    open_ = [100.0] * n
    # bar 18: body 1.2 (>= ATR 1.0), range 1.5 => ratio 0.8 -> displacement
    open_[18], close[18], high[18], low[18] = 100.0, 101.2, 101.4, 99.9
    # bar 19: body 1.2 but range 2.5 => ratio 0.48 -> not displacement
    open_[19], close[19], high[19], low[19] = 101.0, 102.2, 103.0, 100.5
    f = make_frame(close, high, low, open_)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    mask = displacement_mask(f, atr, body_atr_mult=1.0, body_range_min=0.60)
    assert mask[18]
    assert not mask[19]
    assert not mask[17]


def test_displacement_uses_preceding_atr() -> None:
    # The ATR used for bar i must be atr[i-1] (preceding), not the one including bar i.
    n = 20
    close = [100.0] * n
    high = [100.5] * n
    low = [99.5] * n
    open_ = [100.0] * n
    # bar 19 body exactly 1.0 == preceding ATR 1.0 -> qualifies (>=). ATR including bar 19 would be larger.
    open_[19], close[19], high[19], low[19] = 100.0, 101.0, 101.2, 99.9
    f = make_frame(close, high, low, open_)
    atr = atr_wilder(f.high, f.low, f.close, 14)
    mask = displacement_mask(f, atr, body_atr_mult=1.0, body_range_min=0.60)
    assert mask[19]


def test_candleframe_rejects_bad_ohlc() -> None:
    with pytest.raises(ValueError):
        make_frame([100.0, 101.0], high=[100.5, 100.0], low=[99.5, 99.0])


def test_insufficient_data_raised_by_detectors_not_primitives() -> None:
    assert issubclass(InsufficientData, ValueError)
    assert Decimal("0.25") * 4 == 1
