"""Deterministic TA primitives: RSI (Wilder), ATR (Wilder), swing pivots,
displacement mask, VWAP, session levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from trw.ta.types import CandleFrame


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI. NaN for indices < period; index `period` is the seed value."""
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    diff = np.diff(close)
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder ATR. TR[0] is NaN; first ATR at index `period` = mean of TR[1..period]."""
    n = len(high)
    tr = np.full(n, np.nan)
    if n >= 2:
        tr[1:] = np.maximum.reduce(
            [
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ]
        )
    atr = np.full(n, np.nan)
    if n <= period:
        return atr
    atr[period] = float(np.mean(tr[1 : period + 1]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@dataclass(frozen=True)
class Pivot:
    index: int
    kind: Literal["high", "low"]
    price: float
    known_at_index: int


def swing_pivots(frame: CandleFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    """Strict extrema over a left/right window. A pivot is emitted only once its
    full right window of complete bars exists (`index + right <= last_complete_index`)."""
    last = frame.last_complete_index()
    n = last + 1
    high = frame.high[:n]
    low = frame.low[:n]
    out: list[Pivot] = []
    for i in range(left, n - right):
        known_at = i + right
        if known_at > last:
            continue
        h = high[i]
        window_h = np.concatenate([high[i - left : i], high[i + 1 : i + right + 1]])
        if np.all(h > window_h):
            out.append(Pivot(index=i, kind="high", price=float(h), known_at_index=known_at))
        lo = low[i]
        window_l = np.concatenate([low[i - left : i], low[i + 1 : i + right + 1]])
        if np.all(lo < window_l):
            out.append(Pivot(index=i, kind="low", price=float(lo), known_at_index=known_at))
    return out


def displacement_mask(
    frame: CandleFrame,
    atr: np.ndarray,
    body_atr_mult: float = 1.0,
    body_range_min: float = 0.60,
) -> np.ndarray:
    """Bar i qualifies iff body_i >= body_atr_mult * atr[i-1] (preceding ATR;
    NaN -> False) and body_i / range_i >= body_range_min (range 0 -> False)."""
    n = len(frame)
    mask = np.zeros(n, dtype=bool)
    body = np.abs(frame.close - frame.open)
    rng = frame.high - frame.low
    for i in range(1, n):
        prev_atr = atr[i - 1]
        if np.isnan(prev_atr) or rng[i] <= 0:
            continue
        if body[i] >= body_atr_mult * prev_atr and body[i] / rng[i] >= body_range_min:
            mask[i] = True
    return mask


def vwap(frame: CandleFrame) -> np.ndarray:
    """Cumulative typical-price VWAP within the frame. NaN where volume is NaN
    (NaN-volume bars contribute nothing) or cumulative volume is zero."""
    typical = (frame.high + frame.low + frame.close) / 3.0
    vol = frame.volume
    valid = ~np.isnan(vol)
    safe_vol = np.where(valid, vol, 0.0)
    cum_vol = np.cumsum(safe_vol)
    cum_tpv = np.cumsum(typical * safe_vol)
    out = np.empty(len(frame), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(cum_tpv, cum_vol, out=out)
    out[cum_vol <= 0] = np.nan
    out[~valid] = np.nan
    return out


def session_levels(frame: CandleFrame) -> dict[str, float]:
    """High/low/open/close of the frame."""
    return {
        "open": float(frame.open[0]),
        "high": float(np.max(frame.high)),
        "low": float(np.min(frame.low)),
        "close": float(frame.close[-1]),
    }
