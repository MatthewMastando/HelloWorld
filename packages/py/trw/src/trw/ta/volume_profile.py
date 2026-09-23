"""Session volume profile: tick-aligned histogram, POC, value area, HVN/LVN."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from trw.ta.params import VolumeProfileParams
from trw.ta.types import (
    CandleFrame,
    Feature,
    FeatureState,
    FeatureType,
    InstrumentSpec,
    SnapshotRef,
)


def _feature(
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    state: FeatureState,
    params: VolumeProfileParams,
    session_label: str,
    range_start: datetime,
    range_end: datetime,
    natural_key: str,
    levels: dict[str, Decimal],
    payload: dict[str, object],
) -> Feature:
    return Feature(
        feature_type=FeatureType.VOLUME_PROFILE,
        instrument=spec.symbol,
        timeframe="session",
        session=session_label,
        direction=None,
        origin_ts=range_start,
        confirmed_ts=range_end,
        origin_index=0,
        confirmed_index=0,
        levels=levels,
        state=state,
        params=asdict(params),
        snapshot=snapshot,
        natural_key=natural_key,
        payload=payload,
    )


def _value_area(
    prices: list[Decimal],
    vols: np.ndarray,
    poc_i: int,
    total: float,
    value_area: float,
) -> tuple[int, int, float]:
    """Expand from POC bin; returns (val_idx, vah_idx, coverage)."""
    lo = hi = poc_i
    covered = vols[poc_i]
    target = value_area * total
    while covered < target and (lo > 0 or hi < len(prices) - 1):
        above = vols[hi + 1] if hi < len(prices) - 1 else None
        below = vols[lo - 1] if lo > 0 else None
        if above is None:
            lo -= 1
        elif below is None:
            hi += 1
        elif above > below:
            hi += 1
        elif below > above:
            lo -= 1
        else:
            # tie: closer to POC in price distance, then lower price
            d_above = abs(prices[hi + 1] - prices[poc_i])
            d_below = abs(prices[lo - 1] - prices[poc_i])
            if d_above < d_below:
                hi += 1
            elif d_below < d_above:
                lo -= 1
            else:
                lo -= 1  # equal distance -> lower price
        covered = float(vols[lo : hi + 1].sum())
    return lo, hi, covered / total if total > 0 else 0.0


def _local_extrema(
    prices: list[Decimal],
    smoothed: np.ndarray,
    window: int,
    is_max: bool,
    threshold: float,
) -> list[Decimal]:
    """Bins that are local maxima (or minima) within +-window bins and past the
    percentile threshold. Plateau runs collapse to their middle bin
    (lower-middle for even runs)."""
    n = len(prices)
    marks: list[int] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        w = smoothed[lo : hi + 1]
        if is_max:
            ok = smoothed[i] >= w.max() and smoothed[i] >= threshold
        else:
            ok = smoothed[i] <= w.min() and smoothed[i] <= threshold
        if ok:
            marks.append(i)
    # collapse runs of equal smoothed value into their middle bin
    out: list[int] = []
    run: list[int] = []
    for i in marks:
        if run and i == run[-1] + 1 and smoothed[i] == smoothed[run[-1]]:
            run.append(i)
        else:
            if run:
                out.append(run[(len(run) - 1) // 2])
            run = [i]
    if run:
        out.append(run[(len(run) - 1) // 2])
    return [prices[i] for i in out]


def _profile_from_hist(
    hist: dict[Decimal, float],
    spec: InstrumentSpec,
    params: VolumeProfileParams,
) -> tuple[dict[str, Decimal], dict[str, object]]:
    prices = sorted(hist)
    vols = np.array([hist[p] for p in prices], dtype=float)
    total = float(vols.sum())

    poc_i = int(min((i for i in range(len(prices)) if vols[i] == vols.max()), key=lambda i: prices[i]))
    lo, hi, coverage = _value_area(prices, vols, poc_i, total, params.value_area)

    k = params.node_ma_bins
    half = k // 2
    smoothed = np.array([vols[max(0, i - half) : i + half + 1].mean() for i in range(len(prices))])
    hvn_thr = float(np.percentile(smoothed, params.hvn_pct))
    lvn_thr = float(np.percentile(smoothed, params.lvn_pct))
    hvn = _local_extrema(prices, smoothed, params.node_window, is_max=True, threshold=hvn_thr)
    lvn = _local_extrema(prices, smoothed, params.node_window, is_max=False, threshold=lvn_thr)

    levels = {"poc": prices[poc_i], "vah": prices[hi], "val": prices[lo]}
    payload: dict[str, object] = {
        "histogram": [[float(p), float(hist[p])] for p in prices],
        "hvn": [float(p) for p in hvn],
        "lvn": [float(p) for p in lvn],
        "coverage": coverage,
        "bin_width": str(spec.tick_size),
        "total_volume": total,
    }
    return levels, payload


def volume_profile(
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    params: VolumeProfileParams,
    trades: pd.DataFrame | None,
    bars: CandleFrame | None,
    session_label: str,
    range_start: datetime,
    range_end: datetime,
) -> Feature:
    key = f"volprof:{session_label}:{range_start.isoformat()}:{range_end.isoformat()}"

    if trades is None:
        if bars is not None and params.allow_bar_approximation:
            hist: dict[Decimal, float] = {}
            last = bars.last_complete_index()
            for i in range(last + 1):
                v = bars.volume[i]
                if np.isnan(v):
                    continue
                px = spec.to_tick(float(bars.close[i]))
                hist[px] = hist.get(px, 0.0) + float(v)
            levels, payload = _profile_from_hist(hist, spec, params)
            payload["approximation"] = "bar_volume_at_close_bucket"
            payload["labeled"] = True
            payload["exact"] = False
            return _feature(
                spec,
                snapshot,
                FeatureState.CONFIRMED,
                params,
                session_label,
                range_start,
                range_end,
                key,
                levels,
                payload,
            )
        return _feature(
            spec,
            snapshot,
            FeatureState.UNAVAILABLE,
            params,
            session_label,
            range_start,
            range_end,
            key,
            levels={},
            payload={"reason": "no_trade_data", "approximation": None},
        )

    hist = {}
    for _, row in trades.iterrows():
        px = spec.to_tick(float(row["price"]))
        hist[px] = hist.get(px, 0.0) + float(row["size"])
    levels, payload = _profile_from_hist(hist, spec, params)
    payload["exact"] = True
    return _feature(
        spec,
        snapshot,
        FeatureState.CONFIRMED,
        params,
        session_label,
        range_start,
        range_end,
        key,
        levels,
        payload,
    )
