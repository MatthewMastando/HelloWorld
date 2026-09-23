"""Deterministic feature detectors.

All detectors only consider complete bars (indices <= ``frame.last_complete_index()``)
and raise :class:`InsufficientData` when the warm-up requirement is not met.
Emitted price levels are ``Decimal`` via ``spec.to_tick``; natural keys are
deterministic and unique per feature.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import cast

import numpy as np

from trw.ta.params import BOSParams, FVGParams, OrderBlockParams, RSIDivergenceParams, SweepParams
from trw.ta.primitives import Pivot, atr_wilder, displacement_mask, rsi_wilder
from trw.ta.types import (
    CandleFrame,
    DetectorResult,
    Direction,
    EventKind,
    Feature,
    FeatureEvent,
    FeatureState,
    FeatureType,
    InstrumentSpec,
    InsufficientData,
    SnapshotRef,
)


def _ts(frame: CandleFrame, i: int) -> datetime:
    return cast(datetime, frame.ts.iloc[i].to_pydatetime())


def _tick(spec: InstrumentSpec) -> float:
    return float(spec.tick_size)


def _mk_feature(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    ftype: FeatureType,
    direction: Direction | None,
    origin_index: int,
    confirmed_index: int,
    levels: dict[str, Decimal],
    state: FeatureState,
    params: FVGParams | SweepParams | BOSParams | OrderBlockParams | RSIDivergenceParams,
    natural_key: str,
    payload: dict[str, object],
) -> Feature:
    return Feature(
        feature_type=ftype,
        instrument=spec.symbol,
        timeframe=frame.timeframe,
        session=frame.session,
        direction=direction,
        origin_ts=_ts(frame, origin_index),
        confirmed_ts=_ts(frame, confirmed_index),
        origin_index=origin_index,
        confirmed_index=confirmed_index,
        levels=levels,
        state=state,
        params=asdict(params),
        snapshot=snapshot,
        natural_key=natural_key,
        payload=payload,
    )


def _ev(feature: Feature, kind: EventKind, frame: CandleFrame, index: int, **detail: object) -> FeatureEvent:
    return FeatureEvent(
        feature_id=feature.feature_id,
        kind=kind,
        observed_at_ts=_ts(frame, index),
        observed_at_index=index,
        detail=dict(detail),
    )


def _final_state(feature: Feature, events: list[FeatureEvent]) -> None:
    """Set the feature's state to its last transition (confirmed stays CONFIRMED)."""
    last_kind: EventKind = "confirmed"
    for e in events:
        if e.feature_id == feature.feature_id:
            last_kind = e.kind
    feature.state = FeatureState(last_kind)


# --------------------------------------------------------------------------- FVG


def detect_fvg(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    params: FVGParams,
) -> DetectorResult:
    last = frame.last_complete_index()
    if last + 1 < 3:
        raise InsufficientData(f"fvg needs >= 3 complete bars, got {last + 1}")

    high, low, close = frame.high, frame.low, frame.close
    tick = _tick(spec)
    atr = atr_wilder(high, low, close, params.atr_period)
    disp = displacement_mask(frame, atr, params.displacement_body_atr, params.displacement_body_range)

    features: list[Feature] = []
    events: list[FeatureEvent] = []

    for i in range(2, last + 1):
        bull_gap = low[i] - high[i - 2]
        bear_gap = low[i - 2] - high[i]
        direction: Direction | None = None
        bottom = top = 0.0
        if bull_gap >= params.min_ticks * tick - 1e-12:
            direction = Direction.BULLISH
            bottom, top = float(high[i - 2]), float(low[i])
        elif bear_gap >= params.min_ticks * tick - 1e-12:
            direction = Direction.BEARISH
            bottom, top = float(high[i]), float(low[i - 2])
        if direction is None:
            continue

        top_d = spec.to_tick(top)
        bottom_d = spec.to_tick(bottom)
        midpoint = (top_d + bottom_d) / 2
        mid_f = float(midpoint)
        payload: dict[str, object] = {"displacement": bool(disp[i - 1])}
        feat = _mk_feature(
            frame,
            spec,
            snapshot,
            FeatureType.FVG,
            direction,
            origin_index=i - 2,
            confirmed_index=i,
            levels={"top": top_d, "bottom": bottom_d, "midpoint": Decimal(midpoint)},
            state=FeatureState.CONFIRMED,
            params=params,
            natural_key=f"{direction.value}:{i - 2}:{i}",
            payload=payload,
        )
        features.append(feat)
        events.append(_ev(feat, "confirmed", frame, i))

        touched = partial = midpoint_hit = full = False
        invalidated = False
        max_depth = 0.0
        span = top - bottom
        for j in range(i + 1, last + 1):
            if direction == Direction.BULLISH:
                reach = low[j]  # depth into zone measured from the top down
                depth = (top - reach) / span if reach < top else 0.0
                inv = close[j] < bottom
            else:
                reach = high[j]
                depth = (reach - bottom) / span if reach > bottom else 0.0
                inv = close[j] > top

            enters = reach <= top if direction == Direction.BULLISH else reach >= bottom
            if enters and not touched:
                touched = True
                events.append(_ev(feat, "touched", frame, j))
            depth = min(depth, 1.0)
            if depth > max_depth:
                max_depth = depth
                payload["max_fill_depth"] = depth
            if depth > 0 and not partial:
                partial = True
                events.append(_ev(feat, "partial_fill", frame, j, fill_depth=depth))
            crosses_mid = reach <= mid_f if direction == Direction.BULLISH else reach >= mid_f
            if crosses_mid and not midpoint_hit:
                midpoint_hit = True
                events.append(_ev(feat, "midpoint_touched", frame, j))
            crosses_bottom = reach <= bottom if direction == Direction.BULLISH else reach >= top
            if crosses_bottom and not full:
                full = True
                events.append(_ev(feat, "full_fill", frame, j))
            if inv and not invalidated:
                invalidated = True
                events.append(_ev(feat, "invalidated", frame, j))
                break

        _final_state(feat, events)

    events.sort(key=lambda e: e.observed_at_index)
    return DetectorResult(features=features, events=events)


# --------------------------------------------------------------------------- Liquidity pools & sweeps


def _pivot_window(pivots: list[Pivot]) -> int:
    if not pivots:
        return 7
    right = min(p.known_at_index - p.index for p in pivots)
    return 2 * right + 1


def detect_liquidity_sweeps(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    pivots: list[Pivot],
    params: SweepParams,
) -> DetectorResult:
    last = frame.last_complete_index()
    need = params.pivot_left + params.pivot_right + 1
    if last + 1 < need:
        raise InsufficientData(f"sweeps need >= {need} complete bars, got {last + 1}")

    high, low, close = frame.high, frame.low, frame.close
    tick = _tick(spec)
    features: list[Feature] = []
    events: list[FeatureEvent] = []

    # ---- pools: greedily group same-kind pivots --------------------------
    pools: list[Feature] = []

    def flush(group: list[Pivot], kind: str) -> None:
        if len(group) < params.pool_min_touches:
            return
        prices = [p.price for p in group]
        first, second = group[0], group[1]
        feat = _mk_feature(
            frame,
            spec,
            snapshot,
            FeatureType.LIQUIDITY_POOL,
            None,
            origin_index=first.index,
            confirmed_index=second.known_at_index,
            levels={"high": spec.to_tick(max(prices)), "low": spec.to_tick(min(prices))},
            state=FeatureState.CONFIRMED,
            params=params,
            natural_key=f"pool:{kind}:{first.index}:{second.index}",
            payload={"touch_indices": [p.index for p in group], "kind": kind},
        )
        pools.append(feat)
        events.append(_ev(feat, "confirmed", frame, second.known_at_index))

    for kind in ("high", "low"):
        group: list[Pivot] = []
        for p in [p for p in pivots if p.kind == kind]:
            if group:
                sep_ok = p.index - group[-1].index >= params.pool_min_separation_bars
                new_lo = min(min(pr.price for pr in group), p.price)
                new_hi = max(max(pr.price for pr in group), p.price)
                span_ticks = spec.ticks_between(spec.to_tick(new_lo), spec.to_tick(new_hi))
                if sep_ok and span_ticks <= params.pool_max_span_ticks:
                    group.append(p)
                    continue
                flush(group, kind)
                group = [p]
            else:
                group = [p]
        flush(group, kind)

    features.extend(pools)

    # ---- sweeps ------------------------------------------------------------
    swing_highs = [p for p in pivots if p.kind == "high"]
    swing_lows = [p for p in pivots if p.kind == "low"]
    excursion = params.min_excursion_ticks * tick

    def sweep_feat(
        direction: Direction,
        side: str,
        target_kind: str,
        target_key: str,
        origin_index: int,
        j: int,
        level: float,
        exc: float,
    ) -> Feature:
        payload: dict[str, object] = {
            "side": side,
            "target_kind": target_kind,
            "target_feature_key": target_key,
        }
        feat = _mk_feature(
            frame,
            spec,
            snapshot,
            FeatureType.LIQUIDITY_SWEEP,
            direction,
            origin_index=origin_index,
            confirmed_index=j,
            levels={
                "level": spec.to_tick(level),
                "excursion": spec.to_tick(exc),
                "close": spec.to_tick(float(close[j])),
            },
            state=FeatureState.CONFIRMED,
            params=params,
            natural_key=f"{side}:{target_key}:{origin_index}:{j}",
            payload=payload,
        )
        features.append(feat)
        events.append(_ev(feat, "confirmed", frame, j))
        return feat

    for j in range(1, last + 1):
        # swing highs (buy-side liquidity)
        for p in swing_highs:
            if p.known_at_index >= j:
                continue
            if high[j] >= p.price + excursion - 1e-12 and close[j] < p.price:
                sweep_feat(
                    Direction.BEARISH,
                    "buy_side",
                    "swing",
                    f"swing:high:{p.index}",
                    p.index,
                    j,
                    p.price,
                    high[j] - p.price,
                )
        # swing lows (sell-side liquidity)
        for p in swing_lows:
            if p.known_at_index >= j:
                continue
            if low[j] <= p.price - excursion + 1e-12 and close[j] > p.price:
                sweep_feat(
                    Direction.BULLISH,
                    "sell_side",
                    "swing",
                    f"swing:low:{p.index}",
                    p.index,
                    j,
                    p.price,
                    p.price - low[j],
                )
        # pools
        for pool in pools:
            if pool.confirmed_index >= j:
                continue
            ph, pl = float(pool.levels["high"]), float(pool.levels["low"])
            if pool.payload["kind"] == "high" and high[j] >= ph + excursion - 1e-12 and close[j] < ph:
                sweep_feat(
                    Direction.BEARISH,
                    "buy_side",
                    "pool",
                    pool.natural_key,
                    pool.origin_index,
                    j,
                    ph,
                    high[j] - ph,
                )
            if pool.payload["kind"] == "low" and low[j] <= pl - excursion + 1e-12 and close[j] > pl:
                sweep_feat(
                    Direction.BULLISH,
                    "sell_side",
                    "pool",
                    pool.natural_key,
                    pool.origin_index,
                    j,
                    pl,
                    pl - low[j],
                )

    events.sort(key=lambda e: e.observed_at_index)
    return DetectorResult(features=features, events=events)


# --------------------------------------------------------------------------- BOS


def detect_bos(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    pivots: list[Pivot],
    params: BOSParams,
) -> DetectorResult:
    last = frame.last_complete_index()
    need = _pivot_window(pivots)
    if last + 1 < need:
        raise InsufficientData(f"bos needs >= {need} complete bars, got {last + 1}")

    close = frame.close
    tick = _tick(spec)
    features: list[Feature] = []
    events: list[FeatureEvent] = []

    highs_known = sorted((p for p in pivots if p.kind == "high" and p.known_at_index <= last), key=lambda p: p.index)
    lows_known = sorted((p for p in pivots if p.kind == "low" and p.known_at_index <= last), key=lambda p: p.index)
    unconsumed_highs: set[int] = set()
    unconsumed_lows: set[int] = set()
    added_highs: set[int] = set()
    added_lows: set[int] = set()
    high_by_idx = {p.index: p for p in highs_known}
    low_by_idx = {p.index: p for p in lows_known}

    def classify(j: int, direction: Direction) -> str:
        hh = [p for p in highs_known if p.known_at_index < j]
        ll = [p for p in lows_known if p.known_at_index < j]
        if len(hh) < 2 or len(ll) < 2:
            return "unknown"
        h1, h2 = hh[-2], hh[-1]
        l1, l2 = ll[-2], ll[-1]
        if direction == Direction.BULLISH:
            if h2.price > h1.price and l2.price > l1.price:
                return "continuation"
            if h2.price < h1.price and l2.price < l1.price:
                return "possible_change"
        else:
            if h2.price < h1.price and l2.price < l1.price:
                return "continuation"
            if h2.price > h1.price and l2.price > l1.price:
                return "possible_change"
        return "unknown"

    for j in range(1, last + 1):
        for p in highs_known:
            if p.known_at_index < j and p.index not in added_highs:
                added_highs.add(p.index)
                unconsumed_highs.add(p.index)
        for p in lows_known:
            if p.known_at_index < j and p.index not in added_lows:
                added_lows.add(p.index)
                unconsumed_lows.add(p.index)

        # bullish BOS: close breaks a swing high
        broken_hi = [
            high_by_idx[i]
            for i in unconsumed_highs
            if close[j] >= high_by_idx[i].price + params.min_break_ticks * tick - 1e-12
        ]
        if broken_hi:
            top_pivot = max(broken_hi, key=lambda p: p.price)
            payload: dict[str, object] = {
                "classification": classify(j, Direction.BULLISH),
                "consumed_pivot_indices": sorted(p.index for p in broken_hi),
                "swing_index": top_pivot.index,
            }
            feat = _mk_feature(
                frame,
                spec,
                snapshot,
                FeatureType.BOS,
                Direction.BULLISH,
                origin_index=top_pivot.index,
                confirmed_index=j,
                levels={"broken_level": spec.to_tick(top_pivot.price), "close": spec.to_tick(float(close[j]))},
                state=FeatureState.CONFIRMED,
                params=params,
                natural_key=f"bullish:{top_pivot.index}:{j}",
                payload=payload,
            )
            features.append(feat)
            events.append(_ev(feat, "confirmed", frame, j))
            for p in broken_hi:
                unconsumed_highs.discard(p.index)
                events.append(_ev(feat, "consumed", frame, j, pivot_index=p.index, level=str(spec.to_tick(p.price))))

        # bearish BOS: close breaks a swing low
        broken_lo = [
            low_by_idx[i]
            for i in unconsumed_lows
            if close[j] <= low_by_idx[i].price - params.min_break_ticks * tick + 1e-12
        ]
        if broken_lo:
            bot_pivot = min(broken_lo, key=lambda p: p.price)
            payload = {
                "classification": classify(j, Direction.BEARISH),
                "consumed_pivot_indices": sorted(p.index for p in broken_lo),
                "swing_index": bot_pivot.index,
            }
            feat = _mk_feature(
                frame,
                spec,
                snapshot,
                FeatureType.BOS,
                Direction.BEARISH,
                origin_index=bot_pivot.index,
                confirmed_index=j,
                levels={"broken_level": spec.to_tick(bot_pivot.price), "close": spec.to_tick(float(close[j]))},
                state=FeatureState.CONFIRMED,
                params=params,
                natural_key=f"bearish:{bot_pivot.index}:{j}",
                payload=payload,
            )
            features.append(feat)
            events.append(_ev(feat, "confirmed", frame, j))
            for p in broken_lo:
                unconsumed_lows.discard(p.index)
                events.append(_ev(feat, "consumed", frame, j, pivot_index=p.index, level=str(spec.to_tick(p.price))))

    events.sort(key=lambda e: e.observed_at_index)
    return DetectorResult(features=features, events=events)


# --------------------------------------------------------------------------- Order blocks


def detect_order_blocks(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    bos_result: DetectorResult,
    atr: np.ndarray,
    params: OrderBlockParams,
) -> DetectorResult:
    last = frame.last_complete_index()
    non_nan = np.flatnonzero(~np.isnan(atr))
    atr_period = int(non_nan[0]) if non_nan.size else len(atr)
    if last + 1 < atr_period + 2:
        raise InsufficientData(f"order blocks need >= {atr_period + 2} complete bars, got {last + 1}")

    open_, high, low, close = frame.open, frame.high, frame.low, frame.close
    disp = displacement_mask(frame, atr, params.displacement_body_atr, params.displacement_body_range)

    features: list[Feature] = []
    events: list[FeatureEvent] = []

    for bos in bos_result.features:
        if bos.feature_type != FeatureType.BOS:
            continue
        j = bos.confirmed_index
        bullish = bos.direction == Direction.BULLISH
        if j > last or not disp[j]:
            continue

        chosen: int | None = None
        for k in range(j - 1, max(0, j - params.lookback) - 1, -1):
            rng = high[k] - low[k]
            body = abs(close[k] - open_[k])
            if rng <= 0 or body / rng <= params.doji_body_range_max:
                continue
            opposite = close[k] < open_[k] if bullish else close[k] > open_[k]
            if opposite:
                chosen = k
                break
        if chosen is None:
            continue

        k = chosen
        direction = Direction.BULLISH if bullish else Direction.BEARISH
        top, bottom = float(high[k]), float(low[k])
        span = top - bottom
        payload: dict[str, object] = {"bos_feature_id": bos.feature_id}
        feat = _mk_feature(
            frame,
            spec,
            snapshot,
            FeatureType.ORDER_BLOCK,
            direction,
            origin_index=k,
            confirmed_index=j,
            levels={
                "top": spec.to_tick(top),
                "bottom": spec.to_tick(bottom),
                "body_top": spec.to_tick(float(max(open_[k], close[k]))),
                "body_bottom": spec.to_tick(float(min(open_[k], close[k]))),
            },
            state=FeatureState.CONFIRMED,
            params=params,
            natural_key=f"{direction.value}:{k}:{j}",
            payload=payload,
        )
        features.append(feat)
        events.append(_ev(feat, "confirmed", frame, j))

        touched = partial = full = False
        invalidated = False
        max_trav = 0.0
        for m in range(j + 1, last + 1):
            if direction == Direction.BULLISH:
                reach = low[m]
                depth = (top - reach) / span if reach < top else 0.0
                inv = close[m] < bottom
            else:
                reach = high[m]
                depth = (reach - bottom) / span if reach > bottom else 0.0
                inv = close[m] > top

            enters = reach <= top if direction == Direction.BULLISH else reach >= bottom
            if enters and not touched:
                touched = True
                events.append(_ev(feat, "touched", frame, m))
            depth = min(depth, 1.0)
            if depth > max_trav:
                max_trav = depth
                payload["max_traversal"] = depth
            if depth > 0 and not partial:
                partial = True
                events.append(_ev(feat, "partial_fill", frame, m, traversal_depth=depth))
            crosses_bottom = reach <= bottom if direction == Direction.BULLISH else reach >= top
            if crosses_bottom and not full:
                full = True
                events.append(_ev(feat, "full_fill", frame, m))
            if inv and not invalidated:
                invalidated = True
                events.append(_ev(feat, "invalidated", frame, m))
                break

        _final_state(feat, events)

    events.sort(key=lambda e: e.observed_at_index)
    return DetectorResult(features=features, events=events)


# --------------------------------------------------------------------------- RSI divergence


def detect_rsi_divergence(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    pivots: list[Pivot],
    params: RSIDivergenceParams,
) -> DetectorResult:
    last = frame.last_complete_index()
    need = params.rsi_period + 1 + _pivot_window(pivots)
    if last + 1 < need:
        raise InsufficientData(f"rsi divergence needs >= {need} complete bars, got {last + 1}")

    close = frame.close
    low = frame.low
    high = frame.high
    tick = _tick(spec)
    rsi = rsi_wilder(close, params.rsi_period)

    features: list[Feature] = []
    events: list[FeatureEvent] = []

    def emit(
        direction: Direction,
        p1: Pivot,
        p2: Pivot,
        price1: float,
        price2: float,
        hidden: bool,
    ) -> None:
        payload: dict[str, object] = {
            "pivot1_index": p1.index,
            "pivot2_index": p2.index,
            "rsi1": float(rsi[p1.index]),
            "rsi2": float(rsi[p2.index]),
            "hidden": hidden,
        }
        feat = _mk_feature(
            frame,
            spec,
            snapshot,
            FeatureType.RSI_DIVERGENCE,
            direction,
            origin_index=p1.index,
            confirmed_index=p2.known_at_index,
            levels={"price1": spec.to_tick(price1), "price2": spec.to_tick(price2)},
            state=FeatureState.CONFIRMED,
            params=params,
            natural_key=f"{direction.value}:{p1.index}:{p2.index}{':hidden' if hidden else ''}",
            payload=payload,
        )
        features.append(feat)
        events.append(_ev(feat, "confirmed", frame, p2.known_at_index))

    lows = [p for p in pivots if p.kind == "low"]
    highs = [p for p in pivots if p.kind == "high"]

    for p1, p2 in zip(lows, lows[1:]):  # noqa: B905 - adjacent pairs
        sep = p2.index - p1.index
        if not (params.min_separation <= sep <= params.max_separation):
            continue
        if np.isnan(rsi[p1.index]) or np.isnan(rsi[p2.index]):
            continue
        lower_low = low[p2.index] <= low[p1.index] - params.min_price_ticks * tick + 1e-12
        higher_low = low[p2.index] >= low[p1.index] + params.min_price_ticks * tick - 1e-12
        rsi_rise = rsi[p2.index] >= rsi[p1.index] + params.min_rsi_diff
        rsi_fall = rsi[p2.index] <= rsi[p1.index] - params.min_rsi_diff
        if lower_low and rsi_rise:
            emit(Direction.BULLISH, p1, p2, float(low[p1.index]), float(low[p2.index]), hidden=False)
        elif params.hidden and higher_low and rsi_fall:
            emit(Direction.BULLISH, p1, p2, float(low[p1.index]), float(low[p2.index]), hidden=True)

    for p1, p2 in zip(highs, highs[1:]):  # noqa: B905 - adjacent pairs
        sep = p2.index - p1.index
        if not (params.min_separation <= sep <= params.max_separation):
            continue
        if np.isnan(rsi[p1.index]) or np.isnan(rsi[p2.index]):
            continue
        higher_high = high[p2.index] >= high[p1.index] + params.min_price_ticks * tick - 1e-12
        lower_high = high[p2.index] <= high[p1.index] - params.min_price_ticks * tick + 1e-12
        rsi_fall = rsi[p2.index] <= rsi[p1.index] - params.min_rsi_diff
        rsi_rise = rsi[p2.index] >= rsi[p1.index] + params.min_rsi_diff
        if higher_high and rsi_fall:
            emit(Direction.BEARISH, p1, p2, float(high[p1.index]), float(high[p2.index]), hidden=False)
        elif params.hidden and lower_high and rsi_rise:
            emit(Direction.BEARISH, p1, p2, float(high[p1.index]), float(high[p2.index]), hidden=True)

    events.sort(key=lambda e: e.observed_at_index)
    return DetectorResult(features=features, events=events)
