"""Engine orchestration: run every detector over a CandleFrame."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trw.ta.detectors import (
    detect_bos,
    detect_fvg,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_rsi_divergence,
)
from trw.ta.params import TAParams
from trw.ta.primitives import Pivot, atr_wilder, rsi_wilder, swing_pivots, vwap
from trw.ta.types import (
    CandleFrame,
    DetectorResult,
    Feature,
    FeatureEvent,
    InstrumentSpec,
    InsufficientData,
    SnapshotRef,
)
from trw.ta.volume_profile import volume_profile


@dataclass
class EngineResult:
    features: list[Feature]
    events: list[FeatureEvent]
    pivots: list[Pivot]
    warnings: list[str] = field(default_factory=list)
    indicators: dict[str, np.ndarray] = field(default_factory=dict)


def run_all(
    frame: CandleFrame,
    spec: InstrumentSpec,
    snapshot: SnapshotRef,
    params: TAParams | None = None,
    trades: pd.DataFrame | None = None,
) -> EngineResult:
    params = params or TAParams()
    warnings: list[str] = []
    features: list[Feature] = []
    events: list[FeatureEvent] = []

    last = frame.last_complete_index()
    indicators = {
        "rsi": rsi_wilder(frame.close, params.rsi_divergence.rsi_period),
        "atr": atr_wilder(frame.high, frame.low, frame.close, params.fvg.atr_period),
        "vwap": vwap(frame),
    }

    try:
        pivots = swing_pivots(frame, params.sweep.pivot_left, params.sweep.pivot_right)
    except ValueError as exc:
        pivots = []
        warnings.append(f"swing_pivots: {exc}")

    def run(name: str, fn: Callable[[], DetectorResult]) -> DetectorResult | None:
        try:
            return fn()
        except InsufficientData as exc:
            warnings.append(f"{name}: {exc}")
            return None

    bos_res = DetectorResult(features=[], events=[])
    for name, fn in [
        ("fvg", lambda: detect_fvg(frame, spec, snapshot, params.fvg)),
        ("sweeps", lambda: detect_liquidity_sweeps(frame, spec, snapshot, pivots, params.sweep)),
        ("bos", lambda: detect_bos(frame, spec, snapshot, pivots, params.bos)),
    ]:
        res = run(name, fn)
        if res is None:
            continue
        features.extend(res.features)
        events.extend(res.events)
        if name == "bos":
            bos_res = res

    res = run(
        "order_blocks",
        lambda: detect_order_blocks(frame, spec, snapshot, bos_res, indicators["atr"], params.order_block),
    )
    if res is not None:
        features.extend(res.features)
        events.extend(res.events)

    res = run(
        "rsi_divergence",
        lambda: detect_rsi_divergence(frame, spec, snapshot, pivots, params.rsi_divergence),
    )
    if res is not None:
        features.extend(res.features)
        events.extend(res.events)

    vp = volume_profile(
        spec,
        snapshot,
        params.volume_profile,
        trades,
        frame,
        session_label=frame.session,
        range_start=frame.ts.iloc[0].to_pydatetime(),
        range_end=frame.ts.iloc[last].to_pydatetime(),
    )
    features.append(vp)

    events.sort(key=lambda e: e.observed_at_index)
    return EngineResult(features=features, events=events, pivots=pivots, warnings=warnings, indicators=indicators)
