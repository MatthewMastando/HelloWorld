"""Frozen parameter dataclasses for the TA detectors. Detectors store
`dataclasses.asdict(params)` on each emitted Feature."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FVGParams:
    min_ticks: int = 1
    displacement_body_atr: float = 1.0
    displacement_body_range: float = 0.60
    atr_period: int = 14


@dataclass(frozen=True)
class SweepParams:
    pool_min_touches: int = 2
    pool_min_separation_bars: int = 3
    pool_max_span_ticks: int = 2
    min_excursion_ticks: int = 1
    pivot_left: int = 3
    pivot_right: int = 3


@dataclass(frozen=True)
class BOSParams:
    min_break_ticks: int = 1


@dataclass(frozen=True)
class OrderBlockParams:
    lookback: int = 5
    displacement_body_atr: float = 1.0
    displacement_body_range: float = 0.60
    doji_body_range_max: float = 0.10


@dataclass(frozen=True)
class RSIDivergenceParams:
    rsi_period: int = 14
    min_separation: int = 5
    max_separation: int = 60
    min_price_ticks: int = 1
    min_rsi_diff: float = 2.0
    hidden: bool = False


@dataclass(frozen=True)
class VolumeProfileParams:
    value_area: float = 0.70
    node_ma_bins: int = 3
    node_window: int = 2
    hvn_pct: float = 75
    lvn_pct: float = 25
    allow_bar_approximation: bool = False


@dataclass(frozen=True)
class TAParams:
    fvg: FVGParams = FVGParams()
    sweep: SweepParams = SweepParams()
    bos: BOSParams = BOSParams()
    order_block: OrderBlockParams = OrderBlockParams()
    rsi_divergence: RSIDivergenceParams = RSIDivergenceParams()
    volume_profile: VolumeProfileParams = VolumeProfileParams()
