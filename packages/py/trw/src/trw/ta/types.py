"""Core types for the deterministic TA engine.

Every detector returns `Feature` objects. The same objects drive charts,
reports, alerts and replay. Times are UTC. Prices are Decimal-exact via
`tick_size`; internally detectors may use floats but every emitted level is
tick-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

CALC_VERSION = "ta-2026.09.1"


class InstrumentKind(str, Enum):
    FUTURE_CONTRACT = "future_contract"
    EQUITY = "equity"
    ETF = "etf"
    SPOT_CRYPTO = "spot_crypto"


@dataclass(frozen=True)
class InstrumentSpec:
    """Minimum metadata detectors need. Full futures metadata lives in the domain layer."""

    symbol: str
    kind: InstrumentKind
    venue: str
    currency: str
    tick_size: Decimal
    tick_value: Decimal
    point_multiplier: Decimal
    # futures only
    root: str | None = None
    expiry: datetime | None = None

    def to_tick(self, price: float) -> Decimal:
        """Round a float price to the nearest tick (half-up)."""
        q = Decimal(str(price)) / self.tick_size
        return (q.to_integral_value(rounding="ROUND_HALF_UP")) * self.tick_size

    def ticks_between(self, a: Decimal, b: Decimal) -> int:
        return int(((b - a) / self.tick_size).to_integral_value(rounding="ROUND_HALF_UP"))


@dataclass(frozen=True)
class SnapshotRef:
    """Identifies the immutable data a feature was computed from."""

    snapshot_id: str
    data_revision: str
    source: str
    feed_tier: str
    as_of: datetime
    synthetic: bool


class CandleFrame:
    """Validated OHLCV frame.

    Columns: ts (UTC, bar open), open, high, low, close, volume (float, NaN when unknown).
    Rows are sorted, unique on ts. `complete` is a bool column: the final bar may be
    incomplete; detectors only *confirm* on complete bars.
    """

    REQUIRED = ("ts", "open", "high", "low", "close", "volume", "complete")

    def __init__(self, df: pd.DataFrame, timeframe: str, session: str = "rth") -> None:
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"CandleFrame missing columns: {missing}")
        if not pd.api.types.is_datetime64tz_dtype(df["ts"]):
            raise ValueError("ts must be tz-aware (UTC)")
        if not df["ts"].is_monotonic_increasing or df["ts"].duplicated().any():
            raise ValueError("ts must be strictly increasing and unique")
        bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
        if bad.any():
            raise ValueError(f"OHLC invariant violated at rows {list(df.index[bad][:5])}")
        self.df = df.reset_index(drop=True)
        self.timeframe = timeframe
        self.session = session

    def __len__(self) -> int:
        return len(self.df)

    @property
    def ts(self) -> pd.Series:
        return self.df["ts"]

    @property
    def open(self) -> np.ndarray:
        return self.df["open"].to_numpy(dtype=float)

    @property
    def high(self) -> np.ndarray:
        return self.df["high"].to_numpy(dtype=float)

    @property
    def low(self) -> np.ndarray:
        return self.df["low"].to_numpy(dtype=float)

    @property
    def close(self) -> np.ndarray:
        return self.df["close"].to_numpy(dtype=float)

    @property
    def volume(self) -> np.ndarray:
        return self.df["volume"].to_numpy(dtype=float)

    @property
    def complete(self) -> np.ndarray:
        return self.df["complete"].to_numpy(dtype=bool)

    def last_complete_index(self) -> int:
        idx = np.flatnonzero(self.complete)
        if idx.size == 0:
            raise ValueError("no complete bars")
        return int(idx[-1])


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FeatureType(str, Enum):
    SWING_PIVOT = "swing_pivot"
    LIQUIDITY_POOL = "liquidity_pool"
    FVG = "fvg"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BOS = "bos"
    ORDER_BLOCK = "order_block"
    RSI_DIVERGENCE = "rsi_divergence"
    VOLUME_PROFILE = "volume_profile"


class FeatureState(str, Enum):
    CONFIRMED = "confirmed"  # fresh, untouched
    TOUCHED = "touched"
    PARTIAL_FILL = "partial_fill"
    MIDPOINT_TOUCHED = "midpoint_touched"
    FULL_FILL = "full_fill"
    CONSUMED = "consumed"  # e.g. swing broken by BOS
    INVALIDATED = "invalidated"
    UNAVAILABLE = "unavailable"  # e.g. volume profile without trades


@dataclass
class Feature:
    """One computed technical feature.

    `origin_ts` is the bar where the pattern begins (e.g. pivot bar, FVG candle A).
    `confirmed_ts` is the close of the bar at which the feature became *known*;
    replay guarantees a feature is never emitted before `confirmed_ts`.
    `natural_key` is stable across recomputation on the same data revision and is
    what dedupe uses; `feature_id` = sha1(data_revision, type, calc_version, natural_key)[:16].
    """

    feature_type: FeatureType
    instrument: str
    timeframe: str
    session: str
    direction: Direction | None
    origin_ts: datetime
    confirmed_ts: datetime
    origin_index: int
    confirmed_index: int
    levels: dict[str, Decimal]
    state: FeatureState
    params: dict[str, object]
    snapshot: SnapshotRef
    natural_key: str
    payload: dict[str, object] = field(default_factory=dict)
    calc_version: str = CALC_VERSION
    feature_id: str = ""
    as_of: datetime | None = None

    def __post_init__(self) -> None:
        if not self.feature_id:
            import hashlib

            raw = f"{self.snapshot.data_revision}|{self.feature_type.value}|{self.calc_version}|{self.natural_key}"
            self.feature_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
        if self.as_of is None:
            self.as_of = self.snapshot.as_of


EventKind = Literal[
    "confirmed", "touched", "partial_fill", "midpoint_touched", "full_fill", "consumed", "invalidated"
]


@dataclass(frozen=True)
class FeatureEvent:
    """Append-only state transition; `observed_at_ts` is the close of the bar that caused it."""

    feature_id: str
    kind: EventKind
    observed_at_ts: datetime
    observed_at_index: int
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class DetectorResult:
    features: list[Feature]
    events: list[FeatureEvent]
    warnings: list[str] = field(default_factory=list)


class InsufficientData(ValueError):
    """Raised when warm-up requirements are not met; never silently return partial results."""
