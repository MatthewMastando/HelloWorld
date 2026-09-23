"""Hand-calculated tests for the volume profile feature."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from trw.ta.params import VolumeProfileParams
from trw.ta.types import (
    FeatureState,
    FeatureType,
    InstrumentKind,
    InstrumentSpec,
    SnapshotRef,
)
from trw.ta.volume_profile import volume_profile

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
START = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
END = START + timedelta(hours=6)


def trades(*rows: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([START + timedelta(seconds=i) for i, _ in enumerate(rows)], utc=True),
            "price": [r[0] for r in rows],
            "size": [r[1] for r in rows],
        }
    )


def test_unavailable_without_trades_or_approximation() -> None:
    f = volume_profile(SPEC, SNAP, VolumeProfileParams(), None, None, "rth", START, END)
    assert f.feature_type == FeatureType.VOLUME_PROFILE
    assert f.state == FeatureState.UNAVAILABLE
    assert f.payload["reason"] == "no_trade_data"
    assert f.payload["approximation"] is None


def test_poc_tie_goes_to_lower_price() -> None:
    t = trades((100.00, 10), (100.25, 10))
    f = volume_profile(SPEC, SNAP, VolumeProfileParams(), t, None, "rth", START, END)
    assert f.levels["poc"] == Decimal("100.00")


def test_value_area_expansion_and_coverage() -> None:
    # POC 100.00 (10). Above: 100.25 vol 6, 100.50 vol 4. Below: 99.75 vol 6, 99.50 vol 4.
    # value_area 0.70 * total 30 = 21 -> expand: tie 6 vs 6 (equal distance) -> take lower (99.75);
    # then 6 vs 4 -> take above 100.25; covered 22 -> VA [99.75, 100.25], coverage 22/30.
    t = trades(
        (100.00, 10),
        (100.25, 6),
        (99.75, 6),
        (100.50, 4),
        (99.50, 4),
    )
    f = volume_profile(SPEC, SNAP, VolumeProfileParams(value_area=0.70), t, None, "rth", START, END)
    assert f.levels["poc"] == Decimal("100.00")
    assert f.levels["val"] == Decimal("99.75")
    assert f.levels["vah"] == Decimal("100.25")
    assert abs(float(f.payload["coverage"]) - 22 / 30) < 1e-9
    assert f.payload["exact"] is True
    assert f.payload["total_volume"] == 30.0
    assert f.payload["bin_width"] == "0.25"
    assert dict((Decimal(str(p)), v) for p, v in f.payload["histogram"])[Decimal("100.00")] == 10.0


def test_va_tie_picks_closer_to_poc() -> None:
    # POC 100.00 (10). Adjacent histogram bins tie on volume (6): below 99.75 is
    # 0.25 from POC, above 100.50 is 0.50 -> tie-break picks the closer (lower) bin.
    t = trades((100.00, 10), (99.75, 6), (100.50, 6))
    f = volume_profile(SPEC, SNAP, VolumeProfileParams(value_area=0.6), t, None, "rth", START, END)
    # target 13.2: one expansion -> VA [99.75, 100.00]
    assert f.levels["val"] == Decimal("99.75")
    assert f.levels["vah"] == Decimal("100.00")


def test_va_tie_same_distance_picks_lower_price() -> None:
    # POC 100.00 (10). Symmetric volume tie (5 each) at equal distance -> lower price.
    t = trades((100.00, 10), (99.75, 5), (100.25, 5))
    f = volume_profile(SPEC, SNAP, VolumeProfileParams(value_area=0.6), t, None, "rth", START, END)
    # target 12: add 99.75 -> VA [99.75, 100.00]
    assert f.levels["val"] == Decimal("99.75")
    assert f.levels["vah"] == Decimal("100.00")


def test_bar_approximation_is_labeled() -> None:
    frame = make_frame([100.0, 100.5, 100.0])
    params = VolumeProfileParams(allow_bar_approximation=True)
    f = volume_profile(SPEC, SNAP, params, None, frame, "rth", START, END)
    assert f.state == FeatureState.CONFIRMED
    assert f.payload["approximation"] == "bar_volume_at_close_bucket"
    assert f.payload["labeled"] is True
    assert f.payload["exact"] is False
    # volume 100 each: bin 100.00 gets 200, bin 100.50 gets 100
    assert f.levels["poc"] == Decimal("100.00")
