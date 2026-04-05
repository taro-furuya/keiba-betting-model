from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.features.build_features import build_feature_rows, load_input_csv, load_registry


def test_build_feature_rows_generates_required_keys_and_snapshots() -> None:
    registry = load_registry(Path("config/feature_registry_v1_2.yaml"))
    input_rows = load_input_csv(Path("data/samples/features_skeleton_sample.csv"))

    out = build_feature_rows(input_rows, registry)

    assert {"key_race_group_id", "key_snapshot", "key_asof_ts"}.issubset(out[0].keys())
    assert {row["key_snapshot"] for row in out} == {"S60", "S10"}
    assert len(out) == len(input_rows) * 2


def test_snapshot_offsets_are_applied() -> None:
    registry = load_registry(Path("config/feature_registry_v1_2.yaml"))
    input_rows = load_input_csv(Path("data/samples/features_skeleton_sample.csv"))
    out = build_feature_rows(input_rows, registry)

    sample = [r for r in out if r["race_id"] == "R202604050101" and r["horse_id"] == "H001"]
    s60 = next(r for r in sample if r["key_snapshot"] == "S60")
    s10 = next(r for r in sample if r["key_snapshot"] == "S10")

    asof = datetime(2026, 4, 5, 1, 0, tzinfo=timezone.utc)
    assert s60["key_asof_ts"] == asof - timedelta(seconds=60)
    assert s10["key_asof_ts"] == asof - timedelta(seconds=10)
