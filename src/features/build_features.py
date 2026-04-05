from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {"race_id", "horse_id", "asof_ts"}


def _parse_utc_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _to_utc_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_registry(registry_path: Path) -> dict[str, Any]:
    with registry_path.open("r", encoding="utf-8") as f:
        data = json.loads(f.read())

    snapshots = data.get("feature_pipeline", {}).get("snapshots", [])
    if not snapshots:
        raise ValueError("feature_registry does not define feature_pipeline.snapshots")
    return data


def load_input_csv(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"input csv is missing required columns: {sorted(missing)}")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["asof_ts"] = _parse_utc_ts(row["asof_ts"])
        parsed.append(item)
    return parsed


def _row_hash(race_id: str, horse_id: str) -> str:
    raw = f"{race_id}|{horse_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def generate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["key_race_group_id"] = str(row["race_id"])
        out.append(item)
    return out


def generate_ctx_features(rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    version = registry.get("version", "unknown")

    for row in rows:
        item = dict(row)
        item["ctx_pipeline_version"] = version
        item["ctx_model_separation_enabled"] = True
        item["ctx_source_row_hash"] = _row_hash(str(item["race_id"]), str(item["horse_id"]))
        item["ctx_has_odds"] = bool(str(item.get("odds", "")).strip())
        item["ctx_has_popularity"] = bool(str(item.get("popularity", "")).strip())

        # TODO(v1.2-mvp): add family implementations (form/pace/pedigree/course_bias).
        # NOTE: We intentionally avoid current-row leakage feature construction.
        out.append(item)

    return out


def explode_snapshots(rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot_specs = registry["feature_pipeline"]["snapshots"]
    expanded_rows: list[dict[str, Any]] = []

    for row in rows:
        for spec in snapshot_specs:
            item = dict(row)
            item["key_snapshot"] = spec["key_snapshot"]
            item["key_asof_ts"] = row["asof_ts"] - timedelta(seconds=int(spec["offset_seconds"]))
            expanded_rows.append(item)

    return expanded_rows


def build_feature_rows(input_rows: list[dict[str, Any]], registry: dict[str, Any]) -> list[dict[str, Any]]:
    base = generate_keys(input_rows)
    base = generate_ctx_features(base, registry)
    out = explode_snapshots(base, registry)

    ordered = [
        "race_id",
        "horse_id",
        "key_race_group_id",
        "key_snapshot",
        "key_asof_ts",
        "asof_ts",
    ]

    if not out:
        return out

    remaining = [col for col in out[0].keys() if col not in ordered]
    normalized: list[dict[str, Any]] = []
    for row in out:
        normalized.append({k: row.get(k, "") for k in ordered + remaining})
    return normalized


def save_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    serialized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                item[key] = _to_utc_iso(value)
            else:
                item[key] = value
        serialized.append(item)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(serialized[0].keys()))
        writer.writeheader()
        writer.writerows(serialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP feature pipeline outputs")
    parser.add_argument("--input", type=Path, required=True, help="input CSV path")
    parser.add_argument("--output", type=Path, required=True, help="output CSV path")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/feature_registry_v1_2.yaml"),
        help="feature registry YAML path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    input_rows = load_input_csv(args.input)
    output_rows = build_feature_rows(input_rows, registry)
    save_output(output_rows, args.output)


if __name__ == "__main__":
    main()
