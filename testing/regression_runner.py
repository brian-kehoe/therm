# testing/regression_runner.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import zipfile
import pandas as pd
from datetime import datetime, timezone

# -------------------------------
# ✅ PRODUCTION MODULE IMPORTS
# -------------------------------

import ha_loader
import data_loader
import processing
from view_runs import _get_friendly_name
from schema_defs import ROOM_SENSOR_PREFIX, ZONE_SENSOR_PREFIX


# -------------------------------
# ✅ GOLDEN FILE PATHS
# -------------------------------

@dataclass
class SamplePaths:
    base: Path

    @property
    def ha_csv(self) -> Path:
        return self.base / "sample_data" / "ha_input.csv"

    @property
    def ha_profile(self) -> Path:
        return self.base / "sample_data" / "ha_profile.json"

    @property
    def grafana_numeric(self) -> Path:
        return self.base / "sample_data" / "grafana_numeric.csv"

    @property
    def grafana_state(self) -> Path:
        return self.base / "sample_data" / "grafana_state.csv"

    @property
    def grafana_profile(self) -> Path:
        return self.base / "sample_data" / "grafana_profile.json"

    @property
    def artifacts(self) -> Path:
        return self.base / "artifacts"


# -------------------------------
# ✅ BASIC HELPERS
# -------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_artifacts(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def _write_debug_bundle(zip_path: Path, df: pd.DataFrame, debug_json: Dict[str, Any], prefix: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(debug_json, indent=2).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}_merged.csv", csv_bytes)
        zf.writestr(f"{prefix}_debug.json", json_bytes)


def _utc_timestamp() -> str:
    """
    Returns a filesystem-safe UTC timestamp like:
    2025-12-08T13-42-10Z
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# -------------------------------
# ✅ UI SANITY BUILDERS
# -------------------------------

def _summarise_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_runs": len(runs),
        "heating_runs": sum(r.get("run_type") == "Heating" for r in runs),
        "dhw_runs": sum(r.get("run_type") == "DHW" for r in runs),
        "defrost_runs": sum("Defrost" in r.get("run_type", "") for r in runs),
    }


def _summarise_mapping(profile: Dict[str, Any]) -> Dict[str, Any]:
    m = profile.get("mapping", {})
    zones = []
    rooms = []

    for k in m:
        if k.startswith(ZONE_SENSOR_PREFIX):
            zones.append({"key": k, "label": _get_friendly_name(k, profile)})
        if k.startswith(ROOM_SENSOR_PREFIX):
            rooms.append({"key": k, "label": _get_friendly_name(k, profile)})

    bad_zone_labels = [z for z in zones if z["label"].isdigit()]
    bad_room_labels = [r for r in rooms if r["label"].isdigit()]

    return {
        "zone_count": len(zones),
        "room_count": len(rooms),
        "zones": zones,
        "rooms": rooms,
        "bad_zone_labels": bad_zone_labels,
        "bad_room_labels": bad_room_labels,
    }


def _ui_sanity(mode: str, df: pd.DataFrame, runs: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Note: For HA, Heat_HA / COP_Graph_HA should exist.
    For Grafana, these may be None (or you can extend later to use Grafana-specific columns).
    """
    return {
        "mode": mode,
        "rows": len(df),
        "heat_kwh": float(df["Heat_HA"].sum() / 1000 / 60) if "Heat_HA" in df else None,
        "cop_mean": float(df["COP_Graph_HA"].mean()) if "COP_Graph_HA" in df else None,
        "runs": _summarise_runs(runs),
        "mapping": _summarise_mapping(profile),
    }


# -------------------------------
# ✅ ✅ ✅ CORRECT HA PIPELINE ✅ ✅ ✅
# -------------------------------

def run_ha(paths: SamplePaths) -> Tuple[Path, Path]:
    profile = _load_json(paths.ha_profile)

    with paths.ha_csv.open("rb") as f:
        res = ha_loader.process_ha_files([f], profile, progress_cb=None)

    if not res or "df" not in res or res["df"].empty:
        raise RuntimeError("HA regression failed – no data returned")

    df = res["df"]
    runs = res["runs"]
    daily = res.get("daily")
    baselines = res.get("baselines")
    patterns = res.get("patterns")
    raw_history = res.get("raw_history")

    debug_json = {
        "mode": "ha",
        "shape": df.shape,
        "columns": list(df.columns),
        "runs": _summarise_runs(runs),
        "has_daily": daily is not None,
        "has_baselines": baselines is not None,
        "has_patterns": patterns is not None,
        "has_raw_history": raw_history is not None,
    }

    _ensure_artifacts(paths.artifacts)
    ts = _utc_timestamp()

    zip_path = paths.artifacts / f"ha_debug_bundle_{ts}.zip"
    json_path = paths.artifacts / f"ha_ui_sanity_{ts}.json"

    _write_debug_bundle(zip_path, df, debug_json, "ha")

    ui_payload = _ui_sanity("ha", df, runs, profile)
    ui_payload["generated_at_utc"] = ts

    json_path.write_text(
        json.dumps(ui_payload, indent=2),
        encoding="utf-8",
    )

    return zip_path, json_path


# -------------------------------
# ✅ ✅ ✅ CORRECT GRAFANA PIPELINE ✅ ✅ ✅
# -------------------------------

def run_grafana(paths: SamplePaths) -> Tuple[Path, Path]:
    profile = _load_json(paths.grafana_profile)

    with paths.grafana_numeric.open("rb") as f1, paths.grafana_state.open("rb") as f2:
        res = data_loader.load_and_clean_data([f1, f2], profile, progress_cb=None)

    if not res or "df" not in res or res["df"].empty:
        raise RuntimeError("Grafana regression failed – loader returned no data")

    df_raw = res["df"]
    df = processing.apply_gatekeepers(df_raw, profile)
    runs = processing.detect_runs(df, profile)

    debug_json = {
        "mode": "grafana",
        "shape": df.shape,
        "columns": list(df.columns),
        "runs": _summarise_runs(runs),
    }

    _ensure_artifacts(paths.artifacts)
    ts = _utc_timestamp()

    zip_path = paths.artifacts / f"grafana_debug_bundle_{ts}.zip"
    json_path = paths.artifacts / f"grafana_ui_sanity_{ts}.json"

    _write_debug_bundle(zip_path, df, debug_json, "grafana")

    ui_payload = _ui_sanity("grafana", df, runs, profile)
    ui_payload["generated_at_utc"] = ts

    json_path.write_text(
        json.dumps(ui_payload, indent=2),
        encoding="utf-8",
    )

    return zip_path, json_path


# -------------------------------
# ✅ PUBLIC ENTRYPOINT
# -------------------------------

def run_all():
    root = Path(__file__).resolve().parent
    paths = SamplePaths(base=root)

    ha_out = run_ha(paths)
    graf_out = run_grafana(paths)

    print("\n=== ✅ REGRESSION COMPLETE ===")
    print("HA:      ", ha_out)
    print("Grafana: ", graf_out)


if __name__ == "__main__":
    run_all()
