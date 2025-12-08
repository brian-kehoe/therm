# testing/regression_runner.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import zipfile
import pandas as pd
from datetime import datetime, timezone
import subprocess

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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _git_metadata() -> Dict[str, Any]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0

        return {
            "branch": branch,
            "commit": commit,
            "is_dirty": dirty,
        }

    except Exception:
        return {
            "branch": None,
            "commit": None,
            "is_dirty": None,
        }


def _source_module_urls(commit: str | None) -> Dict[str, str]:
    if not commit:
        return {}

    base = f"https://raw.githubusercontent.com/brian-kehoe/THERM/{commit}/"

    modules = [
        "app.py",
        "ha_loader.py",
        "ha_engine.py",
        "data_loader.py",
        "data_normalizer.py",
        "data_resolution.py",
        "processing.py",
        "baselines.py",
        "schema_defs.py",
        "config.py",
        "config_manager.py",
        "view_runs.py",
        "view_quality.py",
        "view_trends.py",
        "mapping_ui.py",
        "utils.py",
    ]

    return {m: base + m for m in modules}


# -------------------------------
# ✅ UI LABEL EXTRACTION (ALL MODES)
# -------------------------------

def _extract_ui_labels_all_modes(
    df: pd.DataFrame,
    daily_df: pd.DataFrame | None,
    runs: List[Dict[str, Any]],
    profile: Dict[str, Any],
) -> Dict[str, Any]:

    # -------- Long-Term Trends --------
    long_term_trends = daily_df.columns.tolist() if daily_df is not None else []

    # -------- Run Inspector: Select Run Dropdown --------
    select_run_dropdown = []
    for r in runs:
        label = f"{r.get('run_type')} | {r.get('start')} → {r.get('end')}"
        select_run_dropdown.append(label)

    # -------- Run Inspector: Core Metrics (legend & tooltips) --------
    core_metrics = [
        c for c in df.columns
        if any(k in c for k in ["Power", "Heat", "COP", "Cost", "Rate"])
    ]

    # -------- Run Inspector: Hydraulics --------
    HYDRAULICS_KEYS = {
        "FlowTemp", "ReturnTemp", "DeltaT",
        "FlowRate", "Power", "Power_Clean",
        "Freq", "ValveMode", "DHW_Mode", "Defrost"
    }
    hydraulics = [c for c in df.columns if c in HYDRAULICS_KEYS]

    # -------- Run Inspector: Rooms & Zones --------
    rooms = []
    zones = []
    for k in profile.get("mapping", {}):
        if k.startswith("Room_"):
            rooms.append(_get_friendly_name(k, profile))
        if k.startswith("Zone_"):
            zones.append(_get_friendly_name(k, profile))

    # -------- Data Quality Audit --------
    data_quality_audit = sorted(df.columns)

    return {
        "long_term_trends": long_term_trends,
        "run_inspector": {
            "select_run_dropdown": select_run_dropdown,
            "core_metrics": core_metrics,
            "hydraulics": hydraulics,
            "rooms": rooms,
            "zones": zones,
        },
        "data_quality_audit": data_quality_audit,
    }


# -------------------------------
# ✅ UI SANITY SUMMARY
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


def _ui_sanity(
    mode: str,
    df: pd.DataFrame,
    runs: List[Dict[str, Any]],
    profile: Dict[str, Any],
    daily_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:

    heat_kwh: float | None = None
    cop_mean: float | None = None

    # Prefer daily stats (matches front-end Trends)
    if daily_df is not None and not daily_df.empty:
        if "Total_Heat_KWh" in daily_df.columns:
            heat_kwh = float(daily_df["Total_Heat_KWh"].sum())
        elif "Total_Heat_kWh" in daily_df.columns:
            heat_kwh = float(daily_df["Total_Heat_kWh"].sum())
        else:
            heat_kwh = float(
                daily_df.get("Heat_Heating_kWh", pd.Series(dtype=float)).sum()
                + daily_df.get("Heat_DHW_kWh", pd.Series(dtype=float)).sum()
            )

        if "Global_SCOP" in daily_df.columns:
            cop_series = pd.to_numeric(daily_df["Global_SCOP"], errors="coerce").dropna()
            if not cop_series.empty:
                cop_mean = float(cop_series.mean())

    # Fallback engine physics
    if heat_kwh is None and "Heat_Clean" in df.columns:
        heat_kwh = float(df["Heat_Clean"].fillna(0).sum() / 1000.0 / 60.0)
    elif heat_kwh is None and "Heat" in df.columns:
        heat_kwh = float(df["Heat"].fillna(0).sum() / 1000.0 / 60.0)

    if cop_mean is None and "COP_Real" in df.columns:
        cop_series = pd.to_numeric(df["COP_Real"], errors="coerce").dropna()
        if not cop_series.empty:
            cop_mean = float(cop_series.mean())
    elif cop_mean is None and "COP_Graph" in df.columns:
        cop_series = pd.to_numeric(df["COP_Graph"], errors="coerce").dropna()
        if not cop_series.empty:
            cop_mean = float(cop_series.mean())

    ui_labels = _extract_ui_labels_all_modes(
        df=df,
        daily_df=daily_df,
        runs=runs,
        profile=profile,
    )

    return {
        "mode": mode,
        "rows": len(df),
        "heat_kwh": heat_kwh,
        "cop_mean": cop_mean,
        "runs": _summarise_runs(runs),
        "mapping": _summarise_mapping(profile),
        "ui_labels": ui_labels,
    }


# -------------------------------
# ✅ HA PIPELINE
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

    ts = _utc_timestamp()
    git_meta = _git_metadata()
    source_urls = _source_module_urls(git_meta.get("commit"))

    debug_json = {
        "mode": "ha",
        "generated_at_utc": ts,
        "git": git_meta,
        "source_modules": source_urls,
        "shape": df.shape,
        "columns": list(df.columns),
        "runs": _summarise_runs(runs),
        "has_daily": daily is not None,
    }

    _ensure_artifacts(paths.artifacts)

    zip_path = paths.artifacts / f"ha_debug_bundle_{ts}.zip"
    json_path = paths.artifacts / f"ha_ui_sanity_{ts}.json"

    _write_debug_bundle(zip_path, df, debug_json, "ha")

    ui_payload = _ui_sanity("ha", df, runs, profile, daily_df=daily)
    ui_payload["generated_at_utc"] = ts
    ui_payload["git"] = git_meta
    ui_payload["source_modules"] = source_urls

    json_path.write_text(json.dumps(ui_payload, indent=2), encoding="utf-8")

    return zip_path, json_path


# -------------------------------
# ✅ GRAFANA PIPELINE
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
    daily = processing.get_daily_stats(df)

    ts = _utc_timestamp()
    git_meta = _git_metadata()
    source_urls = _source_module_urls(git_meta.get("commit"))

    debug_json = {
        "mode": "grafana",
        "generated_at_utc": ts,
        "git": git_meta,
        "source_modules": source_urls,
        "shape": df.shape,
        "columns": list(df.columns),
        "runs": _summarise_runs(runs),
        "has_daily": daily is not None,
    }

    _ensure_artifacts(paths.artifacts)

    zip_path = paths.artifacts / f"grafana_debug_bundle_{ts}.zip"
    json_path = paths.artifacts / f"grafana_ui_sanity_{ts}.json"

    _write_debug_bundle(zip_path, df, debug_json, "grafana")

    ui_payload = _ui_sanity("grafana", df, runs, profile, daily_df=daily)
    ui_payload["generated_at_utc"] = ts
    ui_payload["git"] = git_meta
    ui_payload["source_modules"] = source_urls

    json_path.write_text(json.dumps(ui_payload, indent=2), encoding="utf-8")

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
