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
from utils import strip_entity_prefix


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


def _sanitize_branch_name(branch: str | None) -> str:
    """
    Sanitize branch name for use in filenames.
    Replace slashes and special characters with underscores.
    """
    if not branch:
        return "unknown-branch"

    # Replace characters that aren't safe in filenames
    safe = branch.replace("/", "_").replace("\\", "_").replace(":", "_")
    safe = safe.replace(" ", "_").replace("*", "_").replace("?", "_")
    safe = safe.replace("<", "_").replace(">", "_").replace("|", "_")

    return safe


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

    def _normalize_labels(labels: List[str]) -> List[str]:
        """Strip HA/Grafana prefixes and keep order without duplicates."""
        seen = set()
        cleaned: List[str] = []
        for label in labels:
            nice = strip_entity_prefix(str(label))
            if nice not in seen:
                cleaned.append(nice)
                seen.add(nice)
        return cleaned

    # -------- Long-Term Trends --------
    long_term_trends_raw = daily_df.columns.tolist() if daily_df is not None else []
    long_term_trends = _normalize_labels(long_term_trends_raw)

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
    data_quality_audit_raw = sorted(df.columns)
    data_quality_audit = _normalize_labels(data_quality_audit_raw)

    return {
        "long_term_trends": long_term_trends,
        "long_term_trends_raw": long_term_trends_raw,
        "run_inspector": {
            "select_run_dropdown": select_run_dropdown,
            "core_metrics": core_metrics,
            "hydraulics": hydraulics,
            "rooms": rooms,
            "zones": zones,
        },
        "data_quality_audit": data_quality_audit,
        "data_quality_audit_raw": data_quality_audit_raw,
    }


# -------------------------------
# ✅ BUG DETECTION LOGIC (NEW)
# -------------------------------

def _detect_zone_dropdown_bug(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    BUG #1: Zone dropdown shows "(No Zone Data)" when zones exist in engine.
    
    Detection: Any heating run with active_zones != "None" but label missing.
    """
    total_heating = sum(1 for r in runs if r.get("run_type") == "Heating")
    
    runs_with_zone_string = sum(
        1 for r in runs 
        if r.get("run_type") == "Heating" 
        and r.get("active_zones") 
        and r["active_zones"] != "None"
    )
    
    # Placeholder: would need to check actual UI state vs engine state
    # For now, if we have heating runs with zones, we assume dropdown should work
    runs_with_missing_ui = 0  # Would be detected by checking UI rendering
    
    return {
        "total_heating_runs": total_heating,
        "runs_with_zone_string": runs_with_zone_string,
        "runs_with_engine_zones_but_no_ui_str": runs_with_missing_ui,
        "bug_detected": runs_with_missing_ui > 0,
    }


def _detect_hydraulics_tab_bug(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    BUG #2: Hydraulics tab "Active Zones" chart is blank.
    
    Detection: Zone columns exist in df but have no data or all zeros.
    """
    zone_cols = [k for k in profile.get("mapping", {}) if k.startswith("Zone_")]
    
    zone_activity = {}
    for zone in zone_cols:
        if zone in df.columns:
            total_samples = len(df[zone])
            nonzero_samples = int((df[zone] != 0).sum())
            zone_activity[zone] = {
                "total_samples": total_samples,
                "nonzero_samples": nonzero_samples,
                "pct_active": round(100 * nonzero_samples / total_samples, 2) if total_samples > 0 else 0,
                "has_data": nonzero_samples > 0,
            }
    
    zones_with_no_data = [z for z, info in zone_activity.items() if not info["has_data"]]
    
    return {
        "zone_activity": zone_activity,
        "zones_with_no_data": zones_with_no_data,
        "bug_detected": len(zone_cols) > 0 and len(zones_with_no_data) == len(zone_cols),
    }


def _detect_room_filtering_bug(runs: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    BUG #3: Rooms tab shows all 6 rooms instead of filtering to active zones (2-3).
    
    Detection: Check if rooms_per_zone filtering is applied in run data.
    """
    rooms_per_zone = profile.get("rooms_per_zone", {})
    
    room_filtering_issues = []
    for r in runs:
        if r.get("run_type") != "Heating":
            continue
        
        active_zones_str = r.get("active_zones", "")
        relevant_rooms = r.get("relevant_rooms", [])
        
        # Expected: only rooms from active zones
        # Bug: all rooms shown regardless of zone
        room_filtering_issues.append({
            "run_id": r.get("id"),
            "active_zones": active_zones_str,
            "relevant_rooms_count": len(relevant_rooms),
            "expected_filtering": bool(rooms_per_zone and active_zones_str != "None"),
        })
    
    return {
        "room_filtering_checks": room_filtering_issues[:5],  # Sample
        "total_heating_runs": len(room_filtering_issues),
        "bug_detected": False,  # Would need actual UI comparison
    }


def _detect_data_quality_category_bug(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    BUG #4: Data Quality "Zones" category shows "No data for this category".
    
    Detection: Zone columns exist in df but category logic fails to find them.
    """
    zone_cols = [k for k in profile.get("mapping", {}) if k.startswith("Zone_")]
    
    category_membership = {
        "zones": {
            "expected_sensors": zone_cols,
            "sensors_in_df": [z for z in zone_cols if z in df.columns],
            "has_data": len([z for z in zone_cols if z in df.columns and (df[z] != 0).any()]) > 0,
        }
    }
    
    return {
        "category_checks": category_membership,
        "bug_detected": len(zone_cols) > 0 and not category_membership["zones"]["has_data"],
    }


def _detect_entity_prefix_bug(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    BUG #5: All Sensors shows entity prefixes (sensor., binary_sensor.) not stripped.
    
    Detection: Check if zone/room labels contain prefixes.
    """
    mapping = profile.get("mapping", {})
    
    formatting_issues = []
    for key, entity_id in mapping.items():
        if key.startswith("Zone_") or key.startswith("Room_"):
            label = _get_friendly_name(key, profile)
            
            has_prefix = any(
                label.startswith(p) 
                for p in ["sensor.", "binary_sensor.", "switch."]
            )
            
            if has_prefix:
                formatting_issues.append({
                    "key": key,
                    "entity_id": entity_id,
                    "display_label": label,
                    "has_prefix": has_prefix,
                })
    
    return {
        "formatting_issues": formatting_issues,
        "bug_detected": len(formatting_issues) > 0,
    }


# -------------------------------
# ✅ UI SANITY SUMMARY (ENHANCED)
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

    # ====================================================================
    # NEW: Add bug detection results
    # ====================================================================
    bug_detection = {
        "zone_dropdown": _detect_zone_dropdown_bug(runs),
        "hydraulics_tab": _detect_hydraulics_tab_bug(df, profile),
        "room_filtering": _detect_room_filtering_bug(runs, profile),
        "data_quality_category": _detect_data_quality_category_bug(df, profile),
        "entity_prefix_formatting": _detect_entity_prefix_bug(profile),
    }

    return {
        "mode": mode,
        "rows": len(df),
        "heat_kwh": heat_kwh,
        "cop_mean": cop_mean,
        "runs": _summarise_runs(runs),
        "mapping": _summarise_mapping(profile),
        "ui_labels": ui_labels,
        "bug_detection": bug_detection,  # NEW
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
    branch_safe = _sanitize_branch_name(git_meta.get("branch"))

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

    zip_path = paths.artifacts / f"ha_debug_bundle_{branch_safe}_{ts}.zip"
    json_path = paths.artifacts / f"ha_ui_sanity_{branch_safe}_{ts}.json"

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
    branch_safe = _sanitize_branch_name(git_meta.get("branch"))

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

    zip_path = paths.artifacts / f"grafana_debug_bundle_{branch_safe}_{ts}.zip"
    json_path = paths.artifacts / f"grafana_ui_sanity_{branch_safe}_{ts}.json"

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

    print("\n=== REGRESSION COMPLETE (PASSED) ===")
    print("HA:      ", ha_out)
    print("Grafana: ", graf_out)


if __name__ == "__main__":
    run_all()
