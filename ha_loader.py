# ha_loader.py
# Home Assistant → Engine-ready dataframe loader for THERM
# Fully compatible with v2.75 app structure

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from inspector import is_binary_sensor, safe_smart_parse
import processing


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _infer_dtype(series: pd.Series) -> str:
    """
    Infer dtype for a single sensor based on HA long-form 'state' series.
    Returns: "binary", "numeric", or "string".
    """
    values = series.dropna().astype(str)

    # 1. Binary?
    if is_binary_sensor(values):
        return "binary"

    # 2. Mostly numeric?
    _, mostly_numeric = safe_smart_parse(values)
    if mostly_numeric:
        return "numeric"

    # 3. Fallback
    return "string"


def _convert_value(val, dtype: str):
    """
    Convert HA 'state' values to proper Python values.
    """
    if dtype == "binary":
        s = str(val).strip().lower()
        return 1 if s in ("on", "true", "1", "yes") else 0

    elif dtype == "numeric":
        try:
            return float(val)
        except Exception:
            return np.nan

    else:
        return val


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def process_ha_files(
    files: List[Any],
    user_config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load Home Assistant long-form CSV(s), convert to wide 1-minute data,
    apply user mapping, run physics, detect runs, and compute daily stats.

    Returns dict with:
        df         → engine dataframe
        raw_history→ dataframe pre-physics
        runs       → list of runs
        daily      → daily energy table
        patterns   → None (HA mode does not use patterns yet)
        baselines  → None (HA mode does not use heartbeats yet)
    """

    # ------------------------------------------------------------------
    # 1. LOAD CSVs
    # ------------------------------------------------------------------
    dfs = []
    total = len(files)

    for i, f in enumerate(files):
        if progress_cb:
            progress_cb(f"Reading HA CSV {i+1}/{total}", (i + 1) / total)
        f.seek(0)
        df = pd.read_csv(f)
        dfs.append(df)

    if not dfs:
        return None

    long = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 2. VERIFY REQUIRED COLUMNS
    # ------------------------------------------------------------------
    # HA files usually have: entity_id, state, last_changed
    time_cols = [c for c in ("last_changed", "last_updated", "time") if c in long.columns]
    if "entity_id" not in long.columns or "state" not in long.columns or not time_cols:
        return None

    ts_col = time_cols[0]
    long[ts_col] = pd.to_datetime(long[ts_col], errors="coerce")
    long = long.dropna(subset=[ts_col])

    # ------------------------------------------------------------------
    # 3. INFER DTYPES PER ENTITY
    # ------------------------------------------------------------------
    dtype_map: Dict[str, str] = {}
    for eid, grp in long.groupby("entity_id"):
        dtype_map[eid] = _infer_dtype(grp["state"])

    # ------------------------------------------------------------------
    # 4. CONVERT VALUES
    # ------------------------------------------------------------------
    long["value"] = [
        _convert_value(v, dtype_map.get(eid, "string"))
        for v, eid in zip(long["state"], long["entity_id"])
    ]

    # ------------------------------------------------------------------
    # 5. GROUP DUPLICATES → ONE VALUE PER (timestamp, entity_id)
    # ------------------------------------------------------------------
    def _agg_group(series):
        """Mean for numeric/binary, last valid for strings."""
        if series.dtype == object:
            return series.ffill().iloc[-1]
        else:
            return pd.to_numeric(series, errors="coerce").mean()

    grouped = (
        long.groupby([ts_col, "entity_id"])["value"]
        .apply(_agg_group)
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 6. PIVOT TO WIDE
    # ------------------------------------------------------------------
    wide = grouped.pivot(index=ts_col, columns="entity_id", values="value")
    wide.index = pd.to_datetime(wide.index)

    # ------------------------------------------------------------------
    # 7. RESAMPLE TO 1-MINUTE
    # ------------------------------------------------------------------
    df_wide = wide.resample("1T").ffill()

    # ------------------------------------------------------------------
    # 8. APPLY USER SENSOR MAPPING
    # ------------------------------------------------------------------
    # user_config["mapping"] maps THERM keys → entity_id strings
    mapping = user_config.get("mapping", {})

    # Reverse: entity_id → THERM key
    rename_dict = {}
    for therm_key, entity_id in mapping.items():
        if entity_id in df_wide.columns:
            rename_dict[entity_id] = therm_key

    df = df_wide.rename(columns=rename_dict)

    # ------------------------------------------------------------------
    # 9. COMPUTE DeltaT IF MAPPED
    # ------------------------------------------------------------------
    if "FlowTemp" in df.columns and "ReturnTemp" in df.columns:
        df["DeltaT"] = df["FlowTemp"] - df["ReturnTemp"]

    # Ensure numeric types for engine
    numeric_cols = ["Power", "FlowTemp", "ReturnTemp", "FlowRate", "Freq", "DeltaT"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------------------------------------------------------------------
    # 10. RUN PHYSICS ENGINE
    # ------------------------------------------------------------------
    df_engine = processing.apply_gatekeepers(df, user_config)
    if df_engine is None or df_engine.empty:
        return None

    # ------------------------------------------------------------------
    # 11. RUN DETECTOR + DAILY STATS
    # ------------------------------------------------------------------
    runs = processing.detect_runs(df_engine, user_config)
    daily = processing.get_daily_stats(df_engine)

    # ------------------------------------------------------------------
    # 12. OUTPUT STRUCTURE (same as Grafana loader)
    # ------------------------------------------------------------------
    return {
        "df": df_engine,
        "raw_history": df,   # pre-physics wide dataframe
        "runs": runs,
        "daily": daily,
        "patterns": None,
        "baselines": None,
    }
