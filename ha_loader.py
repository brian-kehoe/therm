# ha_loader.py
# Home Assistant → Engine-ready dataframe loader for THERM
# Updated for public-beta-v2.8
#
# Changes in this version:
#   ✔ Improved dtype inference for raw Modbus 0/1 registers
#   ✔ After pivot/resample, interprets raw Modbus state/mode sensors:
#         - DHW Mode (0/1/2/3)
#         - DHW Status (0/1/360)
#         - Defrost Status (0/2/3/6)
#         - Immersion Mode (0/1)
#         - 3-Way Valve Position (0/1)
#   ✔ Adds clean boolean/label columns:
#         DHW_Mode_Label, DHW_Mode_Is_Active
#         DHW_Status_Is_On
#         Defrost_Is_Active
#         Immersion_Is_On
#         Valve_Is_DHW, Valve_Is_Heating
#
# All raw values are preserved. New interpreted columns appear
# only in the wide dataframe after pivot/resample (Option B2).


import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional

from inspector import is_binary_sensor, safe_smart_parse
import processing


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _infer_dtype(series: pd.Series) -> str:
    """
    Infer dtype for a single sensor based on HA long-form 'state' series.
    Returns: "binary", "numeric", or "string".

    Improvements for Modbus:
        - Numeric series that contain ONLY {0, 1} are treated as binary.
    """
    values = series.dropna().astype(str)

    # 1. Classic HA binary types ("on"/"off", etc.)
    if is_binary_sensor(values):
        return "binary"

    # 2. Mostly numeric?
    parsed, mostly_numeric = safe_smart_parse(values)
    if mostly_numeric:
        try:
            num = pd.to_numeric(parsed, errors="coerce").dropna()
            uniq = set(num.unique().tolist())
            # treat pure 0/1 numeric registers as binary
            if uniq and uniq.issubset({0, 1}):
                return "binary"
        except Exception:
            return "numeric"
        return "numeric"

    # 3. Fallback: treat as string
    return "string"


def _convert_value(val, dtype: str):
    """
    Convert HA 'state' values to appropriate Python values.
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


# ============================================================
# MODBUS INTERPRETATION (Option B2 - applied AFTER pivot)
# ============================================================

def enrich_modbus_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert interpreted columns for raw Modbus state/mode sensors.
    This is applied AFTER pivot/resample on the wide DF.

    Raw Modbus entity ids expected:
        sensor.heat_pump_hot_water_mode
        sensor.heat_pump_hot_water_status
        sensor.heat_pump_defrost_status
        sensor.heat_pump_immersion_heater_mode
        sensor.heat_pump_3way_valve_position

    Raw columns remain untouched.
    New interpreted columns:
        DHW_Mode_Label, DHW_Mode_Is_Active
        DHW_Status_Is_On
        Defrost_Is_Active
        Immersion_Is_On
        Valve_Is_DHW, Valve_Is_Heating
    """

    df = df.copy()

    # ------------------------------------------------------------
    # 1. DHW Mode (register 73)
    # ------------------------------------------------------------
    col = "sensor.heat_pump_hot_water_mode"
    if col in df.columns:
        m = df[col]

        mode_map = {
            0: "Eco",
            1: "Standard",
            2: "Power",
            3: "Force",  # appears on some Samsung EHS units
        }

        df["DHW_Mode_Label"] = m.map(mode_map).fillna("Unknown")
        df["DHW_Mode_Is_Active"] = m.apply(lambda v: 0 if pd.isna(v) else int(v != 0))

    # ------------------------------------------------------------
    # 2. DHW Status (register 72)
    # ------------------------------------------------------------
    col = "sensor.heat_pump_hot_water_status"
    if col in df.columns:
        s = df[col]
        # treat > 0 as "on", allowing values like 1 and 360
        df["DHW_Status_Is_On"] = s.apply(lambda v: 0 if pd.isna(v) else int(v > 0))

    # ------------------------------------------------------------
    # 3. Defrost Status (register 2)
    # ------------------------------------------------------------
    col = "sensor.heat_pump_defrost_status"
    if col in df.columns:
        s = df[col]
        # minimal boolean interpretation: nonzero = defrosting
        df["Defrost_Is_Active"] = s.apply(lambda v: 0 if pd.isna(v) else int(v != 0))

    # ------------------------------------------------------------
    # 4. Immersion Heater Mode (register 85)
    # ------------------------------------------------------------
    col = "sensor.heat_pump_immersion_heater_mode"
    if col in df.columns:
        s = df[col]
        df["Immersion_Is_On"] = s.apply(lambda v: 0 if pd.isna(v) else int(v == 1))

    # ------------------------------------------------------------
    # 5. 3-way Valve Position (register 89)
    # ------------------------------------------------------------
    col = "sensor.heat_pump_3way_valve_position"
    if col in df.columns:
        s = df[col]

        def _valve_label(v):
            if pd.isna(v):
                return None
            if v == 0:
                return "Heating"
            if v == 1:
                return "DHW"
            return f"Unknown({int(v)})"

        df["Valve_Is_DHW"] = s.apply(lambda v: 0 if pd.isna(v) else int(v == 1))
        df["Valve_Is_Heating"] = s.apply(lambda v: 0 if pd.isna(v) else int(v == 0))
        df["Valve_Position_Label"] = s.apply(_valve_label)

    return df


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def process_ha_files(
    files: List[Any],
    user_config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load Home Assistant long-form CSV(s), convert to wide 1-minute data,
    apply Modbus interpretation (Option B2), apply mappings, run engine.
    """

    # ----------------------------------------------
    # 1. Read CSVs into long-form structure
    # ----------------------------------------------
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

    # ----------------------------------------------
    # 2. Validate columns
    # ----------------------------------------------
    time_cols = [c for c in ("last_changed", "last_updated", "time") if c in long.columns]
    if "entity_id" not in long.columns or "state" not in long.columns or not time_cols:
        return None

    ts_col = time_cols[0]
    long[ts_col] = pd.to_datetime(long[ts_col], errors="coerce")
    long = long.dropna(subset=[ts_col])

    # ----------------------------------------------
    # 3. Infer dtype per entity
    # ----------------------------------------------
    dtype_map: Dict[str, str] = {}
    for eid, grp in long.groupby("entity_id"):
        dtype_map[eid] = _infer_dtype(grp["state"])

    # ----------------------------------------------
    # 4. Convert values
    # ----------------------------------------------
    long["value"] = [
        _convert_value(v, dtype_map.get(eid, "string"))
        for v, eid in zip(long["state"], long["entity_id"])
    ]

    # ----------------------------------------------
    # 5. Aggregate duplicate rows
    # ----------------------------------------------
    def _agg_group(series):
        if series.dtype == object:
            return series.ffill().iloc[-1]
        else:
            return pd.to_numeric(series, errors="coerce").mean()

    grouped = (
        long.groupby([ts_col, "entity_id"])["value"]
        .apply(_agg_group)
        .reset_index()
    )

    # ----------------------------------------------
    # 6. Pivot to wide form
    # ----------------------------------------------
    wide = grouped.pivot(index=ts_col, columns="entity_id", values="value")
    wide.index = pd.to_datetime(wide.index)

    # ----------------------------------------------
    # 7. Resample to 1-minute
    # ----------------------------------------------
    df_wide = wide.resample("1T").last().ffill(limit=120)

    # ----------------------------------------------
    # 8. APPLY MODBUS INTERPRETATION (Option B2)
    # ----------------------------------------------
    df_wide = enrich_modbus_interpretation(df_wide)

    # ----------------------------------------------
    # 9. APPLY USER MAPPING
    # ----------------------------------------------
    mapping = user_config.get("mapping", {})
    rename_dict = {}

    for therm_key, entity_id in mapping.items():
        if entity_id in df_wide.columns:
            rename_dict[entity_id] = therm_key

    df = df_wide.rename(columns=rename_dict)

    # ----------------------------------------------
    # 10. Calculate DeltaT if FlowTemp + ReturnTemp mapped
    # ----------------------------------------------
    if "FlowTemp" in df.columns and "ReturnTemp" in df.columns:
        df["DeltaT"] = df["FlowTemp"] - df["ReturnTemp"]

    numeric_cols = ["Power", "FlowTemp", "ReturnTemp", "FlowRate", "Freq", "DeltaT"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------------------------------------
    # 11. Engine processing
    # ----------------------------------------------
    df_engine = processing.apply_gatekeepers(df, user_config)
    if df_engine is None or df_engine.empty:
        return None

    runs = processing.detect_runs(df_engine, user_config)
    daily = processing.get_daily_stats(df_engine)

    # ----------------------------------------------
    # 12. Return output structure
    # ----------------------------------------------
    return {
        "df": df_engine,
        "raw_history": df,   # wide dataframe after interpretation, before physics
        "runs": runs,
        "daily": daily,
        "patterns": None,
        "baselines": None,
    }
