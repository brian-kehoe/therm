# processing.py

import pandas as pd
import numpy as np
import streamlit as st

from config import (
    THRESHOLDS,
    SPECIFIC_HEAT_CAPACITY,
    PHYSICS_THRESHOLDS,
    NIGHT_HOURS,
    TARIFF_STRUCTURE,
)
from utils import safe_div


# --- DYNAMIC HELPERS ---------------------------------------------------------


def get_active_zone_columns(df: pd.DataFrame) -> list:
    """Return columns that look like heating zone state flags (Zone_XXX)."""
    return [c for c in df.columns if c.startswith("Zone_") and len(c) <= 7]


def get_room_columns(df: pd.DataFrame) -> list:
    """Return columns that look like room temperature series (Room_XXX)."""
    return [c for c in df.columns if c.startswith("Room_")]


def get_friendly_name(internal_key, user_config) -> str:
    """
    Robust lookup for friendly names.

    Handles cases where config might be None, empty, or malformed.
    Falls back to the internal key if anything looks off.
    """
    if not isinstance(user_config, dict):
        return str(internal_key)

    mapping = user_config.get("mapping")
    if not isinstance(mapping, dict):
        return str(internal_key)

    val = mapping.get(internal_key, internal_key)
    return str(val)


def calculate_physics_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate physics-derived metrics that can be inferred cheaply from existing columns.
    Currently this just ensures DeltaT exists.
    """
    d = df.copy()

    # Delta T
    if "DeltaT" not in d.columns and {"FlowTemp", "ReturnTemp"}.issubset(d.columns):
        d["DeltaT"] = d["FlowTemp"] - d["ReturnTemp"]

    return d


# --- DEBUG HELPERS (optional) ------------------------------------------------
def _debug_engine_state(d: pd.DataFrame, label: str = "") -> None:
    """
    Lightweight debug helper to inspect the internal physics engine state.
    Controlled by st.session_state['debug_engine']; safe to call in production.
    """
    try:
        import streamlit as st  # already imported at top, but safe
    except Exception:
        return

    if not st.session_state.get("debug_engine", False):
        return

    cols = [c for c in ["Power", "Heat", "FlowRate", "DeltaT", "Freq"] if c in d.columns]
    summary: dict[str, float | int] = {"rows": int(len(d))}
    for col in cols:
        ser = pd.to_numeric(d[col], errors="coerce").fillna(0)
        summary[f"{col}_nonzero"] = int((ser != 0).sum())
        summary[f"{col}_min"] = float(ser.min())
        summary[f"{col}_max"] = float(ser.max())
    st.write(f"🔧 Engine debug – {label}")
    st.json(summary)


# --- HEAT + COP ENGINE -------------------------------------------------------
def _ensure_heat_and_cop(

    d: pd.DataFrame, thresholds: dict | None = None
) -> pd.DataFrame:
    """
    Canonical logic for:
      - Ensuring a 'Heat' channel exists (native or derived from hydraulics)
      - Deriving Power/Heat splits
      - Computing COP_Real and COP_Graph

    This replaces the previously duplicated HEAT OUTPUT blocks and ensures
    we only run this logic once per dataframe.
    """
    if thresholds is None:
        thresholds = {}
    elif not isinstance(thresholds, dict):
        thresholds = {}

    # --- HEAT OUTPUT LOGIC ---
    has_flow = "FlowRate" in d.columns
    has_heat_col = "Heat" in d.columns
    has_heat_data = False

    if has_heat_col:
        # treat NaN as 0 when checking whether we have any signal
        heat_series = pd.to_numeric(d["Heat"], errors="coerce").fillna(0)
        has_heat_data = heat_series.abs().sum() > 0
        d["Heat"] = heat_series  # normalise type

    if not has_heat_data:
        if has_flow:
            # Only derive when we have hydraulics
            min_flow = thresholds.get("min_flow_rate_lpm", 0)
            min_freq = thresholds.get("min_freq_for_heat", 0)

            flow = pd.to_numeric(d["FlowRate"], errors="coerce").fillna(0)
            delta_t = pd.to_numeric(d.get("DeltaT", 0), errors="coerce").fillna(0)
            freq = pd.to_numeric(d.get("Freq", 0), errors="coerce").fillna(0)

        # Basic physics: Heat [W] = c * m_dot * ΔT
        heat_raw = SPECIFIC_HEAT_CAPACITY * flow * delta_t

        # Gatekeepers
        min_dt = thresholds.get("min_valid_delta_t", 0)
        max_dt = thresholds.get("max_valid_delta_t", 999)

        valid = (
            (flow >= min_flow)
            & (freq >= min_freq)
            & (delta_t.abs() >= min_dt)
            & (delta_t.abs() <= max_dt)
        )

        d["Heat"] = 0.0
        d.loc[valid, "Heat"] = heat_raw[valid]

        # Optional: debug gatekeepers
        try:
            import streamlit as st
            if st.session_state.get("debug_engine", False):
                st.write(
                    "🔍 Heat gatekeepers",
                    {
                        "rows_total": int(len(d)),
                        "rows_flow_gt0": int((flow > 0).sum()),
                        "rows_valid": int(valid.sum()),
                        "min_flow_threshold": float(min_flow),
                        "min_freq_threshold": float(min_freq),
                        "min_dt_threshold": float(min_dt),
                        "max_dt_threshold": float(max_dt),
                    },
                )
        except Exception:
            pass


            # No heat when compressor off
            if "is_active" in d.columns:
                d.loc[~d["is_active"].astype(bool), "Heat"] = 0.0
        else:
            # No Heat sensor and no FlowRate → no energy channel
            d["Heat"] = 0.0

    # ------------------------------------------------------------------
    # FIX: Negative Power readings from Home Assistant
    # Some HA monitors report small negative consumption values (e.g. -2.3 W).
    # If not corrected → is_active=0 → no Heat → no COP → blank charts.
    # ------------------------------------------------------------------
    if "Power" in d.columns:
        d["Power"] = pd.to_numeric(d["Power"], errors="coerce").fillna(0)
        d["Power"] = d["Power"].abs()

    # --- Power & Heat Splits + COP ---
    is_heating = d.get("is_heating", 0).astype(bool)
    is_dhw = d.get("is_DHW", 0).astype(bool)

    d["Power_Heating"] = np.where(is_heating, d["Power"], 0)
    d["Power_DHW"] = np.where(is_dhw, d["Power"], 0)


    d["Heat_Heating"] = np.where(is_heating, d["Heat"], 0)
    d["Heat_DHW"] = np.where(is_dhw, d["Heat"], 0)

    d["COP_Real"] = safe_div(d["Heat"], d["Power"])
    d["COP_Graph"] = d["COP_Real"].clip(0, 6)

    return d


# --- TARIFF ENGINE -----------------------------------------------------------


def _parse_tariff_profiles(tariff_structure) -> list:
    """
    Normalise TARIFF_STRUCTURE into a sorted list of profiles:

    [
      {
        "valid_from": date,
        "rules": [
          {"name": str, "start": time, "end": time, "rate": float},
          ...
        ]
      },
      ...
    ]
    """
    if not isinstance(tariff_structure, (list, tuple)):
        return []

    profiles: list[dict] = []

    for p in tariff_structure:
        if not isinstance(p, dict):
            continue
        try:
            vf = pd.to_datetime(p.get("valid_from", "1970-01-01")).date()
        except Exception:
            vf = pd.Timestamp.min.date()

        rules = []
        for r in p.get("rules", []):
            try:
                start = pd.to_datetime(r.get("start", "00:00")).time()
                end = pd.to_datetime(r.get("end", "00:00")).time()
            except Exception:
                continue

            try:
                rate = float(r.get("rate", 0.35))
            except Exception:
                rate = 0.35

            rules.append(
                {
                    "name": r.get("name", ""),
                    "start": start,
                    "end": end,
                    "rate": rate,
                }
            )

        profiles.append({"valid_from": vf, "rules": rules})

    profiles = [p for p in profiles if p["rules"]]
    profiles.sort(key=lambda p: p["valid_from"])
    return profiles


def _compute_tariff_series(
    index: pd.DatetimeIndex,
    tariff_structure,
    default_rate: float = 0.35,
) -> pd.Series:
    """
    Vectorised computation of per-timestamp electricity rate based on:

      - TARIFF_STRUCTURE: list of profiles with valid_from + rules
      - Rules with start/end times (supporting overnight wrap)

    Returns a Series aligned to `index`.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return pd.Series(default_rate, index=index)

    profiles = _parse_tariff_profiles(tariff_structure)
    if not profiles:
        return pd.Series(default_rate, index=index)

    result = pd.Series(default_rate, index=index, dtype="float64")
    dates = index.date

    for i, profile in enumerate(profiles):
        vf = profile["valid_from"]
        if i < len(profiles) - 1:
            next_vf = profiles[i + 1]["valid_from"]
            mask_profile = (dates >= vf) & (dates < next_vf)
        else:
            mask_profile = dates >= vf

        if not mask_profile.any():
            continue

        idx_profile = index[mask_profile]
        secs = (
            idx_profile.hour * 3600
            + idx_profile.minute * 60
            + idx_profile.second
        )
        rates = np.full(len(idx_profile), default_rate, dtype="float64")

        for rule in profile["rules"]:
            start_s = (
                rule["start"].hour * 3600
                + rule["start"].minute * 60
                + rule["start"].second
            )
            end_s = (
                rule["end"].hour * 3600
                + rule["end"].minute * 60
                + rule["end"].second
            )

            if start_s < end_s:
                # Normal same-day window
                mask_rule = (secs >= start_s) & (secs < end_s)
            else:
                # Wrap-around window (e.g. 23:00 → 02:00)
                mask_rule = (secs >= start_s) | (secs < end_s)

            if mask_rule.any():
                rates[mask_rule] = rule["rate"]

        result.loc[idx_profile] = rates

    return result


def get_rate_for_timestamp(ts, tariff_structure, default_rate: float = 0.35) -> float:
    """
    Convenience helper: rate for a single timestamp.

    Uses the same tariff logic as the vectorised engine.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts)

    series = _compute_tariff_series(
        pd.DatetimeIndex([ts]), tariff_structure, default_rate=default_rate
    )
    if series.empty:
        return default_rate
    return float(series.iloc[0])


# --- MAIN GATEKEEPER / ENGINE ------------------------------------------------


def apply_gatekeepers(df: pd.DataFrame, user_config: dict | None = None) -> pd.DataFrame:
    """
    Core "engine" that:
        - Normalises physics fields (DeltaT)
        - Classifies activity + DHW/Heating
        - Ensures Heat + splits + COP
        - Derives Zone configuration strings
        - Infers Immersion activity/power
        - Applies tariff to build incremental cost
    """
    d = df.copy()
    thresholds = PHYSICS_THRESHOLDS if isinstance(PHYSICS_THRESHOLDS, dict) else {}

    # 1. Physics
    d = calculate_physics_metrics(d)
    _debug_engine_state(d, "after physics (DeltaT)")

    # 2. Activity
    power_min = thresholds.get("power_min", 50)
    d["is_active"] = (d["Power"] > power_min).astype(int)


    # 3. Detect DHW Mode
    is_dhw_mask = pd.Series(False, index=d.index)

    if "DHW_Active" in d.columns:
        is_dhw_mask |= d["DHW_Active"] > 0

    if "DHW_Mode" in d.columns:
        if pd.api.types.is_numeric_dtype(d["DHW_Mode"]):
            is_dhw_mask |= d["DHW_Mode"] > 0
        else:
            is_dhw_mask |= (
                d["DHW_Mode"]
                .astype(str)
                .str.lower()
                .str.contains("on|active|hot|dhw", na=False)
            )

    if "ValveMode" in d.columns:
        is_dhw_mask |= (
            d["ValveMode"]
            .astype(str)
            .str.lower()
            .str.contains("hot|dhw", na=False)
        )

    d["is_DHW"] = is_dhw_mask
    d["is_heating"] = (d["is_active"] == 1) & (~d["is_DHW"])

    _debug_engine_state(d, "after activity flags")

    # 4. HEAT + SPLITS + COP (single canonical implementation)
    d = _ensure_heat_and_cop(d, thresholds)
    _debug_engine_state(d, "after heat/COP")


    # 5. Zone Config Strings
    zone_cols = get_active_zone_columns(d)
    if zone_cols:
        # Data Type Fix: Ensure all zone columns are numeric
        for z in zone_cols:
            if not pd.api.types.is_numeric_dtype(d[z]):
                # Silent coercion of 'on'/'off'/boolish strings to 1/0
                d[z] = (
                    d[z]
                    .astype(str)
                    .str.lower()
                    .replace({"on": 1, "off": 0, "true": 1, "false": 0})
                )
                d[z] = pd.to_numeric(d[z], errors="coerce").fillna(0)


        d["Active_Zones_Count"] = d[zone_cols].sum(axis=1)
        z_map = {z: get_friendly_name(z, user_config) for z in zone_cols}

        def get_zone_str(row):
            active = [str(z_map.get(z, z)) for z in zone_cols if row[z] > 0]
            return " + ".join(active) if active else "None"

        d["Zone_Config"] = d.apply(get_zone_str, axis=1)
    else:
        d["Active_Zones_Count"] = 0
        d["Zone_Config"] = "None"

    # 6. Immersion
    if "Immersion_Mode" in d.columns:
        d["Immersion_Active"] = d["Immersion_Mode"] > 0
    elif "Indoor_Power" in d.columns:
        # Heuristic fallback based on indoor power
        d["Immersion_Active"] = d["Indoor_Power"] > 2500
    else:
        d["Immersion_Active"] = False

    if "Indoor_Power" in d.columns:
        d["Immersion_Power"] = np.where(
            d["Immersion_Active"], d["Indoor_Power"], 0
        )
    else:
        # Without Indoor_Power we cannot estimate immersion power reliably.
        # Keep the column but assume 0 W rather than a hard-coded 3 kW.
        d["Immersion_Power"] = 0.0

    # 7. Cost (Tariff-aware)
    d["hour"] = d.index.hour
    # Retain legacy "is_night_rate" for any downstream logic/visuals
    d["is_night_rate"] = d["hour"].isin(NIGHT_HOURS)

    try:
        d["Current_Rate"] = _compute_tariff_series(d.index, TARIFF_STRUCTURE)
    except Exception:
        # Conservative fallback to previous behaviour
        if isinstance(TARIFF_STRUCTURE, dict):
            rate_day = TARIFF_STRUCTURE.get("day_rate", 0.35)
            rate_night = TARIFF_STRUCTURE.get("night_rate", 0.15)
        else:
            rate_day = 0.35
            rate_night = 0.15
        d["Current_Rate"] = np.where(d["is_night_rate"], rate_night, rate_day)

    d["Cost_Inc"] = (d["Power"] / 1000.0 / 60.0) * d["Current_Rate"]

    return d


# --- GLOBAL STATS (CANONICAL) -----------------------------------------------
def compute_global_stats(df: pd.DataFrame) -> dict:
    """
    Canonical global stats for the whole dataset, based on the processed physics engine.

    - Uses `Heat` and `Power` from the fully-processed dataframe (after apply_gatekeepers),
      so all the existing logic (immersion removal, DHW vs heating, defrost protection)
      has already been applied.
    - If `is_active` is present, we only count periods where the heat pump is actually running.
      This makes the COP comparable to per-run COP.
    - Energy is computed as W-minutes -> kWh, consistent with run and daily stats.

    Returns:
        {
            "total_heat_kwh": float,
            "total_elec_kwh": float,
            "global_cop": float,
        }
    """
    if df is None or df.empty:
        return {
            "total_heat_kwh": 0.0,
            "total_elec_kwh": 0.0,
            "global_cop": 0.0,
        }

    d = df.copy()

    # Restrict to HP actually running if we have the engine flag
    if "is_active" in d.columns:
        d = d[d["is_active"] == 1]

    # Normalise series
    power = pd.to_numeric(d.get("Power", 0), errors="coerce").fillna(0.0)
    heat = pd.to_numeric(d.get("Heat", 0), errors="coerce").fillna(0.0)

    # W-minutes → kWh (60 minutes/hour * 1000 W/kW)
    total_elec_kwh = power.sum() / 60000.0
    total_heat_kwh = heat.sum() / 60000.0
    global_cop = safe_div(total_heat_kwh, total_elec_kwh)

    return {
        "total_heat_kwh": float(total_heat_kwh),
        "total_elec_kwh": float(total_elec_kwh),
        "global_cop": float(global_cop),
    }


# --- RUN DETECTION -----------------------------------------------------------


def detect_runs(df: pd.DataFrame, user_config: dict | None = None) -> list[dict]:
    """
    Group contiguous active periods into "runs" and derive per-run metrics
    (heat, electricity, COP, hydraulics, zone configuration, room deltas).
    """
    # Safety check on user_config type
    if isinstance(user_config, dict):
        rooms_per_zone = user_config.get("rooms_per_zone", {})
    else:
        rooms_per_zone = {}

    df = df.copy()

    df["run_change"] = (
        (df["is_active"].diff().ne(0))
        | (df["is_DHW"].ne(df["is_DHW"].shift()))
    )
    df["run_id"] = df["run_change"].cumsum()

    runs: list[dict] = []
    active_groups = df[df["is_active"] == 1].groupby("run_id")

    zone_cols = get_active_zone_columns(df)
    room_cols = get_room_columns(df)

    for run_id, group in active_groups:
        if len(group) < 5:
            continue

        run_type = "DHW" if group["is_DHW"].any() else "Heating"

        # Averages
        avg_outdoor = (
            group["OutdoorTemp"].mean()
            if "OutdoorTemp" in group.columns
            else 0
        )
        avg_flow = group["FlowTemp"].mean() if "FlowTemp" in group.columns else 0
        avg_flow_rate = (
            group["FlowRate"].mean()
            if "FlowRate" in group.columns
            else 0
        )

        # Metrics
        heat_kwh = group["Heat"].sum() / 60000.0
        elec_kwh = group["Power"].sum() / 60000.0
        cop = safe_div(heat_kwh, elec_kwh)

        # Friendly Zones
        active_zones_list = []
        if run_type == "Heating" and zone_cols:
            for z in zone_cols:
                if (group[z].sum() / len(group)) > 0.5:
                    active_zones_list.append(z)

        friendly_zones = [
            str(get_friendly_name(z, user_config)) for z in active_zones_list
        ]
        dominant_zones_str = (
            " + ".join(friendly_zones)
            if friendly_zones
            else ("None" if run_type == "Heating" else "DHW")
        )

        # Friendly Rooms
        relevant_rooms: list[str] = []
        if run_type == "Heating" and rooms_per_zone and active_zones_list:
            for z in active_zones_list:
                relevant_rooms.extend(rooms_per_zone.get(z, []))
            relevant_rooms = list(set(relevant_rooms))

        # Room Deltas
        room_deltas: dict[str, float] = {}
        for r in room_cols:
            series = group[r].dropna()
            if len(series) > 0:
                start_t = series.iloc[0]
                end_t = series.iloc[-1]
                friendly_r = get_friendly_name(r, user_config)
                room_deltas[friendly_r] = round(end_t - start_t, 2)

        runs.append(
            {
                "id": int(run_id),
                "start": group.index[0],
                "end": group.index[-1],
                "duration_mins": len(group),
                "run_type": run_type,
                "avg_outdoor": round(avg_outdoor, 1),
                "avg_flow_temp": round(avg_flow, 1),
                "avg_dt": round(group.get("DeltaT", pd.Series(0)).mean(), 1),
                "avg_flow_rate": round(avg_flow_rate, 1),
                "run_cop": round(cop, 2),
                "heat_kwh": heat_kwh,
                "electricity_kwh": elec_kwh,
                "active_zones": dominant_zones_str,
                "dominant_zones": dominant_zones_str,
                "room_deltas": room_deltas,
                "relevant_rooms": relevant_rooms,
                "immersion_kwh": group["Immersion_Power"].sum() / 60000.0,
                "immersion_mins": group["Immersion_Active"].sum(),
            }
        )

    return runs


# --- DAILY STATS -------------------------------------------------------------


def get_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level data into daily metrics.

    Now robust to being called on any dataframe with:
      - a DatetimeIndex, and
      - ideally a 'Power' column (for DQ_Score).

    If some of the internal engine columns are missing (e.g. Power_Heating),
    they are simply omitted from aggregation and the derived kWh fields fall
    back to 0 via .get(...).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("get_daily_stats expects a DatetimeIndex on the input.")

    # Base daily index + row counts (used for DQ_Score)
    if "Power" in df.columns:
        base_counts = df["Power"].resample("D").count().rename("row_count")
    else:
        base_counts = df.resample("D").size().rename("row_count")

    daily = base_counts.to_frame()

    # Core engine aggregates (only include if columns exist)
    base_agg = {
        "Power_Heating": ["sum"],
        "Power_DHW": ["sum"],
        "Heat_Heating": ["sum"],
        "Heat_DHW": ["sum"],
        "Cost_Inc": ["sum"],
        "is_active": ["sum"],
        "Immersion_Power": ["sum"],
        "is_DHW": ["sum"],
        "is_heating": ["sum"],
    }

    agg_dict: dict[str, list[str]] = {}
    for col, funcs in base_agg.items():
        if col in df.columns:
            agg_dict[col] = funcs

    # Additional columns: summary stats per column
    exclude = list(base_agg.keys()) + [
        "last_changed",
        "entity_id",
        "state",
        "Zone_Config",
    ]

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = ["mean", "min", "max", "count"]
        else:
            agg_dict[col] = ["count"]

    if agg_dict:
        daily_aggs = df.resample("D").agg(agg_dict)
        daily_aggs.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in daily_aggs.columns
        ]
        daily = daily.join(daily_aggs, how="left")

    # Normalise core column names (some may not exist; rename ignores missing)
    for core_col in [
        "Power_Heating",
        "Power_DHW",
        "Heat_Heating",
        "Heat_DHW",
    ]:
        if core_col in daily.columns and f"{core_col}_sum" not in daily.columns:
            daily = daily.rename(columns={core_col: f"{core_col}_sum"})

    rename_map = {
        "Power_Heating_sum": "Electricity_Heating_Wmin",
        "Power_DHW_sum": "Electricity_DHW_Wmin",
        "Heat_Heating_sum": "Heat_Heating_Wmin",
        "Heat_DHW_sum": "Heat_DHW_Wmin",
        "Cost_Inc_sum": "Daily_Cost_Euro",
        "is_active_sum": "Active_Mins",
        "is_DHW_sum": "DHW_Mins",
        "is_heating_sum": "Heating_Mins",
        "Immersion_Power_sum": "Immersion_Wh",
        "OutdoorTemp_mean": "Outdoor_Avg",
        "OutdoorTemp_min": "Outdoor_Min",
        "OutdoorTemp_max": "Outdoor_Max",
    }
    daily = daily.rename(columns=rename_map)

    # kWh conversions (safe even if *_Wmin columns are missing)
    daily["Electricity_Heating_kWh"] = (
        daily.get("Electricity_Heating_Wmin", 0) / 60000.0
    )
    daily["Electricity_DHW_kWh"] = (
        daily.get("Electricity_DHW_Wmin", 0) / 60000.0
    )
    daily["Heat_Heating_kWh"] = daily.get("Heat_Heating_Wmin", 0) / 60000.0
    daily["Heat_DHW_kWh"] = daily.get("Heat_DHW_Wmin", 0) / 60000.0
    daily["Immersion_kWh"] = daily.get("Immersion_Wh", 0) / 60000.0

    daily["Total_Electricity_kWh"] = (
        daily["Electricity_Heating_kWh"]
        + daily["Electricity_DHW_kWh"]
        + daily["Immersion_kWh"]
    )
    daily["Total_Heat_kWh"] = (
        daily["Heat_Heating_kWh"] + daily["Heat_DHW_kWh"]
    )

    daily["Global_SCOP"] = safe_div(
        daily["Total_Heat_kWh"], daily["Total_Electricity_kWh"]
    )

    # Data Quality Score / Tier
    if "row_count" in daily.columns:
        daily["DQ_Score"] = (
            (daily["row_count"] / 1440.0) * 100.0
        ).clip(0, 100)
        daily = daily.drop(columns=["row_count"])
    else:
        daily["DQ_Score"] = 0

    daily["DQ_Tier"] = np.where(
        daily["DQ_Score"] > 90, "Tier 1 (Gold)", "Tier 3 (Bronze)"
    )

    return daily
