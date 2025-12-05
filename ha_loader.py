# ha_loader.py
#
# Home Assistant CSV loader for v2.5
# Goal: produce a wide, 1-minute dataframe with the same column
# names the existing processing pipeline expects (Freq, Power, Heat,
# FlowTemp, ReturnTemp, DHW_Temp, ValveMode, etc.).
#
# This is distilled from your "Heat Pump Analysis - 11pm 02-12-2025
# Claude Refactor 3" logic, but simplified for HA-only use. :contentReference[oaicite:2]{index=2}

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable, Iterable, Optional, Dict, Any

import streamlit as st  # same pattern as data_loader/processing

from config import ENTITY_MAP  # friendly-name mapping for your entities
from processing import apply_gatekeepers, detect_runs, get_daily_stats


def _normalize_ha_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise Home Assistant CSV columns to:
      - last_changed (datetime)
      - entity_id (lowercase string)
      - state (string/numeric)
    """
    df = df.copy()

    # Trim weird characters / BOMs
    cleaned_cols = []
    for c in df.columns:
        c = str(c).strip().strip('"').strip("'").replace("\r", "").replace("\n", "").replace("\ufeff", "")
        cleaned_cols.append(c)
    df.columns = cleaned_cols

    # Normalise time column
    if "last_changed" in df.columns:
        time_col = "last_changed"
    elif "last_updated" in df.columns:
        df = df.rename(columns={"last_updated": "last_changed"})
        time_col = "last_changed"
    elif "time" in df.columns:
        df = df.rename(columns={"time": "last_changed"})
        time_col = "last_changed"
    else:
        # Fallback: assume first column is timestamp
        time_col = df.columns[0]
        df = df.rename(columns={time_col: "last_changed"})
        time_col = "last_changed"

    # Normalise state/value column
    if "state" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "state"})
    if "state" not in df.columns:
        # If still missing, create a dummy state (will be mostly useless)
        df["state"] = np.nan

    # Normalise entity column
    if "entity_id" not in df.columns and "entity" in df.columns:
        df = df.rename(columns={"entity": "entity_id"})
    if "entity_id" not in df.columns:
        # Try to infer from a column name like "sensor.heat_pump_power_ch1"
        # This is a best-effort fallback for wide HA exports, but the expectation
        # is that you export in long format (entity_id column present).
        raise ValueError("HA CSV must contain an 'entity_id' (or 'entity') column.")

    # Parse time + basic cleaning
    df["last_changed"] = pd.to_datetime(df["last_changed"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["last_changed"])

    df["entity_id"] = df["entity_id"].astype(str).str.lower()
    df["state"] = df["state"].astype(str).str.strip()

    return df[["last_changed", "entity_id", "state"]].copy()


def _clean_binary_state(x: Any) -> float:
    """
    Convert HA-style binary states to 0/1 (or numeric value if already numeric).
    """
    s = str(x).strip().lower()
    if s in ("on", "true", "1"):
        return 1.0
    if s in ("off", "false", "0", ""):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def load_ha_dataframe(
    uploaded_files: Iterable[Any],
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load one or more Home Assistant CSV exports and return a
    wide, 1-minute-resampled dataframe with friendly column names.

    This is intentionally simpler than the Grafana loader:
    - No pairing of Numeric/State files
    - Assumes long-format HA history (last_changed/entity_id/state)
    """
    files = list(uploaded_files)
    if not files:
        return None

    if progress_cb:
        progress_cb("Loading Home Assistant CSV…", 10)

    dfs = []
    for f in files:
        f.seek(0)
        df = pd.read_csv(f)
        df = _normalize_ha_columns(df)

        # Filter to mapped entities only
        df["entity_id"] = df["entity_id"].astype(str).str.lower()
        df = df[df["entity_id"].isin(ENTITY_MAP.keys())].copy()

        if df.empty:
            continue

        dfs.append(df)

    if not dfs:
        return None

    full_df = pd.concat(dfs, ignore_index=True).sort_values("last_changed")

    # Binary entities: on/off → 1/0
    binary_mask = (
        full_df["entity_id"].str.startswith("binary_")
        | full_df["entity_id"].str.contains("defrost")
    )
    full_df.loc[binary_mask, "state"] = full_df.loc[binary_mask, "state"].map(_clean_binary_state)

    if progress_cb:
        progress_cb("Pivoting to wide format…", 30)

    # Pivot to wide (entity_id columns)
    df_pivot = full_df.pivot_table(
        index="last_changed",
        columns="entity_id",
        values="state",
        aggfunc="last",
    )

    # 1-minute resample; for downsampled HA data this creates a continuous series
    # We use a simple forward-fill; the physics gatekeepers will still impose
    # thresholds on Power/Freq/etc when deciding what is "active".
    df_resampled = df_pivot.resample("1min").ffill()

    if progress_cb:
        progress_cb("Renaming entities…", 50)

    # Map HA entity_ids → friendly column names used by processing.py
    df_resampled = df_resampled.rename(columns=ENTITY_MAP)

    # Normalise a few categorical/state-like columns before numeric coercion
    if "Immersion_Mode" in df_resampled.columns:
        df_resampled["Immersion_Mode"] = df_resampled["Immersion_Mode"].apply(
            lambda x: 1 if str(x).strip().lower() == "on" else 0
        )

    if "Quiet_Mode" in df_resampled.columns:
        df_resampled["Quiet_Mode"] = df_resampled["Quiet_Mode"].apply(
            lambda x: 1 if str(x).strip().lower() == "on" else 0
        )

    # DHW_Mode normalisation (Economic/Standard/Power)
    if "DHW_Mode" in df_resampled.columns:
        def _norm_dhw_mode(val: Any) -> Optional[str]:
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            if s in ("economic", "eco"):
                return "Economic"
            if s in ("standard", "std"):
                return "Standard"
            if s in ("power", "boost", "forced"):
                return "Power"
            return np.nan

        df_resampled["DHW_Mode"] = df_resampled["DHW_Mode"].apply(_norm_dhw_mode)

    # Convert everything except clearly non-numeric columns to numeric
    non_numeric = {"ValveMode", "DHW_Mode"}
    for col in df_resampled.columns:
        if col in non_numeric:
            continue
        df_resampled[col] = pd.to_numeric(df_resampled[col], errors="coerce")

    if progress_cb:
        progress_cb("HA data loaded", 60)

    return df_resampled


def process_ha_files(
    uploaded_files: Iterable[Any],
) -> Optional[Dict[str, Any]]:
    """
    High-level helper to:
      1. Load HA CSV(s) → df_ha (wide, 1-min, friendly names),
      2. Run apply_gatekeepers (physics + DHW/heating flags),
      3. Detect runs (including DHW runs),
      4. Compute daily stats (heat, electricity, COP, DHW_SCOP, etc.).

    Returns a dict like:
      {
        "df":     minute-level dataframe (after gatekeepers),
        "runs":   list of per-run dicts (including DHW runs),
        "daily":  daily aggregation (heat, COP, DHW_SCOP, etc.)
      }
    or None if nothing usable was found.
    """
    # Simple progress bar integration mirroring data_loader
    pbar = st.progress(0, text="Loading Home Assistant data…")

    def _update(text: str, pct: int) -> None:
        pbar.progress(pct, text=text)

    df_ha = load_ha_dataframe(uploaded_files, progress_cb=_update)
    if df_ha is None or df_ha.empty:
        pbar.empty()
        return None

    _update("Applying physics gatekeepers…", 75)
    df = apply_gatekeepers(df_ha)

    _update("Detecting runs (Heating vs DHW)…", 85)
    runs_list = detect_runs(df)

    _update("Computing daily statistics (Heat & COP)…", 100)
    daily_df = get_daily_stats(df)

    pbar.empty()

    return {
        "df": df,
        "runs": runs_list,
        "daily": daily_df,
    }
