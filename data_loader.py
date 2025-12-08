# data_loader.py
"""
Core CSV loader for THERM.

- Handles both numeric ("value") and state ("state") CSVs.
- Normalises time to a 'Time' index.
- Pivots to wide format (index=Time, columns=entity_id).
- Resamples to 1-minute cadence.
- Applies user-config mapping.
- Provides hooks for unified Modbus interpretation across HA + Grafana paths.
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# TIME NORMALISATION
# ----------------------------------------------------------------------

# Accept a broader set of time column names, including Home Assistant history
# exports. This list is intentionally aligned with the main branch + HA exports.
_TIME_CANDIDATES = [
    "Time",
    "time",
    "timestamp",
    "Timestamp",
    "date",
    "datetime",
    "DateTime",
    "Date",
    # Home Assistant history export timestamp
    "last_changed",
]


def _normalise_time_column(
    temp: pd.DataFrame,
    filename: str,
) -> Optional[pd.DataFrame]:
    """
    Ensure the dataframe has a valid 'Time' column in datetime format.

    - Looks for a set of common timestamp column names.
    - Parses to datetime with day-first semantics.
    - Drops rows with invalid timestamps.
    - Renames the chosen column to 'Time'.

    Returns:
        Cleaned dataframe, or None if no usable time column is found.
    """
    time_col: Optional[str] = None

    for cand in _TIME_CANDIDATES:
        if cand in temp.columns:
            time_col = cand
            break

    if time_col is None:
        st.error(
            f"{filename} has no recognised time column "
            f"(expected one of: {_TIME_CANDIDATES}) – skipping this file."
        )
        return None

    temp = temp.copy()
    temp[time_col] = pd.to_datetime(
        temp[time_col],
        dayfirst=True,
        errors="coerce",
    )
    temp = temp.dropna(subset=[time_col])

    if temp.empty:
        st.error(
            f"{filename} has no valid timestamps after parsing – skipping this file."
        )
        return None

    if time_col != "Time":
        temp = temp.rename(columns={time_col: "Time"})

    return temp


# ----------------------------------------------------------------------
# MODBUS INTERPRETATION HOOK (SKELETON)
# ----------------------------------------------------------------------

def apply_modbus_interpretation(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, Any]] = None,
    source_hint: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified hook for interpreting raw Modbus registers into synthetic columns.

    This is intentionally a NO-OP skeleton for now – it is safe to call in
    existing workflows without changing behaviour.

    Intended responsibilities (to implement later, mirroring ha_loader.py):
        - Use raw Modbus integer registers (0/1, small integer enums, etc.).
        - Derive synthetic boolean / label columns such as:
            * DHW_Mode_Label, DHW_Mode_Is_Active
            * DHW_Status_Is_On
            * Defrost_Is_Active
            * Immersion_Is_On
            * Valve_Is_DHW, Valve_Is_Heating, Valve_Position_Label
        - Work for both:
            * Home Assistant history CSV exports (raw Modbus entities)
            * Grafana/Influx exports (same entities via Influx bridge)
        - Optionally use `mapping` and/or `source_hint` to:
            * Disambiguate where multiple registers are present
            * Support future dual-source architectures
              ("raw Modbus" primary, "HA template" fallback)

    Args:
        df: Wide dataframe indexed by Time, columns = entity_id or mapped names.
        mapping: Optional user mapping dictionary from config, if needed.
        source_hint: Optional free-text hint ("ha_csv", "grafana_csv", etc.).

    Returns:
        DataFrame with the same index/columns for now. Future versions will
        add synthetic columns but will avoid destructive changes.
    """
    # NOTE: This is currently a stub. Implementation will live here or will
    # call a separate shared module (e.g., modbus_interpreter.py) so that
    # HA and Grafana paths share identical decoding rules.
    return df


# ----------------------------------------------------------------------
# NUMERIC COERCION
# ----------------------------------------------------------------------

def _coerce_numeric_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """ Coerce sensor-like columns to numeric while preserving explicit text modes.
    Mirrors the main-branch invariant:
    - FlowRate, Power, temperatures, etc. should be numeric.
    - Only a small, explicit set of mode/label columns stay as text.
    """
    # Extend this set as you add more textual mode/label columns.
    text_cols = {
        "ValveMode",
        "DHW_Mode",
        # Future examples (once synthetic Modbus columns are added):
        # "Valve_Position_Label",
        # "DHW_Mode_Label",
    }

    # Tokens we consider as binary-like states (lowercased & stripped)
    binary_token_map = {
        "on": 1,
        "off": 0,
        "true": 1,
        "false": 0,
        "running": 1,
        "not running": 0,
        "yes": 1,
        "no": 0,
        "open": 1,
        "closed": 0,
        "1": 1,
        "0": 0,
    }

    for col in df.columns:
        if col in text_cols:
            # Explicitly textual columns stay as-is
            continue

        series = df[col]

        # If it's already numeric, leave it alone
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Try to detect binary-like columns (on/off, 0/1, running/not running, etc.)
        non_null = series.dropna()
        if not non_null.empty:
            norm = (
                non_null.astype(str)
                .str.strip()
                .str.lower()
            )
            unique_vals = set(norm.unique())

            # Only treat as binary if small cardinality and all values are in our token map
            if 0 < len(unique_vals) <= 6 and unique_vals.issubset(binary_token_map.keys()):
                df[col] = (
                    series.astype(str)
                    .str.strip()
                    .str.lower()
                    .map(binary_token_map)
                    .astype("float64")
                )
                continue

        # Fallback: numeric coercion as before
        try:
            df[col] = pd.to_numeric(series, errors="coerce")
        except Exception:
            # If conversion fails for some weird column, leave it as-is.
            # Downstream code should ignore non-numeric channels.
            pass

    return df



# ----------------------------------------------------------------------
# PUBLIC LOADER API
# ----------------------------------------------------------------------

def load_and_clean_data(
    files,
    user_config: Optional[Dict[str, Any]],
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Robust data loader that handles both Numeric (Float) and State (Text) CSV files.

    Steps:
        1) Read each uploaded CSV and normalise the time column to 'Time'.
        2) Split into numeric ('value') vs state ('state') tables.
        3) Pivot each to wide format: index=Time, columns=entity_id.
        4) Resample to 1-minute resolution:
            - numeric: mean + short interpolation
            - state: forward-fill (state persists until changed)
        5) Merge numeric + state with an outer join on Time.
        6) Apply mapping (simple column rename from config).
        7) Apply unified Modbus interpretation hook (currently NO-OP).
        8) Coerce sensor-like columns to numeric, preserve explicit text columns.
        9) Fill small gaps with limited forward-fill and return the dataset.

    Returns:
        dict with keys:
            - "df": processed wide dataframe
            - "raw_history": copy of df (for future use)
            - "baselines": None (placeholder)
            - "patterns": None (placeholder)
        or None if no usable data was found.
    """
    if not files:
        return None

    numeric_dfs: list[pd.DataFrame] = []
    state_dfs: list[pd.DataFrame] = []

    # 1. Read and classify files
    for i, f in enumerate(files):
        try:
            # Read CSV
            f.seek(0)
            temp = pd.read_csv(f)

            # Normalise / parse time column
            temp = _normalise_time_column(
                temp,
                getattr(f, "name", "uploaded file"),
            )
            if temp is None:
                # Skip files with no usable time column
                continue

            # Classify based on columns
            if "state" in temp.columns:
                state_dfs.append(temp)
            elif "value" in temp.columns:
                numeric_dfs.append(temp)

            if progress_cb:
                progress_cb(
                    f"Read {getattr(f, 'name', 'uploaded file')}",
                    (i / max(len(files), 1)) * 0.2,
                )
        except Exception as e:
            st.error(f"Error reading {getattr(f, 'name', 'uploaded file')}: {e}")

    # 2. Process Numeric Data (Resample: Mean)
    df_numeric_wide = pd.DataFrame()
    if numeric_dfs:
        if progress_cb:
            progress_cb("Processing numeric data...", 0.3)

        df_num = pd.concat(numeric_dfs, ignore_index=True)

        # Pivot: Index=Time, Columns=entity_id, Values=value
        # We group by Time/entity_id first to handle any duplicate timestamps.
        df_numeric_wide = (
            df_num.groupby(["Time", "entity_id"])["value"]
            .mean()
            .unstack()
        )

        # Resample to 1 minute, interpolating missing values
        df_numeric_wide = (
            df_numeric_wide
            .resample("1min")
            .mean()
            .interpolate(limit=30)
        )

    # 3. Process State Data (Resample: FFill)
    df_state_wide = pd.DataFrame()
    if state_dfs:
        if progress_cb:
            progress_cb("Processing state data...", 0.5)

        df_state = pd.concat(state_dfs, ignore_index=True)

        # Pivot: Index=Time, Columns=entity_id, Values=state
        # For state, we take the 'last' known state if duplicates exist
        df_state_wide = (
            df_state.groupby(["Time", "entity_id"])["state"]
            .last()
            .unstack()
        )

        # Resample to 1 minute, FORWARD FILLING the state
        # (state persists until changed)
        df_state_wide = df_state_wide.resample("1min").ffill()

    # 4. Merge
    if progress_cb:
        progress_cb("Merging datasets...", 0.6)

    if df_numeric_wide.empty and df_state_wide.empty:
        return None
    elif df_numeric_wide.empty:
        combined_df = df_state_wide
    elif df_state_wide.empty:
        combined_df = df_numeric_wide
    else:
        # Outer join to align timestamps
        combined_df = df_numeric_wide.join(df_state_wide, how="outer")

    # 5. Apply Mapping (simple column rename)
    if progress_cb:
        progress_cb("Applying sensor mapping...", 0.8)

    if user_config and "mapping" in user_config:
        # The config has {Friendly: Raw}; we need {Raw: Friendly}
        forward_map = user_config.get("mapping", {}) or {}
        if isinstance(forward_map, dict) and forward_map:
            reverse_map = {v: k for k, v in forward_map.items()}
            combined_df = combined_df.rename(columns=reverse_map)

    # 5b. Apply unified Modbus interpretation hook (currently NO-OP)
    # NOTE: When implementing, we may want to pass an explicit source_hint
    # such as "grafana_csv" vs "ha_csv" depending on upstream caller.
    combined_df = apply_modbus_interpretation(
        combined_df,
        mapping=user_config.get("mapping") if user_config else None,
        source_hint=None,
    )

    # 6. Final cleanup & small-gap filling
    combined_df = combined_df.sort_index()
    combined_df = _coerce_numeric_sensors(combined_df)
    combined_df = combined_df.ffill(limit=60)

    return {
        "df": combined_df,
        "raw_history": combined_df.copy(),
        "baselines": None,
        "patterns": None,
    }
