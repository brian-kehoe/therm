# data_loader.py

import pandas as pd
import streamlit as st


# Accept a broader set of time column names, including Home Assistant history exports
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


def _normalise_time_column(temp: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    """
    Ensure the dataframe has a valid 'Time' column in datetime format.

    - Looks for a set of common timestamp column names.
    - Parses to datetime with day-first semantics.
    - Drops rows with invalid timestamps.
    - Renames the chosen column to 'Time'.

    Returns:
        Cleaned dataframe, or None if no usable time column is found.
    """
    time_col = None
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


def load_and_clean_data(files, user_config, progress_cb=None):
    """
    Robust data loader that handles both Numeric (Float) and State (Text) CSV files.

    - Reads each uploaded CSV.
    - Ensures a valid 'Time' column exists (or skips the file).
    - Splits into numeric ('value') vs state ('state') tables.
    - Pivots to wide format: index=Time, columns=entity_id.
    - Resamples to 1-minute resolution (mean for numeric, ffill for state).
    - Merges numeric + state.
    - Applies simple column renaming based on user_config["mapping"].
    - Finally, coerces all sensor-like columns to numeric, except explicit
      textual mode columns (ValveMode, DHW_Mode).

    Returns:
        {
          "df": combined_df,
          "raw_history": combined_df.copy(),
          "baselines": None,
          "patterns": None,
        }
    """
    if not files:
        return None

    numeric_dfs = []
    state_dfs = []

    # 1. Read and Classify Files
    for i, f in enumerate(files):
        try:
            # Read CSV
            f.seek(0)
            temp = pd.read_csv(f)

            # Normalise / parse time column
            temp = _normalise_time_column(temp, getattr(f, "name", "uploaded file"))
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
            df_numeric_wide.resample("1min").mean().interpolate(limit=30)
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

        # Resample to 1 minute, FORWARD FILLING the state (state persists until changed)
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
        # Create reverse map: {Raw_ID: Friendly_Column}
        # The config has {Friendly: Raw}
        forward_map = user_config.get("mapping", {}) or {}
        if isinstance(forward_map, dict) and forward_map:
            reverse_map = {v: k for k, v in forward_map.items()}
            combined_df = combined_df.rename(columns=reverse_map)

    # 6. Final Cleanup

    # Ensure index is sorted
    combined_df = combined_df.sort_index()

    # --- Global numeric coercion (aligned with main branch) ----------------
    #
    # At this point combined_df contains:
    #   - Numeric channels from Grafana exports ("value" path)
    #   - State-based channels from Home Assistant history ("state" path)
    #
    # We want all sensor-like columns to be numeric where possible, and keep
    # only a small, explicit set of textual mode columns as strings.
    #
    # This mirrors the main-branch invariant: processing.py can assume that
    # FlowRate, Power, temperatures, etc. are numeric when doing physics and
    # run detection.
    TEXT_COLS = {"ValveMode", "DHW_Mode"}  # extend if you add more textual modes

    for col in combined_df.columns:
        if col in TEXT_COLS:
            # Explicitly textual columns stay as-is
            continue
        try:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")
        except Exception:
            # If conversion fails for some weird column, leave it as-is.
            # Downstream code should ignore non-numeric channels.
            pass
    # ----------------------------------------------------------------------

    # Fill small gaps (limited ffill to avoid masking real outages)
    combined_df = combined_df.ffill(limit=60)

    return {
        "df": combined_df,
        "raw_history": combined_df.copy(),
        "baselines": None,
        "patterns": None,
    }
