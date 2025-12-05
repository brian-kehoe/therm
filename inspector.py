# inspector.py
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# SAFE PARSER
# ---------------------------------------------------------------------
def safe_smart_parse(series):
    """
    Convert numeric-capable values to floats, leave other values unchanged.
    Returns parsed series + flag indicating if sensor is mostly numeric.
    """
    parsed = []
    numeric_count = 0
    total = len(series)

    for val in series.astype(str):
        try:
            f = float(val)
            parsed.append(f)
            numeric_count += 1
        except:
            parsed.append(val)

    s = pd.Series(parsed, index=series.index)
    mostly_numeric = (numeric_count / total) >= 0.70 if total else False
    return s, mostly_numeric


# ---------------------------------------------------------------------
# BINARY DETECTION HELPER
# ---------------------------------------------------------------------
def is_binary_sensor(values):
    """
    Returns True only if ALL non-null values are binary-like strings
    ("on","off","true","false","0","1") OR if unique count ≤ 3 AND all
    values are short strings without decimals.
    """
    allowed = {"on", "off", "true", "false", "0", "1"}

    str_vals = [str(v).lower() for v in values if str(v).lower() != "unavailable"]

    # rule 1: all values in allowed set
    if all(v in allowed for v in str_vals):
        return True

    # rule 2: very small unique set & no decimals
    if len(str_vals) <= 3:
        if all("." not in v for v in str_vals):
            return True

    return False


# ============================================================================
# MAIN INSPECTOR
# ============================================================================
def inspect_raw_files(uploaded_files):
    summary_stats = []
    file_details = {}

    # --- FILE-LEVEL METADATA -------------------------------------------------
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)

        file_stat = {
            "Filename": uploaded_file.name,
            "Size (KB)": round(uploaded_file.size / 1024, 1),
        }

        try:
            df_raw = pd.read_csv(uploaded_file)

            file_stat["Rows"] = len(df_raw)
            file_stat["Columns"] = len(df_raw.columns)

            ts_col = next(
                (c for c in df_raw.columns if any(x in c.lower() for x in
                ["time", "date", "timestamp", "last_changed", "last_updated"])),
                None
            )

            if ts_col:
                try:
                    start_ts = pd.to_datetime(df_raw[ts_col].iloc[0], errors="coerce")
                    end_ts   = pd.to_datetime(df_raw[ts_col].iloc[-1], errors="coerce")
                    file_stat["Start Time"] = str(start_ts)
                    file_stat["End Time"]   = str(end_ts)
                except:
                    file_stat["Start Time"] = "Parse Error"
                    file_stat["End Time"]   = "Parse Error"
            else:
                file_stat["Start Time"] = "Not Found"
                file_stat["End Time"]   = "Not Found"

            cols_lower = [str(c).lower() for c in df_raw.columns]

            if "entity_id" in cols_lower:
                structure = "Long Format (State)"
                e_col = df_raw.columns[cols_lower.index("entity_id")]
                entities = sorted(df_raw[e_col].dropna().astype(str).unique().tolist())
            else:
                structure = "Wide Format (Table)"
                ignore = ["time","timestamp","last_changed","last_updated","value","series"]
                entities = [c for c in df_raw.columns if c.lower() not in ignore]

            file_details[uploaded_file.name] = {
                "structure": structure,
                "columns_raw": list(df_raw.columns),
                "entities_found": entities,
            }

        except Exception as e:
            file_stat["Rows"] = "Error"
            file_stat["Start Time"] = "Error"
            file_stat["End Time"] = "Error"
            file_details[uploaded_file.name] = {"error": str(e)}

        summary_stats.append(file_stat)
        uploaded_file.seek(0)

    summary_df = pd.DataFrame(summary_stats).astype(str)

    # --- SENSOR DEBUG --------------------------------------------------------
    sensor_debug = {}

    for fname, info in file_details.items():
        if "error" in info:
            sensor_debug[fname] = {"error": info["error"]}
            continue

        # Reload file
        df_raw = None
        for f in uploaded_files:
            if f.name == fname:
                f.seek(0)
                df_raw = pd.read_csv(f)
                break
        if df_raw is None:
            continue

        long_format = info["structure"] == "Long Format (State)"
        sensors_info = {}

        ts_col = next(
            (c for c in df_raw.columns if any(x in c.lower() for x in
            ["time","date","timestamp","last_changed","last_updated"])),
            None
        )
        ts = pd.to_datetime(df_raw[ts_col], errors="coerce") if ts_col else None

        # ---------------------------------------------------------------------
        # LONG FORMAT
        # ---------------------------------------------------------------------
        if long_format:

            entity_col = next(c for c in df_raw.columns if c.lower() == "entity_id")
            state_col  = next(c for c in df_raw.columns if c.lower() == "state")

            for entity_name, g in df_raw.groupby(entity_col):

                raw = g[state_col].astype(str)
                parsed, is_numeric = safe_smart_parse(raw)

                non_null = parsed[parsed != "unavailable"]
                total_rows = len(parsed)
                entry_count = non_null.count()
                availability = round(entry_count / total_rows * 100, 2) if total_rows else 0

                info_dict = {
                    "dtype": "numeric" if is_numeric else "string",
                    "total_rows": total_rows,
                    "non_null_rows": int(entry_count),
                    "availability_%": availability,
                    "unique_values": non_null.nunique(),
                    "example_values": non_null.head(5).tolist(),
                }

                # Numeric stats
                if is_numeric:
                    nv = pd.to_numeric(non_null, errors="coerce").dropna()
                    if not nv.empty:
                        info_dict["min"] = float(nv.min())
                        info_dict["max"] = float(nv.max())
                        info_dict["mean"] = float(nv.mean())

                # Sampling interval
                if ts is not None:
                    sensor_ts = ts.loc[g.index]
                    diffs = sensor_ts.diff().dt.total_seconds().dropna()
                    if len(diffs):
                        med = float(diffs.median())
                        info_dict["median_interval_sec"] = med
                        info_dict["hourly_downsampled"] = med >= 3600
                    else:
                        info_dict["median_interval_sec"] = None
                        info_dict["hourly_downsampled"] = False

                # Strict binary detection
                if is_binary_sensor(non_null.unique().tolist()):
                    info_dict["dtype"] = "binary"
                    info_dict["binary_candidate_values"] = sorted(
                        list({str(v).lower() for v in non_null.unique()})
                    )
                    info_dict["binary_valid"] = True

                sensors_info[entity_name] = info_dict

        # ---------------------------------------------------------------------
        # WIDE FORMAT
        # ---------------------------------------------------------------------
        else:
            for col in info["columns_raw"]:
                series = df_raw[col].astype(str)
                parsed, is_numeric = safe_smart_parse(series)
                non_null = parsed[parsed != "unavailable"]

                total = len(parsed)
                entry_count = non_null.count()

                c_info = {
                    "dtype": "numeric" if is_numeric else "string",
                    "non_null_rows": int(entry_count),
                    "availability_%": round(entry_count / total * 100, 2) if total else 0,
                    "unique_values": non_null.nunique(),
                }

                if is_numeric:
                    nv = pd.to_numeric(non_null, errors="coerce").dropna()
                    if not nv.empty:
                        c_info["min"] = float(nv.min())
                        c_info["max"] = float(nv.max())
                        c_info["mean"] = float(nv.mean())

                sensors_info[col] = c_info

        sensor_debug[fname] = sensors_info

    # -------------------------------------------------------------------------
    return summary_df, {
        "file_details": file_details,
        "sensor_debug": sensor_debug,
    }
