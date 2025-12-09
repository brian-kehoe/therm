# view_quality.py

import streamlit as st
import pandas as pd
import numpy as np
import os

from config import (
    SENSOR_EXPECTATION_MODE,
    SENSOR_GROUPS,
    SENSOR_ROLES,
    BASELINE_JSON_PATH,
)
from utils import availability_pct, strip_entity_prefix
import baselines


def render_data_quality(
    daily_df: pd.DataFrame,
    df: pd.DataFrame,
    unmapped_entities: list,
    patterns: dict | None,
    heartbeat_path: str | None,
) -> None:
    """
    Data Quality Studio:
    - Overview scorecard (DQ_Tier + category availability)
    - Category drill-down
    - Master sensor matrix
    - Heartbeat baselines
    - Unmapped data
    """
    st.title("️ Data Quality Studio")

    if daily_df is None or daily_df.empty:
        st.warning("No data loaded.")
        return

    # ------------------------------------------------------------------
    # Partial Day Logic
    # ------------------------------------------------------------------
    # Denominator is based on actual system-on minutes per day rather than
    # blindly hardcoding 1440 mins. This allows 100% uptime on partial days.
    system_on_minutes = daily_df.apply(
        lambda r: max(
            r.get("Recorded_Minutes", 0),
            r.get("DQ_Power_Count", 0),
            1,
        ),
        axis=1,
    )

    def expected_window_series(sensor_name: str, system_on_minutes_series: pd.Series):
        """
        Compute the expected sample count per day for a sensor, based on:
        - Learned heartbeat baselines (if available in session state)
        - SENSOR_EXPECTATION_MODE (system/heating/dhw/system_slow/event_only)
        """
        baseline_all = st.session_state.get("heartbeat_baseline", {})
        baseline_data = (baseline_all or {}).get(sensor_name)

        # Scenario A: Baseline exists (e.g. sparse sensor like OWM)
        if baseline_data and baseline_data.get("expected_minutes"):
            # Scale expected minutes by partial-day ratio
            baseline_ratio = baseline_data["expected_minutes"] / 1440.0
            base = system_on_minutes_series * baseline_ratio
            return (
                base.replace(0, np.nan)
                .apply(lambda x: max(1.0, x) if x > 0 else np.nan)
            )

        # Scenario B: Mode-driven expectation
        mode = SENSOR_EXPECTATION_MODE.get(sensor_name, "system")

        if mode == "heating_active":
            base = daily_df.get("Active_Mins", system_on_minutes_series)
        elif mode == "dhw_active":
            base = daily_df.get("Total_DHW_Mins", system_on_minutes_series)
        elif mode == "system_slow":
            # Very slow sensors: effectively "hourly-ish"
            base = (system_on_minutes_series / 60.0).apply(np.ceil)
        else:
            base = system_on_minutes_series

        return base.replace(0, np.nan)

    def format_dq_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        df_out.index.name = "Date"
        try:
            df_out.index = df_out.index.strftime("%d-%m-%Y")
        except Exception:
            # If not a datetime index, leave as-is
            pass
        return df_out

    dq_tab1, dq_tab2, dq_tab3, dq_tab4, dq_tab5 = st.tabs(
        ["Overview", "Category Drill-Down", "All Sensors", "Heartbeats", "⚠️ Unmapped Data"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Overview
    # ------------------------------------------------------------------
    with dq_tab1:
        st.markdown("### System Health Scorecard")

        dq_avg = float(daily_df.get("DQ_Score", 0).mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Data Health", f"{dq_avg:.1f}%")

        gold = len(
            daily_df[
                daily_df.get("DQ_Tier", "")
                .astype(str)
                .str.contains("Gold", na=False)
            ]
        )
        silver = len(
            daily_df[
                daily_df.get("DQ_Tier", "")
                .astype(str)
                .str.contains("Silver", na=False)
            ]
        )
        c2.metric("Gold Days", gold)
        c3.metric("Silver Days", silver)

        overview_df = daily_df[["DQ_Tier"]].copy()
        group_cols: list[str] = []

        for group_name, sensors in SENSOR_GROUPS.items():
            if "Events" in group_name or "Event" in group_name:
                continue

            # Identify sensors that actually have count columns
            valid_sensors = [
                s
                for s in sensors
                if f"DQ_{s}_Count" in daily_df.columns
                or f"{s}_count" in daily_df.columns
            ]
            if not valid_sensors:
                continue

            group_pcts = []
            for s in valid_sensors:
                col = (
                    f"DQ_{s}_Count"
                    if f"DQ_{s}_Count" in daily_df.columns
                    else f"{s}_count"
                )
                pct = availability_pct(
                    daily_df[col],
                    expected_window_series(s, system_on_minutes),
                )
                group_pcts.append(pct)

            if group_pcts:
                overview_df[group_name] = (
                    pd.concat(group_pcts, axis=1).mean(axis=1).round(0)
                )
                group_cols.append(group_name)

        overview_disp = format_dq_df(overview_df[["DQ_Tier"] + group_cols])

        st.dataframe(
            overview_disp.style.background_gradient(
                subset=group_cols,
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
            ).format("{:.0f}", subset=group_cols),
            width="stretch",
        )

    # ------------------------------------------------------------------
    # TAB 2: Category Drill-Down
    # ------------------------------------------------------------------
    with dq_tab2:
        st.markdown("### Category Inspector")

        cat = st.selectbox("Select System Category", list(SENSOR_GROUPS.keys()))
        selected = SENSOR_GROUPS.get(cat, [])

        cat_df = pd.DataFrame(index=daily_df.index)
        valid_cols: list[str] = []

        for sensor in selected:
            col_name = (
                f"DQ_{sensor}_Count"
                if f"DQ_{sensor}_Count" in daily_df.columns
                else f"{sensor}_count"
            )
            if col_name not in daily_df.columns:
                continue

            mode = SENSOR_EXPECTATION_MODE.get(sensor, "system")

            if mode == "event_only" or "defrost" in sensor.lower():
                # Event-style sensors: report raw count
                cat_df[sensor] = daily_df[col_name].fillna(0).astype(int)
            else:
                exp = expected_window_series(sensor, system_on_minutes)
                cat_df[sensor] = availability_pct(
                    daily_df[col_name], exp
                ).round(0)

            valid_cols.append(sensor)

        cat_disp = format_dq_df(cat_df)

        if not cat_disp.empty:
            normal_cols = [
                c
                for c in valid_cols
                if SENSOR_EXPECTATION_MODE.get(c, "system") != "event_only"
            ]
            styler = cat_disp.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(
                    subset=normal_cols,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=100,
                )
            st.dataframe(styler, width="stretch")
        else:
            st.info("No data for this category.")

    # ------------------------------------------------------------------
    # TAB 3: Master Sensor Matrix (with categorisation from main branch)
    # ------------------------------------------------------------------
    with dq_tab3:
        st.markdown("### Master Sensor Matrix")

        # --- Helper: resolve mapped / display names ---
        config_obj = st.session_state.get("system_config", {}) or {}
        if isinstance(config_obj, dict):
            mapping = config_obj.get("mapping", {}) or {}
        else:
            mapping = {}

        mapped_keys = set(mapping.keys())

        # Canonical UI labels for standardised core data points
        STANDARD_LABELS = {
            "Power": "Outdoor Power",
            "Indoor_Power": "Indoor Power",
            "Heat": "Heat Output",
            "FlowTemp": "Flow Temp",
            "ReturnTemp": "Return Temp",
            "ValveMode": "3 Way Valve",
            "OutdoorTemp": "Outdoor Temp",
            "DHW_Temp": "Hot Water Temp",
            "Freq": "Compressor Freq",
        }

        # Derived / internal-only names that should NEVER appear in All Sensors
        DERIVED_NAMES = {
            "DeltaT",
            "COP_Real",
            "COP_Graph",
            "Active_Zones_Count",
            "hour",
            "is_night_rate",
            "Current_Rate",
            "Zone_Config",
            "Immersion_Active",
            "Immersion_Power",
            "Cost_Inc",
            "Electricity_Heating_Wmin",
            "Electricity_DHW_Wmin",
            "Heat_Heating_Wmin",
            "Heat_DHW_Wmin",
            "Immersion_Wh",
        }

        def sensor_is_allowed(name: str) -> bool:
            """
            Only include:
            - Sensors that the user explicitly mapped in the profile
            - AND are not in the derived/internal blacklist.
            """
            if not name:
                return False
            if name in DERIVED_NAMES:
                return False
            return name in mapped_keys

        def display_label_for(internal_name: str) -> str:
            """
            What the user sees on the front end.

            - Zones & Rooms: show the mapped sensor name (entity_id) with prefixes stripped.
            - Standard core sensors: show FRIENDLY labels (Outdoor Power, Flow Temp, etc.).
            - Everything else: mapped entity_id (prefix-stripped), or internal name if unmapped.
            """
            if internal_name.startswith("Zone_") or internal_name.startswith("Room_"):
                return strip_entity_prefix(str(mapping.get(internal_name, internal_name)))

            if internal_name in STANDARD_LABELS:
                return STANDARD_LABELS[internal_name]

            return strip_entity_prefix(str(mapping.get(internal_name, internal_name)))


        def tooltip_label_for(internal_name: str) -> str:
            """
            Tooltip content: show the mapped sensor (entity_id),
            and optionally the internal name for debugging context.
            """
            entity = mapping.get(internal_name, internal_name)
            if entity == internal_name:
                return str(entity)
            return f"{entity} (internal: {internal_name})"

        # 1. Build Data (counts -> availability % or raw event counts)
        count_cols = [
            c for c in daily_df.columns
            if c.endswith("_count") or c.endswith("_Count")
        ]
        if count_cols:
            flat_data: dict[str, pd.Series] = {}

            for c in count_cols:
                clean_name = (
                    c.replace("DQ_", "")
                    .replace("_Count", "")
                    .replace("_count", "")
                )

                # Skip internal diagnostic counters, like short-cycle trackers
                if "short_cycle" in clean_name.lower():
                    continue

                # Only keep "real" sensors that the user mapped
                if not sensor_is_allowed(clean_name):
                    continue

                mode = SENSOR_EXPECTATION_MODE.get(clean_name, "system")

                # Event-style sensors: raw count
                if mode == "event_only":
                    flat_data[clean_name] = daily_df[c].fillna(0).astype(int)
                else:
                    flat_data[clean_name] = availability_pct(
                        daily_df[c],
                        expected_window_series(clean_name, system_on_minutes),
                    ).round(0)

            df_flat = pd.DataFrame(flat_data, index=daily_df.index)

            if df_flat.empty:
                st.info("No mapped sensor count columns found in daily data.")
            else:
                # 2. Re-construct Ordered Columns (Events moved to end)
                column_meta = []   # list of dicts: {category, internal, display, tooltip}
                valid_data_cols: list[str] = []
                events_cat: str | None = None
                events_list: list[str] = []
                zones_cat_name: str | None = None

                # A. Normal Groups (except Zones + Events)
                for cat_name, sensors in SENSOR_GROUPS.items():
                    if "Event" in cat_name or "Events" in cat_name:
                        events_cat = cat_name
                        events_list = sensors
                        continue

                    # Option B: treat "Zones" as a generic group, not fixed names
                    if "Zones" in cat_name or " Zone" in cat_name:
                        zones_cat_name = cat_name
                        continue

                    found_sensors = [s for s in sensors if s in df_flat.columns]
                    for s in found_sensors:
                        column_meta.append(
                            {
                                "category": cat_name,
                                "internal": s,
                                "display": display_label_for(s),
                                "tooltip": tooltip_label_for(s),
                            }
                        )
                        valid_data_cols.append(s)

                # B. Zones (generic: any Zone_* present in df_flat)
                if zones_cat_name:
                    zone_cols = sorted(
                        [
                            c for c in df_flat.columns
                            if c.startswith("Zone_") and c not in valid_data_cols
                        ]
                    )
                    for z in zone_cols:
                        column_meta.append(
                            {
                                "category": zones_cat_name,
                                "internal": z,
                                "display": display_label_for(z),
                                "tooltip": tooltip_label_for(z),
                            }
                        )
                        valid_data_cols.append(z)

                # C. Rooms (any Room_* not yet placed)
                room_cols = sorted(
                    [
                        c for c in df_flat.columns
                        if c.startswith("Room_") and c not in valid_data_cols
                    ]
                )
                for r in room_cols:
                    column_meta.append(
                        {
                            "category": "️ Rooms",
                            "internal": r,
                                # Show friendly label or entity ID, not "Room_1"/"1"
                            "display": display_label_for(r),
                            "tooltip": tooltip_label_for(r),
                        }
                    )
                    valid_data_cols.append(r)

                # D. Others (still mapped, but not in any category above)
                remaining = sorted(
                    [
                        c for c in df_flat.columns
                        if c not in valid_data_cols
                        and (not events_list or c not in events_list)
                    ]
                )
                for rem in remaining:
                    column_meta.append(
                        {
                            "category": "Other",
                            "internal": rem,
                            "display": display_label_for(rem),
                            "tooltip": tooltip_label_for(rem),
                        }
                    )
                    valid_data_cols.append(rem)

                # E. Events (appended last)
                if events_cat:
                    found_events = [s for s in events_list if s in df_flat.columns]
                    for s in found_events:
                        column_meta.append(
                            {
                                "category": events_cat,
                                "internal": s,
                                "display": display_label_for(s),
                                "tooltip": tooltip_label_for(s),
                            }
                        )
                        valid_data_cols.append(s)

                # 3. Build Final DataFrame
                df_final = df_flat[valid_data_cols].copy()
                df_final = format_dq_df(df_final)

                # Build MultiIndex columns from category + display label
                multi_cols = pd.MultiIndex.from_tuples(
                    [(m["category"], m["display"]) for m in column_meta]
                )
                df_final.columns = multi_cols

                # 4. Apply Styles (normal vs event-style)
                event_cols = []
                normal_cols = []
                internal_names = [m["internal"] for m in column_meta]

                for col_tuple, internal_name in zip(df_final.columns, internal_names):
                    mode = SENSOR_EXPECTATION_MODE.get(internal_name, "system")
                    if mode == "event_only" or "defrost" in internal_name.lower():
                        event_cols.append(col_tuple)
                    else:
                        normal_cols.append(col_tuple)

            styler = df_final.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(
                    subset=normal_cols, cmap="RdYlGn", vmin=0, vmax=100,
                )
            if event_cols:
                # Grey background for event-style sensors
                styler = styler.map(
                    lambda x: "background-color: #e0e0e0; color: #555555",
                    subset=event_cols,
                )

            # NOTE:
            # We intentionally do NOT use Styler tooltips here because Streamlit
            # will escape the HTML and show it in the cells instead of rendering
            # proper hover-tooltips. Column headers already show the correct
            # standard labels (or entity_ids for Rooms/Zones).

            st.dataframe(styler, width="stretch")

        else:
            st.info("No count-based columns found in daily data.")





    # ------------------------------------------------------------------
    # TAB 4: Heartbeats
    # ------------------------------------------------------------------
    with dq_tab4:
        st.markdown("### ❤️ Sensor Heartbeats")

        # Try to load existing baseline into memory if not already
        baseline = st.session_state.get("heartbeat_baseline")

        if heartbeat_path and not baseline:
            try:
                baseline = baselines.load_baseline_file(heartbeat_path)
                st.session_state["heartbeat_baseline"] = baseline
                st.info(f"Loaded existing baseline from: {heartbeat_path}")
            except Exception as e:
                st.warning(f"Could not load baseline file: {e}")
                baseline = None

        # --- Build / Rebuild button ---
        build_clicked = st.button("Generate / Rebuild Heartbeat Baseline")

        if build_clicked:
            # Prefer raw long history if present
            history_df = st.session_state.get("raw_history_df")

            if history_df is None or history_df.empty:
                st.warning(
                    "No raw history available in session. "
                    "Using the current processed dataframe instead."
                )
                history_df = df

            if history_df is None or history_df.empty:
                st.error("No data available to build baselines.")
            else:
                with st.spinner("Building baselines..."):
                    new_bl = baselines.build_offline_aware_seasonal_baseline(
                        history_df, SENSOR_ROLES
                    )
                    st.session_state["heartbeat_baseline"] = new_bl
                    baseline = new_bl
                st.success("Heartbeat baseline updated in memory.")

        if baseline:
            # Show summary table
            rows = []
            for sensor_name, meta in baseline.items():
                rows.append(
                    {
                        "Sensor": sensor_name,
                        "Role": meta.get("role", SENSOR_ROLES.get(sensor_name, "")),
                        "Expected Minutes / Day": meta.get("expected_minutes", None),
                        "Mode": SENSOR_EXPECTATION_MODE.get(sensor_name, "system"),
                    }
                )
            hb_df = pd.DataFrame(rows).sort_values("Sensor")
            st.dataframe(hb_df, width="stretch")

            # Allow user to download to a location of their choice
            import json as _json

            st.download_button(
                label="💾 Download Heartbeat Baseline (JSON)",
                data=_json.dumps(baseline, indent=2),
                file_name="therm_heartbeat_baseline.json",
                mime="application/json",
            )
        else:
            st.info(
                "No heartbeat baseline loaded yet. "
                "Click the button above to generate one."
            )

        # Optional: show the pattern analysis table if available
        if patterns:
            st.markdown("### Detected Sensor Patterns")
            pat_data = []
            for sensor, details in patterns.items():
                pat_data.append(
                    {
                        "Sensor": sensor,
                        "Type": details["report_type"],
                        "Interval (s)": round(details["normal_interval_sec"], 1),
                        "Gap Limit (s)": round(details["gap_threshold_sec"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(pat_data), width="stretch")


    # ------------------------------------------------------------------
    # TAB 5: Unmapped Data
    # ------------------------------------------------------------------
    with dq_tab5:
        st.markdown("### Unmapped / Dropped Entities")

        if not unmapped_entities:
            st.info("No unmapped entities found in the source files.")
        else:
            st.write(
                "The following entities were present in the upload but not "
                "mapped to any internal sensor role:"
            )
            st.dataframe(
                pd.DataFrame(
                    sorted(set(unmapped_entities)), columns=["Entity ID"]
                ),
                width="stretch",
            )
