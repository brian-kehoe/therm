# view_runs.py - Fixed zone and room naming



import streamlit as st

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import json

import numpy as np

from datetime import datetime, timezone



from config import (
    CALC_VERSION,
    THRESHOLDS,
)

from utils import safe_div, strip_entity_prefix







def _build_system_context(user_config: dict | None, include_heating_note: bool) -> str:
    """Compose AI context from user-provided freetext; add DHW heating note only if detected."""
    parts: list[str] = []
    if isinstance(user_config, dict):
        ai_ctx = user_config.get("ai_context") or {}
        for key in ("hp_model", "property_context", "operational_goals"):
            val = ai_ctx.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        tariff = user_config.get("tariff_structure")
        currency = user_config.get("currency", "?")
        if isinstance(tariff, dict):
            day = tariff.get("day_rate")
            night = tariff.get("night_rate", day)
            parts.append(f"Tariff: day/night rates {currency}{day} / {currency}{night}.")
        elif isinstance(tariff, list) and tariff:
            rules = tariff[0].get("rules", [])
            if rules:
                summary = "; ".join(
                    f"{r.get('name','')}: {r.get('start','')}-{r.get('end','')} @ {currency}{r.get('rate','')}"
                    for r in rules
                )
                parts.append(f"Tariff bands: {summary}")
    if include_heating_note:
        parts.append(
            "Heating during DHW detected: zone pumps active during DHW can cause return mixing and low COP; attribute DHW efficiency penalties accordingly."
        )
    return "\n".join(parts) if parts else "No additional system context supplied."

def _get_friendly_name(internal_key: str, user_config: dict) -> str:

    """

    Get the friendly display name for a zone or room.

    

    For Zone_1, Zone_2, etc. or Room_1, Room_2, etc.:

    - Returns the mapped entity_id from user config

    - Strips "binary_sensor." and "sensor." prefixes for cleaner display

    - Falls back to the internal key if not mapped

    

    Args:

        internal_key: Internal name like "Zone_1" or "Room_3"

        user_config: User's system configuration dict

        

    Returns:

        Friendly name string (cleaned entity_id or internal key)

    """

    if not isinstance(user_config, dict):

        return str(internal_key)

    

    mapping = user_config.get("mapping", {})

    if not isinstance(mapping, dict):

        return str(internal_key)



    # Get the mapped entity_id and strip HA/Grafana prefixes for display

    entity_id = str(mapping.get(internal_key, internal_key))

    return strip_entity_prefix(entity_id)





def _get_rooms_per_zone_config(user_config: dict) -> dict:

    """

    Get the user's rooms_per_zone configuration.

    

    Returns:

        dict mapping zone keys (e.g., "Zone_1") to lists of room keys (e.g., ["Room_3"])

    """

    if not isinstance(user_config, dict):

        return {}

    

    return user_config.get("rooms_per_zone", {})





def render_run_inspector(df, runs_list):

    """

    Run Inspector:

    - Sidebar global stats

    - Run picker with Previous/Next navigation

    - Tabs:

        ⚡ Efficiency

        Hydraulics (incl. Heating during DHW)
        Rooms

        AI Data (per-run JSON payload)

    """

    st.title(" Run Inspector")

    

    # Get user config from session state for friendly names

    user_config = st.session_state.get("system_config", {})

    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}

    has_zones_mapped = any(k.startswith("Zone_") for k in mapping)



    total_kwh_in = safe_div(df.get("Power_Clean", df["Power"]).sum(), 1000.0) / 60.0

    total_kwh_out = safe_div(df.get("Heat_Clean", df["Heat"]).sum(), 1000.0) / 60.0

    global_cop = safe_div(total_kwh_out, total_kwh_in)



    if not runs_list:

        st.info("No runs detected.")

        return



    runs_list = sorted(runs_list, key=lambda x: x["start"], reverse=True)



    # --- Run Selection Logic ---

    run_options = {}

    for r in runs_list:

        start_str = r["start"].strftime("%d/%m/%Y %H:%M")



        heating_during_dhw = bool(
            r.get("heating_during_dhw_detected") or r.get("heating_during_dhw_power_detected")
        )

        if r["run_type"] == "DHW":
            # Use a water drop to represent DHW runs
            icon = "💧"
        else:
            icon = "🔥"



        zone_raw = r.get("active_zones", r.get("dominant_zones", "None"))



        # For DHW runs, hide zone info unless heating during DHW was detected.

        show_zone = (

            has_zones_mapped

            and (

                r["run_type"] != "DHW"

                or r.get("heating_during_dhw_detected")

                or r.get("heating_during_dhw_power_detected")

            )

        )

        if show_zone:
            zone_label = zone_raw if zone_raw and str(zone_raw).lower() != "none" else "No Zone Data"
        else:
            zone_label = ""

        # Avoid redundant "(DHW)" suffix when the run itself is DHW
        if heating_during_dhw and r["run_type"] == "DHW":
            if zone_label.strip().lower() == "dhw":
                zone_label = ""

        label = f"{start_str} | {r['duration_mins']}m | {icon} {r['run_type']}"
        if heating_during_dhw and r["run_type"] == "DHW":
            label = f"{label} (Heating Active❗)"
        if zone_label:
            label = f"{label} ({zone_label})"

        run_options[label] = r



    option_labels = list(run_options.keys())

    if "run_selector_idx" not in st.session_state:

        st.session_state["run_selector_idx"] = 0



    st.session_state["run_selector_idx"] = min(

        st.session_state["run_selector_idx"], len(option_labels) - 1

    )



    nav_prev, nav_select, nav_next = st.columns([1, 4, 1])



    with nav_prev:

        if st.button(

            "← Previous", disabled=st.session_state["run_selector_idx"] <= 0

        ):

            st.session_state["run_selector_idx"] = max(

                0, st.session_state["run_selector_idx"] - 1

            )



    with nav_next:

        if st.button(

            "Next →",

            disabled=st.session_state["run_selector_idx"] >= len(option_labels) - 1,

        ):

            st.session_state["run_selector_idx"] = min(

                len(option_labels) - 1, st.session_state["run_selector_idx"] + 1

            )



    with nav_select:

        selected_label = st.selectbox(

            "Select Run",

            options=option_labels,

            index=st.session_state["run_selector_idx"],

        )

        st.session_state["run_selector_idx"] = option_labels.index(selected_label)



    selected_run = run_options[selected_label]

    run_data = df.loc[selected_run["start"] : selected_run["end"]]



    # --- Top-level stats ---

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Duration", f"{selected_run['duration_mins']}m")

    c2.metric("COP", f"{selected_run['run_cop']:.2f}")

    c3.metric("Avg ΔT", f"{selected_run['avg_dt']:.1f}°")

    avg_flow_lpm = selected_run.get("avg_flow_rate", 0)

    c4.metric("Avg Flow", f"{avg_flow_lpm:.1f} L/m")



    tight_layout = dict(margin=dict(l=10, r=10, t=30, b=10), height=350)



    # --- TABS ---

    tab1, tab2, tab3, tab4 = st.tabs(

        ["⚡ Efficiency", " Hydraulics", " Rooms", " AI Data"]

    )



    # ------------------------------------------------------------------

    # TAB 1: Efficiency

    # ------------------------------------------------------------------

    with tab1:

        fig = make_subplots(specs=[[{"secondary_y": True}]])



        fig.add_trace(

            go.Scatter(

                x=run_data.index,

                y=run_data.get("Heat_Clean", run_data["Heat"]),

                name="Heat",

                fill="tozeroy",

                line=dict(color="orange", width=0),

            ),

            secondary_y=False,

        )



        fig.add_trace(

            go.Scatter(

                x=run_data.index,

                y=run_data.get("Power_Clean", run_data["Power"]),

                name="Power",

                line=dict(color="red", width=1),

            ),

            secondary_y=False,

        )



        if "Indoor_Power" in run_data.columns:

            fig.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["Indoor_Power"],

                    name="Indoor",

                    line=dict(color="purple", width=1, dash="dot"),

                ),

                secondary_y=False,

            )



        if "COP_Graph" in run_data.columns:

            fig.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["COP_Graph"],

                    name="COP",

                    line=dict(color="blue", dash="dot", width=1),

                ),

                secondary_y=True,

            )



        fig.update_layout(

            **tight_layout,

            title="Power & Efficiency",

            hovermode="x unified",

        )

        st.plotly_chart(fig, width="stretch", key="run_power_chart")



    # ------------------------------------------------------------------

    # TAB 2: Hydraulics (incl. Heating during DHW)
    # ------------------------------------------------------------------

    with tab2:

        if selected_run["run_type"] == "DHW":

            detection_source = selected_run.get("heating_during_dhw_detection_source", "none")

            detected_by_zones = (

                selected_run.get("heating_during_dhw_detected")

                if detection_source == "zones"

                else None

            )

            detected_by_power = (

                selected_run.get("heating_during_dhw_power_detected")

                if detection_source == "power"

                else None

            )



            if detection_source != "none":

                detected = bool(detected_by_zones) or bool(detected_by_power)

                if detected:

                    if detection_source == "zones":

                        st.markdown("**Heating during DHW (Zones):** ⚠️ **Detected**")

                    elif detection_source == "power":

                        st.markdown("**Heating during DHW (Indoor Power):** ⚠️ **Detected**")

                    st.caption("⚠️ *Heating zones appear to be active during hot water production. This reduces efficiency.*")

                    st.divider()



        is_dhw = selected_run["run_type"] == "DHW"

        # ΔT chart (own legend)

        if "DeltaT" in run_data.columns:

            fig_dt = go.Figure()

            fig_dt.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["DeltaT"],

                    name="ΔT",

                    line=dict(color="green"),

                )

            )

            fig_dt.add_hline(y=5.0, line_dash="dash", line_color="red")

            fig_dt.update_layout(

                title="Delta T",

                margin=dict(l=10, r=10, t=30, b=10),

                height=200,

                hovermode="x unified",

                showlegend=True,

                legend=dict(

                    orientation="h",

                    yanchor="top",

                    y=-0.12,

                    xanchor="left",

                    x=0,

                ),

            )

            st.plotly_chart(fig_dt, width="stretch", key="run_hydro_dt")



        # Flow Rate chart (own legend)

        if "FlowRate" in run_data.columns:

            fig_flow = go.Figure()

            fig_flow.add_trace(

                go.Scatter(

                    x=run_data.index,

                    y=run_data["FlowRate"],

                    name="Flow",

                    line=dict(color="cyan"),

                )

            )

            fig_flow.update_layout(
                title="Flow Rate",
                margin=dict(l=10, r=10, t=30, b=80),
                height=200,
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.35,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(fig_flow, width="stretch", key="run_hydro_flow")



        # Active Zones chart (own legend)

        zone_cols = [

            c for c in run_data.columns 

            if c.startswith("Zone_") and c != "Zone_Config"

        ]

        has_dhw = "DHW_Active" in run_data.columns



        if zone_cols:

            fig_zones = go.Figure()



            # Build friendly zone labels from user mapping

            zone_labels = {}

            if has_dhw:

                zone_labels["DHW_Active"] = "Hot Water"

            for z in zone_cols:

                zone_labels[z] = _get_friendly_name(z, user_config)



            ordered_keys = []

            if has_dhw:

                ordered_keys.append("DHW_Active")

            ordered_keys.extend(sorted(zone_cols))

            zone_offsets = {key: idx for idx, key in enumerate(ordered_keys)}



            for key in ordered_keys:

                if key not in run_data.columns:

                    continue

                base_y = zone_offsets[key]



                def _zone_active(val):

                    """Check if zone is active, handling both numeric and string values."""

                    if pd.isna(val):

                        return None

                    if isinstance(val, str):

                        v = val.strip().lower()

                        is_active = v in ("on", "true", "1", "yes", "active")

                        return base_y + 0.8 if is_active else None

                    try:

                        is_active = float(val) > 0

                        return base_y + 0.8 if is_active else None

                    except:

                        return None



                y_vals = run_data[key].apply(_zone_active)

                fig_zones.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=y_vals,

                        name=zone_labels[key],

                        mode="lines",

                        line=dict(width=15),

                        connectgaps=False,

                    )

                )



            y_tick_vals = [zone_offsets[key] + 0.4 for key in ordered_keys]

            y_tick_labels = [zone_labels[key] for key in ordered_keys]

            fig_zones.update_yaxes(

                tickvals=y_tick_vals,

                ticktext=y_tick_labels,

                range=[0, len(ordered_keys)],

            )

            fig_zones.update_layout(
                title="Active Zones",
                margin=dict(l=10, r=10, t=30, b=80),
                height=max(220, 140 + 20 * len(ordered_keys)),
                hovermode="x unified",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.35,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(fig_zones, width="stretch", key="run_hydro_zones")



        # DHW / Return temps chart (own legend)

        if is_dhw and ("DHW_Temp" in run_data.columns or "ReturnTemp" in run_data.columns):

            fig_temp = go.Figure()

            if "DHW_Temp" in run_data.columns:

                fig_temp.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["DHW_Temp"],

                        name="DHW Tank",

                        line=dict(color="orange", width=2),

                    )

                )

            if "ReturnTemp" in run_data.columns:

                fig_temp.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["ReturnTemp"],

                        name="Return",

                        line=dict(color="grey", width=1, dash="dot"),

                    )

                )

            fig_temp.update_layout(

                title="Hot Water / Return Temps",

                margin=dict(l=10, r=10, t=30, b=100),

                height=220,

                hovermode="x unified",

                showlegend=True,

                legend=dict(

                    orientation="h",

                    yanchor="top",

                    y=-0.36,

                    xanchor="left",

                    x=0,

                ),

            )

            st.plotly_chart(fig_temp, width="stretch", key="run_hydro_temps")



    # ------------------------------------------------------------------

    # TAB 3: Rooms

    # ------------------------------------------------------------------

    with tab3:

        if selected_run["run_type"] == "Heating":

            fig3 = go.Figure()



            # Dynamically detect zones (exclude Zone_Config which is a string column)

            detected_zones = [

                z for z in run_data.columns 

                if z.startswith("Zone_") and z != "Zone_Config"

            ]



            # Which zones were active?

            active_zones = [

                z for z in detected_zones if run_data[z].sum() > 0

            ]



            # ================================================================

            #   FIXED: USE USER'S rooms_per_zone CONFIGURATION

            # ================================================================

            rooms_per_zone = _get_rooms_per_zone_config(user_config)

            

            # Build allowed rooms from user's configuration

            allowed_rooms = set()

            

            if rooms_per_zone:

                # Use user's explicit zone → rooms mapping

                for z in active_zones:

                    rooms_in_zone = rooms_per_zone.get(z, [])

                    allowed_rooms.update(rooms_in_zone)

            

            # If no explicit mapping or empty result, show all rooms

            if not allowed_rooms:

                allowed_rooms = set([

                    c for c in run_data.columns if c.startswith("Room_")

                ])



            room_cols = [

                c for c in run_data.columns

                if c.startswith("Room_") and c in allowed_rooms

            ]



            deltas = selected_run.get("room_deltas", {}) or {}



            # ================================================================

            #   FIXED: USE FRIENDLY ROOM NAMES FROM USER MAPPING

            # ================================================================

            for col in room_cols:

                # Get friendly name from user mapping (e.g., entity_id)

                friendly_name = _get_friendly_name(col, user_config)

                

                # Check if this room is relevant (has delta data)

                is_relevant = col in deltas



                fig3.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data[col],

                        name=(f"* {friendly_name}" if is_relevant else friendly_name),

                        mode="lines+markers",

                        line=dict(width=3 if is_relevant else 1),

                        opacity=1.0 if is_relevant else 0.5,

                    )

                )



            if "OutdoorTemp" in run_data.columns:

                fig3.add_trace(

                    go.Scatter(

                        x=run_data.index,

                        y=run_data["OutdoorTemp"],

                        name="Outdoor",

                        line=dict(color="grey", width=2, dash="dash"),

                        yaxis="y2",

                    )

                )



            fig3.update_layout(

                title="Temperature Response",

                hovermode="x unified",

                yaxis2=dict(

                    title="Outdoor",

                    overlaying="y",

                    side="right",

                    showgrid=False,

                ),

                height=350,

            )

            st.plotly_chart(fig3, width="stretch", key="run_rooms_chart")

        else:

            st.info("Room temperature analysis is skipped for DHW runs.")



    # ------------------------------------------------------------------

    # TAB 4: AI Data

    # ------------------------------------------------------------------

    with tab4:

        st.markdown("### Single Run AI Context")

        st.info("Copy this JSON to analyze this specific run with the AI.")



        run_date_str = str(selected_run["start"].date())

        active_tag = "baseline_v1"

        active_note = "Initial commissioning."

        # Get history from the loaded profile, not the legacy config
        config_history = user_config.get("config_history", []) if isinstance(user_config, dict) else []
        for entry in sorted(config_history, key=lambda x: str(x.get("start", ""))):
            if run_date_str >= entry["start"]:
                active_tag = entry["config_tag"]
                active_note = entry.get("change_note", "")
            else:
                break

        # Derive tariff name from profile, not legacy config
        tariff_name = "Unknown"
        tariff = user_config.get("tariff_structure")
        if isinstance(tariff, list) and tariff:
            tariff_name = tariff[0].get("name", "Custom")
        elif isinstance(tariff, dict):
            if tariff.get("day_rate") == tariff.get("night_rate", tariff.get("day_rate")):
                tariff_name = "Flat Rate"
            else:
                tariff_name = "Day/Night"


        # Economics

        run_cost = (

            run_data["Cost_Inc"].sum()

            if "Cost_Inc" in run_data.columns

            else 0

        )

        total_rows = len(run_data)



        # Compressor stats

        if total_rows > 0 and "Freq" in run_data.columns:

            pct_low_hz = (

                len(run_data[run_data["Freq"] < 25]) / total_rows * 100

            )

            pct_high_hz = (

                len(run_data[run_data["Freq"] > 45]) / total_rows * 100

            )

            avg_hz = run_data["Freq"].mean()

        else:

            pct_low_hz, pct_high_hz, avg_hz = 0, 0, 0



        # Hydraulics

        avg_flow_temp = (

            run_data["FlowTemp"].mean()

            if "FlowTemp" in run_data.columns

            else 0

        )

        max_flow_temp = (

            run_data["FlowTemp"].max()

            if "FlowTemp" in run_data.columns

            else 0

        )



        # Target flow

        if selected_run["run_type"] == "DHW":

            target_flow = 50.0

        elif "Target_Flow" in run_data.columns:

            target_flow = run_data["Target_Flow"].mean()

        else:

            target_flow = None



        ai_payload = {

            "meta": {

                "report_type": "SINGLE_RUN_INSPECTOR",

                "generated_at": datetime.now(timezone.utc).isoformat(),

                "calc_version": CALC_VERSION,

                "run_id": selected_run["id"],

                "timestamp_start": str(selected_run["start"]),

                "timestamp_end": str(selected_run["end"]),

                "duration_minutes": selected_run["duration_mins"],

                "run_type": selected_run["run_type"],

                "active_zones": selected_run.get("dominant_zones"),

            },

            "system_context": _build_system_context(
                user_config, bool(selected_run.get("heating_during_dhw_detected"))
            ),
            "config_history": user_config.get("config_history", []) if isinstance(user_config, dict) else [],

            "configuration_state": {

                "active_profile_tag": active_tag,

                "change_note_if_new": active_note,
                "tariff_profile": tariff_name,
            },

            "economics": {

                "run_cost_euro": round(run_cost, 3),

                "kwh_electricity": round(

                    selected_run.get("electricity_kwh", 0), 2

                ),

                "kwh_heat": round(selected_run.get("heat_kwh", 0), 2),

                "effective_cop": round(selected_run.get("run_cop", 0), 2),

                "cost_per_kwh_heat_euro": (
                    round(selected_run.get("cost_per_kwh_heat"), 3)
                    if selected_run.get("cost_per_kwh_heat") is not None
                    else None
                ),
                "immersion_kwh_estimated": selected_run.get(
                    "immersion_kwh", 0
                ),
                "immersion_was_active": bool(
                    selected_run.get("immersion_detected", False)
                ),
                "immersion_active_minutes": selected_run.get(
                    "immersion_mins", 0
                ),
            },
            "diagnostics_physics": {

                "avg_flow_rate_lpm": round(selected_run.get("avg_flow_rate", 0), 1),

                "avg_delta_t": round(selected_run.get("avg_dt", 0), 2),

                "avg_flow_temp_c": round(avg_flow_temp, 1),

                "max_flow_temp_c": round(max_flow_temp, 1),

                "avg_return_temp_c": (

                    round(selected_run.get("avg_return_temp"), 1)

                    if selected_run.get("avg_return_temp") is not None

                    else None

                ),

                "min_return_temp_c": (

                    round(selected_run.get("min_return_temp"), 1)

                    if selected_run.get("min_return_temp") is not None

                    else None

                ),

                "return_temp_range_c": (

                    round(selected_run.get("return_temp_range"), 1)

                    if selected_run.get("return_temp_range") is not None

                    else None

                ),

                "target_flow_temp_avg": (

                    round(target_flow, 1) if target_flow else None

                ),

                "compressor_stats": {

                    "avg_hz": round(avg_hz, 1),

                    "min_hz": (

                        round(selected_run.get("min_freq"), 1)

                        if selected_run.get("min_freq") is not None

                        else None

                    ),

                    "max_hz": (

                        round(selected_run.get("max_freq"), 1)

                        if selected_run.get("max_freq") is not None

                        else None

                    ),

                    "std_dev_hz": (

                        round(selected_run.get("freq_std_dev"), 1)

                        if selected_run.get("freq_std_dev") is not None

                        else None

                    ),

                    "pct_time_low_modulation (<25Hz)": round(

                        pct_low_hz, 1

                    ),

                    "pct_time_high_modulation (>45Hz)": round(

                        pct_high_hz, 1

                    ),

                },

            },

            "control_modes": {

                "dhw_mode_value": selected_run.get("dhw_mode", None),

                "quiet_mode_active": selected_run.get(

                    "quiet_mode_active", False

                ),

            },

            "run_characteristics": {

                "is_short_cycle": selected_run.get("is_short_cycle", False),

                "short_cycle_threshold_minutes": THRESHOLDS.get("short_cycle_min", 20),

            },

            "environmental_conditions": {
                "outdoor_temp_avg": round(
                    run_data["OutdoorTemp"].mean(), 1
                )
                if "OutdoorTemp" in run_data.columns
                else None,
                "outdoor_temp_min": (

                    round(selected_run.get("outdoor_temp_min"), 1)

                    if selected_run.get("outdoor_temp_min") is not None

                    else None

                ),

                "outdoor_temp_max": (

                    round(selected_run.get("outdoor_temp_max"), 1)

                    if selected_run.get("outdoor_temp_max") is not None

                    else None

                ),

                "outdoor_temp_change_c": (

                    round(selected_run.get("outdoor_temp_change"), 1)

                    if selected_run.get("outdoor_temp_change") is not None

                    else None

                ),

                "outdoor_humidity_avg": round(

                    run_data["Outdoor_Humidity"].mean(), 1

                )

                if "Outdoor_Humidity" in run_data.columns

                else None,

                "wind_speed_avg": round(
                    run_data["Wind_Speed"].mean(), 1
                )
                if "Wind_Speed" in run_data.columns
                else None,
            },
        }

        if selected_run["run_type"] == "Heating":
            ai_payload["room_response_deltas"] = selected_run.get(
                "room_deltas", {}
            )

        if selected_run["run_type"] == "DHW":
            ai_payload["dhw_temperature_profile"] = {
                "start_c": (
                    round(selected_run.get("dhw_temp_start"), 1)
                    if selected_run.get("dhw_temp_start") is not None
                    else None
                ),
                "end_c": (
                    round(selected_run.get("dhw_temp_end"), 1)
                    if selected_run.get("dhw_temp_end") is not None
                    else None
                ),
                "rise_c": (
                    round(selected_run.get("dhw_rise"), 1)
                    if selected_run.get("dhw_rise") is not None
                    else None
                ),
                "stratification_range_c": (
                    round(selected_run.get("dhw_stratification_range"), 1)
                    if selected_run.get("dhw_stratification_range") is not None
                    else None
                ),
            }

            ai_payload["hydraulic_integrity"] = {
                "heating_during_dhw_detected": (
                    bool(selected_run.get("heating_during_dhw_detected"))
                    if selected_run.get("heating_during_dhw_detected") is not None
                    else None
                ),
                "heating_during_dhw_power_detected": (
                    bool(selected_run.get("heating_during_dhw_power_detected"))
                    if selected_run.get("heating_during_dhw_power_detected") is not None
                    else None
                ),
                "heating_during_dhw_pct": selected_run.get(
                    "heating_during_dhw_pct", 0.0
                ),
            }


        st.json(ai_payload)
