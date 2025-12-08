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
    AI_SYSTEM_CONTEXT,
    TARIFF_PROFILE_ID,
    CONFIG_HISTORY,
)
from utils import safe_div


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
    
    # Get the mapped entity_id
    entity_id = str(mapping.get(internal_key, internal_key))
    
    # Strip common Home Assistant prefixes for cleaner display
    if entity_id.startswith("binary_sensor."):
        return entity_id.replace("binary_sensor.", "", 1)
    elif entity_id.startswith("sensor."):
        return entity_id.replace("sensor.", "", 1)
    
    return entity_id


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
        Hydraulics (incl. ghost pumping)
        Rooms
        AI Data (per-run JSON payload)
    """
    st.title(" Run Inspector")
    
    # Get user config from session state for friendly names
    user_config = st.session_state.get("system_config", {})

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

        if r["run_type"] == "DHW":
            icon = ""
            if r.get("heating_during_dhw_detected") or r.get(
                "ghost_pumping_power_detected"
            ):
                icon += "⚠️ (Heating Active)"
            else:
                icon = "♨️"
        else:
            icon = "🔥"

        zone_raw = r.get("active_zones", r.get("dominant_zones", "None"))
        if r["run_type"] == "DHW" and not r.get(
            "heating_during_dhw_detected", False
        ):
            zone_label = "No Zone Data"
        else:
            zone_label = (
                zone_raw if zone_raw and zone_raw.lower() != "none" else "No Zone Data"
            )

        label = (
            f"{start_str} | {r['duration_mins']}m | {icon} "
            f"{r['run_type']} ({zone_label})"
        )
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
        st.plotly_chart(fig, use_container_width=True, key="run_power_chart")

    # ------------------------------------------------------------------
    # TAB 2: Hydraulics (incl. Ghost Pumping)
    # ------------------------------------------------------------------
    with tab2:
        if selected_run["run_type"] == "DHW":
            is_ghost_sensor = selected_run.get(
                "heating_during_dhw_detected", False
            )
            is_ghost_power = selected_run.get(
                "ghost_pumping_power_detected", False
            )

            state_sensor = "⚠️ **Detected**" if is_ghost_sensor else "✅ Clear"
            state_power = "⚠️ **Detected**" if is_ghost_power else "✅ Clear"

            st.markdown(
                f"**Ghost Pumping Check:**   Sensors: {state_sensor}   |   "
                f"Power Proxy: {state_power}"
            )

            if is_ghost_sensor or is_ghost_power:
                st.caption(
                    "⚠️ *Warning: Heating zones appear to be active during "
                    "Hot Water generation. This reduces efficiency.*"
                )

            st.divider()

        is_dhw = selected_run["run_type"] == "DHW"
        num_rows = 4 if is_dhw else 3
        row_titles = ["Delta T", "Flow Rate", "Active Zones"]
        if is_dhw:
            row_titles.append("Hot Water / Return Temps")

        chart_height = 650 if is_dhw else 500

        fig2 = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=row_titles,
            specs=[[{"secondary_y": False}]] * num_rows,
        )

        # ΔT
        if "DeltaT" in run_data.columns:
            fig2.add_trace(
                go.Scatter(
                    x=run_data.index,
                    y=run_data["DeltaT"],
                    name="ΔT",
                    line=dict(color="green"),
                ),
                row=1,
                col=1,
            )
            fig2.add_hline(
                y=5.0,
                line_dash="dash",
                line_color="red",
                row=1,
                col=1,
            )

        # Flow Rate
        if "FlowRate" in run_data.columns:
            fig2.add_trace(
                go.Scatter(
                    x=run_data.index,
                    y=run_data["FlowRate"],
                    name="Flow",
                    line=dict(color="cyan"),
                ),
                row=2,
                col=1,
            )

        # ================================================================
        #   FIXED: DYNAMIC ZONE LABELS WITH USER MAPPING
        # ================================================================
        # Exclude Zone_Config which is a string column for display
        zone_cols = [
            c for c in run_data.columns 
            if c.startswith("Zone_") and c != "Zone_Config"
        ]
        has_dhw = "DHW_Active" in run_data.columns

        # Build friendly zone labels from user mapping
        zone_labels = {}
        
        # DHW label
        if has_dhw:
            zone_labels["DHW_Active"] = "Hot Water"
        
        # Zone labels from user mapping (e.g., "binary_sensor.underfloor_pump")
        for z in zone_cols:
            zone_labels[z] = _get_friendly_name(z, user_config)

        # Order: DHW (if present) first, then sorted zones
        ordered_keys = []
        if has_dhw:
            ordered_keys.append("DHW_Active")
        ordered_keys.extend(sorted(zone_cols))

        # Add thick zone bars
        zone_offsets = {key: idx for idx, key in enumerate(ordered_keys)}

        for key in ordered_keys:
            if key not in run_data.columns:
                continue
            base_y = zone_offsets[key]
            
            # Convert zone values to numeric (handles HA string values like "on"/"off")
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
            fig2.add_trace(
                go.Scatter(
                    x=run_data.index,
                    y=y_vals,
                    name=zone_labels[key],
                    mode="lines",
                    line=dict(width=15),
                    connectgaps=False,
                ),
                row=3,
                col=1,
            )

        # Y-axis labels for zones
        y_tick_vals = [zone_offsets[key] + 0.4 for key in ordered_keys]
        y_tick_labels = [zone_labels[key] for key in ordered_keys]
        
        fig2.update_yaxes(
            tickvals=y_tick_vals,
            ticktext=y_tick_labels,
            range=[0, len(ordered_keys)],
            row=3,
            col=1,
        )

        # DHW / Return temps
        if is_dhw:
            if "DHW_Temp" in run_data.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=run_data.index,
                        y=run_data["DHW_Temp"],
                        name="DHW Tank",
                        line=dict(color="orange", width=2),
                    ),
                    row=4,
                    col=1,
                )
            if "ReturnTemp" in run_data.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=run_data.index,
                        y=run_data["ReturnTemp"],
                        name="Return",
                        line=dict(color="grey", width=1, dash="dot"),
                    ),
                    row=4,
                    col=1,
                )

        fig2.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=chart_height,
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True, key="run_hydro_chart")

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
            st.plotly_chart(fig3, use_container_width=True, key="run_rooms_chart")
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

        for entry in sorted(CONFIG_HISTORY, key=lambda x: x["start"]):
            if run_date_str >= entry["start"]:
                active_tag = entry["config_tag"]
                active_note = entry.get("change_note", "")
            else:
                break

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
            "system_context": AI_SYSTEM_CONTEXT.strip(),
            "configuration_state": {
                "active_profile_tag": active_tag,
                "change_note_if_new": active_note,
                "tariff_profile": TARIFF_PROFILE_ID,
            },
            "economics": {
                "run_cost_euro": round(run_cost, 3),
                "kwh_electricity": round(
                    selected_run.get("electricity_kwh", 0), 2
                ),
                "kwh_heat": round(selected_run.get("heat_kwh", 0), 2),
                "effective_cop": round(selected_run.get("run_cop", 0), 2),
                "immersion_kwh_estimated": selected_run.get(
                    "immersion_kwh", 0
                ),
                "immersion_was_active": selected_run.get(
                    "immersion_detected", False
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
                "target_flow_temp_avg": (
                    round(target_flow, 1) if target_flow else None
                ),
                "compressor_stats": {
                    "avg_hz": round(avg_hz, 1),
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
            "environmental_conditions": {
                "outdoor_temp_avg": round(
                    run_data["OutdoorTemp"].mean(), 1
                )
                if "OutdoorTemp" in run_data.columns
                else None,
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
            "dhw_temperature_profile": {
                "start_c": round(
                    selected_run.get("dhw_temp_start", 0), 1
                ),
                "end_c": round(selected_run.get("dhw_temp_end", 0), 1),
                "rise_c": round(selected_run.get("dhw_rise", 0), 1),
            },
        }

        if selected_run["run_type"] == "Heating":
            ai_payload["room_response_deltas"] = selected_run.get(
                "room_deltas", {}
            )

        if selected_run["run_type"] == "DHW":
            ai_payload["hydraulic_integrity"] = {
                "heating_during_dhw_detected": selected_run.get(
                    "heating_during_dhw_detected", False
                ),
                "ghost_pumping_power_detected": selected_run.get(
                    "ghost_pumping_power_detected", False
                ),
                "heating_during_dhw_pct": selected_run.get(
                    "heating_during_dhw_pct", 0.0
                ),
            }

        st.json(ai_payload)