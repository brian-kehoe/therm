# view_runs.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import json
from config import CALC_VERSION, AI_SYSTEM_CONTEXT, TARIFF_PROFILE_ID, CONFIG_HISTORY
from utils import safe_div


def render_run_inspector(df, runs_list):
    st.title("Run Inspector")

    caps = st.session_state.get("capabilities", {})
    has_flowrate = caps.get("has_flowrate", True)

    total_kwh_in = (df["Power"].sum() / 1000) / 60 if "Power" in df.columns else 0
    total_kwh_out = (df["Heat"].sum() / 1000) / 60 if "Heat" in df.columns else 0
    has_heat_data = total_kwh_out > 0
    show_cop = has_flowrate or has_heat_data

    global_cop = safe_div(total_kwh_out, total_kwh_in) if has_heat_data else 0
    global_cop_str = f"{global_cop:.2f}" if has_heat_data else "N/A"

    st.sidebar.markdown("### Global Stats")
    st.sidebar.metric("Runs Detected", len(runs_list) if runs_list else 0)
    st.sidebar.metric("Total Heat Output", f"{total_kwh_out:.1f} kWh" if has_heat_data else "N/A")
    st.sidebar.metric("Total Electricity Input", f"{total_kwh_in:.1f} kWh")
    st.sidebar.metric("Global COP", global_cop_str)

    if not runs_list:
        st.info("No runs detected.")
        return

    runs_list.sort(key=lambda x: x["start"], reverse=True)

    run_options = {}
    for r in runs_list:
        try:
            start_str = r["start"].strftime("%d/%m/%Y %H:%M")
        except Exception:
            start_str = str(r["start"])

        if r["run_type"] == "DHW":
            icon = "💧 DHW"
            if r.get("heating_during_dhw_detected") or r.get("ghost_pumping_power_detected"):
                icon += " ⚠"
        else:
            icon = "🔥 Heating"
        zone_raw = r.get("active_zones", r.get("dominant_zones", "None"))
        zone_label = zone_raw if zone_raw and str(zone_raw).lower() != "none" else "No Zone Data"
        label = f"{start_str} | {r['duration_mins']}m | {icon} ({zone_label})"
        run_options[label] = r

    option_labels = list(run_options.keys())
    if "run_selector_idx" not in st.session_state:
        st.session_state["run_selector_idx"] = 0
    st.session_state["run_selector_idx"] = min(st.session_state["run_selector_idx"], len(option_labels) - 1)

    nav_prev, nav_select, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("◀ Prev", disabled=st.session_state["run_selector_idx"] <= 0):
            st.session_state["run_selector_idx"] = max(0, st.session_state["run_selector_idx"] - 1)
    with nav_next:
        if st.button("Next ▶", disabled=st.session_state["run_selector_idx"] >= len(option_labels) - 1):
            st.session_state["run_selector_idx"] = min(len(option_labels) - 1, st.session_state["run_selector_idx"] + 1)
    with nav_select:
        selected_label = st.selectbox("Select Run", options=option_labels, index=st.session_state["run_selector_idx"])
    st.session_state["run_selector_idx"] = option_labels.index(selected_label)

    selected_run = run_options[selected_label]
    run_data = df.loc[selected_run["start"]: selected_run["end"]]

    # Display Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{selected_run['duration_mins']}m")
    if show_cop:
        c2.metric("COP", f"{selected_run.get('run_cop', 0):.2f}")
    else:
        c2.metric("COP", "N/A")
    c3.metric("Avg Delta T", f"{selected_run.get('avg_dt', 0):.1f} deg")
    c4.metric("Avg Flow", f"{selected_run.get('avg_flow_rate', 0):.1f} L/m")

    tight_layout = dict(margin=dict(l=10, r=10, t=30, b=10), height=350)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Efficiency", "Hydraulics", "Rooms", "AI Data"])

    with tab1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        heat_col = "Heat_Clean" if "Heat_Clean" in run_data.columns else "Heat"
        power_col = "Power_Clean" if "Power_Clean" in run_data.columns else "Power"

        if heat_col in run_data.columns:
            fig.add_trace(go.Scatter(x=run_data.index, y=run_data[heat_col], name="Heat", fill="tozeroy", line=dict(color="orange", width=0)), secondary_y=False)

        if power_col in run_data.columns:
            fig.add_trace(go.Scatter(x=run_data.index, y=run_data[power_col], name="Power", line=dict(color="red", width=1)), secondary_y=False)

        if "Indoor_Power" in run_data.columns:
            fig.add_trace(go.Scatter(x=run_data.index, y=run_data["Indoor_Power"], name="Indoor", line=dict(color="purple", width=1, dash="dot")), secondary_y=False)

        if show_cop and "COP_Graph" in run_data.columns:
            fig.add_trace(go.Scatter(x=run_data.index, y=run_data["COP_Graph"], name="COP", line=dict(color="blue", dash="dot", width=1)), secondary_y=True)

        fig.update_layout(**tight_layout, title="Power & Efficiency", hovermode="x unified")
        st.plotly_chart(fig, width="stretch", key="run_power_chart")

    with tab2:
        is_dhw = selected_run["run_type"] == "DHW"
        num_rows = 4 if is_dhw else 3
        row_titles = ["Delta T", "Flow Rate", "Active Zones"]
        if is_dhw:
            row_titles.append("Hot Water / Return Temps")

        chart_height = 650 if is_dhw else 500
        fig2 = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=row_titles, specs=[[{"secondary_y": False}]] * num_rows)

        if "DeltaT" in run_data.columns:
            fig2.add_trace(go.Scatter(x=run_data.index, y=run_data["DeltaT"], name="Delta T", line=dict(color="green")), row=1, col=1)
            fig2.add_hline(y=5.0, line_dash="dash", line_color="red", row=1, col=1)

        if "FlowRate" in run_data.columns:
            fig2.add_trace(go.Scatter(x=run_data.index, y=run_data["FlowRate"], name="Flow", line=dict(color="cyan")), row=2, col=1)

        # Dynamic zone/flag plotting (supports Zone_*, DHW_Active, ValveMode-derived)
        zone_cols = [c for c in run_data.columns if c.startswith("Zone_") or c == "DHW_Active"]
        zone_cols = sorted(zone_cols)
        if zone_cols:
            zone_offsets = {z: i for i, z in enumerate(zone_cols)}
            for z_col in zone_cols:
                base_y = zone_offsets[z_col]
                series = pd.to_numeric(run_data[z_col], errors="coerce").fillna(0)
                y_vals = series.apply(lambda x: base_y + 0.8 if x > 0 else None)
                fig2.add_trace(
                    go.Scatter(
                        x=run_data.index,
                        y=y_vals,
                        name=z_col.replace("Zone_", "Zone "),
                        mode="lines",
                        line=dict(width=15),
                        connectgaps=False,
                    ),
                    row=3,
                    col=1,
                )
            tick_positions = [v + 0.4 for v in zone_offsets.values()]
            tick_labels = [zc.replace("Zone_", "Zone ") if zc != "DHW_Active" else "Hot Water" for zc in zone_offsets.keys()]
            fig2.update_yaxes(tickvals=tick_positions, ticktext=tick_labels, range=[0, len(zone_cols)+0.5], row=3, col=1)

        if is_dhw:
            if "DHW_Temp" in run_data.columns:
                fig2.add_trace(go.Scatter(x=run_data.index, y=run_data["DHW_Temp"], name="DHW Tank", line=dict(color="orange", width=2)), row=4, col=1)
            if "ReturnTemp" in run_data.columns:
                fig2.add_trace(go.Scatter(x=run_data.index, y=run_data["ReturnTemp"], name="Return", line=dict(color="grey", width=1, dash="dot")), row=4, col=1)

        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=chart_height, hovermode="x unified")
        st.plotly_chart(fig2, width="stretch", key="run_hydro_chart")

    with tab3:
        if selected_run["run_type"] == "Heating":
            fig3 = go.Figure()

            room_cols = [c for c in run_data.columns if c.startswith("Room_")]
            deltas = selected_run.get("room_deltas", {})
            relevant = selected_run.get("relevant_rooms", [])

            for col in room_cols:
                clean_name = col.replace("Room_", "")
                is_relevant = col in relevant if relevant else (col in deltas)

                if run_data[col].notna().any():
                    fig3.add_trace(go.Scatter(
                        x=run_data.index,
                        y=run_data[col],
                        name=f"* {clean_name}" if is_relevant else clean_name,
                        mode="lines+markers",
                        line=dict(width=3 if is_relevant else 1),
                        opacity=1.0 if is_relevant else 0.5
                    ))

            if "OutdoorTemp" in run_data.columns:
                fig3.add_trace(go.Scatter(x=run_data.index, y=run_data["OutdoorTemp"], name="Outdoor", line=dict(color="grey", width=2, dash="dash"), yaxis="y2"))

            fig3.update_layout(title="Temperature Response", hovermode="x unified", yaxis2=dict(title="Outdoor", overlaying="y", side="right", showgrid=False), height=350)
            st.plotly_chart(fig3, width="stretch", key="run_rooms_chart")
        else:
            st.info("Room temperature analysis is skipped for DHW runs.")

    with tab4:
        st.markdown("### Single Run AI Context")
        st.info("Copy this JSON to analyze this specific run with the AI.")

        run_date_str = str(selected_run["start"].date())
        active_tag = "baseline_v1"
        active_note = "Initial commissioning."

        if CONFIG_HISTORY:
            for entry in sorted(CONFIG_HISTORY, key=lambda x: x["start"]):
                if run_date_str >= entry["start"]:
                    active_tag = entry["config_tag"]
                    active_note = entry.get("change_note", "")
                else:
                    break

        run_cost = run_data["Cost_Inc"].sum() if "Cost_Inc" in run_data.columns else 0
        total_rows = len(run_data)

        if total_rows > 0 and "Freq" in run_data.columns:
            pct_low_hz = (len(run_data[run_data["Freq"] < 25]) / total_rows * 100)
            pct_high_hz = (len(run_data[run_data["Freq"] > 45]) / total_rows * 100)
            avg_hz = run_data["Freq"].mean()
        else:
            pct_low_hz, pct_high_hz, avg_hz = 0, 0, 0

        avg_flow_temp = run_data["FlowTemp"].mean() if "FlowTemp" in run_data.columns else 0
        max_flow_temp = run_data["FlowTemp"].max() if "FlowTemp" in run_data.columns else 0

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
                "run_id": selected_run.get("id"),
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
                "tariff_profile": TARIFF_PROFILE_ID
            },
            "economics": {
                "run_cost_euro": round(run_cost, 3),
                "kwh_electricity": round(selected_run.get("electricity_kwh", 0), 2),
                "kwh_heat": round(selected_run.get("heat_kwh", 0), 2),
                "effective_cop": round(selected_run.get("run_cop", 0), 2) if show_cop else None,
                "immersion_kwh_estimated": selected_run.get("immersion_kwh", 0),
                "immersion_was_active": selected_run.get("immersion_mins", 0) > 0,
                "immersion_active_minutes": selected_run.get("immersion_mins", 0)
            },
            "diagnostics_physics": {
                "avg_flow_rate_lpm": round(selected_run.get("avg_flow_rate", 0), 1),
                "avg_delta_t": round(selected_run.get("avg_dt", 0), 2),
                "avg_flow_temp_c": round(avg_flow_temp, 1),
                "max_flow_temp_c": round(max_flow_temp, 1),
                "target_flow_temp_avg": round(target_flow, 1) if target_flow else None,
                "compressor_stats": {
                    "avg_hz": round(avg_hz, 1),
                    "pct_time_low_modulation (<25Hz)": round(pct_low_hz, 1),
                    "pct_time_high_modulation (>45Hz)": round(pct_high_hz, 1)
                }
            },
            "control_modes": {
                "dhw_mode_value": selected_run.get("dhw_mode"),
                "quiet_mode_active": selected_run.get("quiet_mode_active", False)
            },
            "environmental_conditions": {
                "outdoor_temp_avg": round(run_data["OutdoorTemp"].mean(), 1) if "OutdoorTemp" in run_data else None,
                "outdoor_humidity_avg": round(run_data["Outdoor_Humidity"].mean(), 1) if "Outdoor_Humidity" in run_data else None,
                "wind_speed_avg": round(run_data["Wind_Speed"].mean(), 1) if "Wind_Speed" in run_data else None
            },
            "dhw_temperature_profile": {
                "start_c": round(selected_run.get("dhw_temp_start", 0), 1),
                "end_c": round(selected_run.get("dhw_temp_end", 0), 1),
                "rise_c": round(selected_run.get("dhw_rise", 0), 1)
            }
        }

        if selected_run["run_type"] == "Heating":
            ai_payload["room_response_deltas"] = selected_run.get("room_deltas", {})

        if selected_run["run_type"] == "DHW":
            ai_payload["hydraulic_integrity"] = {
                "heating_during_dhw_detected": selected_run.get("heating_during_dhw_detected", False),
                "ghost_pumping_power_detected": selected_run.get("ghost_pumping_power_detected", False),
                "heating_during_dhw_pct": selected_run.get("heating_during_dhw_pct", 0.0)
            }

        st.json(ai_payload)
