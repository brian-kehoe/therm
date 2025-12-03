# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import os
import warnings
import json
import streamlit as st

# Suppress noisy RuntimeWarning for sparse data averages
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)

# Import our modularised files
from config import (CALC_VERSION, TARIFF_PROFILE_ID, SENSOR_EXPECTATION_MODE, 
                    THRESHOLDS, SENSOR_ROLES, AI_SYSTEM_CONTEXT, 
                    CONFIG_HISTORY, ZONE_TO_ROOM_MAP, SENSOR_GROUPS)
from utils import safe_div, availability_pct
import data_loader
import processing
import baselines

#st.write("Step 1: App Started")  # <--- Debug Line 1

# Page Setup
st.set_page_config(page_title="Heat Pump Analytics", layout="wide", page_icon="🔥")
pd.set_option('future.no_silent_downcasting', True)
pd.options.display.float_format = lambda x: f"{x:.2f}"

def process_uploaded_files_once(uploaded_files):
    file_key = tuple(sorted((f.name, getattr(f, "size", None)) for f in uploaded_files))
    if "cached_processing" in st.session_state and st.session_state["cached_processing"].get("file_key") == file_key:
        return st.session_state["cached_processing"]

    pbar = st.progress(0, text="Loading data...")
    def update_progress(text, pct): pbar.progress(pct, text=text)

    # 1. Load Data
    result = data_loader.load_and_clean_data(uploaded_files, progress_cb=update_progress)
    if result is None: 
        pbar.empty()
        return None
        
    df_clean = result["df"]
    
    # Store auxiliary data in session state for Baselines tool
    st.session_state["raw_history_df"] = result["raw_history"]
    st.session_state["heartbeat_baseline"] = result["baselines"]
    st.session_state["heartbeat_baseline_path"] = result["baseline_path"]
    st.session_state["sensor_patterns"] = result["patterns"]
    st.session_state["sensor_baselines"] = result["baselines"]

    # 2. Physics & Logic
    pbar.progress(40, text="Analyzing hydraulics...")
    df = processing.apply_gatekeepers(df_clean)
    
    pbar.progress(60, text="Detecting runs...")
    runs_list = processing.detect_runs(df)
    
    pbar.progress(80, text="Compiling daily stats...")
    daily_df = processing.get_daily_stats(df)
    
    pbar.progress(100, text="Done")
    cache = {"file_key": file_key, "df": df, "runs": runs_list, "daily": daily_df}
    st.session_state["cached_processing"] = cache
    return cache

# === MAIN UI ===
st.sidebar.title("Heat Pump Analytics")
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", accept_multiple_files=True, type="csv")

if uploaded_files:
    cached = process_uploaded_files_once(uploaded_files)
    #st.write("Step 2: Processing Complete")  # <--- Debug Line 2
    if cached and cached.get("df") is not None:
        df = cached["df"]
        runs_list = cached["runs"]
        daily_df = cached["daily"]
        
        mode = st.sidebar.radio("Analysis Mode", ["Long-Term Trends", "Run Inspector", "Data Quality Audit"])
        
        # ==========================================
        # 1. LONG-TERM TRENDS
        # ==========================================
        if mode == "Long-Term Trends":
            st.title("📈 Long-Term Performance")
            if not daily_df.empty:
                tab_main, tab_ai = st.tabs(["Performance", "AI Report"])
                
                with tab_main:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Days", len(daily_df))
                    k2.metric("Total Heat", f"{daily_df['Total_Heat_kWh'].sum():.0f} kWh")
                    k3.metric("Total Cost", f"€{daily_df['Daily_Cost_Euro'].sum():.2f}")
                    scop = safe_div(daily_df['Total_Heat_kWh'].sum(), daily_df['Total_Electricity_kWh'].sum())
                    k4.metric("Period SCOP", f"{scop:.2f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=daily_df.index, y=daily_df['Heat_Heating_kWh'], name='Heat', marker_color='#ffa600'))
                    fig.add_trace(go.Bar(x=daily_df.index, y=daily_df['Electricity_Heating_kWh'], name='Elec', marker_color='#003f5c'))
                    fig.update_layout(title="Daily Energy", barmode='group')
                    st.plotly_chart(fig, width="stretch")

                    c1, c2 = st.columns(2)
                    with c1:
                        fig_env = make_subplots(specs=[[{"secondary_y": True}]])
                        if 'Wind_Avg' in daily_df:
                            fig_env.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Wind_Avg'], name="Wind", line=dict(color='grey')), secondary_y=False)
                        if 'Humidity_Avg' in daily_df:
                            fig_env.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Humidity_Avg'], name="Humidity", line=dict(color='blue', dash='dot')), secondary_y=True)
                        fig_env.update_layout(title="Wind & Humidity", height=300)
                        st.plotly_chart(fig_env, width="stretch")
                    with c2:
                        fig_sol = make_subplots(specs=[[{"secondary_y": True}]])
                        if 'Solar_Avg' in daily_df:
                            fig_sol.add_trace(go.Bar(x=daily_df.index, y=daily_df['Solar_Avg'], name="Solar", marker_color='orange'), secondary_y=False)
                        fig_sol.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Global_SCOP'], name="SCOP", line=dict(color='green')), secondary_y=True)
                        fig_sol.update_layout(title="Solar Gain vs Efficiency", height=300)
                        st.plotly_chart(fig_sol, width="stretch")
                    
                    st.subheader("Optimization: Weather Compensation Curve")
                    
                    # Sample data for performance
                    scatter_data = df[(df['is_active']) & (~df['is_DHW'])].sample(frac=0.1) 
                    
                    fig_wc = px.scatter(
                        scatter_data, 
                        x='OutdoorTemp', 
                        y='FlowTemp', 
                        color='COP_Graph',
                        color_continuous_scale='RdYlGn',
                        title="Flow Temp vs Outdoor Temp",
                        opacity=0.6,
                        labels={'OutdoorTemp': 'Outdoor Temp (°C)', 'FlowTemp': 'Flow Temp (°C)'}
                    )

                    # Improved "Inefficient Zone" Visual
                    fig_wc.add_shape(
                        type="rect",
                        x0=10, y0=43, x1=20, y1=60,
                        line=dict(width=0),       # No border line
                        fillcolor="Red",
                        opacity=0.08,             # Very subtle background tint
                        layer="below"             # Sit behind the data points
                    )

                    # Clean Label inside the zone
                    fig_wc.add_annotation(
                        x=15, y=58,               # Centered near the top of the zone
                        text="Inefficient Zone (>43°C)",
                        showarrow=False,
                        font=dict(color="darkred", size=10),
                        bgcolor="rgba(255, 255, 255, 0.5)" # Semi-transparent backing for readability
                    )
                    
                    st.plotly_chart(fig_wc, width="stretch")

                with tab_ai:
                    # 1. Prepare Data
                    json_ready = daily_df.copy().reset_index()
                    json_ready = json_ready.rename(columns={json_ready.columns[0]: 'date'})
                    json_ready['date'] = json_ready['date'].astype(str)
                    
                    float_cols = json_ready.select_dtypes(include=[float]).columns
                    json_ready[float_cols] = json_ready[float_cols].round(2)
                    
                    period_summary = {
                        "days": int(len(json_ready)),
                        "total_heat_kwh": round(float(daily_df['Total_Heat_kWh'].sum()), 2),
                        "total_electricity_kwh": round(float(daily_df['Total_Electricity_kWh'].sum()), 2),
                        "total_cost_eur": round(float(daily_df['Daily_Cost_Euro'].sum()), 2),
                        "period_scop": round(float(safe_div(daily_df['Total_Heat_kWh'].sum(), daily_df['Total_Electricity_kWh'].sum())), 2)
                    }
                    
                    ai_payload = {
                        "meta": {
                            "report_type": "LONG_TERM_TRENDS", 
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "calc_version": CALC_VERSION
                        }, 
                        "system_context": AI_SYSTEM_CONTEXT, 
                        "period_summary": period_summary, 
                        "daily_metrics": json_ready.to_dict(orient='records')
                    }
                    
                    # 2. Add Download Button (FIXED)
                    st.download_button(
                        label="📥 Download JSON for AI Analysis",
                        data=json.dumps(ai_payload, indent=2),
                        file_name=f"heat_pump_long_term_ai_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

                    # 3. Display JSON
                    st.json(ai_payload)
            else:
                st.warning("No valid daily data found.")

        # ==========================================
        # 2. RUN INSPECTOR (Restored Visuals)
        # ==========================================
        elif mode == "Run Inspector":
            st.title("🔍 Run Inspector")
            
            # --- Sidebar Global Stats ---
            total_kwh_in = (df['Power_Clean'].sum() / 1000) / 60
            total_kwh_out = (df['Heat_Clean'].sum() / 1000) / 60
            global_cop = safe_div(total_kwh_out, total_kwh_in)
            
            st.sidebar.markdown("### Global Stats")
            st.sidebar.metric("Runs Detected", len(runs_list) if runs_list else 0)
            st.sidebar.metric("Total Heat Output", f"{total_kwh_out:.1f} kWh")
            st.sidebar.metric("Total Electricity Input", f"{total_kwh_in:.1f} kWh")
            st.sidebar.metric("Global COP", f"{global_cop:.2f}")

            runs_list = runs_list or []
            if runs_list:
                runs_list.sort(key=lambda x: x['start'], reverse=True)
                run_options = {}
                for r in runs_list:
                    start_str = r['start'].strftime('%d/%m %H:%M')
                    
                    if r['run_type'] == "DHW":
                        icon = "🚿"
                        if r.get('heating_during_dhw_detected') or r.get('ghost_pumping_power_detected'):
                            icon += "⚠️ (Heating Active)"
                    else:
                        icon = "♨️"
                        
                    zone_raw = r.get('active_zones', r.get('dominant_zones', 'None'))
                    if r['run_type'] == "DHW" and not r.get('heating_during_dhw_detected', False):
                        zone_label = "No Zone Data"
                    else:
                        zone_label = zone_raw if zone_raw and zone_raw.lower() != "none" else "No Zone Data"
                    
                    label = f"{start_str} | {r['duration_mins']}m | {icon} {r['run_type']} ({zone_label})"
                    run_options[label] = r

                option_labels = list(run_options.keys())
                if "run_selector_idx" not in st.session_state: st.session_state["run_selector_idx"] = 0
                st.session_state["run_selector_idx"] = min(st.session_state["run_selector_idx"], len(option_labels) - 1)

                nav_prev, nav_select, nav_next = st.columns([1, 4, 1])
                with nav_prev:
                    if st.button("Previous", disabled=st.session_state["run_selector_idx"] <= 0):
                        st.session_state["run_selector_idx"] = max(0, st.session_state["run_selector_idx"] - 1)
                with nav_next:
                    if st.button("Next", disabled=st.session_state["run_selector_idx"] >= len(option_labels) - 1):
                        st.session_state["run_selector_idx"] = min(len(option_labels) - 1, st.session_state["run_selector_idx"] + 1)
                with nav_select:
                    selected_label = st.selectbox("Select Run", options=option_labels, index=st.session_state["run_selector_idx"])
                st.session_state["run_selector_idx"] = option_labels.index(selected_label)

                selected_run = run_options[selected_label]
                run_data = df.loc[selected_run['start'] : selected_run['end']]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Duration", f"{selected_run['duration_mins']}m")
                c2.metric("COP", f"{selected_run['run_cop']:.2f}")
                c3.metric("Avg ΔT", f"{selected_run['avg_dt']:.1f}°")
                c4.metric("Avg Flow", f"{selected_run['avg_flow']:.1f} L/m")
                
                # Layout helper for consistent chart styling
                tight_layout = dict(margin=dict(l=10, r=10, t=30, b=10), height=350)

                tab1, tab2, tab3, tab4 = st.tabs(["⚡ Efficiency", "💧 Hydraulics", "🏠 Rooms", "🤖 AI Data"])
                
                with tab1:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Heat_Clean'], name="Heat", fill='tozeroy', line=dict(color='orange', width=0)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Power_Clean'], name="Power", line=dict(color='red', width=1)), secondary_y=False)
                    if 'Indoor_Power' in run_data.columns:
                        fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Indoor_Power'], name="Indoor", line=dict(color='purple', width=1, dash='dot')), secondary_y=False)
                    fig.add_trace(go.Scatter(x=run_data.index, y=run_data['COP_Graph'], name="COP", line=dict(color='blue', dash='dot', width=1)), secondary_y=True)
                    fig.update_layout(**tight_layout, title="Power & Efficiency", hovermode="x unified")
                    st.plotly_chart(fig, width="stretch")
                
                with tab2:
                    if selected_run['run_type'] == "DHW":
                        k1, k2 = st.columns(2)
                        is_ghost_sensor = selected_run.get('heating_during_dhw_detected', False)
                        is_ghost_power = selected_run.get('ghost_pumping_power_detected', False)
                        k1.metric("Ghost (Sensors)", "Detected" if is_ghost_sensor else "Clear", delta="⚠️" if is_ghost_sensor else None, delta_color="inverse")
                        k2.metric("Ghost (Power Proxy)", "Detected" if is_ghost_power else "Clear", help=f"Threshold: {THRESHOLDS['ghost_power_threshold']}W", delta="⚠️" if is_ghost_power else None, delta_color="inverse")

                    # --- RESTORED: 4-Row Forensic Subplot ---
                    fig2 = make_subplots(
                        rows=4, 
                        cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Delta T", "Flow Rate", "Active Zones", "Hot Water / Return Temps"),
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
                    )
                    
                    # Row 1: Delta T
                    fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['DeltaT'], name="ΔT", line=dict(color='green')), row=1, col=1)
                    fig2.add_hline(y=5.0, line_dash="dash", line_color="red", row=1, col=1)
                    
                    # Row 2: Flow Rate
                    if 'FlowRate' in run_data.columns: 
                        fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['FlowRate'], name="Flow", line=dict(color='cyan')), row=2, col=1)
                    
                    # Row 3: Active Zones
                    zone_map = {'DHW_Active': 'Hot Water', 'Zone_UFH': 'Kitchen', 'Zone_DS': 'Downstairs', 'Zone_US': 'Upstairs'}
                    zone_offsets = {'DHW_Active': 0, 'Zone_UFH': 1, 'Zone_DS': 2, 'Zone_US': 3}
                    for z_col, z_name in zone_map.items():
                        if z_col in run_data.columns:
                            base_y = zone_offsets[z_col]
                            y_vals = run_data[z_col].apply(lambda x: base_y + 0.8 if x > 0 else None)
                            fig2.add_trace(go.Scatter(x=run_data.index, y=y_vals, name=z_name, mode='lines', line=dict(width=15), connectgaps=False), row=3, col=1)
                    fig2.update_yaxes(tickvals=[0.4, 1.4, 2.4, 3.4], ticktext=["Hot Water", "Kitchen", "Downstairs", "Upstairs"], range=[0, 4], row=3, col=1)
                    
                    # Row 4: DHW / Return Temps (Restored)
                    dhw_avail = 'DHW_Temp' in run_data.columns and run_data['DHW_Temp'].notna().any()
                    ret_avail = 'ReturnTemp' in run_data.columns and run_data['ReturnTemp'].notna().any()
                    
                    if dhw_avail:
                        fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['DHW_Temp'], name="DHW Tank", line=dict(color='orange', width=2)), row=4, col=1)
                    if ret_avail:
                        fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['ReturnTemp'], name="Return", line=dict(color='grey', width=1, dash='dot')), row=4, col=1)
                    
                    if not (dhw_avail or ret_avail):
                        fig2.update_yaxes(showticklabels=False, row=4, col=1)
                    else:
                        fig2.update_yaxes(title_text="Temp (°C)", row=4, col=1)

                    fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=650, hovermode="x unified")
                    st.plotly_chart(fig2, width="stretch")
                
                with tab3:
                    if selected_run['run_type'] == "Heating":
                        fig3 = go.Figure()
                        
                        # --- RESTORED: Smart Room Filtering & Styling ---
                        active_zones = [z for z in ['Zone_UFH', 'Zone_DS', 'Zone_US'] if z in run_data.columns and run_data[z].sum() > 0]
                        allowed_rooms = set()
                        for z in active_zones:
                            allowed_rooms.update(ZONE_TO_ROOM_MAP.get(z, []))
                        
                        room_cols = [c for c in run_data.columns if c.startswith('Room_')]
                        if allowed_rooms:
                            room_cols = [c for c in room_cols if c in allowed_rooms]

                        # Use 'room_deltas' to highlight responding rooms
                        deltas = selected_run.get('room_deltas', {})
                        
                        for col in room_cols:
                            clean_name = col.replace("Room_", "")
                            is_relevant = col in deltas
                            
                            fig3.add_trace(go.Scatter(
                                x=run_data.index, 
                                y=run_data[col], 
                                name=f"* {clean_name}" if is_relevant else clean_name,
                                mode='lines+markers',
                                line=dict(width=3 if is_relevant else 1, dash='solid'),
                                opacity=1.0 if is_relevant else 0.5,
                                marker=dict(size=4 if is_relevant else 0)
                            ))
                            
                        # Restore Outdoor Temp Overlay
                        if 'OutdoorTemp' in run_data.columns:
                             fig3.add_trace(go.Scatter(x=run_data.index, y=run_data['OutdoorTemp'], name="Outdoor", line=dict(color='grey', width=2, dash='dash'), yaxis="y2"))

                        # Restore Layout
                        fig3.update_layout(
                            **tight_layout,
                            title="Temperature Response",
                            hovermode="x unified",
                            yaxis2=dict(title="Outdoor", overlaying="y", side="right", showgrid=False),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0)
                        )
                        st.plotly_chart(fig3, width="stretch")
                    else:
                        st.info("Room temperature analysis is skipped for DHW runs.")

                with tab4:
                    st.markdown("### 🤖 Single Run AI Context")
                    st.info("Copy this JSON to analyze this specific run with the AI.")

                    run_date_str = str(selected_run['start'].date())
                    active_tag = "baseline_v1"
                    active_note = "Initial commissioning."
                    for entry in sorted(CONFIG_HISTORY, key=lambda x: x["start"]):
                        if run_date_str >= entry["start"]:
                            active_tag = entry["config_tag"]
                            active_note = entry.get("change_note", "")
                        else:
                            break

                    run_cost = run_data['Cost_Inc'].sum() if 'Cost_Inc' in run_data.columns else 0
                    
                    total_rows = len(run_data)
                    if total_rows > 0 and 'Freq' in run_data.columns:
                        pct_low_hz = (len(run_data[run_data['Freq'] < 25]) / total_rows * 100)
                        pct_high_hz = (len(run_data[run_data['Freq'] > 45]) / total_rows * 100)
                        avg_hz = run_data['Freq'].mean()
                    else:
                        pct_low_hz = 0
                        pct_high_hz = 0
                        avg_hz = 0

                    ai_payload = {
                        "meta": {
                            "report_type": "SINGLE_RUN_INSPECTOR",
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "calc_version": CALC_VERSION,
                            "run_id": selected_run['id'],
                            "timestamp_start": str(selected_run['start']),
                            "timestamp_end": str(selected_run['end']),
                            "duration_minutes": selected_run['duration_mins'],
                            "run_type": selected_run['run_type'],
                            "active_zones": selected_run.get('dominant_zones'),
                        },
                        "system_context": AI_SYSTEM_CONTEXT.strip(),
                        "configuration_state": {
                            "active_profile_tag": active_tag,
                            "change_note_if_new": active_note,
                            "tariff_profile": TARIFF_PROFILE_ID
                        },
                        "economics": {
                            "run_cost_euro": round(run_cost, 3),
                            "kwh_electricity": round(selected_run.get('electricity_kwh', 0), 2),
                            "kwh_heat": round(selected_run.get('heat_kwh', 0), 2),
                            "effective_cop": round(selected_run.get('run_cop', 0), 2),
                            "immersion_kwh_estimated": selected_run.get('immersion_kwh', 0),
                            "immersion_was_active": selected_run.get('immersion_detected', False),
                            "immersion_active_minutes": selected_run.get('immersion_mins', 0)
                        },
                        "diagnostics_physics": {
                            "avg_flow_rate_lpm": round(selected_run.get('avg_flow', 0), 1),
                            "avg_delta_t": round(selected_run.get('avg_dt', 0), 2),
                            "avg_flow_temp_c": round(run_data['FlowTemp'].mean(), 1) if 'FlowTemp' in run_data else None,
                            "max_flow_temp_c": round(run_data['FlowTemp'].max(), 1) if 'FlowTemp' in run_data else None,
                            "target_flow_temp_avg": (
                                50.0 if selected_run['run_type'] == "DHW"
                                else round(run_data['Target_Flow'].mean(), 1) if 'Target_Flow' in run_data
                                else None
                            ),
                            "compressor_stats": {
                                "avg_hz": round(avg_hz, 1),
                                "pct_time_low_modulation (<25Hz)": round(pct_low_hz, 1),
                                "pct_time_high_modulation (>45Hz)": round(pct_high_hz, 1)
                            }
                        },
                        "control_modes": {
                            "dhw_mode_value": selected_run.get('dhw_mode', None),
                            "quiet_mode_active": selected_run.get('quiet_mode_active', False)
                        },
                        "environmental_conditions": {
                            "outdoor_temp_avg": round(run_data['OutdoorTemp'].mean(), 1) if 'OutdoorTemp' in run_data else None,
                            "outdoor_humidity_avg": round(run_data['Outdoor_Humidity'].mean(), 1) if 'Outdoor_Humidity' in run_data else None,
                            "wind_speed_avg": round(run_data['Wind_Speed'].mean(), 1) if 'Wind_Speed' in run_data else None
                        },
                        "dhw_temperature_profile": {
                            "start_c": round(selected_run.get('dhw_temp_start', 0), 1),
                            "end_c": round(selected_run.get('dhw_temp_end', 0), 1),
                            "rise_c": round(selected_run.get('dhw_rise', 0), 1)
                        }
                    }

                    if selected_run['run_type'] == "Heating":
                        ai_payload["room_response_deltas"] = selected_run.get('room_deltas', {})

                    if selected_run['run_type'] == "DHW":
                        ai_payload["hydraulic_integrity"] = {
                            "heating_during_dhw_detected": selected_run.get('heating_during_dhw_detected', False),
                            "ghost_pumping_power_detected": selected_run.get('ghost_pumping_power_detected', False),
                            "heating_during_dhw_pct": selected_run.get('heating_during_dhw_pct', 0.0)
                        }

                    st.json(ai_payload)

            else:
                st.info("No runs detected.")

        # ==========================================
        # 3. DATA QUALITY (Events Excluded from Overview)
        # ==========================================
        elif mode == "Data Quality Audit":
            st.title("🛡️ Data Quality Studio")
            if not daily_df.empty:
                system_on_minutes = daily_df.apply(lambda r: max(r.get('Recorded_Minutes', 0), r.get('DQ_Power_Count', 0), 1), axis=1)
                
                def expected_window_series(sensor_name, system_on_minutes):
                    # 1. Try to get a smart target from the learned Baseline
                    # (This aligns the score with what is "normal" for this specific sensor)
                    baseline_data = st.session_state.get("heartbeat_baseline", {}).get(sensor_name)
                    
                    if baseline_data and baseline_data.get('expected_minutes'):
                        # Calculate expected ratio (e.g. 0.05 updates per minute)
                        # We use 1440.0 as the standard day reference for the baseline
                        baseline_ratio = baseline_data['expected_minutes'] / 1440.0
                        
                        # Apply that ratio to the actual active time today
                        # If system was on for 1000 mins, and sensor usually reports 10% of the time, expect 100 updates.
                        base = system_on_minutes * baseline_ratio
                        
                        # Ensure we expect at least 1 update if the system was on
                        return base.replace(0, np.nan).apply(lambda x: max(1.0, x) if x > 0 else np.nan)

                    # 2. Fallback: Use Manual Rules (if no baseline exists yet)
                    mode = SENSOR_EXPECTATION_MODE.get(sensor_name, 'system')
                    
                    if mode == 'heating_active': 
                        base = daily_df.get('Active_Mins', system_on_minutes)
                    elif mode == 'dhw_active': 
                        base = daily_df.get('Total_DHW_Mins', system_on_minutes)
                    elif mode == 'system_slow':
                        # Manual fallback for slow sensors
                        base = (system_on_minutes / 60.0).apply(np.ceil)
                    else: 
                        # Default 'system' = expect 1 update per minute
                        base = system_on_minutes
                        
                    return base.replace(0, np.nan)

                # --- Helper to Format Index ---
                def format_dq_df(df_in):
                    df_out = df_in.copy()
                    df_out.index.name = "Date"
                    df_out.index = df_out.index.strftime('%d-%m-%Y')
                    return df_out

                dq_tab1, dq_tab2, dq_tab3, dq_tab4 = st.tabs(["Overview", "Category Drill-Down", "All Sensors", "Heartbeats"])

                with dq_tab1:
                    st.markdown("### System Health Scorecard")
                    
                    dq_avg = daily_df['DQ_Score'].mean()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Average Data Health", f"{dq_avg:.1f}%", delta="Target: 100%")
                    
                    gold_days = len(daily_df[daily_df['DQ_Tier'].astype(str).str.contains("Gold", na=False)])
                    silver_days = len(daily_df[daily_df['DQ_Tier'].astype(str).str.contains("Silver", na=False)])
                    
                    c2.metric("Gold Days", gold_days, help="Full Power + Heat Meter Data")
                    c3.metric("Silver Days", silver_days, help="Power + Hydraulic Calculation Data")

                    # Dynamic Overview Table based on SENSOR_GROUPS
                    overview_df = daily_df[['DQ_Tier']].copy()
                    
                    # Select one representative sensor per group to act as the "Group Health" proxy
                    group_cols = []
                    
                    for group_name, sensors in SENSOR_GROUPS.items():
                        # --- FIX: Explicitly Skip 'Events' from Overview ---
                        if "Events" in group_name or "Event" in group_name:
                            continue

                        # Find valid columns for this group
                        valid_sensors = [s for s in sensors if f"DQ_{s}_Count" in daily_df.columns or f"{s}_count" in daily_df.columns]
                        if not valid_sensors:
                            continue
                            
                        # Calculate availability for ALL sensors in the group and take the MEAN
                        group_pcts = []
                        for s in valid_sensors:
                            col = f"DQ_{s}_Count" if f"DQ_{s}_Count" in daily_df.columns else f"{s}_count"
                            pct = availability_pct(daily_df[col], expected_window_series(s, system_on_minutes))
                            group_pcts.append(pct)
                        
                        # Create a composite score column for the group
                        if group_pcts:
                            avg_group_pct = pd.concat(group_pcts, axis=1).mean(axis=1).round(0)
                            overview_df[group_name] = avg_group_pct
                            group_cols.append(group_name)

                    # Apply formatting helper
                    overview_disp = format_dq_df(overview_df[['DQ_Tier'] + group_cols])

                    st.dataframe(
                        overview_disp.style
                        .background_gradient(subset=group_cols, cmap='RdYlGn', vmin=0, vmax=100)
                        .format("{:.0f}", subset=group_cols),
                        width="stretch"
                    )

                with dq_tab2:
                    st.markdown("### Category Inspector")
                    
                    cat_options = list(SENSOR_GROUPS.keys())
                    cat = st.selectbox("Select System Category", cat_options)
                    
                    selected_sensors = SENSOR_GROUPS.get(cat, [])
                    cat_df = pd.DataFrame(index=daily_df.index)
                    
                    valid_cols_for_display = []
                    for sensor in selected_sensors:
                        col_name = f"DQ_{sensor}_Count" if f"DQ_{sensor}_Count" in daily_df.columns else f"{sensor}_count"
                        if col_name in daily_df.columns:
                            # Pass raw count for events/defrost, percentage for others
                            if 'defrost' in sensor.lower():
                                cat_df[sensor] = daily_df[col_name].fillna(0).astype(int)
                            else:
                                expected = expected_window_series(sensor, system_on_minutes)
                                cat_df[sensor] = availability_pct(daily_df[col_name], expected).round(0)
                            valid_cols_for_display.append(sensor)
                    
                    # Apply formatting helper
                    cat_disp = format_dq_df(cat_df)

                    styler = cat_disp.style.format("{:.0f}", na_rep="N/A")
                    
                    # Gradient for standard sensors
                    normal_cols = [c for c in valid_cols_for_display if 'defrost' not in c.lower()]
                    if normal_cols:
                        styler = styler.background_gradient(subset=normal_cols, cmap='RdYlGn', vmin=0, vmax=100)
                    
                    # Grey for events
                    defrost_cols = [c for c in valid_cols_for_display if 'defrost' in c.lower()]
                    if defrost_cols:
                        styler = styler.map(lambda x: "background-color: #e0e0e0; color: #555555", subset=defrost_cols)

                    st.dataframe(styler, width="stretch")

                with dq_tab3:
                    st.markdown("### 🧬 Master Sensor Matrix")
                    st.caption("Grouped by system function for easier scanning.")
                    
                    count_cols = [c for c in daily_df.columns if c.endswith('_count') or c.endswith('_Count')]
                    if count_cols:
                        # 1. Calculate raw percentages
                        # 1. Calculate raw percentages
                        flat_data = {}
                        for c in count_cols:
                            clean_name = c.replace('DQ_', '').replace('_Count', '').replace('_count', '')
                            
                            # SKIP short cycles calc
                            if 'short_cycle' in clean_name.lower(): 
                                continue
                            
                            # Check the CONFIG for the mode
                            mode = SENSOR_EXPECTATION_MODE.get(clean_name, 'system')
                            
                            # If it's an Event (like Defrost, ValveMode), just show raw count
                            if mode == 'event_only':
                                flat_data[clean_name] = daily_df[c].fillna(0).astype(int)
                            else:
                                expected = expected_window_series(clean_name, system_on_minutes)
                                flat_data[clean_name] = availability_pct(daily_df[c], expected).round(0)
                        
                        df_flat = pd.DataFrame(flat_data, index=daily_df.index)

                        # 2. Build Multi-Index Columns
                        new_columns = []
                        valid_data_cols = []
                        
                        events_group_name = None
                        events_sensors = []

                        # A. Process defined groups (EXCEPT Events)
                        for cat_name, sensors in SENSOR_GROUPS.items():
                            if "Event" in cat_name:
                                events_group_name = cat_name
                                events_sensors = sensors
                                continue

                            # FIX: Removed 'sorted()' to respect the order in config.py
                            found_sensors = [s for s in sensors if s in df_flat.columns]
                            
                            for s in found_sensors:
                                new_columns.append((cat_name, s))
                                valid_data_cols.append(s)

                        # B. Process Rooms (Keep alphabetical as it makes sense for rooms)
                        room_cols = sorted([c for c in df_flat.columns if c.startswith('Room_') and c not in valid_data_cols])
                        for r in room_cols:
                            short_name = r.replace('Room_', '')
                            new_columns.append(("🌡️ Rooms", short_name))
                            valid_data_cols.append(r)

                        # C. Process 'Other' (Keep alphabetical)
                        remaining = sorted([c for c in df_flat.columns if c not in valid_data_cols and (not events_sensors or c not in events_sensors)])
                        for rem in remaining:
                            new_columns.append(("Other", rem))
                            valid_data_cols.append(rem)

                        # D. Process Events (LAST)
                        if events_group_name:
                            found_events = [s for s in events_sensors if s in df_flat.columns]
                            for s in found_events:
                                new_columns.append((events_group_name, s))
                                valid_data_cols.append(s)

                        # 3. Construct Final DataFrame
                        df_final = df_flat[valid_data_cols].copy()
                        df_final.columns = pd.MultiIndex.from_tuples(new_columns)
                        df_final = format_dq_df(df_final)

                        # 4. Render with Styling
                        # Identify 'Event' columns dynamically based on Config
                        event_cols = [
                            col for col in df_final.columns 
                            if SENSOR_EXPECTATION_MODE.get(col[1], 'system') == 'event_only'
                        ]
                        normal_cols = [col for col in df_final.columns if col not in event_cols]

                        styler = df_final.style.format("{:.0f}", na_rep="-")
                        
                        if normal_cols:
                            styler = styler.background_gradient(subset=normal_cols, cmap='RdYlGn', vmin=0, vmax=100)
                        
                        if event_cols:
                            # Apply Neutral Grey to all Event sensors (Valve, DHW Mode, Defrost)
                            styler = styler.map(lambda x: "background-color: #e0e0e0; color: #555555", subset=event_cols)

                        st.dataframe(styler, width="stretch")

                with dq_tab4:
                    st.markdown("### ❤️ Sensor Heartbeats")
                    st.caption("Baseline vs Current Reporting Intervals")

                    if st.button("Generate Heartbeat Baseline"):
                        history_df = st.session_state.get("raw_history_df")
                        if history_df is None or history_df.empty:
                            st.warning("No raw history available. Please load data first.")
                        else:
                            with st.spinner("Building offline-aware seasonal baselines..."):
                                days_in_history = history_df['last_changed'].dt.normalize().nunique()
                                bl = baselines.build_offline_aware_seasonal_baseline(history_df, SENSOR_ROLES)
                                path = baselines.save_heartbeat_baseline_to_json(bl, None, days_in_history=days_in_history)
                                st.success(f"Baseline saved: {os.path.basename(path)}")
                                st.session_state["heartbeat_baseline"] = bl
                    
                    patterns = st.session_state.get("sensor_patterns", {})
                    if patterns:
                        pat_data = []
                        for sensor, details in patterns.items():
                            pat_data.append({
                                "Sensor": sensor,
                                "Type": details['report_type'],
                                "Interval (s)": round(details['normal_interval_sec'], 1),
                                "Gap Limit (s)": round(details['gap_threshold_sec'], 1)
                            })
                        st.dataframe(pd.DataFrame(pat_data), width="stretch")

else:
    st.info("Please upload Grafana CSV exports to begin.")