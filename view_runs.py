#view_runs.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd # Needed for to_numeric
from plotly.subplots import make_subplots
from config import CALC_VERSION, AI_SYSTEM_CONTEXT, TARIFF_PROFILE_ID, CONFIG_HISTORY
from utils import safe_div
import json
from datetime import datetime, timezone

def render_run_inspector(df, runs_list):
    st.title("🔍 Run Inspector")
    
    # Retrieve mapping for friendly names
    # Config is stored as {Internal_Key: User_Sensor_Name}
    user_mapping = st.session_state.get("system_config", {}).get("mapping", {})
    
    def get_friendly_name(internal_key):
        val = user_mapping.get(internal_key, internal_key)
        return str(val)

    if 'Power_Clean' in df.columns:
        total_kwh_in = (df['Power_Clean'].sum() / 1000) / 60
    else:
        total_kwh_in = 0
        
    if 'Heat_Clean' in df.columns:
        total_kwh_out = (df['Heat_Clean'].sum() / 1000) / 60
    else:
        total_kwh_out = 0
        
    global_cop = safe_div(total_kwh_out, total_kwh_in)
    
    st.sidebar.markdown("### Global Stats")
    st.sidebar.metric("Runs Detected", len(runs_list) if runs_list else 0)
    st.sidebar.metric("Total Heat Output", f"{total_kwh_out:.1f} kWh")
    st.sidebar.metric("Total Electricity Input", f"{total_kwh_in:.1f} kWh")
    st.sidebar.metric("Global COP", f"{global_cop:.2f}")

    if not runs_list:
        st.info("No runs detected.")
        return

    runs_list.sort(key=lambda x: x['start'], reverse=True)
    
    run_options = {r['id']: f"Run {r['id']} | {r['start'].strftime('%d %b %H:%M')} | {r['run_type']}" for r in runs_list}
    
    selected_id = st.selectbox("Select Run", [r['id'] for r in runs_list], format_func=lambda x: run_options[x])
    
    # Get Data
    selected_run = next(r for r in runs_list if r['id'] == selected_id)
    mask = (df.index >= selected_run['start']) & (df.index <= selected_run['end'])
    run_data = df.loc[mask]
    
    # Display Run Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", selected_run['run_type'])
    c2.metric("Duration", f"{selected_run['duration_mins']} mins")
    c3.metric("COP", f"{selected_run['run_cop']:.2f}")
    c4.metric("Zones", selected_run['active_zones'])
    
    # --- CHART 1: Temperatures ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=run_data.index, y=run_data['FlowTemp'], name='Flow', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=run_data.index, y=run_data['ReturnTemp'], name='Return', line=dict(color='orange')))
    if 'OutdoorTemp' in run_data.columns:
        fig1.add_trace(go.Scatter(x=run_data.index, y=run_data['OutdoorTemp'], name='Outdoor', line=dict(color='blue', dash='dot')))
    
    # DHW Temp if available
    if 'DHW_Temp' in run_data.columns and run_data['DHW_Temp'].mean() > 10:
        fig1.add_trace(go.Scatter(x=run_data.index, y=run_data['DHW_Temp'], name='DHW Tank', line=dict(color='purple')))

    fig1.update_layout(title="System Temperatures", height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- CHART 2: Power & Heat ---
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['Power'], name='Electric (W)', fill='tozeroy', line=dict(color='blue')), secondary_y=False)
    fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['Heat'], name='Heat Output (W)', line=dict(color='gold')), secondary_y=False)
    
    if 'FlowRate' in run_data.columns:
        fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['FlowRate'], name='Flow Rate (L/min)', line=dict(color='cyan', dash='dot')), secondary_y=True)
        
    fig2.update_layout(title="Power, Heat & Flow", height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- CHART 3: Room Response ---
    # Find relevant room columns
    all_rooms = [c for c in run_data.columns if c.startswith('Room_')]
    if all_rooms:
        fig3 = go.Figure()
        
        # Only show relevant rooms if identified, otherwise show all
        relevant_rooms = selected_run.get('relevant_rooms', [])
        rooms_to_show = relevant_rooms if relevant_rooms else all_rooms
        
        for col in all_rooms:
            is_relevant = col in rooms_to_show
            # Use Friendly Name
            friendly_room = get_friendly_name(col)
            
            fig3.add_trace(go.Scatter(
                x=run_data.index, y=run_data[col], 
                name=friendly_room, 
                mode='lines', 
                line=dict(width=3 if is_relevant else 1), 
                opacity=1.0 if is_relevant else 0.3,
                visible=True if is_relevant else "legendonly"
            ))
        
        if 'OutdoorTemp' in run_data.columns:
            fig3.add_trace(go.Scatter(x=run_data.index, y=run_data['OutdoorTemp'], name="Outdoor", line=dict(color='grey', dash='dash'), yaxis="y2"))
        
        fig3.update_layout(title="Room Response", hovermode="x unified", yaxis2=dict(overlaying="y", side="right"), height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    # --- AI DOWNLOAD SECTION ---
    with st.expander("🤖 Generate AI Analysis for this Run"):
        ai_payload = {
            "meta": {
                "report_type": "SINGLE_RUN_ANALYSIS",
                "calc_version": CALC_VERSION,
                "run_id": selected_run['id'],
                "timestamp": selected_run['start'].isoformat()
            },
            "system_context": AI_SYSTEM_CONTEXT,
            "run_metrics": {
                "type": selected_run['run_type'],
                "duration_min": selected_run['duration_mins'],
                "cop": selected_run['run_cop'],
                "avg_flow_temp": selected_run['avg_flow_temp'],
                "avg_outdoor": selected_run['avg_outdoor'],
                "zones_active": selected_run['active_zones']
            },
            "control_modes": {
                "dhw_mode_value": selected_run.get('dhw_mode', None),
            },
            "environmental_conditions": {
                "outdoor_temp_avg": round(run_data['OutdoorTemp'].mean(), 1) if 'OutdoorTemp' in run_data else None,
            },
            "time_series_sample": run_data[['Power', 'FlowTemp', 'ReturnTemp']].resample('5min').mean().to_dict()
        }
        
        st.download_button(
            label="📥 Download Run JSON",
            data=json.dumps(ai_payload, indent=2),
            file_name=f"run_{selected_run['id']}_analysis.json",
            mime="application/json"
        )