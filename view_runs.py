# view_runs.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd # Needed for to_numeric
from plotly.subplots import make_subplots
from config import CALC_VERSION, AI_SYSTEM_CONTEXT, TARIFF_PROFILE_ID, CONFIG_HISTORY

def render_run_inspector(df, runs_list):
    st.title("🔍 Run Inspector")
    
    if 'Power_Clean' in df.columns:
        total_kwh_in = (df['Power_Clean'].sum() / 1000) / 60
    else:
        total_kwh_in = 0
        
    if 'Heat_Clean' in df.columns:
        total_kwh_out = (df['Heat_Clean'].sum() / 1000) / 60
    else:
        total_kwh_out = 0
    
    if not runs_list:
        st.info("No runs detected.")
        return

    runs_list.sort(key=lambda x: x['start'], reverse=True)
    
    run_options = {}
    for r in runs_list:
        start_str = r['start'].strftime('%d/%m/%Y %H:%M')
        icon = "🚿" if r['run_type'] == "DHW" else "♨️"
        zone_label = r.get('active_zones', 'None')
        label = f"{start_str} | {r['duration_mins']}m | {icon} {r['run_type']} ({zone_label})"
        run_options[label] = r

    option_labels = list(run_options.keys())
    if "run_selector_idx" not in st.session_state: st.session_state["run_selector_idx"] = 0
    st.session_state["run_selector_idx"] = min(st.session_state["run_selector_idx"], len(option_labels) - 1)

    nav_prev, nav_select, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("Previous", disabled=st.session_state["run_selector_idx"] <= 0):
            st.session_state["run_selector_idx"] = max(0, st.session_state["run_selector_idx"] - 1)
            st.rerun()
    with nav_next:
        if st.button("Next", disabled=st.session_state["run_selector_idx"] >= len(option_labels) - 1):
            st.session_state["run_selector_idx"] = min(len(option_labels) - 1, st.session_state["run_selector_idx"] + 1)
            st.rerun()
    with nav_select:
        selected_label = st.selectbox("Select Run", options=option_labels, index=st.session_state["run_selector_idx"])
    st.session_state["run_selector_idx"] = option_labels.index(selected_label)

    selected_run = run_options[selected_label]
    run_data = df.loc[selected_run['start'] : selected_run['end']]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{selected_run['duration_mins']}m")
    c2.metric("COP", f"{selected_run['run_cop']:.2f}")
    c3.metric("Avg ΔT", f"{selected_run['avg_dt']:.1f}°")
    
    flow_val = selected_run.get('avg_flow_rate', selected_run.get('avg_flow', 0))
    c4.metric("Avg Flow", f"{flow_val:.1f} L/m")
    
    tight = dict(margin=dict(l=10, r=10, t=30, b=10), height=350)
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Efficiency", "💧 Hydraulics", "🏠 Rooms", "🤖 AI Data"])
    
    with tab1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Heat_Clean'], name="Heat", fill='tozeroy', line=dict(color='orange', width=0)), secondary_y=False)
        fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Power_Clean'], name="Power", line=dict(color='red', width=1)), secondary_y=False)
        if 'Indoor_Power' in run_data.columns:
            fig.add_trace(go.Scatter(x=run_data.index, y=run_data['Indoor_Power'], name="Indoor", line=dict(color='purple', width=1, dash='dot')), secondary_y=False)
        fig.add_trace(go.Scatter(x=run_data.index, y=run_data['COP_Graph'], name="COP", line=dict(color='blue', dash='dot', width=1)), secondary_y=True)
        fig.update_layout(**tight, title="Power & Efficiency", hovermode="x unified")
        st.plotly_chart(fig, width="stretch", key="run_power")

    with tab2:
        is_dhw = selected_run['run_type'] == "DHW"
        rows = 4 if is_dhw else 3
        titles = ["Delta T", "Flow Rate", "Active Zones"]
        if is_dhw: titles.append("DHW Temps")
        fig2 = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=titles)
        
        fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['DeltaT'], name="ΔT", line=dict(color='green')), row=1, col=1)
        if 'FlowRate' in run_data.columns:
            fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['FlowRate'], name="Flow", line=dict(color='cyan')), row=2, col=1)
        
        zone_cols = [c for c in run_data.columns if c.startswith('Zone_')]
        for i, z in enumerate(zone_cols):
            # FIX: Ensure numeric data for plotting
            series_numeric = pd.to_numeric(run_data[z], errors='coerce').fillna(0)
            y_vals = series_numeric.apply(lambda x: i + 0.8 if x > 0 else None)
            
            fig2.add_trace(go.Scatter(x=run_data.index, y=y_vals, name=z, mode='lines', line=dict(width=15)), row=3, col=1)
            
        if is_dhw and 'DHW_Temp' in run_data.columns:
             fig2.add_trace(go.Scatter(x=run_data.index, y=run_data['DHW_Temp'], name="DHW Tank", line=dict(color='orange')), row=4, col=1)

        fig2.update_layout(height=600 if is_dhw else 450, hovermode="x unified")
        st.plotly_chart(fig2, width="stretch", key="run_hydro")

    with tab3:
        if selected_run['run_type'] == "Heating":
            fig3 = go.Figure()
            relevant_rooms = selected_run.get('relevant_rooms', [])
            all_rooms = [c for c in run_data.columns if c.startswith('Room_')]
            rooms_to_show = relevant_rooms if relevant_rooms else all_rooms
            
            for col in all_rooms:
                is_relevant = col in rooms_to_show
                clean_name = col.replace("Room_", "Room ")
                fig3.add_trace(go.Scatter(
                    x=run_data.index, y=run_data[col], 
                    name=clean_name, 
                    mode='lines', 
                    line=dict(width=3 if is_relevant else 1), 
                    opacity=1.0 if is_relevant else 0.3,
                    visible=True if is_relevant else "legendonly"
                ))
            
            if 'OutdoorTemp' in run_data.columns:
                fig3.add_trace(go.Scatter(x=run_data.index, y=run_data['OutdoorTemp'], name="Outdoor", line=dict(color='grey', dash='dash'), yaxis="y2"))
            
            fig3.update_layout(title="Room Response", hovermode="x unified", yaxis2=dict(overlaying="y", side="right"), height=400)
            st.plotly_chart(fig3, width="stretch", key="run_rooms")
        else:
            st.info("Room analysis skipped for DHW runs.")

    with tab4:
        st.write("AI Context Payload (Preview):")
        st.json(selected_run)