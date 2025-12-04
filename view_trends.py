#view_trends.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import json
import pandas as pd
from config import CALC_VERSION, AI_SYSTEM_CONTEXT
from utils import safe_div

def get_friendly_name(internal_key):
    """Helper to resolve internal keys to friendly names from session state."""
    config = st.session_state.get("system_config", {})
    if not config or "mapping" not in config:
        return internal_key
    return str(config["mapping"].get(internal_key, internal_key))

def render_long_term_trends(daily_df, raw_df, runs_list):
    st.title("📈 Long-Term Performance")
    
    if daily_df.empty:
        st.warning("No valid daily data found.")
        return

    # Use tabs to control rendering
    tabs = ["Performance", "Temperatures", "AI Report"]
    
    # Check if DHW data exists to show tab
    has_dhw = 'DHW_Mins' in daily_df.columns and daily_df['DHW_Mins'].sum() > 0
    if has_dhw:
        tabs.insert(1, "DHW Analysis")
        
    st_tabs = st.tabs(tabs)
    
    # --- Performance Tab Logic ---
    with st_tabs[0]:
        # 1. KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Days", len(daily_df))
        k2.metric("Total Heat", f"{daily_df['Total_Heat_kWh'].sum():.0f} kWh")
        k3.metric("Total Cost", f"€{daily_df['Daily_Cost_Euro'].sum():.2f}")
        scop = safe_div(daily_df['Total_Heat_kWh'].sum(), daily_df['Total_Electricity_kWh'].sum())
        k4.metric("Period SCOP", f"{scop:.2f}")
        
        # --- CHART 1: DAILY ENERGY (Stacked) ---
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily_df.index, y=daily_df['Electricity_Heating_kWh'], name='Heating Elec'))
        if has_dhw:
            fig.add_trace(go.Bar(x=daily_df.index, y=daily_df['Electricity_DHW_kWh'], name='DHW Elec'))
        
        fig.update_layout(title="Daily Electricity Consumption (kWh)", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

        # --- CHART 2: COP vs OUTDOOR ---
        fig_cop = px.scatter(daily_df, x='Outdoor_Avg', y='Global_SCOP', 
                         size='Total_Heat_kWh', color='Total_Heat_kWh',
                         title="Daily COP vs Outdoor Temperature",
                         labels={'Global_SCOP': 'COP', 'Outdoor_Avg': 'Outdoor Temp (°C)'})
        fig_cop.add_hline(y=3.0, line_dash="dot", annotation_text="Target COP 3.0")
        st.plotly_chart(fig_cop, use_container_width=True)

    # --- DHW Tab Logic (Conditional) ---
    if has_dhw:
        # Find the index of DHW tab
        dhw_tab_idx = tabs.index("DHW Analysis")
        with st_tabs[dhw_tab_idx]:
            st.subheader("Domestic Hot Water Analysis")
            c1, c2 = st.columns(2)
            
            # Chart: DHW Runs per Day
            if runs_list:
                run_df = pd.DataFrame(runs_list)
                if not run_df.empty:
                    dhw_runs = run_df[run_df['run_type'] == 'DHW']
                    if not dhw_runs.empty:
                        daily_counts = dhw_runs.set_index('start').resample('D').size()
                        fig1 = px.bar(x=daily_counts.index, y=daily_counts.values, 
                                      title="DHW Cycles per Day", labels={'y': 'Cycles'})
                        c1.plotly_chart(fig1, use_container_width=True)
                    else:
                        c1.info("No DHW runs found in run list.")
            
            # Chart: DHW Energy
            fig2 = px.bar(daily_df, x=daily_df.index, y='Electricity_DHW_kWh', title="DHW Electricity (kWh)")
            c2.plotly_chart(fig2, use_container_width=True)

    # --- Temperatures Tab Logic ---
    temp_tab_idx = tabs.index("Temperatures")
    with st_tabs[temp_tab_idx]:
        # Find Room Columns
        room_cols = [c for c in daily_df.columns if c.startswith('Room_') and c.endswith('_mean')]
        
        if room_cols:
            fig_temp = go.Figure()
            for r in room_cols:
                # Clean key: "Room_1_mean" -> "Room_1"
                base_key = r.replace("_mean", "")
                label = get_friendly_name(base_key)
                fig_temp.add_trace(go.Scatter(x=daily_df.index, y=daily_df[r], name=label))
            
            fig_temp.update_layout(title="Average Daily Room Temperatures")
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No room temperature sensors mapped.")

    # --- AI Report Tab Logic ---
    ai_tab_idx = tabs.index("AI Report")
    with st_tabs[ai_tab_idx]:
        # Prepare JSON for AI
        json_ready = daily_df.copy()
        # Convert index to string
        json_ready.index = json_ready.index.strftime('%Y-%m-%d')
        # Drop internal columns
        cols_to_drop = [c for c in json_ready.columns if 'Internal' in c or 'Unnamed' in c]
        json_ready = json_ready.drop(columns=cols_to_drop, errors='ignore')
        
        # Round floats
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
        
        st.download_button(
            label="📥 Download JSON for AI Analysis",
            data=json.dumps(ai_payload, indent=2),
            file_name=f"heat_pump_long_term_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        with st.expander("View JSON Preview"):
            st.json(period_summary)