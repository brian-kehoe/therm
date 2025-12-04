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

def render_long_term_trends(daily_df, raw_df, runs_list):
    st.title("📈 Long-Term Performance")
    
    if daily_df.empty:
        st.warning("No valid daily data found.")
        return

    # Use tabs to control rendering
    tab_main, tab_ai = st.tabs(["Performance", "AI Report"])
    
    # --- Performance Tab Logic ---
    with tab_main:
        # 1. KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Days", len(daily_df))
        k2.metric("Total Heat", f"{daily_df['Total_Heat_kWh'].sum():.0f} kWh")
        k3.metric("Total Cost", f"€{daily_df['Daily_Cost_Euro'].sum():.2f}")
        scop = safe_div(daily_df['Total_Heat_kWh'].sum(), daily_df['Total_Electricity_kWh'].sum())
        k4.metric("Period SCOP", f"{scop:.2f}")
        
        # --- CHART 1: DAILY ENERGY (Stacked) (UNCHANGED) ---
        fig = go.Figure()
        
        # Define hovers for consistency (1 decimal place for kWh)
        heat_hover = "<b>Heat Output</b><br>Space Heat: %{y:.1f} kWh<extra></extra>"
        dhw_hover = "<b>Heat Output</b><br>DHW Heat: %{y:.1f} kWh<extra></extra>"
        elec_hover = "<b>Electricity Input</b><br>Space Elec: %{y:.1f} kWh<extra></extra>"
        dhw_elec_hover = "<b>Electricity Input</b><br>DHW Elec: %{y:.1f} kWh<extra></extra>"
        immersion_hover = "<b>Electricity Input</b><br>Immersion: %{y:.1f} kWh<extra></extra>"

        # Left Stack: Heat Output
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Heat_Heating_kWh'], 
            name='Space Heat', marker_color='#ffa600', offsetgroup=0, legendgroup='Output',
            hovertemplate=heat_hover
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Heat_DHW_kWh'], 
            name='DHW Heat', marker_color='#ffd580', offsetgroup=0, 
            base=daily_df['Heat_Heating_kWh'], legendgroup='Output',
            hovertemplate=dhw_hover
        ))

        # Right Stack: Energy Input
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Electricity_Heating_kWh'], 
            name='Space Elec', marker_color='#003f5c', offsetgroup=1, legendgroup='Input',
            hovertemplate=elec_hover
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Electricity_DHW_kWh'], 
            name='DHW Elec', marker_color='#58508d', offsetgroup=1, 
            base=daily_df['Electricity_Heating_kWh'], legendgroup='Input',
            hovertemplate=dhw_elec_hover
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Immersion_kWh'], 
            name='Immersion', marker_color='#bc5090', offsetgroup=1, 
            base=daily_df['Electricity_Heating_kWh'] + daily_df['Electricity_DHW_kWh'], legendgroup='Input',
            hovertemplate=immersion_hover
        ))
        
        fig.update_layout(
            title="Daily Energy Balance (Stacked by Component)",
            yaxis_title="Energy (kWh)",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch", key="daily_energy_chart")

        # --- CHART 2: ENVIRONMENTAL (UNCHANGED) ---
        c1, c2 = st.columns(2)
        with c1:
            fig_env = make_subplots(specs=[[{"secondary_y": True}]])
            if 'Wind_Avg' in daily_df:
                fig_env.add_trace(go.Scatter(
                    x=daily_df.index, y=daily_df['Wind_Avg'], 
                    name="Wind", line=dict(color='grey'), 
                    connectgaps=True,
                    hovertemplate="Wind: %{y:.1f} m/s<extra></extra>"
                ), secondary_y=False)
            if 'Humidity_Avg' in daily_df:
                fig_env.add_trace(go.Scatter(
                    x=daily_df.index, y=daily_df['Humidity_Avg'], 
                    name="Humidity", line=dict(color='blue', dash='dot'),
                    connectgaps=True,
                    hovertemplate="Humidity: %{y:.1f} %<extra></extra>" 
                ), secondary_y=True)
            
            fig_env.update_layout(title="Wind & Humidity", height=300, hovermode='x unified')
            st.plotly_chart(fig_env, width="stretch", key="env_chart")
            
        with c2:
            fig_sol = make_subplots(specs=[[{"secondary_y": True}]])
            if 'Solar_Avg' in daily_df:
                fig_sol.add_trace(go.Bar(
                    x=daily_df.index, y=daily_df['Solar_Avg'], name="Solar", marker_color='orange',
                    hovertemplate="Solar: %{y:.1f} W/m²<extra></extra>"
                ), secondary_y=False)
            fig_sol.add_trace(go.Scatter(
                x=daily_df.index, y=daily_df['Global_SCOP'], name="SCOP", line=dict(color='green'),
                hovertemplate="SCOP: %{y:.2f}<extra></extra>"
            ), secondary_y=True)
            fig_sol.update_layout(title="Solar Gain vs Efficiency", height=300, hovermode='x unified')
            st.plotly_chart(fig_sol, width="stretch", key="solar_chart")
        
        st.divider()

        # --- CHART 3: HEATING CURVE (Space Heating Only) ---
        # NOTE: Subheader and Chart Title changed for consistency
        st.subheader("1. Space Heating Run Averages: Weather Compensation Curve")
        st.caption("Target: Diagonal line downwards (One dot representing the average of each run).")
        
        heating_runs = [
            r for r in runs_list 
            if r['run_type'] == 'Heating' and r['avg_flow_temp'] > 25
        ]
        
        if heating_runs:
            df_heat_scatter = pd.DataFrame(heating_runs)
            
            fig_wc = px.scatter(
                df_heat_scatter, 
                x='avg_outdoor', 
                y='avg_flow_temp', 
                color='run_cop',
                color_continuous_scale='RdYlGn',
                # FIX: Chart title changed for consistency
                title="Space Heating Run Averages: Flow Temp vs Outdoor Temp",
                opacity=0.9,
                labels={'avg_outdoor': 'Avg Outdoor Temp (°C)', 'avg_flow_temp': 'Avg Flow Temp (°C)', 'run_cop': 'COP'},
                hover_data={
                    'avg_outdoor': ':.1f',  
                    'avg_flow_temp': ':.1f',     
                    'run_cop': ':.2f',    
                    # FIX: Changed 'id' to 'start' for run start time
                    'start': '|%d-%m-%Y %H:%M'         
                }
            )
            # Inefficient Zone Visual
            fig_wc.add_shape(
                type="rect", x0=10, y0=43, x1=20, y1=60,
                line=dict(color="red", width=1, dash="dot"),
                fillcolor="rgba(0,0,0,0)", opacity=0.3, layer="below"
            )
            fig_wc.add_annotation(
                x=15, y=58, text="Inefficient Zone (>43°C)",
                showarrow=False, font=dict(color="red", size=9), opacity=0.6
            )
            st.plotly_chart(fig_wc, width="stretch", key="wc_heating_chart")
        else:
            st.info("No Space Heating runs detected.")

        # --- CHART 4: DHW CURVE (Hot Water Only) ---
        # NOTE: Subheader and Chart Title changed for consistency
        st.subheader("2. Hot Water Run Averages: Temperature Consistency")
        st.caption("Target: Flat horizontal cluster (One dot representing the average of each run).")

        dhw_runs = [
            r for r in runs_list 
            if r['run_type'] == 'DHW' and r['avg_flow_temp'] > 25
        ]
        
        if dhw_runs:
            df_dhw_scatter = pd.DataFrame(dhw_runs)
            
            fig_dhw = px.scatter(
                df_dhw_scatter, 
                x='avg_outdoor', 
                y='avg_flow_temp', 
                color='run_cop',
                color_continuous_scale='RdYlGn',
                # FIX: Chart title changed for consistency
                title="Hot Water Run Averages: Flow Temp vs Outdoor Temp",
                opacity=0.9, 
                labels={'avg_outdoor': 'Avg Outdoor Temp (°C)', 'avg_flow_temp': 'Avg Flow Temp (°C)', 'run_cop': 'COP'},
                hover_data={
                    'avg_outdoor': ':.1f',  
                    'avg_flow_temp': ':.1f',     
                    'run_cop': ':.2f',    
                    # FIX: Changed 'id' to 'start' for run start time
                    'start': '|%d-%m-%Y %H:%M'
                }
            )
            # Reference Line at 50C
            fig_dhw.add_hline(y=50.0, line_dash="dot", line_color="grey", annotation_text="Typical Target (50°C)")
            
            st.plotly_chart(fig_dhw, width="stretch", key="wc_dhw_chart")
        else:
            st.info("No Hot Water (DHW) runs detected.")

    # --- AI Report Tab Logic (UNCHANGED) ---
    with tab_ai:
        st.markdown("### 🤖 Download AI System Context")
        st.info("The JSON below contains all the data required for a full long-term analysis.")

        # Prepare Data for JSON (This calculation is fast, but the st.json render is slow)
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
        
        st.download_button(
            label="📥 Download JSON for AI Analysis",
            data=json.dumps(ai_payload, indent=2),
            file_name=f"heat_pump_long_term_ai_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        # CRITICAL CHANGE: Only render the JSON viewer if explicitly requested
        if st.checkbox("Show Raw JSON Payload", value=False):
            st.json(ai_payload)