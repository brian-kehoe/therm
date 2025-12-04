#view_trends.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import json
from config import CALC_VERSION, AI_SYSTEM_CONTEXT
from utils import safe_div

def render_long_term_trends(daily_df, raw_df):
    st.title("📈 Long-Term Performance")
    
    if daily_df.empty:
        st.warning("No valid daily data found.")
        return

    tab_main, tab_ai = st.tabs(["Performance", "AI Report"])
    
    with tab_main:
        # 1. KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Days", len(daily_df))
        k2.metric("Total Heat", f"{daily_df['Total_Heat_kWh'].sum():.0f} kWh")
        k3.metric("Total Cost", f"€{daily_df['Daily_Cost_Euro'].sum():.2f}")
        scop = safe_div(daily_df['Total_Heat_kWh'].sum(), daily_df['Total_Electricity_kWh'].sum())
        k4.metric("Period SCOP", f"{scop:.2f}")
        
        # --- CHART 1: DAILY ENERGY (Stacked) ---
        fig = go.Figure()
        
        # Left Stack: Heat Output
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Heat_Heating_kWh'], 
            name='Space Heat', marker_color='#ffa600', offsetgroup=0, legendgroup='Output'
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Heat_DHW_kWh'], 
            name='DHW Heat', marker_color='#ffd580', offsetgroup=0, 
            base=daily_df['Heat_Heating_kWh'], legendgroup='Output'
        ))

        # Right Stack: Energy Input
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Electricity_Heating_kWh'], 
            name='Space Elec', marker_color='#003f5c', offsetgroup=1, legendgroup='Input'
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Electricity_DHW_kWh'], 
            name='DHW Elec', marker_color='#58508d', offsetgroup=1, 
            base=daily_df['Electricity_Heating_kWh'], legendgroup='Input'
        ))
        fig.add_trace(go.Bar(
            x=daily_df.index, y=daily_df['Immersion_kWh'], 
            name='Immersion', marker_color='#bc5090', offsetgroup=1, 
            base=daily_df['Electricity_Heating_kWh'] + daily_df['Electricity_DHW_kWh'], legendgroup='Input'
        ))
        
        fig.update_layout(
            title="Daily Energy Balance (Stacked by Component)",
            yaxis_title="Energy (kWh)",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width="stretch", key="daily_energy_chart")

        # --- CHART 2: ENVIRONMENTAL ---
        c1, c2 = st.columns(2)
        with c1:
            fig_env = make_subplots(specs=[[{"secondary_y": True}]])
            if 'Wind_Avg' in daily_df:
                fig_env.add_trace(go.Scatter(
                    x=daily_df.index, y=daily_df['Wind_Avg'], 
                    name="Wind", line=dict(color='grey'), 
                    connectgaps=True 
                ), secondary_y=False)
            if 'Humidity_Avg' in daily_df:
                fig_env.add_trace(go.Scatter(
                    x=daily_df.index, y=daily_df['Humidity_Avg'], 
                    name="Humidity", line=dict(color='blue', dash='dot'),
                    connectgaps=True 
                ), secondary_y=True)
            
            fig_env.update_layout(title="Wind & Humidity", height=300)
            st.plotly_chart(fig_env, width="stretch", key="env_chart")
            
        with c2:
            fig_sol = make_subplots(specs=[[{"secondary_y": True}]])
            if 'Solar_Avg' in daily_df:
                fig_sol.add_trace(go.Bar(x=daily_df.index, y=daily_df['Solar_Avg'], name="Solar", marker_color='orange'), secondary_y=False)
            fig_sol.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Global_SCOP'], name="SCOP", line=dict(color='green')), secondary_y=True)
            fig_sol.update_layout(title="Solar Gain vs Efficiency", height=300)
            st.plotly_chart(fig_sol, width="stretch", key="solar_chart")
        
        st.divider()

        # --- CHART 3: HEATING CURVE (Space Heating Only) ---
        st.subheader("1. Space Heating: Weather Compensation Curve")
        st.caption("Target: Diagonal line downwards (One dot representing the average of each run).")
        
        heat_data = raw_df[(raw_df['is_heating']) & (~raw_df['is_DHW']) & (raw_df['FlowTemp'] > 25)]
        
        if not heat_data.empty and 'run_id' in heat_data.columns:
            scatter_heat = heat_data.groupby('run_id')[['OutdoorTemp', 'FlowTemp', 'COP_Graph']].mean().reset_index()
            
            fig_wc = px.scatter(
                scatter_heat, 
                x='OutdoorTemp', 
                y='FlowTemp', 
                color='COP_Graph',
                color_continuous_scale='RdYlGn',
                title="Flow Temp vs Outdoor Temp (Run Averages)",
                opacity=0.9,
                labels={'OutdoorTemp': 'Avg Outdoor Temp (°C)', 'FlowTemp': 'Avg Flow Temp (°C)', 'COP_Graph': 'COP'},
                hover_data={
                    'OutdoorTemp': ':.1f',  
                    'FlowTemp': ':.1f',     
                    'COP_Graph': ':.2f',    
                    'run_id': True          
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
            st.info("No Space Heating runs detected in this dataset.")

        # --- CHART 4: DHW CURVE (Hot Water Only) ---
        st.subheader("2. Hot Water: Temperature Consistency")
        st.caption("Target: Flat horizontal cluster (One dot representing the average of each run).")

        dhw_data = raw_df[(raw_df['is_heating']) & (raw_df['is_DHW']) & (raw_df['FlowTemp'] > 25)]
        
        if not dhw_data.empty and 'run_id' in dhw_data.columns:
            scatter_dhw = dhw_data.groupby('run_id')[['OutdoorTemp', 'FlowTemp', 'COP_Graph']].mean().reset_index()
            
            fig_dhw = px.scatter(
                scatter_dhw, 
                x='OutdoorTemp', 
                y='FlowTemp', 
                color='COP_Graph',
                color_continuous_scale='RdYlGn',
                opacity=0.9, 
                labels={'OutdoorTemp': 'Avg Outdoor Temp (°C)', 'FlowTemp': 'Avg Flow Temp (°C)', 'COP_Graph': 'COP'},
                hover_data={
                    'OutdoorTemp': ':.1f',  
                    'FlowTemp': ':.1f',     
                    'COP_Graph': ':.2f',    
                    'run_id': True
                }
            )
            # Reference Line at 50C
            fig_dhw.add_hline(y=50.0, line_dash="dot", line_color="grey", annotation_text="Typical Target (50°C)")
            
            st.plotly_chart(fig_dhw, width="stretch", key="wc_dhw_chart")
        else:
            st.info("No Hot Water (DHW) runs detected in this dataset.")

    with tab_ai:
        # Prepare Data for JSON
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
        st.json(ai_payload)