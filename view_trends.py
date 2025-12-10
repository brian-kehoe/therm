# view_trends.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import json
import pandas as pd

from config import CALC_VERSION
from utils import safe_div


def _build_system_context(user_config: dict | None, include_heating_note: bool) -> str:
    """Compose AI system context from user-provided freetext; add DHW note only if detected."""
    parts: list[str] = []
    if isinstance(user_config, dict):
        ai_ctx = user_config.get("ai_context") or {}
        for key in ("hp_model", "property_context", "operational_goals"):
            val = ai_ctx.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        tariff = user_config.get("tariff_structure")
        currency = user_config.get("currency", "€")
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


def render_long_term_trends(daily_df: pd.DataFrame, raw_df: pd.DataFrame, runs_list: list, user_config: dict | None = None) -> None:
    """
    Long-term performance view:
    - KPI cards
    - Daily stacked energy balance
    - Environmental charts
    - Weather-compensation scatter (heating)
    - DHW run scatter
    - AI JSON export for long-term analysis
    """
    st.title(" Long-Term Performance")

    if daily_df is None or daily_df.empty:
        st.warning("No valid daily data found.")
        return

    # Tabs: main performance vs AI export
    tab_main, tab_ai = st.tabs(["Performance", "AI Report"])

    # ----------------------------------------------------------------------
    # PERFORMANCE TAB
    # ----------------------------------------------------------------------
    with tab_main:
        # 1. KPI Metrics
        k1, k2, k3, k4 = st.columns(4)

        total_heat = float(daily_df.get("Total_Heat_kWh", 0).sum())
        total_elec = float(daily_df.get("Total_Electricity_kWh", 0).sum())
        currency = "€"
        if isinstance(user_config, dict):
            currency = user_config.get("currency", currency)

        total_cost = float(daily_df.get("Daily_Cost_Euro", 0).sum())
        scop = safe_div(total_heat, total_elec)

        k1.metric("Days", len(daily_df))
        k2.metric("Total Heat", f"{total_heat:.0f} kWh")
        k3.metric("Total Cost", f"{currency}{total_cost:.2f}")
        k4.metric("Period SCOP", f"{scop:.2f}")

        # 2. Daily Energy (Stacked)
        st.subheader("Daily Energy Balance")

        has_heat_energy = (
            ("Heat_Heating_kWh" in daily_df.columns)
            or ("Heat_DHW_kWh" in daily_df.columns)
        )

        if has_heat_energy:
            fig = go.Figure()

            # Hover templates (1 decimal place for kWh)
            heat_hover = "Heat Output<br>Space Heat: %{y:.1f} kWh"
            dhw_hover = "Heat Output<br>DHW Heat: %{y:.1f} kWh"
            elec_hover = "Electricity Input<br>Space Elec: %{y:.1f} kWh"
            dhw_elec_hover = "Electricity Input<br>DHW Elec: %{y:.1f} kWh"
            immersion_hover = "Electricity Input<br>Immersion: %{y:.1f} kWh"

            # Left stack: Heat Output
            if "Heat_Heating_kWh" in daily_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Heat_Heating_kWh"].fillna(0),
                        name="Space Heat",
                        marker_color="#ffa600",
                        offsetgroup=0,
                        legendgroup="Output",
                        hovertemplate=heat_hover,
                    )
                )

            if "Heat_DHW_kWh" in daily_df.columns:
                base_heat = daily_df.get("Heat_Heating_kWh", 0)
                if not isinstance(base_heat, pd.Series):
                    base_heat = pd.Series(base_heat, index=daily_df.index)

                fig.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Heat_DHW_kWh"].fillna(0),
                        name="DHW Heat",
                        marker_color="#ffd580",
                        offsetgroup=0,
                        base=base_heat.fillna(0),
                        legendgroup="Output",
                        hovertemplate=dhw_hover,
                    )
                )

            # Right stack: Electricity Input
            base_elec = daily_df.get(
                "Electricity_Heating_kWh",
                pd.Series(0, index=daily_df.index),
            ).fillna(0)

            if "Electricity_Heating_kWh" in daily_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Electricity_Heating_kWh"].fillna(0),
                        name="Space Elec",
                        marker_color="#003f5c",
                        offsetgroup=1,
                        legendgroup="Input",
                        hovertemplate=elec_hover,
                    )
                )

            if "Electricity_DHW_kWh" in daily_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Electricity_DHW_kWh"].fillna(0),
                        name="DHW Elec",
                        marker_color="#58508d",
                        offsetgroup=1,
                        base=base_elec,
                        legendgroup="Input",
                        hovertemplate=dhw_elec_hover,
                    )
                )
                base_elec = base_elec + daily_df["Electricity_DHW_kWh"].fillna(0)

            if "Immersion_kWh" in daily_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Immersion_kWh"].fillna(0),
                        name="Immersion",
                        marker_color="#bc5090",
                        offsetgroup=1,
                        base=base_elec,
                        legendgroup="Input",
                        hovertemplate=immersion_hover,
                    )
                )

            fig.update_layout(
                title="Daily Energy Balance (Stacked by Component)",
                yaxis_title="Energy (kWh)",
                barmode="group",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch", key="daily_energy_chart")
        else:
            st.info(
                "Daily energy balance is disabled because no Heat output is "
                "available (no Flow Rate or Heat sensor mapped)."
            )

        # 3. Environmental Charts
        st.divider()
        c1, c2 = st.columns(2)

        # Respect user-selected units for wind speed (default m/s)
        wind_unit = "m/s"
        wind_factor = 1.0
        try:
            cfg_units = st.session_state.get("system_config", {}).get("units", {})
        except Exception:
            cfg_units = {}
        if isinstance(cfg_units, dict):
            user_unit = cfg_units.get("Wind_Speed")
            if user_unit in ["m/s", "km/h", "mph"]:
                wind_unit = user_unit
                wind_factor = {"m/s": 1.0, "km/h": 3.6, "mph": 2.23693629}[wind_unit]

        with c1:
            fig_env = make_subplots(specs=[[{"secondary_y": True}]])
            if "Wind_Avg" in daily_df.columns:
                wind_series = pd.to_numeric(daily_df["Wind_Avg"], errors="coerce") * wind_factor
                fig_env.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=wind_series,
                        name="Wind",
                        line=dict(color="grey"),
                        connectgaps=True,
                        hovertemplate=f"Wind: %{{y:.1f}} {wind_unit}",
                    ),
                    secondary_y=False,
                )

            if "Humidity_Avg" in daily_df.columns:
                fig_env.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Humidity_Avg"],
                        name="Humidity",
                        line=dict(color="blue", dash="dot"),
                        connectgaps=True,
                        hovertemplate="Humidity: %{y:.1f} %",
                    ),
                    secondary_y=True,
                )

            if "Global_SCOP" in daily_df.columns:
                fig_env.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Global_SCOP"],
                        name="SCOP",
                        line=dict(color="green"),
                        connectgaps=True,
                        hovertemplate="SCOP: %{y:.2f}",
                    ),
                    secondary_y=True,
                )

            fig_env.update_layout(
                title="Wind & Humidity",
                height=300,
                hovermode="x unified",
            )
            fig_env.update_yaxes(title_text=f"Wind ({wind_unit})", secondary_y=False)
            fig_env.update_yaxes(title_text="Humidity (%) / SCOP", secondary_y=True)
            st.plotly_chart(fig_env, width="stretch", key="env_chart")

        with c2:
            fig_sol = make_subplots(specs=[[{"secondary_y": True}]])
            if "Solar_Avg" in daily_df.columns:
                fig_sol.add_trace(
                    go.Bar(
                        x=daily_df.index,
                        y=daily_df["Solar_Avg"],
                        name="Solar",
                        marker_color="orange",
                        hovertemplate="Solar: %{y:.1f} W/m²",
                    ),
                    secondary_y=False,
                )

            if has_heat_energy and "Global_SCOP" in daily_df.columns:
                fig_sol.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Global_SCOP"],
                        name="SCOP",
                        line=dict(color="green"),
                        hovertemplate="SCOP: %{y:.2f}",
                    ),
                    secondary_y=True,
                )

            fig_sol.update_layout(
                title="Solar Gain vs Efficiency",
                height=300,
                hovermode="x unified",
            )
            st.plotly_chart(fig_sol, width="stretch", key="solar_chart")

        # 4. Heating Weather Compensation (space heating runs)
        st.divider()
        st.subheader("1. Space Heating Run Averages: Weather Compensation Curve")
        st.caption(
            "Target: Diagonal line downwards (one dot representing the "
            "average of each run)."
        )

        heating_runs = [
            r
            for r in (runs_list or [])
            if r.get("run_type") == "Heating" and r.get("avg_flow_temp", 0) > 25
        ]

        if heating_runs:
            df_heat_scatter = pd.DataFrame(heating_runs)
            fig_wc = px.scatter(
                df_heat_scatter,
                x="avg_outdoor",
                y="avg_flow_temp",
                color="run_cop",
                color_continuous_scale="RdYlGn",
                title="Space Heating Run Averages: Flow Temp vs Outdoor Temp",
                opacity=0.9,
                labels={
                    "avg_outdoor": "Avg Outdoor Temp (°C)",
                    "avg_flow_temp": "Avg Flow Temp (°C)",
                    "run_cop": "COP",
                },
                hover_data={
                    "avg_outdoor": ":.1f",
                    "avg_flow_temp": ":.1f",
                    "run_cop": ":.2f",
                    # Run start timestamp
                    "start": "|%d-%m-%Y %H:%M",
                },
            )

            # Inefficient Zone (> 43°C)
            fig_wc.add_shape(
                type="rect",
                x0=10,
                y0=43,
                x1=20,
                y1=60,
                line=dict(color="red", width=1, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
                opacity=0.3,
                layer="below",
            )
            fig_wc.add_annotation(
                x=15,
                y=58,
                text="Inefficient Zone (>43°C)",
                showarrow=False,
                font=dict(color="red", size=9),
                opacity=0.6,
            )
            st.plotly_chart(fig_wc, width="stretch", key="wc_heating_chart")
        else:
            st.info("No Space Heating runs detected.")

        # 5. DHW Temperature Consistency
        st.subheader("2. Hot Water Run Averages: Temperature Consistency")
        st.caption(
            "Target: Flat horizontal cluster (one dot representing the "
            "average of each run)."
        )

        dhw_runs = [
            r
            for r in (runs_list or [])
            if r.get("run_type") == "DHW" and r.get("avg_flow_temp", 0) > 25
        ]

        if dhw_runs:
            df_dhw_scatter = pd.DataFrame(dhw_runs)
            fig_dhw = px.scatter(
                df_dhw_scatter,
                x="avg_outdoor",
                y="avg_flow_temp",
                color="run_cop",
                color_continuous_scale="RdYlGn",
                title="Hot Water Run Averages: Flow Temp vs Outdoor Temp",
                opacity=0.9,
                labels={
                    "avg_outdoor": "Avg Outdoor Temp (°C)",
                    "avg_flow_temp": "Avg Flow Temp (°C)",
                    "run_cop": "COP",
                },
                hover_data={
                    "avg_outdoor": ":.1f",
                    "avg_flow_temp": ":.1f",
                    "run_cop": ":.2f",
                    "start": "|%d-%m-%Y %H:%M",
                },
            )

            # Reference line at 50°C
            fig_dhw.add_hline(
                y=50.0,
                line_dash="dot",
                line_color="grey",
                annotation_text="Typical Target (50°C)",
            )
            st.plotly_chart(fig_dhw, width="stretch", key="wc_dhw_chart")
        else:
            st.info("No Hot Water (DHW) runs detected.")

    # ----------------------------------------------------------------------
    # AI REPORT TAB
    # ----------------------------------------------------------------------
    with tab_ai:
        st.markdown("### Download AI System Context")
        st.info(
            "The JSON below contains all the data required for a full "
            "long-term analysis."
        )

        # Prepare daily_df for JSON
        json_ready = daily_df.copy().reset_index()
        if json_ready.columns.size > 0:
            json_ready = json_ready.rename(
                columns={json_ready.columns[0]: "date"}
            )
        json_ready["date"] = json_ready["date"].astype(str)

        float_cols = json_ready.select_dtypes(include=[float]).columns
        json_ready[float_cols] = json_ready[float_cols].round(2)

        period_summary = {
            "days": int(len(json_ready)),
            "total_heat_kwh": round(total_heat, 2),
            "total_electricity_kwh": round(total_elec, 2),
            "total_cost_eur": round(total_cost, 2),
            "period_scop": round(float(scop), 2),
        }

        include_heating_note = any(
            bool(r.get("heating_during_dhw_detected")) for r in runs_list or []
        )

        ai_payload = {
            "meta": {
                "report_type": "LONG_TERM_TRENDS",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "calc_version": CALC_VERSION,
            },
            "system_context": _build_system_context(user_config, include_heating_note),
            "period_summary": period_summary,
            "daily_metrics": json_ready.to_dict(orient="records"),
        }

        st.download_button(
            label=" Download JSON for AI Analysis",
            data=json.dumps(ai_payload, indent=2),
            file_name=f"heat_pump_long_term_ai_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

        # Rendering the full JSON is slow; only do it on demand
        if st.checkbox("Show Raw JSON Payload", value=False):
            st.json(ai_payload)
