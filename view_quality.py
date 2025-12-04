#view_quality.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from config import SENSOR_ROLES, BASELINE_JSON_PATH
from utils import availability_pct
import baselines

def render_data_quality(daily_df, df, unmapped_entities, patterns, heartbeat_path):
    st.title("🛡️ Data Quality Studio")
    
    if daily_df.empty:
        st.warning("No data loaded.")
        return

    # --- PARTIAL DAY LOGIC ---
    # Calculate denominator based on actual recorded minutes
    system_on_minutes = daily_df.apply(lambda r: max(r.get('Recorded_Minutes', 0), r.get('Active_Mins', 0), 1), axis=1)
    
    def expected_window_series(sensor_name, system_on_minutes):
        baseline_data = st.session_state.get("heartbeat_baseline", {}).get(sensor_name)
        
        # Scenario A: Baseline exists
        if baseline_data and baseline_data.get('expected_minutes'):
            # Scale the expected minutes by the partial day ratio if needed
            # For simplicity in this reconstruction, we return system_on_minutes as fallback
            # In a full implementation, you'd scale this.
            return system_on_minutes 
            
        # Scenario B: Known Role (DHW Only)
        role = SENSOR_ROLES.get(sensor_name, 'standard')
        if role == 'dhw_active':
            # FIX: Use DHW_Mins from processing.py output
            return daily_df.get('DHW_Mins', system_on_minutes)
            
        return system_on_minutes

    dq_tab1, dq_tab5 = st.tabs(["Daily Completeness", "Unmapped Entities"])
    
    with dq_tab1:
        st.markdown("### Sensor Availability (%)")
        
        # Identify sensors to check
        core_sensors = [c for c in df.columns if c in SENSOR_ROLES or c.startswith('Room_') or c.startswith('Zone_')]
        
        # Calculate scores
        scores = {}
        for s in core_sensors:
            # Count minutes where value is not null/zero (depending on sensor type)
            # For simplicity, just count valid non-NaN samples per day
            daily_counts = df[s].notna().resample('D').sum()
            
            # Get expected window
            expected = expected_window_series(s, system_on_minutes)
            
            # Calculate %
            pct_series = availability_pct(daily_counts, expected)
            scores[s] = round(pct_series.mean(), 1)
            
        # Display as dataframe
        score_df = pd.DataFrame(list(scores.items()), columns=['Sensor', 'Availability %'])
        score_df = score_df.sort_values('Availability %', ascending=False)
        st.dataframe(score_df, width=600)

        # Baseline Controls
        with st.expander("Baseline Management"):
            if st.button("Build/Update Baselines"):
                history_df = st.session_state.get("raw_history_df")
                if history_df is not None:
                    with st.spinner("Building baselines..."):
                        bl = baselines.build_offline_aware_seasonal_baseline(history_df, SENSOR_ROLES)
                        baselines.save_heartbeat_baseline_to_json(bl, None)
                        st.session_state["heartbeat_baseline"] = bl
                        st.success("Baseline updated.")
        
        if patterns:
            pat_data = []
            for sensor, details in patterns.items():
                pat_data.append({
                    "Sensor": sensor,
                    "Type": details['report_type'],
                    "Interval (s)": round(details['normal_interval_sec'], 1),
                    "Gap Limit (s)": round(details['gap_threshold_sec'], 1)
                })
            st.dataframe(pd.DataFrame(pat_data), use_container_width=True)

    with dq_tab5:
        st.markdown("### ⚠️ Unmapped Entities")
        if unmapped_entities:
            with st.expander("See List", expanded=False):
                st.code("\n".join(unmapped_entities), language="text")
            
            stats = []
            for e in unmapped_entities:
                if e in df.columns:
                    s = df[e]
                    stats.append({"Entity": e, "Samples": s.notna().sum(), "Mean": s.mean() if pd.api.types.is_numeric_dtype(s) else "N/A"})
            st.dataframe(pd.DataFrame(stats))
        else:
            st.success("All entities in the file are mapped!")