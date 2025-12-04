#view_quality.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from config import SENSOR_EXPECTATION_MODE, SENSOR_GROUPS, SENSOR_ROLES, BASELINE_JSON_PATH
from utils import availability_pct
import baselines

def render_data_quality(daily_df, df, unmapped_entities, patterns, heartbeat_path):
    st.title("🛡️ Data Quality Studio")
    
    if daily_df.empty:
        st.warning("No data loaded.")
        return

    # --- PARTIAL DAY LOGIC ---
    # We calculate the denominator based on actual recorded minutes (e.g. 720 mins for a half day)
    # rather than hardcoding 1440 mins. This ensures 100% uptime is possible even on partial days.
    system_on_minutes = daily_df.apply(lambda r: max(r.get('Recorded_Minutes', 0), r.get('DQ_Power_Count', 0), 1), axis=1)
    
    def expected_window_series(sensor_name, system_on_minutes):
        baseline_data = st.session_state.get("heartbeat_baseline", {}).get(sensor_name)
        
        # Scenario A: Baseline exists (e.g. sparse sensor like OWM)
        if baseline_data and baseline_data.get('expected_minutes'):
            # Scale the expected minutes by the partial day ratio
            baseline_ratio = baseline_data['expected_minutes'] / 1440.0
            base = system_on_minutes * baseline_ratio
            return base.replace(0, np.nan).apply(lambda x: max(1.0, x) if x > 0 else np.nan)
            
        # Scenario B: Standard System Sensor
        mode = SENSOR_EXPECTATION_MODE.get(sensor_name, 'system')
        if mode == 'heating_active': base = daily_df.get('Active_Mins', system_on_minutes)
        elif mode == 'dhw_active': base = daily_df.get('Total_DHW_Mins', system_on_minutes)
        elif mode == 'system_slow': base = (system_on_minutes / 60.0).apply(np.ceil)
        else: base = system_on_minutes
        
        return base.replace(0, np.nan)

    def format_dq_df(df_in):
        df_out = df_in.copy()
        df_out.index.name = "Date"
        df_out.index = df_out.index.strftime('%d-%m-%Y')
        return df_out

    dq_tab1, dq_tab2, dq_tab3, dq_tab4, dq_tab5 = st.tabs(["Overview", "Category Drill-Down", "All Sensors", "Heartbeats", "⚠️ Unmapped Data"])

    with dq_tab1:
        st.markdown("### System Health Scorecard")
        dq_avg = daily_df['DQ_Score'].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Data Health", f"{dq_avg:.1f}%")
        gold = len(daily_df[daily_df['DQ_Tier'].astype(str).str.contains("Gold", na=False)])
        silver = len(daily_df[daily_df['DQ_Tier'].astype(str).str.contains("Silver", na=False)])
        c2.metric("Gold Days", gold)
        c3.metric("Silver Days", silver)
        
        overview_df = daily_df[['DQ_Tier']].copy()
        group_cols = []
        for group_name, sensors in SENSOR_GROUPS.items():
            if "Events" in group_name or "Event" in group_name: continue
            valid_sensors = [s for s in sensors if f"DQ_{s}_Count" in daily_df.columns or f"{s}_count" in daily_df.columns]
            if not valid_sensors: continue
            group_pcts = []
            for s in valid_sensors:
                col = f"DQ_{s}_Count" if f"DQ_{s}_Count" in daily_df.columns else f"{s}_count"
                pct = availability_pct(daily_df[col], expected_window_series(s, system_on_minutes))
                group_pcts.append(pct)
            if group_pcts:
                overview_df[group_name] = pd.concat(group_pcts, axis=1).mean(axis=1).round(0)
                group_cols.append(group_name)
        
        overview_disp = format_dq_df(overview_df[['DQ_Tier'] + group_cols])
        st.dataframe(overview_disp.style.background_gradient(subset=group_cols, cmap='RdYlGn', vmin=0, vmax=100).format("{:.0f}", subset=group_cols), width="stretch")

    with dq_tab2:
        st.markdown("### Category Inspector")
        cat = st.selectbox("Select System Category", list(SENSOR_GROUPS.keys()))
        selected = SENSOR_GROUPS.get(cat, [])
        cat_df = pd.DataFrame(index=daily_df.index)
        valid_cols = []
        
        for sensor in selected:
            col_name = f"DQ_{sensor}_Count" if f"DQ_{sensor}_Count" in daily_df.columns else f"{sensor}_count"
            if col_name in daily_df.columns:
                mode = SENSOR_EXPECTATION_MODE.get(sensor, 'system')
                if mode == 'event_only' or 'defrost' in sensor.lower():
                    cat_df[sensor] = daily_df[col_name].fillna(0).astype(int)
                else:
                    exp = expected_window_series(sensor, system_on_minutes)
                    cat_df[sensor] = availability_pct(daily_df[col_name], exp).round(0)
                valid_cols.append(sensor)
        
        cat_disp = format_dq_df(cat_df)
        if not cat_disp.empty:
            normal_cols = [c for c in valid_cols if SENSOR_EXPECTATION_MODE.get(c, 'system') != 'event_only']
            styler = cat_disp.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(subset=normal_cols, cmap='RdYlGn', vmin=0, vmax=100)
            st.dataframe(styler, width="stretch")
        else:
            st.info("No data for this category.")

    with dq_tab3:
        st.markdown("### 🧬 Master Sensor Matrix")
        
        # 1. Build Data
        count_cols = [c for c in daily_df.columns if c.endswith('_count') or c.endswith('_Count')]
        if count_cols:
            flat_data = {}
            for c in count_cols:
                clean_name = c.replace('DQ_', '').replace('_Count', '').replace('_count', '')
                if 'short_cycle' in clean_name.lower(): continue
                mode = SENSOR_EXPECTATION_MODE.get(clean_name, 'system')
                
                if mode == 'event_only': 
                    flat_data[clean_name] = daily_df[c].fillna(0).astype(int)
                else: 
                    flat_data[clean_name] = availability_pct(daily_df[c], expected_window_series(clean_name, system_on_minutes)).round(0)
            
            df_flat = pd.DataFrame(flat_data, index=daily_df.index)
            
            # 2. Re-construct Ordered Columns (Events Moved to End)
            new_columns = []
            valid_data_cols = []
            events_cat = None
            events_list = []
            
            # A. Normal Groups
            for cat_name, sensors in SENSOR_GROUPS.items():
                if "Event" in cat_name or "Events" in cat_name:
                    events_cat = cat_name
                    events_list = sensors
                    continue
                    
                found_sensors = [s for s in sensors if s in df_flat.columns]
                for s in found_sensors:
                    new_columns.append((cat_name, s))
                    valid_data_cols.append(s)

            # B. Rooms
            room_cols = sorted([c for c in df_flat.columns if c.startswith('Room_') and c not in valid_data_cols])
            for r in room_cols:
                new_columns.append(("🌡️ Rooms", r.replace('Room_', '')))
                valid_data_cols.append(r)

            # C. Others
            remaining = sorted([c for c in df_flat.columns if c not in valid_data_cols and (not events_list or c not in events_list)])
            for rem in remaining:
                new_columns.append(("Other", rem))
                valid_data_cols.append(rem)

            # D. Events (Appended Last)
            if events_cat:
                found_events = [s for s in events_list if s in df_flat.columns]
                for s in found_events:
                    new_columns.append((events_cat, s))
                    valid_data_cols.append(s)

            # 3. Build Final DataFrame
            df_final = df_flat[valid_data_cols].copy()
            df_final.columns = pd.MultiIndex.from_tuples(new_columns)
            df_final = format_dq_df(df_final)

            # 4. Apply Styles
            event_cols = []
            normal_cols = []
            
            for col_tuple in df_final.columns:
                sensor_name = col_tuple[1] 
                check_name = f"Room_{sensor_name}" if col_tuple[0] == "🌡️ Rooms" else sensor_name
                
                mode = SENSOR_EXPECTATION_MODE.get(check_name, 'system')
                if mode == 'event_only' or 'defrost' in str(check_name).lower():
                    event_cols.append(col_tuple)
                else:
                    normal_cols.append(col_tuple)

            styler = df_final.style.format("{:.0f}", na_rep="-")
            
            if normal_cols:
                styler = styler.background_gradient(subset=normal_cols, cmap='RdYlGn', vmin=0, vmax=100)
            if event_cols:
                styler = styler.map(lambda x: "background-color: #e0e0e0; color: #555555", subset=event_cols)
                
            st.dataframe(styler, width="stretch")

    with dq_tab4:
        st.markdown("### ❤️ Sensor Heartbeats")
        if st.button("Generate Heartbeat Baseline"):
            history_df = st.session_state.get("raw_history_df")
            if history_df is None or history_df.empty:
                st.warning("No raw history available.")
            else:
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
            st.dataframe(pd.DataFrame(pat_data), width="stretch")

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
            st.dataframe(pd.DataFrame(stats).set_index("Entity"), width="stretch")
        else:
            st.success("All entities mapped.")