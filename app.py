# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings

# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
pd.set_option('future.no_silent_downcasting', True)

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import processing

st.set_page_config(page_title="therm v2 beta", layout="wide", page_icon="🔥")

# === SIDEBAR ===
st.sidebar.image("assets/therm_logo.png", use_container_width=True)
st.sidebar.markdown("**Thermal Health & Efficiency Reporting Module v2 beta**")

# === FILE UPLOADER ===
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", accept_multiple_files=True, type="csv")
show_inspector = st.sidebar.checkbox("Show File Inspector", value=False)

# === MANUAL CACHING LOGIC ===
def get_processed_data(files, user_config):
    """
    Manually manages the cache to prevent UI re-renders of the loading screen.
    """
    # Create a unique key based on file properties AND config
    if not files: return None
    
    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config) 
    combined_key = (files_key, config_key)
    
    # Check Cache
    if "cached" in st.session_state:
        if st.session_state["cached"]["key"] == combined_key:
            return st.session_state["cached"]
    
    # --- IF CACHE MISS: RUN PROCESSING ---
    
    # Use st.status ONLY here, so it never replays on cache hits
    status_container = st.status("Processing Data...", expanded=True)
    try:
        # 1. Load
        status_container.write("📂 Loading and merging files (Numeric + State)...")
        progress_cb = lambda t, p: status_container.write(f"Reading: {t}")
        
        res = data_loader.load_and_clean_data(files, user_config, progress_cb)
        if not res:
            status_container.update(label="Error: No data found", state="error")
            return None
            
        # 2. Hydraulics
        status_container.write("⚙️ Applying physics engine...")
        df = processing.apply_gatekeepers(res["df"], user_config)
        
        # 3. Runs
        status_container.write("🏃 Detecting Runs (DHW/Heating)...")
        runs = processing.detect_runs(df, user_config)
        
        # 4. Daily Stats
        status_container.write("Calculating daily stats...")
        daily = processing.get_daily_stats(df)

        # --- NEW: capability detection ---
        has_flowrate = 'FlowRate' in df.columns and df['FlowRate'].notna().any()
        caps = st.session_state.get("capabilities", {})
        caps["has_flowrate"] = has_flowrate
        st.session_state["capabilities"] = caps

        status_container.update(label="Processing Complete!", state="complete", expanded=False)
        
        # Save to Cache
        cache = {
            "key": combined_key,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": res["patterns"]
        }
        st.session_state["cached"] = cache
        
        # Save Baselines
        st.session_state["heartbeat_baseline"] = res["baselines"]
        st.session_state["heartbeat_baseline_path"] = res.get("baseline_path")
        
        return cache

    except Exception as e:
        status_container.update(label="Processing Failed", state="error")
        st.error(f"An error occurred: {e}")
        return None

# === MAIN LOGIC ===
if uploaded_files:
    if show_inspector:
        st.title("🕵️ Pre-Flight Inspector")
        summary, details = inspector.inspect_raw_files(uploaded_files)
        st.dataframe(summary, width="stretch")
        for f, d in details.items():
            with st.expander(f"📄 {f}"):
                st.write(f"Entities found: {len(d['entities_found'])}")
                st.code("\n".join(d['entities_found']))
    else:
        # --- CONFIGURATION WORKFLOW ---
        if "system_config" not in st.session_state:
            config_object = mapping_ui.render_configuration_interface(uploaded_files)
            if config_object:
                st.session_state["system_config"] = config_object
                st.rerun()
        else:
            # 1. Sidebar Config
            with st.sidebar:
                st.divider()
                st.markdown("### ⚙️ Configuration")
                st.caption(f"Profile: **{st.session_state['system_config'].get('profile_name')}**")
                mapping_ui.render_config_download(st.session_state["system_config"])
                if st.button("🔄 Change Profile / Remap"):
                    del st.session_state["system_config"]
                    if "cached" in st.session_state: del st.session_state["cached"]
                    st.rerun()

            # 2. GET DATA (Manual Cache Check)
            data = get_processed_data(uploaded_files, st.session_state["system_config"])

            # 3. Debugger
            with st.sidebar.expander("🛠️ Data Debugger", expanded=False):
                st.write("**Full Config:**")
                st.json(st.session_state["system_config"])
                if data and "df" in data:
                    st.write("**Columns:**", list(data["df"].columns))
                    if data.get("runs"):
                        st.write(f"Detected {len(data['runs'])} runs")
                        st.write(f"Run 0 Type: {data['runs'][0]['run_type']}")

            # 4. Render Dashboard
            if data:
                caps = st.session_state.get("capabilities", {})
                has_flowrate = caps.get("has_flowrate", True)

                if not has_flowrate:
                    st.info("Flow sensor not mapped - energy output (Heat kWh), COP and SCOP are disabled. The dashboard is running in **Power & Temps only** mode.")

                mode = st.sidebar.radio("Analysis Mode", ["Long-Term Trends", "Run Inspector", "Data Quality Audit"])
                
                # Context injection
                st.session_state["ai_context_user"] = st.session_state["system_config"].get("ai_context", {})

                if mode == "Long-Term Trends":
                    view_trends.render_long_term_trends(data["daily"], data["df"], data["runs"])
                elif mode == "Run Inspector":
                    view_runs.render_run_inspector(data["df"], data["runs"])
                elif mode == "Data Quality Audit":
                    hb_path = st.session_state.get("heartbeat_baseline_path")
                    view_quality.render_data_quality(
                        data["daily"], data["df"], [], 
                        data["patterns"], hb_path
                    )
else:
    st.info("Upload CSV files to begin.")
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About therm"):
        st.markdown("**therm v2.0** - Heat Pump Performance Analysis")



