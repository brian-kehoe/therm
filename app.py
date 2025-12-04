# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import json

# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
pd.set_option('future.no_silent_downcasting', True)

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

# --- MODULE IMPORTS ---
import data_loader
import processing
import inspector
import view_trends
import view_runs
import view_quality
import mapping_ui
import config_manager

st.set_page_config(page_title="therm v2 beta", layout="wide", page_icon="🔥")

# === SIDEBAR ===
st.sidebar.image("assets/therm_logo.png", use_container_width=True)
st.sidebar.markdown("**Thermal Health & Efficiency Reporting Module v2 beta**")

# === FILE UPLOADER ===
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", accept_multiple_files=True, type="csv")
show_inspector = st.sidebar.checkbox("Show File Inspector", value=False)

# === CACHING LOGIC ===
@st.cache_data(show_spinner=False) 
def process_data(files, user_config):
    # Create a unique key based on file properties AND the configuration profile
    # This ensures if you change the mapping, the cache invalidates and re-runs.
    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config) # Hash full config including rooms_per_zone
    combined_key = (files_key, config_key)

    if "cached" in st.session_state and st.session_state["cached"]["key"] == combined_key:
        return st.session_state["cached"]
    
    # Placeholder container for progress bar
    placeholder = st.empty()
    with placeholder.container():
        pbar = st.progress(0, "Loading...")
        progress_cb = lambda t, p: pbar.progress(p, t)
        
        # 1. Load & Normalize
        res = data_loader.load_and_clean_data(files, user_config, progress_cb)
        if not res: 
            placeholder.empty()
            return None
        
        # 2. Hydraulics
        pbar.progress(40, "Hydraulics...")
        df = processing.apply_gatekeepers(res["df"])
        
        # 3. Runs
        pbar.progress(60, "Runs...")
        # === PASS ROOM MAPPING HERE ===
        rooms_map = user_config.get("rooms_per_zone", {})
        runs = processing.detect_runs(df, rooms_map)
        
        # 4. Daily Stats
        pbar.progress(80, "Daily Stats...")
        daily = processing.get_daily_stats(df)
        
        pbar.progress(100, "Done")
        placeholder.empty()

    # Store history (outside cache object)
    st.session_state["raw_history_df"] = res["raw_history"]
    st.session_state["heartbeat_baseline"] = res["baselines"]
    st.session_state["heartbeat_baseline_path"] = res.get("baseline_path")
    
    cache = {
        "key": combined_key, 
        "df": df, 
        "runs": runs, 
        "daily": daily, 
        "unmapped": [], # Deprecated in v2
        "patterns": res["patterns"]
    }
    st.session_state["cached"] = cache
    return cache

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
        # --- NEW: CONFIGURATION WORKFLOW ---
        
        # State A: No Configuration -> Show Wizard
        if "system_config" not in st.session_state:
            config_object = mapping_ui.render_configuration_interface(uploaded_files)
            
            if config_object:
                st.session_state["system_config"] = config_object
                st.rerun()
        
        # State B: Configuration Loaded -> Show Dashboard
        else:
            # 1. Show Configuration Controls in Sidebar
            with st.sidebar:
                st.divider()
                st.markdown("### ⚙️ Configuration")
                st.caption(f"Profile: **{st.session_state['system_config'].get('profile_name')}**")
                
                # Download Button
                mapping_ui.render_config_download(st.session_state["system_config"])
                
                # Reset Button
                if st.button("🔄 Change Profile / Remap"):
                    del st.session_state["system_config"]
                    if "cached" in st.session_state:
                        del st.session_state["cached"]
                    st.rerun()

            # 2. Process Data (using the config)
            data = process_data(uploaded_files, st.session_state["system_config"])

            # === DEBUGGER ===
            with st.sidebar.expander("🛠️ Data Debugger", expanded=False):
                st.write("**Full Config:**")
                st.json(st.session_state["system_config"])
                if data and "df" in data:
                    st.write("**Columns:**", list(data["df"].columns))
                    if data.get("runs"):
                        st.write(f"Detected {len(data['runs'])} runs")
                        # Show relevant rooms to verify linkage
                        st.write(data["runs"][0].get('relevant_rooms', 'No rooms linked'))
            # ================
            
            # 3. Inject AI Context
            if "system_config" in st.session_state:
                st.session_state["ai_context_user"] = st.session_state["system_config"].get("ai_context", {})

            # 4. Render Dashboard
            if data:
                mode = st.sidebar.radio("Analysis Mode", ["Long-Term Trends", "Run Inspector", "Data Quality Audit"])
                
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
    
    # About Section
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About therm"):
        st.markdown(
            """
            **therm** is an open-source tool for **heat pump performance analysis**.
            
            Upload your data logs to visualize performance, diagnose issues, and audit data quality.
            
            *Version 2.0 (Public Beta)*
            """
        )