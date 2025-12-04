# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings

# --- FIX: Console Error Suppression ---
# 1. Suppress "Mean of empty slice" spam from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

# 2. Suppress Pandas "Downcasting" FutureWarning
pd.set_option('future.no_silent_downcasting', True)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import os
import json

# Import our modularised files
import data_loader
import processing
import inspector
import view_trends
import view_runs
import view_quality

st.set_page_config(page_title="therm", layout="wide", page_icon="🔥")

# === SIDEBAR ===
st.sidebar.image("assets/therm_logo.png", use_container_width=True)
st.sidebar.markdown(
    "**Thermal Health & Efficiency Reporting Module**"
)
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", accept_multiple_files=True, type="csv")
show_inspector = st.sidebar.checkbox("Show File Inspector", value=False)

# === CACHING LOGIC ===
@st.cache_data(show_spinner=False) 
def process_data(files):
    key = tuple(sorted((f.name, f.size) for f in files))
    if "cached" in st.session_state and st.session_state["cached"]["key"] == key:
        return st.session_state["cached"]
    
    # START OF CHANGE: Create a placeholder container
    placeholder = st.empty()
    with placeholder.container():
        pbar = st.progress(0, "Loading...")
        # Define a callback that updates the bar within the placeholder
        progress_cb = lambda t, p: pbar.progress(p, t)
        
        # NOTE: If we use the placeholder, we must define the cache logic slightly differently,
        # or rely only on the session state check to handle memoization outside of the Streamlit cache decorator.
        
        # Reset progress bar state at the start
        
        res = data_loader.load_and_clean_data(files, progress_cb)
        if not res: 
            placeholder.empty() # Clear the placeholder on error
            return None
        
        pbar.progress(40, "Hydraulics...")
        df = processing.apply_gatekeepers(res["df"])
        pbar.progress(60, "Runs...")
        runs = processing.detect_runs(df)
        pbar.progress(80, "Daily Stats...")
        daily = processing.get_daily_stats(df)
        pbar.progress(100, "Done")
        
        # CRITICAL FIX: Clear the placeholder after the process is truly complete
        placeholder.empty()

    # END OF CHANGE (The rest of the cache logic remains outside the placeholder context)
    
    # Store history for baselines (moved outside the placeholder container)
    st.session_state["raw_history_df"] = res["raw_history"]
    st.session_state["heartbeat_baseline"] = res["baselines"]
    st.session_state["heartbeat_baseline_path"] = res.get("baseline_path")
    
    cache = {
        "key": key, "df": df, "runs": runs, "daily": daily, 
        "unmapped": res["unmapped_entities"], "patterns": res["patterns"]
    }
    st.session_state["cached"] = cache
    return cache

# === MAIN LOGIC ===
if uploaded_files:
    if show_inspector:
        st.title("🕵️ Pre-Flight Inspector")
        summary, details = inspector.inspect_raw_files(uploaded_files)
        st.dataframe(summary, use_container_width=True)
        for f, d in details.items():
            with st.expander(f"📄 {f}"):
                st.write(f"Entities found: {len(d['entities_found'])}")
                st.code("\n".join(d['entities_found']))
    else:
        data = process_data(uploaded_files)
        if data:
            mode = st.sidebar.radio("Analysis Mode", ["Long-Term Trends", "Run Inspector", "Data Quality Audit"])
            
            if mode == "Long-Term Trends":
                # START OF CHANGE: Pass the pre-computed runs list
                view_trends.render_long_term_trends(data["daily"], data["df"], data["runs"])
                # END OF CHANGE
            elif mode == "Run Inspector":
                view_runs.render_run_inspector(data["df"], data["runs"])
            elif mode == "Data Quality Audit":
                # Pass path if available
                hb_path = st.session_state.get("heartbeat_baseline_path")
                view_quality.render_data_quality(
                    data["daily"], data["df"], data["unmapped"], 
                    data["patterns"], hb_path
                )
else:
    st.info("Upload CSV files to begin.")

# === ABOUT SECTION (Placed at the bottom of the sidebar) ===

# Using st.expander is a common way to include an "About" section in the sidebar
st.sidebar.markdown("---") # Optional separator

with st.sidebar.expander("ℹ️ About therm"):
    st.markdown(
        """
        **therm** (Thermal Health & Efficiency Reporting Module) is an open-source tool dedicated to **heat pump performance analysis**.
        
        Simply upload your heat pump data logs (in CSV format) to instantly visualize key performance metrics, diagnose hydraulic and control issues, and audit data quality.
        
        Our goal is to help homeowners, installers, and researchers **maximize the efficiency** and lifespan of heat pump systems by providing accessible, insightful analytics.
        
        *Made by the community, for the community.*
        """
    )