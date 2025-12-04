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

st.set_page_config(page_title="T.H.E.R.M.", layout="wide", page_icon="🔥")

# === SIDEBAR ===
st.sidebar.title("Heat Pump Analytics")
uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", accept_multiple_files=True, type="csv")
show_inspector = st.sidebar.checkbox("Show File Inspector", value=False)

# === CACHING LOGIC ===
@st.cache_data(show_spinner=False) # <--- Added st.cache_data here for safety, though it's called by the session state logic.
def process_data(files):
    key = tuple(sorted((f.name, f.size) for f in files))
    if "cached" in st.session_state and st.session_state["cached"]["key"] == key:
        return st.session_state["cached"]
    
    pbar = st.progress(0, "Loading...")
    # NOTE: The progress bar function is defined here, so the cache decorator needs to be managed carefully.
    # We rely on the session state check to handle the memoization, but adding the decorator is standard practice.
    
    res = data_loader.load_and_clean_data(files, lambda t, p: pbar.progress(p, t))
    if not res: return None
    
    # Store history for baselines
    st.session_state["raw_history_df"] = res["raw_history"]
    st.session_state["heartbeat_baseline"] = res["baselines"]
    st.session_state["heartbeat_baseline_path"] = res.get("baseline_path") # Pass path for UI
    
    pbar.progress(40, "Hydraulics...")
    df = processing.apply_gatekeepers(res["df"])
    pbar.progress(60, "Runs...")
    runs = processing.detect_runs(df)
    pbar.progress(80, "Daily Stats...")
    daily = processing.get_daily_stats(df)
    pbar.progress(100, "Done")
    
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