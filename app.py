# app.py
import warnings
import pandas as pd
import numpy as np
import streamlit as st

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import processing
from utils import safe_div


# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
pd.set_option("future.no_silent_downcasting", True)

# --- UI helpers ---------------------------------------------------------------
def _scroll_to_top_if_requested() -> None:
    """
    If a previous step requested a scroll-to-top (e.g. after 'Process Uploaded Data'),
    inject a small JS snippet once and then clear the flag.
    """
    if st.session_state.get("scroll_to_top"):
        st.markdown(
            "<script>window.scrollTo(0, 0);</script>",
            unsafe_allow_html=True,
        )
        st.session_state["scroll_to_top"] = False
# --- end scroll helper --------------------------------------------------------

st.set_page_config(page_title="therm v2 beta", layout="wide", page_icon="assets/therm_logo_browser_tab.png")

# Decide whether we're in System Setup (no processed config yet)
in_system_setup = "system_config" not in st.session_state

# Apply any one-shot scroll request
_scroll_to_top_if_requested()

# Versioned key to allow hard-reset of the file uploader widget
if "csv_uploader_version" not in st.session_state:
    st.session_state["csv_uploader_version"] = 0


# === SIDEBAR HEADER ===
st.sidebar.image("assets/therm_logo.png", width="stretch")
st.sidebar.markdown("**Thermal Health & Efficiency Reporting Module v2 beta**")

with st.sidebar.expander("Data source & files", expanded=in_system_setup):
    # Use a versioned key so we can hard-reset the uploader
    uploaded_files = st.file_uploader(
        "Upload CSV(s)",
        accept_multiple_files=True,
        type="csv",
        key=f"csv_uploader_{st.session_state['csv_uploader_version']}",
    )

    show_inspector = st.checkbox("Show File Inspector", value=False)

    # Persist the filenames so we can show them even after processing
    if uploaded_files:
        st.session_state["uploaded_filenames"] = [f.name for f in uploaded_files]

    if "uploaded_filenames" in st.session_state:
        st.markdown("**Loaded files:**")
        for name in st.session_state["uploaded_filenames"]:
            st.markdown(f"- `{name}`")

        # Optional: allow the user to hard-reset everything
        if st.button("Clear files and start again", type="secondary"):
            for key in [
                "system_config",
                "cached",
                "uploaded_filenames",
                "capabilities",
            ]:
                st.session_state.pop(key, None)

            # Force the file_uploader to re-mount with a fresh key
            st.session_state["csv_uploader_version"] += 1

            st.rerun()



# === MANUAL CACHING LOGIC ===
def get_processed_data(files, user_config):
    """
    Manually manages the cache to prevent UI re-renders of the loading screen.
    """
    if not files:
        return None

    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config)
    combined_key = (files_key, config_key)

    if "cached" in st.session_state:
        if st.session_state["cached"]["key"] == combined_key:
            return st.session_state["cached"]

    status_container = st.status("Processing Data...", expanded=True)
    try:
        # 1. Load
        status_container.write("Loading and merging files (Numeric + State)...")
        progress_cb = lambda t, p: status_container.write(f"Reading: {t}")

        res = data_loader.load_and_clean_data(files, user_config, progress_cb)
        if not res:
            status_container.update(label="Error: No data found", state="error")
            return None

        # 2. Hydraulics
        status_container.write("Applying physics engine...")
        df = processing.apply_gatekeepers(res["df"], user_config)

        # 3. Runs
        status_container.write("Detecting Runs (DHW/Heating)...")
        runs = processing.detect_runs(df, user_config)

        # 4. Daily Stats
        status_container.write("Calculating daily stats...")
        daily = processing.get_daily_stats(df)

        # --- NEW: capability detection ---
        has_flowrate = "FlowRate" in df.columns and df["FlowRate"].notna().any()
        has_heat_sensor = "Heat" in df.columns and pd.to_numeric(df["Heat"], errors="coerce").fillna(0).abs().sum() > 0
        caps = st.session_state.get("capabilities", {})
        caps["has_flowrate"] = has_flowrate
        caps["has_heat_sensor"] = has_heat_sensor
        caps["has_energy_channel"] = has_flowrate or has_heat_sensor
        st.session_state["capabilities"] = caps

        status_container.update(label="Processing Complete!", state="complete", expanded=False)

        cache = {
            "key": combined_key,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": res["patterns"],
        }
        st.session_state["cached"] = cache

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
        st.title("Pre-Flight Inspector")
        summary, details = inspector.inspect_raw_files(uploaded_files)
        st.dataframe(summary, width="stretch")
        for f, d in details.items():
            with st.expander(f"Details: {f}"):
                st.write(f"Entities found: {len(d['entities_found'])}")
                st.code("\n".join(d["entities_found"]))
    else:
        # --- CONFIGURATION WORKFLOW ---
        if "system_config" not in st.session_state:
            config_object = mapping_ui.render_configuration_interface(uploaded_files)
            if config_object:
                st.session_state["system_config"] = config_object
                st.rerun()

        else:
            # 1. GET DATA (Manual Cache Check)
            data = get_processed_data(uploaded_files, st.session_state["system_config"])

            # 2. Sidebar: Analysis Mode, Global Stats (canonical), Configuration, Debugger
            with st.sidebar:
                # --- Analysis Mode ---
                st.markdown("### Analysis Mode")
                if data and "df" in data:
                    mode = st.radio(
                        "Select view",
                        ["Long-Term Trends", "Run Inspector", "Data Quality Audit"],
                    )

                    # Canonical global stats from processing engine
                    stats = processing.compute_global_stats(data["df"])
                    total_heat = stats["total_heat_kwh"]
                    total_elec = stats["total_elec_kwh"]
                    global_cop = stats["global_cop"]

                    runs_detected = len(data.get("runs") or [])
                else:
                    mode = None
                    st.info("Upload data and configure your system to begin analysis.")
                    total_heat = total_elec = global_cop = 0.0
                    runs_detected = 0

                # --- Global Stats (GREEN BLOCK ONLY, shared between LT Trends + Run Inspector) ---
                if data and mode in ("Long-Term Trends", "Run Inspector"):
                    st.markdown("### Global Stats")

                    # Runs Detected as headline
                    st.metric("Runs Detected", f"{runs_detected}")

                    # Detailed stats
                    st.metric("Total Heat Output", f"{total_heat:.1f} kWh")
                    st.metric("Total Electricity Input", f"{total_elec:.1f} kWh")
                    st.metric("Global COP", f"{global_cop:.2f}")

                # --- Configuration block ---
                st.markdown("### Configuration")
                profile_name = st.session_state["system_config"].get("profile_name", "Unnamed Profile")
                st.caption(f"Profile: **{profile_name}**")

                # Back to System Setup instead of Change Profile / Remap here
                if st.button("↩ Back to System Setup"):
                    st.session_state.pop("system_config", None)
                    st.session_state.pop("cached", None)
                    st.session_state.pop("capabilities", None)
                    st.rerun()

                # NOTE:
                # In Analysis Mode screens we intentionally do NOT show
                # the "Download Profile" button. Profile download remains
                # available as part of the System Setup / mapping UI.

                # --- Data Debugger at the bottom ---
                with st.expander("Data Debugger", expanded=False):
                    st.write("**Full Config:**")
                    st.json(st.session_state["system_config"])
                    if data and "df" in data:
                        st.write("**Columns:**", list(data["df"].columns))
                    if data and data.get("runs"):
                        st.write(f"Detected {len(data['runs'])} runs")
                        st.write(f"Run 0 Type: {data['runs'][0]['run_type']}")


            # 3. Render Dashboard (main panel)
            if data and mode:
                caps = st.session_state.get("capabilities", {})
                has_flowrate = caps.get("has_flowrate", True)
                has_energy_channel = caps.get("has_energy_channel", True)

                if not has_energy_channel:
                    st.info(
                        "No Flow Rate or Heat output sensor mapped — energy output (Heat kWh), COP and SCOP are disabled. "
                        "The dashboard is running in Power & Temps only mode."
                    )

                # Context injection for AI
                st.session_state["ai_context_user"] = st.session_state["system_config"].get("ai_context", {})

                if mode == "Long-Term Trends":
                    view_trends.render_long_term_trends(data["daily"], data["df"], data["runs"])
                elif mode == "Run Inspector":
                    view_runs.render_run_inspector(data["df"], data["runs"])
                elif mode == "Data Quality Audit":
                    hb_path = st.session_state.get("heartbeat_baseline_path")
                    view_quality.render_data_quality(
                        data["daily"], data["df"], [], data["patterns"], hb_path
                    )

else:
    st.info("Upload CSV files to begin.")
    st.sidebar.markdown("---")
    with st.sidebar.expander("About therm"):
        st.markdown("**therm v2.0** - Heat Pump Performance Analysis")