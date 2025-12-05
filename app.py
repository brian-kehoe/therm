# app.py

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import traceback  # NEW: for detailed debug

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import ha_loader   # NEW: Home Assistant loader
import processing
from utils import safe_div

# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
pd.set_option("future.no_silent_downcasting", True)

# --- UI helpers ---------------------------------------------------------------
def _scroll_to_top_if_requested() -> None:
    """ If a previous step requested a scroll-to-top (e.g. after 'Process Uploaded Data'), inject a small JS snippet once and then clear the flag. """
    if st.session_state.get("scroll_to_top"):
        st.markdown(
            """
            <script>
              window.scrollTo({ top: 0, behavior: 'smooth' });
            </script>
            """,
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

        # ------------------------------------------------------------------
        # New: per-file source selector (Grafana vs Home Assistant)
        # ------------------------------------------------------------------
        file_sources = st.session_state.get("file_sources", {})
        st.markdown("**File source type:**")
        for idx, f in enumerate(uploaded_files):
            fname = getattr(f, "name", f"file_{idx}")
            default_source = file_sources.get(fname, "grafana")
            source_label = st.radio(
                label=f"Source for `{fname}`",
                options=["Grafana / Influx CSV", "Home Assistant CSV"],
                index=0 if default_source == "grafana" else 1,
                key=f"file_source_{idx}",
            )
            internal_code = "grafana" if source_label.startswith("Grafana") else "ha"
            file_sources[fname] = internal_code
        st.session_state["file_sources"] = file_sources
        # ------------------------------------------------------------------

    if "uploaded_filenames" in st.session_state:
        st.markdown("**Loaded files:**")
        for name in st.session_state["uploaded_filenames"]:
            st.markdown(f"- `{name}`")

    if st.button("Clear files and start again", type="secondary"):
        for key in [
            "system_config",
            "cached",
            "uploaded_filenames",
            "capabilities",
            "loaded_profile_signature",
            "available_sensors",
            "available_sensors_files_key",
        ]:
            st.session_state.pop(key, None)
        st.session_state["csv_uploader_version"] += 1
        st.rerun()

# === MANUAL CACHING & SOURCE-ROUTED ENGINE LOGIC ===
def get_processed_data(files, user_config):
    """ Manages cache and routes file loading through Grafana or HA loader depending on user selection. """
    if not files:
        return None

    # Determine dataset source from per-file selection
    file_sources = st.session_state.get("file_sources", {})
    sources = set()
    for f in files:
        fname = getattr(f, "name", None)
        src = file_sources.get(fname, "grafana")
        sources.add(src)

    if len(sources) > 1:
        st.error(
            "Mixed data sources detected. Please set **all files** to either "
            "`Grafana / Influx CSV` or `Home Assistant CSV`, not a mixture."
        )
        return None

    dataset_source = sources.pop() if sources else "grafana"

    # Build cache key including dataset source
    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config)
    combined_key = (files_key, config_key, dataset_source)

    if "cached" in st.session_state:
        if st.session_state["cached"].get("key") == combined_key:
            return st.session_state["cached"]

    status_container = st.status("Processing data...", expanded=True)

    try:
        if dataset_source == "ha":
            status_container.update(label="Loading Home Assistant history…", state="running")
            df_res = ha_loader.process_ha_files(files, user_config, progress_cb=lambda t, p: status_container.update(label=f"Reading HA file {t}", state="running"))
            if not df_res or df_res.get("df") is None:
                status_container.update(label="Error: No usable HA data", state="error")
                st.error("No usable HA data found in the uploaded file(s). Please check you exported the correct HA history CSV.")
                return None
            df = df_res["df"]
            runs = df_res["runs"]
            daily = df_res["daily"]
            raw_history = df_res.get("raw_history")
            patterns = None
            baselines = None
        else:
            status_container.update(label="Loading & cleaning Grafana / Influx CSV data…", state="running")
            res = data_loader.load_and_clean_data(files, user_config, progress_cb=lambda t, p: status_container.write(f"Reading: {t}"))
            if not res or res.get("df") is None:
                status_container.update(label="Error: No data found", state="error")
                return None
            df_pre = res["df"]
            status_container.update(label="Applying physics engine...", state="running")
            df = processing.apply_gatekeepers(df_pre, user_config)
            status_container.update(label="Detecting Runs (DHW/Heating)…", state="running")
            runs = processing.detect_runs(df, user_config)
            status_container.update(label="Calculating daily stats…", state="running")
            daily = processing.get_daily_stats(df)
            raw_history = res.get("raw_history")
            patterns = res.get("patterns")
            baselines = res.get("baselines")

        status_container.update(label="Processing complete!", state="complete", expanded=False)

        cache = {
            "key": combined_key,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": patterns,
            "raw_history": raw_history,
            "source": dataset_source,
        }
        st.session_state["cached"] = cache
        st.session_state["heartbeat_baseline"] = baselines
        st.session_state["heartbeat_baseline_path"] = None

        return cache

    except Exception as e:
        status_container.update(label="Processing failed", state="error")
        st.error(f"An error occurred during data processing: {e}")
        tb = traceback.format_exc()
        with st.expander("Debug: Full traceback", expanded=False):
            st.code(tb)
        return None


# === MAIN APP LOGIC ===
if uploaded_files:
    if show_inspector:
        st.title("Pre-Flight Inspector")
        summary, details_all = inspector.inspect_raw_files(uploaded_files)
        st.dataframe(summary, width="stretch")
        file_details = details_all.get("file_details", {})
        sensor_debug = details_all.get("sensor_debug", {})

        # FILE-LEVEL DETAILS
        for fname, info in file_details.items():
            with st.expander(f"File Details: {fname}", expanded=False):
                entities = info.get("entities_found", [])
                st.write(f"Entities found: {len(entities)}")
                if entities:
                    st.code("\n".join(entities))

                cols = info.get("columns_raw", [])
                if cols:
                    st.markdown("**Columns:**")
                    st.code("\n".join(cols))

        # SENSOR-LEVEL DEBUG
        for fname, sensors in sensor_debug.items():
            with st.expander(f"Sensor Debug: {fname}", expanded=False):
                if "error" in sensors:
                    st.error(sensors["error"])
                    continue
                st.write(f"Sensors detected: {len(sensors)}")
                st.json(sensors)
    else:
        # CONFIGURATION WORKFLOW
        if "system_config" not in st.session_state:
            config_object = mapping_ui.render_configuration_interface(uploaded_files)
            if config_object:
                st.session_state["system_config"] = config_object
                st.rerun()
        else:
            # 1. GET DATA (Manual Cache / Loader Switch)
            data = get_processed_data(uploaded_files, st.session_state["system_config"])

            # 2. Sidebar: Analysis Mode / Global Stats / Debug
            if data and "df" in data:
                mode = st.radio(
                    "Select view",
                    ["Long-Term Trends", "Run Inspector", "Data Quality Audit"],
                )
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

            # Global Stats (common)
            if data and mode in ("Long-Term Trends", "Run Inspector"):
                st.markdown("### Global Stats")
                st.metric("Runs Detected", f"{runs_detected}")
                st.metric("Total Heat Output", f"{total_heat:.1f} kWh")
                st.metric("Total Electricity Input", f"{total_elec:.1f} kWh")
                st.metric("Global COP", f"{global_cop:.2f}")

            if st.button("↩ Back to System Setup"):
                for key in [
                    "system_config",
                    "cached",
                    "capabilities",
                    "loaded_profile_signature",
                    "available_sensors",
                    "available_sensors_files_key",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

            # Data Debugger / Download Options
            with st.expander("Data Debugger", expanded=False):
                debug_flag = st.checkbox(
                    "Enable engine debug traces",
                    value=st.session_state.get("debug_engine", False),
                )
                st.session_state["debug_engine"] = debug_flag

                config = st.session_state.get("system_config")
                if config is not None:
                    st.write("**Full Config:**")
                    st.json(config)
                else:
                    st.write("**Full Config:** (no system_config in session state yet)")

                if data is not None and "df" in data:
                    df_dbg = data["df"]
                    st.write("**Columns:**", list(df_dbg.columns))
                    core_cols = [
                        c
                        for c in [
                            "Power",
                            "Heat",
                            "FlowTemp",
                            "ReturnTemp",
                            "FlowRate",
                            "DeltaT",
                            "is_active",
                            "is_DHW",
                            "is_heating",
                        ]
                        if c in df_dbg.columns
                    ]
                    if core_cols:
                        st.write("**Core engine columns (head):**")
                        st.dataframe(df_dbg[core_cols].head(50))

                    # Download merged engine dataframe
                    ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M")
                    engine_csv = df_dbg.to_csv(index=True).encode("utf-8")
                    st.download_button(
                        "⬇ Download merged engine dataframe (CSV)",
                        data=engine_csv,
                        file_name=f"Data Debugger MERGED {ts}_export.csv",
                        mime="text/csv",
                        key="download_merged_engine_df",
                    )

                    raw_history = data.get("raw_history")
                    if isinstance(raw_history, pd.DataFrame) and not raw_history.empty:
                        raw_csv = raw_history.to_csv(index=True).encode("utf-8")
                        st.download_button(
                            "⬇ Download pre-physics merged dataframe (raw_history CSV)",
                            data=raw_csv,
                            file_name=f"Data Debugger RAW {ts}_export.csv",
                            mime="text/csv",
                            key="download_raw_history_df",
                        )

                if data is not None and data.get("runs"):
                    st.write(f"Detected {len(data['runs'])} runs")
                    st.write(f"Run 0 Type: {data['runs'][0].get('run_type', 'n/a')}")

            # 3. Render Dashboard (main panel)
            if data and mode:
                caps = st.session_state.get("capabilities", {})
                has_flowrate = caps.get("has_flowrate", True)
                has_energy_channel = caps.get("has_energy_channel", True)

                if not has_energy_channel:
                    st.info(
                        "No Flow Rate or Heat output sensor mapped — energy output (Heat kWh), COP and SCOP are disabled.\n"
                        "The dashboard is running in Power & Temps only mode."
                    )

                st.session_state["ai_context_user"] = st.session_state["system_config"].get("ai_context", {})

                if mode == "Long-Term Trends":
                    view_trends.render_long_term_trends(
                        data["daily"],
                        data["df"],
                        data["runs"]
                    )

                elif mode == "Run Inspector":
                    view_runs.render_run_inspector(
                        data["df"],
                        data["runs"]
                    )

                elif mode == "Data Quality Audit":
                    hb_path = st.session_state.get("heartbeat_baseline_path")
                    view_quality.render_data_quality(
                        data["daily"],
                        data["df"],
                        [],
                        data["patterns"],
                        hb_path,
                    )

            else:
                st.info("Upload CSV files to begin.")

    # Sidebar about panel
    st.sidebar.markdown("---")
    with st.sidebar.expander("About therm"):
        st.markdown("**therm v2.0** - Heat Pump Performance Analysis")
