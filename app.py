# app.py
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import traceback  # NEW: for detailed debug
import json  # NEW: for app debug bundle export
import io     # NEW: for in-memory ZIPs
import zipfile  # NEW: for ZIP downloads

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import ha_loader
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

    This targets the main Streamlit app container in the parent document,
    which is the element the user actually scrolls.
    """
    if st.session_state.get("scroll_to_top"):
        st.markdown(
            """
            <script>
            (function() {
                try {
                    // Streamlit main view container in the parent document
                    var app = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                    if (app && app.scrollTo) {
                        app.scrollTo({top: 0, behavior: 'smooth'});
                    } else if (window.parent && window.parent.scrollTo) {
                        // Fallback
                        window.parent.scrollTo({top: 0, behavior: 'smooth'});
                    }
                } catch (e) {
                    // Silently ignore any cross-origin or selector errors
                    console && console.warn && console.warn("Scroll-to-top failed:", e);
                }
            })();
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
        # New: auto-detect per-file source (Grafana vs Home Assistant),
        # with manual override.
        # ------------------------------------------------------------------
        file_sources = st.session_state.get("file_sources", {})

        st.markdown("**File source type:**")
        for idx, f in enumerate(uploaded_files):
            fname = getattr(f, "name", f"file_{idx}")

            auto_detected = data_loader.detect_file_source(f)
            # Prefer previously selected source, else auto-detected, else grafana
            default_source = file_sources.get(fname, auto_detected or "grafana")

            source_label = st.radio(
                label=f"Source for `{fname}`",
                options=["Grafana / Influx CSV", "Home Assistant CSV"],
                index=0 if default_source == "grafana" else 1,
                key=f"file_source_{idx}",
                help=f"Auto-detected: {auto_detected or 'unknown'}",
            )

            internal_code = "grafana" if source_label.startswith("Grafana") else "ha"
            file_sources[fname] = internal_code

        st.session_state["file_sources"] = file_sources
        # ------------------------------------------------------------------

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
            "loaded_profile_signature",      # NEW: forget which profile was loaded
            "available_sensors",             # optional: force re-scan
            "available_sensors_files_key",   # optional: force re-scan
        ]:
            st.session_state.pop(key, None)

        # Force the file_uploader to re-mount with a fresh key
        st.session_state["csv_uploader_version"] += 1
        st.rerun()





# === MANUAL CACHING LOGIC ===
def get_processed_data(files, user_config):
    """
    Manually manages the cache to prevent UI re-renders of the loading screen.
    Now also routes between Grafana/Influx and Home Assistant CSV modes.
    """
    if not files:
        return None

    import time
    def _log(msg: str) -> None:
        try:
            import sys
            sys.stdout.write(f"[process] {msg}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. Decide which data source is being used for this run
    # ------------------------------------------------------------------
    # file_sources is set in the sidebar "File source type" radios
    file_sources = st.session_state.get("file_sources", {})
    sources = set()
    for f in files:
        fname = getattr(f, "name", None)
        src = file_sources.get(fname, "grafana")
        sources.add(src)

    if len(sources) > 1:
        st.error(
            "Mixed data sources detected.\n\n"
            "For a given run, please set **all files** to either "
            "`Grafana / Influx CSV` or `Home Assistant CSV`, not a mixture."
        )
        return None

    dataset_source = sources.pop() if sources else "grafana"
    # Store for downstream debug bundle
    st.session_state["dataset_source"] = dataset_source


    # ------------------------------------------------------------------
    # 2. Manual cache key (include dataset_source so HA vs Grafana
    #    runs don't collide in the cache)
    # ------------------------------------------------------------------
    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config)
    combined_key = (files_key, config_key, dataset_source)

    if "cached" in st.session_state:
        if st.session_state["cached"].get("key") == combined_key:
            return st.session_state["cached"]

    status_container = st.status("Processing Data...", expanded=True)

    try:
        t_start = time.time()
        # ------------------------------------------------------------------
        # 3. Source-specific loading
        # ------------------------------------------------------------------
        if dataset_source == "ha":
            # Home Assistant mode
            status_container.write("Loading Home Assistant history CSV data...")
            t_ha = time.time()

            def progress_cb_ha(label: str, frac: float) -> None:
                """Progress callback expected by ha_loader."""
                try:
                    pct = int(max(0, min(frac * 100.0, 100.0)))
                except Exception:
                    pct = 0
                status_container.write(f"{label} ({pct}%)")

            res = ha_loader.process_ha_files(files, user_config, progress_cb=progress_cb_ha)
            _log(f"ha_loader.process_ha_files secs={time.time()-t_ha:.3f}")
            if not res or res.get("df") is None or res["df"].empty:
                status_container.update(
                    label="Error: No usable Home Assistant data found.",
                    state="error",
                )
                return None

            # ha_loader already applies physics + runs + daily
            df = res["df"]
            runs = res["runs"]
            daily = res["daily"]
            patterns = res.get("patterns")
            raw_history = res.get("raw_history")
            baselines = res.get("baselines")
            baseline_path = None

        else:
            # Grafana / Influx mode (existing pipeline)
            status_container.write("Loading and merging files (Numeric + State)...")
            progress_cb = lambda t, p: status_container.write(f"Reading: {t}")

            t_load = time.time()
            res = data_loader.load_and_clean_data(files, user_config, progress_cb)
            _log(f"data_loader.load_and_clean_data secs={time.time()-t_load:.3f}")
            if not res or res.get("df") is None or res["df"].empty:
                status_container.update(label="Error: No data found", state="error")
                return None

            status_container.write("Applying physics engine...")
            t_phys = time.time()
            df = processing.apply_gatekeepers(res["df"], user_config)
            _log(f"processing.apply_gatekeepers secs={time.time()-t_phys:.3f}")

            status_container.write("Detecting Runs (DHW/Heating)...")
            t_runs = time.time()
            runs = processing.detect_runs(df, user_config)
            _log(f"processing.detect_runs secs={time.time()-t_runs:.3f}")

            status_container.write("Calculating daily stats...")
            t_daily = time.time()
            daily = processing.get_daily_stats(df)
            _log(f"processing.get_daily_stats secs={time.time()-t_daily:.3f}")

            patterns = res["patterns"]
            raw_history = res.get("raw_history")
            baselines = res["baselines"]
            baseline_path = res.get("baseline_path")

        # ------------------------------------------------------------------
        # 4. Capability detection (unchanged logic, applied to df)
        # ------------------------------------------------------------------
        has_flowrate = "FlowRate" in df.columns and df["FlowRate"].notna().any()
        has_heat_sensor = (
            "Heat" in df.columns
            and pd.to_numeric(df["Heat"], errors="coerce").fillna(0).abs().sum() > 0
        )
        caps = st.session_state.get("capabilities", {})
        caps["has_flowrate"] = has_flowrate
        caps["has_heat_sensor"] = has_heat_sensor
        caps["has_energy_channel"] = has_flowrate or has_heat_sensor
        st.session_state["capabilities"] = caps

        # ------------------------------------------------------------------
        # 5. Finalise & cache
        # ------------------------------------------------------------------
        status_container.update(label="Processing Complete!", state="complete", expanded=False)
        cache = {
            "key": combined_key,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": patterns,
            # keep pre-physics merged dataframe for deep debugging
            "raw_history": raw_history,
        }
        st.session_state["cached"] = cache
        st.session_state["heartbeat_baseline"] = baselines
        st.session_state["heartbeat_baseline_path"] = baseline_path
        _log(f"total secs={time.time()-t_start:.3f} source={dataset_source} rows={len(df) if df is not None else 0}")
        return cache

    except Exception as e:
        status_container.update(label="Processing Failed", state="error")
        st.error(f"An error occurred: {e}")

        # Detailed traceback for debugging
        tb = traceback.format_exc()
        with st.expander("Debug: Full traceback", expanded=False):
            st.code(tb)

        return None

def _build_engine_debug_traces(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    """
    Build a compact set of 'interesting' engine rows for JSON debug traces.

    Heuristics:
    - COP_Real < 0 or > 7
    - Heat < 0
    - is_heating and is_DHW both True
    Falls back to the last `max_rows` rows if nothing matches.
    """
    if df is None or df.empty:
        return []

    # Candidate columns – we keep whichever are present
    candidate_cols = [
        "Heat", "Heat_Heating", "Heat_DHW",
        "Power", "Power_Heating", "Power_DHW",
        "FlowTemp", "ReturnTemp", "DeltaT", "FlowRate",
        "ValveMode", "DHW_Mode",
        "is_heating", "is_DHW", "is_active",
        "COP_Real", "COP_Graph",
    ]
    cols = [c for c in candidate_cols if c in df.columns]

    if not cols:
        return []

    # Build masks for "interesting" rows
    idx = df.index
    cop = pd.to_numeric(df.get("COP_Real", pd.Series(index=idx)), errors="coerce")
    heat = pd.to_numeric(df.get("Heat", pd.Series(index=idx)), errors="coerce")

    masks = []

    if not cop.isna().all():
        masks.append(cop < 0)
        masks.append(cop > 7)

    if not heat.isna().all():
        masks.append(heat < 0)

    if "is_heating" in df.columns and "is_DHW" in df.columns:
        masks.append(df["is_heating"] & df["is_DHW"])

    if masks:
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
        interesting = df[mask]
    else:
        interesting = df

    if interesting.empty:
        interesting = df.tail(max_rows)
    else:
        interesting = interesting.head(max_rows)

    # Serialise to a JSON-friendly list of dicts, including the index as Time_index
    records: list[dict] = []
    for idx, row in interesting[cols].iterrows():
        rec: dict[str, object] = {}
        # index as a label
        if isinstance(idx, pd.Timestamp):
            rec["Time_index"] = idx.isoformat()
        else:
            rec["Time_index"] = str(idx)

        for c in cols:
            val = row[c]
            if isinstance(val, pd.Timestamp):
                val = val.isoformat()
            elif isinstance(val, (np.floating, float, int, np.integer)):
                val = float(val)
            rec[c] = val
        records.append(rec)

    return records


# === MAIN LOGIC ===
if uploaded_files:
    if show_inspector:
        st.title("Pre-Flight Inspector")
        summary, details_all = inspector.inspect_raw_files(uploaded_files)

        # Store inspector outputs for downstream debug bundle (optional, AI-facing)
        st.session_state["inspector_summary"] = summary
        st.session_state["inspector_details"] = details_all

        st.dataframe(summary, width="stretch")
        file_details = details_all.get("file_details", {})
        sensor_debug = details_all.get("sensor_debug", {})

        # ---- FILE-LEVEL DETAILS ----
        for fname, info in file_details.items():
            with st.expander(f"File Details: {fname}", expanded=False):
                entities = info.get("entities_found", [])
                st.write(f"Entities found: {len(entities)}")
                if entities:
                    st.code("\n".join(entities))

                # Optional: show raw column list
                cols = info.get("columns_raw", [])
                if cols:
                    st.markdown("**Columns:**")
                    st.code("\n".join(cols))

        # ---- SENSOR-LEVEL DEBUG ----
        for fname, sensors in sensor_debug.items():
            with st.expander(f"Sensor Debug: {fname}", expanded=False):
                if "error" in sensors:
                    st.error(sensors["error"])
                    continue

                st.write(f"Sensors detected: {len(sensors)}")
                st.json(sensors)

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

                # Back to System Setup instead of Change Profile / Remap here
                if st.button("↩ Back to System Setup"):
                    for key in [
                        "system_config",
                        "cached",
                        "capabilities",
                        "loaded_profile_signature",      # NEW: allow same profile to be loaded again
                        "available_sensors",             # optional: force re-scan if files changed
                        "available_sensors_files_key",
                    ]:
                        st.session_state.pop(key, None)
                    st.rerun()


                # NOTE:
                # In Analysis Mode screens we intentionally do NOT show
                # the "Download Profile" button. Profile download remains
                # available as part of the System Setup / mapping UI.

                # --- Data Debugger at the bottom ---
                with st.expander("Data Debugger", expanded=False):
                    # Toggle for engine-level debug traces in processing.py
                    debug_flag = st.checkbox(
                        "Enable engine debug traces (JSON only)",
                        value=st.session_state.get("debug_engine", True),  # default ON
                    )
                    st.session_state["debug_engine"] = debug_flag

                    # Only expose download bundles here (no inline debug tables / JSON)
                    if data is not None and "df" in data:
                        df_dbg = data["df"]
                        ts_label = pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M")

                        # 1) Merged engine dataframe (post-physics, post-tariff)
                        merged_csv_bytes = df_dbg.to_csv(index=True).encode("utf-8")

                        # 2) Pre-physics merged dataframe (raw_history) if available
                        raw_history = data.get("raw_history")
                        raw_csv_bytes: bytes | None = None
                        if isinstance(raw_history, pd.DataFrame) and not raw_history.empty:
                            raw_csv_bytes = raw_history.to_csv(index=True).encode("utf-8")
                        else:
                            raw_history = None

                        # Build pre-physics sensor coverage + active_stats for JSON bundle only
                        sensor_cols = ["Power", "FlowTemp", "ReturnTemp", "FlowRate", "Freq", "DeltaT"]
                        coverage: dict[str, dict] = {}
                        active_stats: dict[str, dict] = {}

                        if isinstance(raw_history, pd.DataFrame) and not raw_history.empty:
                            total_rows = len(raw_history)
                            for col in sensor_cols:
                                if col in raw_history.columns:
                                    series = pd.to_numeric(raw_history[col], errors="coerce")
                                    non_null = int(series.notna().sum())
                                    non_zero = int((series != 0).sum())
                                    coverage[col] = {
                                        "total_rows": int(total_rows),
                                        "non_null": non_null,
                                        "non_null_pct": float(100 * non_null / total_rows)
                                        if total_rows
                                        else 0.0,
                                        "non_zero": non_zero,
                                        "non_zero_pct": float(100 * non_zero / total_rows)
                                        if total_rows
                                        else 0.0,
                                    }

                            # Active sample stats (Power > 500W) – for JSON bundle only
                            if "Power" in raw_history.columns:
                                power_series = pd.to_numeric(raw_history["Power"], errors="coerce")
                                active_mask = power_series > 500
                                if active_mask.any():
                                    active_sample = raw_history[active_mask].head(20)
                                    display_cols = [c for c in sensor_cols if c in active_sample.columns]
                                    for col in display_cols:
                                        vals = (
                                            pd.to_numeric(active_sample[col], errors="coerce")
                                            .dropna()
                                        )
                                        if len(vals) > 0:
                                            active_stats[col] = {
                                                "min": float(vals.min()),
                                                "max": float(vals.max()),
                                                "mean": float(vals.mean()),
                                                "count": int(len(vals)),
                                            }

                        # 3) Build and expose App Debug Bundle (JSON) + README inside ZIPs
                        try:
                            config = st.session_state.get("system_config")
                            caps = st.session_state.get("capabilities", {})
                            runs_list = data.get("runs") or []
                            dataset_source = st.session_state.get("dataset_source", "unknown")

                            # Canonical global stats for the debug bundle
                            try:
                                global_stats = processing.compute_global_stats(df_dbg)
                            except Exception:
                                global_stats = {}

                            # Mapping summary (logical role -> entity_id)
                            mapping_obj = {}
                            if isinstance(config, dict):
                                mapping_obj = config.get("mapping") or {}

                            # Inspector outputs, if inspector has been run this session
                            inspector_summary = st.session_state.get("inspector_summary")
                            inspector_details = st.session_state.get("inspector_details")
                            if hasattr(inspector_summary, "to_dict"):
                                inspector_summary_payload = inspector_summary.to_dict(orient="list")
                            else:
                                inspector_summary_payload = inspector_summary

                            # Build engine debug traces only when the checkbox is enabled
                            try:
                                if debug_flag:
                                    engine_traces_payload = _build_engine_debug_traces(df_dbg)
                                else:
                                    engine_traces_payload = []
                            except Exception:
                                # Never let trace building break the export
                                engine_traces_payload = []

                            debug_bundle = {
                                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                                "app_version": "public-beta-v2.8",
                                "dataset_source": dataset_source,
                                "config": config,
                                "mapping": mapping_obj,
                                "global_stats": global_stats,
                                "runs": runs_list,
                                "sensor_coverage": coverage,
                                "active_sample_stats": active_stats,
                                "capabilities": caps,
                                "inspector_summary": inspector_summary_payload,
                                "inspector_details": inspector_details,
                                # Per-stage engine debug traces derived from the engine dataframe
                                "engine_debug_traces": engine_traces_payload,
                            }


                            debug_json_bytes = json.dumps(debug_bundle, default=str).encode("utf-8")

                            # Single README text for all ZIPs
                            readme_text = (
                                "THERM Debug Export README\n"
                                f"Version: public-beta-v2.8\n"
                                f"Generated at: {pd.Timestamp.now(tz='UTC').isoformat()}\n\n"
                                "Files in this archive:\n\n"
                                "1) therm_merged_engine.csv\n"
                                "   - Final engine dataframe after physics, flags, and tariff logic.\n"
                                "   - One row per time step (usually 1 minute).\n"
                                "   - Use this for detailed time-series analysis: Power, Heat, COP,\n"
                                "     DHW vs Heating, tariffs, immersion, zone flags, etc.\n\n"
                                "2) therm_debug_bundle.json\n"
                                "   - Structured metadata for this analysis run.\n"
                                "   - Includes: configuration, mapping, global statistics, run summaries,\n"
                                "     sensor coverage, capabilities, inspector details, and internal\n"
                                "     engine debug traces.\n"
                                "   - Use this to understand how the app interpreted and processed the data,\n"
                                "     not for raw time-series.\n\n"
                                "3) therm_raw_prephysics.csv (if present)\n"
                                "   - Pre-physics dataframe created by the loader/resampler before\n"
                                "     gatekeepers, COP calculations, or run detection.\n"
                                "   - Use this only when debugging data ingestion or resampling issues.\n\n"
                                "Usage with an LLM:\n"
                                "- Always upload therm_debug_bundle.json alongside at least\n"
                                "  therm_merged_engine.csv.\n"
                                "- Use therm_debug_bundle.json for structure, summary, and diagnostics.\n"
                                "- Use therm_merged_engine.csv when you need to inspect or recompute\n"
                                "  time-series values.\n"
                                "- Include therm_raw_prephysics.csv only when investigating\n"
                                "  loader/resampler problems.\n"
                            )
                            readme_bytes = readme_text.encode("utf-8")

                            # --- ZIP #1: merged CSV + debug JSON + README ---
                            buf1 = io.BytesIO()
                            with zipfile.ZipFile(buf1, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                zf.writestr("therm_readme.txt", readme_bytes)
                                zf.writestr("therm_merged_engine.csv", merged_csv_bytes)
                                zf.writestr("therm_debug_bundle.json", debug_json_bytes)
                            buf1.seek(0)

                            st.download_button(
                                "⬇ Download merged CSV + debug JSON (ZIP)",
                                data=buf1,
                                file_name=f"THERM_debug_merged_and_json_{ts_label}.zip",
                                mime="application/zip",
                                key="download_zip_merged_json",
                            )

                            # --- ZIP #2: all three files (if raw is available) + README ---
                            if raw_csv_bytes is not None:
                                buf2 = io.BytesIO()
                                with zipfile.ZipFile(buf2, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                    zf.writestr("therm_readme.txt", readme_bytes)
                                    zf.writestr("therm_merged_engine.csv", merged_csv_bytes)
                                    zf.writestr("therm_raw_prephysics.csv", raw_csv_bytes)
                                    zf.writestr("therm_debug_bundle.json", debug_json_bytes)
                                buf2.seek(0)

                                st.download_button(
                                    "⬇ Download all debug files (merged + raw + JSON) (ZIP)",
                                    data=buf2,
                                    file_name=f"THERM_debug_all_{ts_label}.zip",
                                    mime="application/zip",
                                    key="download_zip_all",
                                )
                            else:
                                # Provide a disabled button so the UI still shows the option
                                st.download_button(
                                    "⬇ Download all debug files (merged + raw + JSON) (ZIP)",
                                    data=b"",
                                    file_name=f"THERM_debug_all_{ts_label}.zip",
                                    mime="application/zip",
                                    key="download_zip_all",
                                    disabled=True,
                                    help="Raw pre-physics dataframe not available for this run.",
                                )

                        except Exception:
                            # Debug export must never break the main UI
                            pass




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
