# mapping_ui.py
import csv
import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from schema_defs import (
    REQUIRED_SENSORS, RECOMMENDED_SENSORS, OPTIONAL_SENSORS, 
    ZONE_SENSORS, ENVIRONMENTAL_SENSORS, AI_CONTEXT_PROMPTS, ROOM_SENSOR_PREFIX
)
import config_manager
import config


def _log(msg: str) -> None:
    """Best-effort console logger for profiling (disabled by default)."""
    return

def _friendly_entity_label(entity: str) -> str:
    """Trim common HA prefixes for display only."""
    if not isinstance(entity, str):
        return str(entity)
    prefixes = (
        "sensor.",
        "binary_sensor.",
        "switch.",
        "climate.",
        "input_boolean.",
        "input_number.",
    )
    for p in prefixes:
        if entity.startswith(p):
            return entity[len(p):]
    return entity


# -------------------------------------------------------------------
# FAST ENTITY SCAN (streaming) TO AVOID LOADING HUGE HA FILES
# -------------------------------------------------------------------
def _quick_entity_scan(file_obj, max_rows: int = 50_000, max_entities: int = 200):
    """
    Extract unique entity_id values from a long-form HA CSV without loading it
    fully. Stops early once a reasonable number of unique entities have been
    found, and caps total rows scanned to keep the UI responsive even for
    large history exports. We only need the first occurrence of each entity_id.
    """
    # Remember position so we can restore
    try:
        start_pos = file_obj.tell()
    except Exception:
        start_pos = None

    entities: set[str] = set()
    rows_seen = 0

    try:
        reader = csv.reader(file_obj)
        header = next(reader, None)
        if not header:
            return []

        # Locate entity_id column (case-insensitive)
        try:
            eid_idx = [h.lower() for h in header].index("entity_id")
        except ValueError:
            return []

        for row in reader:
            rows_seen += 1
            if rows_seen > max_rows:
                break
            if eid_idx < len(row):
                val = row[eid_idx].strip()
                if val and val not in entities:
                    entities.add(val)
            if len(entities) >= max_entities:
                break

    except Exception:
        return []
    finally:
        if start_pos is not None:
            try:
                file_obj.seek(start_pos)
            except Exception:
                pass

    return sorted(entities)


def get_all_unique_entities(uploaded_files):
    """
    Entity discovery with caching and chunked parsing for long-form files.
    """
    entity_cache = st.session_state.get("entity_cache", {})
    debug_scans = []
    found_entities = set()
    for file_obj in uploaded_files:
        file_obj.seek(0)
        sig = (getattr(file_obj, "name", None), getattr(file_obj, "size", None))

        # Always refresh when invoked (cache is still updated for future calls)
        cached = entity_cache.get(sig)

        t0 = time.time()
        try:
            df_head = pd.read_csv(file_obj, nrows=200)
            cols_lower = {c.lower(): c for c in df_head.columns}

            # --- LONG FORM: entity_id or similar ---
            entity_col = None
            for cand in ["entity_id", "entity id", "entity"]:
                if cand in cols_lower:
                    entity_col = cols_lower[cand]
                    break

            if entity_col:
                file_obj.seek(0)
                ents_set: set[str] = set()
                for chunk in pd.read_csv(file_obj, usecols=[entity_col], chunksize=50_000):
                    vals = (
                        chunk[entity_col]
                        .astype(str)
                        .dropna()
                        .str.strip()
                        .tolist()
                    )
                    ents_set.update([v for v in vals if v])
                    if len(ents_set) >= 5000:
                        break
                ents = sorted(ents_set)

            # --- LONG FORM: Grafana series ---
            else:
                if "series" in cols_lower:
                    series_col = cols_lower["series"]
                    file_obj.seek(0)
                    ents = set()
                    for chunk in pd.read_csv(file_obj, usecols=[series_col], chunksize=50_000):
                        vals = (
                            chunk[series_col]
                            .astype(str)
                            .dropna()
                            .str.strip()
                            .tolist()
                        )
                        ents.update([v for v in vals if v])
                        if len(ents) >= 1000:
                            break
                    ents = sorted(ents)
                else:
                    # --- WIDE FORM ---
                    ents = [
                        c
                        for c in df_head.columns
                        if c.lower()
                        not in [
                            "time",
                            "date",
                            "timestamp",
                            "datetime",
                            "last_changed",
                            "last_updated",
                            "value",
                            "state",
                        ]
                    ]

            found_entities.update(ents)
            # Update cache with the newly discovered entities (or fall back to cached)
            if found_entities:
                entity_cache[sig] = sorted(set(ents)) if 'ents' in locals() else []
                debug_scans.append({
                    "file": sig[0],
                    "cached": False,
                    "entities": len(entity_cache[sig]),
                    "secs": time.time() - t0,
                })
            elif cached:
                found_entities.update(cached)
                debug_scans.append({
                    "file": sig[0],
                    "cached": True,
                    "entities": len(cached),
                    "secs": 0.0,
                })
        except Exception:
            if cached:
                found_entities.update(cached)
                debug_scans.append({
                    "file": sig[0],
                    "cached": True,
                    "entities": len(cached),
                    "secs": 0.0,
                })
        file_obj.seek(0)
    st.session_state["entity_cache"] = entity_cache
    st.session_state["entity_scan_debug"] = debug_scans
    return sorted(list(found_entities))

def render_sensor_row(label, internal_key, options, defaults, required=False, help_text=None, formatter=None):
    c1, c2 = st.columns([2, 1])
    
    widget_key = f"map_{internal_key}"
    selectbox_args = {}
    
    if widget_key not in st.session_state:
        saved_val = defaults["mapping"].get(internal_key, "None")
        idx = options.index(saved_val) if saved_val in options else 0
        selectbox_args["index"] = idx

    with c1:
        sel = st.selectbox(
            f"**{label}**" + (" *" if required else ""), 
            options, 
            key=widget_key, 
            help=help_text,
            format_func=formatter or _friendly_entity_label,
            **selectbox_args
        )
    
    unit_key = f"unit_{internal_key}"
    unit_val = None
    from schema_defs import get_unit_options
    u_opts = get_unit_options(internal_key)
    
    if len(u_opts) > 1 and u_opts != ["unknown"]:
        with c2:
            u_args = {}
            if unit_key not in st.session_state:
                saved_u = defaults["units"].get(internal_key, u_opts[0])
                u_idx = u_opts.index(saved_u) if saved_u in u_opts else 0
                u_args["index"] = u_idx
            
            unit_val = st.selectbox("Unit", u_opts, key=unit_key, **u_args)
    else:
        if u_opts: unit_val = u_opts[0]
            
    return sel, unit_val

def render_configuration_interface(uploaded_files):
    t_render_start = time.time()
    _log("render_configuration_interface start")

    # Two containers: the top block stays sticky; the rest scrolls normally.
    top_section = st.container()
    body_section = st.container()

    # Defaults that can be overridden by a loaded profile
    defaults = {
        "mapping": {},
        "units": {},
        "ai_context": {},
        "profile_name": "My Heat Pump",
        "rooms_per_zone": {},
        "thresholds": {},
        "physics_thresholds": {},
        "tariff_structure": config.TARIFF_STRUCTURE,
    }

    # Cached entity list (may be empty if skipping scan)
    available_entities = st.session_state.get("available_sensors", []) or []

    # If a config already exists (user came back from analysis), preload it
    existing_cfg = st.session_state.get("system_config")
    if isinstance(existing_cfg, dict):
        defaults.update(existing_cfg)
        for k, v in (existing_cfg.get("mapping") or {}).items():
            st.session_state.setdefault(f"map_{k}", v)
        for k, v in (existing_cfg.get("units") or {}).items():
            st.session_state.setdefault(f"unit_{k}", v)
        for z_key, links in (existing_cfg.get("rooms_per_zone") or {}).items():
            st.session_state.setdefault(f"link_{z_key}", links)
    # Auto-refresh entity cache when new files are provided or cache is empty
    if uploaded_files:
        files_key = sorted(
            (getattr(f, "name", ""), getattr(f, "size", 0)) for f in uploaded_files
        )
        cached_key = st.session_state.get("available_sensors_files_key")
        if (not available_entities) or (cached_key != files_key):
            try:
                refreshed = get_all_unique_entities(uploaded_files)
                # Filter out blanks/None
                refreshed = [e for e in refreshed if isinstance(e, str) and e.strip()]
                if not refreshed:
                    # Fallback: naive column/entity scrape so dropdowns aren't empty
                    fallback_entities: set[str] = set()
                    for f in uploaded_files:
                        try:
                            f.seek(0)
                            df_head = pd.read_csv(f, nrows=200)
                            cols = list(df_head.columns)
                            cols_lower = [c.lower() for c in cols]
                            if "entity_id" in cols_lower:
                                e_col = cols[cols_lower.index("entity_id")]
                                vals = (
                                    df_head[e_col]
                                    .astype(str)
                                    .dropna()
                                    .str.strip()
                                    .tolist()
                                )
                                fallback_entities.update([v for v in vals if v])
                            elif "series" in cols_lower:
                                s_col = cols[cols_lower.index("series")]
                                vals = (
                                    df_head[s_col]
                                    .astype(str)
                                    .dropna()
                                    .str.strip()
                                    .tolist()
                                )
                                fallback_entities.update([v for v in vals if v])
                            else:
                                ignore = {"time", "timestamp", "date", "datetime", "last_changed", "last_updated"}
                                for c in cols:
                                    if c.lower() not in ignore:
                                        fallback_entities.add(c)
                        except Exception:
                            continue
                        finally:
                            try:
                                f.seek(0)
                            except Exception:
                                pass
                    if fallback_entities:
                        refreshed = sorted(fallback_entities)

                if refreshed:
                    available_entities = refreshed
                    st.session_state["available_sensors"] = available_entities
                    st.session_state["available_sensors_files_key"] = files_key
            except Exception:
                # Best-effort: keep existing cache if scan fails
                available_entities = st.session_state.get("available_sensors", []) or []
    profile_loaded = False
    loaded_profile_name = None

    with top_section:
        st.markdown("<div class='setup-sticky'>", unsafe_allow_html=True)
        st.markdown("## System Setup")

        col_load, col_name = st.columns([1, 2])

        # --- Load Profile (JSON) ---
        with col_load:
            uploaded_config = st.file_uploader(" Load Profile", type="json", key="cfg_up")
            if uploaded_config:
                t_profile = time.time()
                try:
                    loaded = json.load(uploaded_config)
                    defaults.update(loaded)
                    _log(f"profile_load file={getattr(uploaded_config,'name',None)} secs={time.time()-t_profile:.3f}")

                    # Build a simple signature for the currently uploaded profile
                    current_sig = (uploaded_config.name, getattr(uploaded_config, "size", None))
                    prev_sig = st.session_state.get("loaded_profile_signature")

                    # Only push mapping/unit values into session_state when:
                    #   - a profile is loaded for the first time, or
                    #   - a *different* profile file is chosen
                    if current_sig != prev_sig:
                        st.session_state["loaded_profile_signature"] = current_sig

                        # 1. Clear any existing mapping/unit keys (only those used)
                        for k in defaults.get("mapping", {}):
                            map_key = f"map_{k}"
                            if map_key in st.session_state:
                                del st.session_state[map_key]

                        for k in defaults.get("units", {}):
                            unit_key = f"unit_{k}"
                            if unit_key in st.session_state:
                                del st.session_state[unit_key]

                        # 2. Apply defaults freshly
                        for k, v in defaults.get("mapping", {}).items():
                            st.session_state[f"map_{k}"] = v

                        for k, v in defaults.get("units", {}).items():
                            st.session_state[f"unit_{k}"] = v

                        # 3. Apply room links per zone
                        for z_key, links in (defaults.get("rooms_per_zone") or {}).items():
                            st.session_state[f"link_{z_key}"] = links

                        # 4. Sync profile name when loading a new/different profile
                        st.session_state["profile_name_input"] = defaults["profile_name"]

                    _log(f"profile_apply_defaults secs={time.time()-t_profile:.3f}")
                    profile_loaded = True
                    loaded_profile_name = defaults["profile_name"]
                except Exception:
                    st.error("Failed to load profile JSON.")

        # Manual refresh button for entity discovery (skips auto-scan on profile load)
        if uploaded_files:
            with st.expander("Entity Discovery", expanded=False):
                files_key = sorted(
                    (getattr(f, "name", ""), getattr(f, "size", 0)) for f in uploaded_files
                )
                if st.button("Refresh entities from uploaded files", type="secondary"):
                    t_scan = time.time()
                    st.session_state["available_sensors"] = get_all_unique_entities(uploaded_files)
                    st.session_state["available_sensors_files_key"] = files_key
                    st.success(f"Entities refreshed in {time.time()-t_scan:.3f}s")
                # Show cached results if present
                cached_entities = st.session_state.get("available_sensors", [])
                if cached_entities:
                    st.caption(f"{len(cached_entities)} entities cached.")
                else:
                    st.caption("Entities will be loaded from profile mapping unless refreshed.")

        # Build options list:
        #   - Entities from the loaded profile mapping
        #   - PLUS any entities discovered/cached from files (if refreshed)
        profile_entities = [v for v in (defaults.get("mapping") or {}).values() if v]
        combined_entities = sorted(set(profile_entities + available_entities), key=lambda x: str(x).lower())
        options = ["None"] + combined_entities

        # Friendly label map with collision handling (avoid duplicate-looking entries)
        friendly_lookup = {}
        label_counts: dict[str, int] = {}
        for opt in combined_entities:
            label = _friendly_entity_label(opt)
            label_counts[label] = label_counts.get(label, 0) + 1
            if label_counts[label] > 1:
                friendly_lookup[opt] = f"{label} ({opt})"
            else:
                friendly_lookup[opt] = label

        def format_opt(opt: str) -> str:
            return friendly_lookup.get(opt, _friendly_entity_label(opt))

        # Quick diagnostics for discovery
        st.caption(f"Entities available: {len(available_entities)} (profile: {len(profile_entities)})")

        # --- Profile name ---
        if "profile_name_input" not in st.session_state:
            st.session_state["profile_name_input"] = defaults["profile_name"]
        with col_name:
            profile_name = st.text_input("Profile Name", key="profile_name_input")
            if profile_loaded and loaded_profile_name:
                st.success(f"Loaded {loaded_profile_name}")
        # Keep defaults in sync with the current input value
        defaults["profile_name"] = st.session_state.get("profile_name_input", defaults.get("profile_name", "My Heat Pump"))

        # Top action bar (will be sticky with CSS)
        action_bar_top = st.container()
        st.markdown("</div>", unsafe_allow_html=True)

    # Everything below scrolls independently of the sticky section
    with body_section:
        user_map: dict = {}
        user_units: dict = {}

        # ------------------------------------------------------------------
        # 1. Critical Sensors
        # ------------------------------------------------------------------
        st.subheader("1. Critical Sensors")
        for key, d in REQUIRED_SENSORS.items():
            s, u = render_sensor_row(
                d["label"],
                key,
                options,
                defaults,
                required=True,
                help_text=d["description"],
                formatter=format_opt,
            )
            if s != "None":
                user_map[key] = s
                if u:
                    user_units[key] = u

        # ------------------------------------------------------------------
        # 2. Recommended
        # ------------------------------------------------------------------
        st.subheader("2. Recommended")
        for key, d in RECOMMENDED_SENSORS.items():
            s, u = render_sensor_row(
                d["label"],
                key,
                options,
                defaults,
                required=False,
                help_text=d["description"],
                formatter=format_opt,
            )
            if s != "None":
                user_map[key] = s
                if u:
                    user_units[key] = u

        # ------------------------------------------------------------------
        # 3. Zones & Rooms
        # ------------------------------------------------------------------
        st.subheader("3. Zones & Rooms")

        # Always define this so it exists when Step B runs
        mapped_rooms_labels: dict = {}

        # Step A: Map Room Sensors
        with st.expander("Step A: Map Room Sensors", expanded=True):
            r_cols = st.columns(2)
            for i in range(1, 9):
                r_key = f"{ROOM_SENSOR_PREFIX}{i}"
                with r_cols[(i - 1) % 2]:
                    r_s, _ = render_sensor_row(
                        f"Room Sensor {i}",
                        r_key,
                        options,
                        defaults,
                        required=False,
                        help_text=None,
                        formatter=format_opt,
                    )
                    if r_s != "None":
                        user_map[r_key] = r_s
                        mapped_rooms_labels[r_key] = r_s

        # Step B: Configure Zones & Link Rooms
        with st.expander("Step B: Configure Zones & Link Rooms", expanded=True):
            rooms_per_zone: dict = {}

            # Friendly labels for rooms: "Kitchen (Room_1)" etc.
            friendly_room_options = {
                k: f"{v} ({k})" for k, v in mapped_rooms_labels.items()
            }
            room_keys_list = list(friendly_room_options.keys())

            for z_key, z_d in ZONE_SENSORS.items():
                st.markdown(f"**{z_d['label']}**")

                z_s, z_u = render_sensor_row(
                    "Zone Pump/Valve (Binary)",
                    z_key,
                    options,
                    defaults,
                    required=False,
                    help_text=z_d["description"],
                    formatter=format_opt,
                )
                if z_s != "None":
                    user_map[z_key] = z_s
                    if u:
                        user_units[z_key] = z_u
                else:
                    # If zone unmapped, clear any lingering room links in session state
                    if f"link_{z_key}" in st.session_state:
                        st.session_state[f"link_{z_key}"] = []

                saved_links = defaults.get("rooms_per_zone", {}).get(z_key, [])
                valid_defaults = [r for r in saved_links if r in room_keys_list] if z_s != "None" else []

                # Avoid Streamlit warning when default + session_state both set:
                # only pass default if the key is not already in session_state.
                multiselect_kwargs = {
                    "format_func": lambda x: friendly_room_options[x],
                    "key": f"link_{z_key}",
                    "help": "Select which room sensors belong to this zone.",
                }
                if f"link_{z_key}" not in st.session_state:
                    multiselect_kwargs["default"] = valid_defaults

                selected_keys = st.multiselect(
                    f"Select rooms controlled by {z_d['label']}:",
                    options=room_keys_list,
                    **multiselect_kwargs,
                )
                rooms_per_zone[z_key] = selected_keys

                st.markdown("---")

            # If a zone was unmapped (set to None), clear its room links to avoid stale associations
            for z_key in list(rooms_per_zone.keys()):
                if z_key not in user_map:
                    rooms_per_zone[z_key] = []

        # ------------------------------------------------------------------
        # 4. Advanced / Environmental Sensors (alphabetical)
        # ------------------------------------------------------------------
        with st.expander("Advanced / Environmental Sensors"):
            merged = {**OPTIONAL_SENSORS, **ENVIRONMENTAL_SENSORS}
            for key, d in sorted(
                merged.items(), key=lambda kv: kv[1].get("label", kv[0]).lower()
            ):
                s, u = render_sensor_row(
                    d["label"],
                    key,
                    options,
                    defaults,
                    required=False,
                    help_text=d.get("description", None),
                    formatter=format_opt,
                )
                if s != "None":
                    user_map[key] = s
                    if u:
                        user_units[key] = u

        # ------------------------------------------------------------------
        # 5. Thresholds (policy + advanced)
        # ------------------------------------------------------------------
        thresholds_cfg: dict = {}
        physics_cfg: dict = {}

        user_thresholds = defaults.get("thresholds") or {}
        user_phys = defaults.get("physics_thresholds") or {}

        def _tval(key, default):
            return user_thresholds.get(key, default)

        def _pval(key, default):
            return user_phys.get(key, default)

        with st.expander("Thresholds"):
            # Helper to keep inputs from stretching full width
            def _narrow_number_input(label, *, value, help, step, min_value=None, max_value=None, fmt=None):
                col_input, _ = st.columns([1, 3])
                kwargs = {
                    "value": value,
                    "help": help,
                    "step": step,
                    "min_value": min_value,
                    "max_value": max_value,
                }
                if fmt:
                    kwargs["format"] = fmt
                with col_input:
                    return st.number_input(label, **kwargs)

            # Group 1: Basic Run Detection (Power only)
            with st.expander("Basic Run Detection (Power)", expanded=True):
                thresholds_cfg["minimum_run_duration_min"] = _narrow_number_input(
                    "Minimum run duration (min)",
                    value=float(_tval("minimum_run_duration_min", config.THRESHOLDS.get("minimum_run_duration_min", 5))),
                    help="Cycles shorter than this are ignored as noise.",
                    step=1.0,
                    min_value=0.0,
                )
                thresholds_cfg["short_cycle_min"] = _narrow_number_input(
                    "Short cycle threshold (min)",
                    value=float(_tval("short_cycle_min", config.THRESHOLDS.get("short_cycle_min", 20))),
                    help="Heating runs shorter than this are flagged as short cycles.",
                    step=1.0,
                    min_value=0.0,
                )
                thresholds_cfg["very_short_cycle_min"] = _narrow_number_input(
                    "Very short cycle threshold (min)",
                    value=float(_tval("very_short_cycle_min", config.THRESHOLDS.get("very_short_cycle_min", 10))),
                    help="Heating runs shorter than this are flagged as critical short cycles.",
                    step=1.0,
                    min_value=0.0,
                )
                physics_cfg["power_on_W"] = _narrow_number_input(
                    "Power on threshold (W)",
                    value=float(_pval("power_on_W", config.ENGINE_STATE_THRESHOLDS.get("power_on_W", 300.0))),
                    help="Minimum electrical power required to consider the unit active.",
                    step=10.0,
                    min_value=0.0,
                )
                physics_cfg["coast_down_window_min"] = _narrow_number_input(
                    "Coast-down lookback (min)",
                    value=float(_pval("coast_down_window_min", config.ENGINE_STATE_THRESHOLDS.get("coast_down_window_min", 5))),
                    help="Minutes to look back to decide if a run is coasting after power drops.",
                    step=1.0,
                    min_value=0.0,
                )

            # Group 2: Hydraulics & Efficiency (Flow/Temps)
            with st.expander("Hydraulics & Efficiency (Flow/Temps)", expanded=False):
                physics_cfg["min_flow_rate_lpm"] = _narrow_number_input(
                    "Min flow rate to count water movement (L/min)",
                    value=float(_pval("min_flow_rate_lpm", config.PHYSICS_THRESHOLDS.get("min_flow_rate_lpm", 3.0))),
                    help="Heat output forced to 0 if flow is below this value (filters idle noise).",
                    step=0.1,
                    min_value=0.0,
                )
                thresholds_cfg["flow_limit_tolerance"] = _narrow_number_input(
                    "Flow limit tolerance (°C)",
                    value=float(_tval("flow_limit_tolerance", config.THRESHOLDS.get("flow_limit_tolerance", 2.0))),
                    help="Gap allowed between target and actual flow before flagging a flow limit issue.",
                    step=0.1,
                    min_value=0.0,
                )
                thresholds_cfg["flow_limit_min_duration"] = _narrow_number_input(
                    "Flow limit minimum duration (min)",
                    value=float(_tval("flow_limit_min_duration", config.THRESHOLDS.get("flow_limit_min_duration", 15))),
                    help="Duration required to trigger a flow limit warning.",
                    step=1.0,
                    min_value=0.0,
                )
                physics_cfg["delta_on_C"] = _narrow_number_input(
                    "DeltaT on threshold (°C)",
                    value=float(_pval("delta_on_C", config.ENGINE_STATE_THRESHOLDS.get("delta_on_C", 1.0))),
                    help="Minimum DeltaT required to confirm active heating.",
                    step=0.1,
                    min_value=0.0,
                )
                thresholds_cfg["dhw_scop_low"] = _narrow_number_input(
                    "Low DHW COP threshold",
                    value=float(_tval("dhw_scop_low", config.THRESHOLDS.get("dhw_scop_low", 2.2))),
                    help="DHW runs with efficiency below this are flagged as poor.",
                    step=0.1,
                    min_value=0.0,
                )

            # Group 3: Advanced Diagnostics (Optional sensors)
            with st.expander("Advanced Diagnostics (Optional sensors)", expanded=False):
                thresholds_cfg["heating_during_dhw_power_threshold"] = _narrow_number_input(
                    "Heating during DHW power threshold (W)",
                    value=float(_tval("heating_during_dhw_power_threshold", config.THRESHOLDS.get("heating_during_dhw_power_threshold", 120))),
                    help="Indoor power above this during DHW (with no zones) implies heating running during DHW.",
                    step=10.0,
                    min_value=0.0,
                )
                thresholds_cfg["heating_during_dhw_detection_pct"] = _narrow_number_input(
                    "Heating during DHW detection fraction",
                    value=float(_tval("heating_during_dhw_detection_pct", config.THRESHOLDS.get("heating_during_dhw_detection_pct", 0.15))),
                    help="Fraction of a DHW run that must show heating activity to trigger the warning (e.g., 0.15 = 15%).",
                    step=0.01,
                    min_value=0.0,
                    max_value=1.0,
                    fmt="%.2f",
                )
                thresholds_cfg["hdd_base_temp"] = _narrow_number_input(
                    "Heating degree day base temp (°C)",
                    value=float(_tval("hdd_base_temp", config.THRESHOLDS.get("hdd_base_temp", 18.0))),
                    help="Base temperature for heating degree day calculations.",
                    step=0.5,
                )
                physics_cfg["freq_on_hz"] = _narrow_number_input(
                    "Freq on threshold (Hz)",
                    value=float(_pval("freq_on_hz", config.ENGINE_STATE_THRESHOLDS.get("freq_on_hz", 10.0))),
                    help="Compressor frequency above this is considered active.",
                    step=0.5,
                    min_value=0.0,
                )
                physics_cfg["freq_off_hz"] = _narrow_number_input(
                    "Freq off threshold (Hz)",
                    value=float(_pval("freq_off_hz", config.ENGINE_STATE_THRESHOLDS.get("freq_off_hz", 5.0))),
                    help="Compressor frequency below this is considered inactive.",
                    step=0.5,
                    min_value=0.0,
                )
                thresholds_cfg["flow_over_43c_pct_high"] = _narrow_number_input(
                    "High flow temp share (%)",
                    value=float(_tval("flow_over_43c_pct_high", config.THRESHOLDS.get("flow_over_43c_pct_high", 20))),
                    help="Warn if flow temperature exceeds 43°C for more than this share of a run.",
                    step=1.0,
                    min_value=0.0,
                    max_value=100.0,
                )
                thresholds_cfg["min_heating_run_minutes_with_no_zones"] = _narrow_number_input(
                    "Min heating minutes when zones unknown",
                    value=float(_tval("min_heating_run_minutes_with_no_zones", config.THRESHOLDS.get("min_heating_run_minutes_with_no_zones", 8))),
                    help="If no zone data, runs shorter than this are ignored to avoid phantom startups.",
                    step=1.0,
                    min_value=0.0,
                )
                thresholds_cfg["min_heating_run_heat_kwh_with_no_zones"] = _narrow_number_input(
                    "Min heating kWh when zones unknown",
                    value=float(_tval("min_heating_run_heat_kwh_with_no_zones", config.THRESHOLDS.get("min_heating_run_heat_kwh_with_no_zones", 0.25))),
                    help="If no zone data, runs with less heat than this are ignored to avoid phantom startups.",
                    step=0.05,
                    min_value=0.0,
                )
                thresholds_cfg["high_night_share"] = _narrow_number_input(
                    "High night-rate share (fraction)",
                    value=float(_tval("high_night_share", config.THRESHOLDS.get("high_night_share", 0.50))),
                    help="Night-rate share above this is considered high for load-shifting analysis.",
                    step=0.05,
                    min_value=0.0,
                    max_value=1.0,
                    fmt="%.2f",
                )
                # Additional engine/physics tuning (keep available for power users)
                physics_cfg["flow_on_lpm"] = _narrow_number_input(
                    "Flow on threshold (L/min)",
                    value=float(_pval("flow_on_lpm", config.ENGINE_STATE_THRESHOLDS.get("flow_on_lpm", 2.0))),
                    help="Flow rate above this implies pump running (fallback detection).",
                    step=0.1,
                    min_value=0.0,
                )
                physics_cfg["flow_off_lpm"] = _narrow_number_input(
                    "Flow off threshold (L/min)",
                    value=float(_pval("flow_off_lpm", config.ENGINE_STATE_THRESHOLDS.get("flow_off_lpm", 1.0))),
                    help="Flow rate below this implies pump off (fallback detection).",
                    step=0.1,
                    min_value=0.0,
                )
                physics_cfg["delta_coast_min_C"] = _narrow_number_input(
                    "DeltaT coast threshold (°C)",
                    value=float(_pval("delta_coast_min_C", config.ENGINE_STATE_THRESHOLDS.get("delta_coast_min_C", 0.5))),
                    help="DeltaT threshold to consider a run coasting after compressor stop.",
                    step=0.1,
                    min_value=0.0,
                )
                physics_cfg["min_freq_for_delta_t"] = _narrow_number_input(
                    "Min freq to trust DeltaT (Hz)",
                    value=float(_pval("min_freq_for_delta_t", config.PHYSICS_THRESHOLDS.get("min_freq_for_delta_t", 5.0))),
                    help="Minimum compressor frequency to trust DeltaT calculation.",
                    step=0.5,
                    min_value=0.0,
                )
                physics_cfg["min_freq_for_heat"] = _narrow_number_input(
                    "Min freq to calculate heat (Hz)",
                    value=float(_pval("min_freq_for_heat", config.PHYSICS_THRESHOLDS.get("min_freq_for_heat", 7.5))),
                    help="Minimum compressor frequency to calculate heat output.",
                    step=0.5,
                    min_value=0.0,
                )
                physics_cfg["max_valid_delta_t"] = _narrow_number_input(
                    "Max valid DeltaT (°C)",
                    value=float(_pval("max_valid_delta_t", config.PHYSICS_THRESHOLDS.get("max_valid_delta_t", 15.0))),
                    help="DeltaT above this is treated as sensor error and ignored.",
                    step=0.5,
                    min_value=0.0,
                )
                physics_cfg["min_valid_delta_t"] = _narrow_number_input(
                    "Min valid DeltaT (°C)",
                    value=float(_pval("min_valid_delta_t", config.PHYSICS_THRESHOLDS.get("min_valid_delta_t", 0.2))),
                    help="DeltaT below this is treated as noise and ignored.",
                    step=0.05,
                    min_value=0.0,
                )

        # ------------------------------------------------------------------
        # 6. Tariff / Cost Structure
        # ------------------------------------------------------------------
        tariff_structure_cfg = defaults.get("tariff_structure", config.TARIFF_STRUCTURE)
        currency_default = defaults.get("currency", "€")
        currency_options = ["€", "£", "$"]
        currency = st.selectbox(
            "Currency",
            options=currency_options,
            index=currency_options.index(currency_default) if currency_default in currency_options else 0,
            help="Currency symbol used for tariff rates.",
        )

        # Determine default mode
        def _infer_mode(ts):
            if isinstance(ts, dict):
                return "Day/Night" if "day_rate" in ts or "night_rate" in ts else "Flat"
            if isinstance(ts, list) and ts:
                return "Custom bands"
            return "Flat"

        tariff_mode = st.radio(
            "Tariff / Cost Structure",
            options=["Flat", "Day/Night", "Custom bands"],
            index=["Flat", "Day/Night", "Custom bands"].index(_infer_mode(tariff_structure_cfg)),
            help="Define how electricity cost varies over time.",
        )

        if tariff_mode == "Flat":
            flat_rate = st.number_input(
                f"Flat rate ({currency}/kWh)",
                value=float(tariff_structure_cfg.get("day_rate", 0.33) if isinstance(tariff_structure_cfg, dict) else 0.33),
                step=0.01,
                min_value=0.0,
            )
            tariff_structure_cfg = {"day_rate": flat_rate, "night_rate": flat_rate}

        elif tariff_mode == "Day/Night":
            if isinstance(tariff_structure_cfg, dict):
                day_rate_default = float(tariff_structure_cfg.get("day_rate", 0.33))
                night_rate_default = float(tariff_structure_cfg.get("night_rate", 0.15))
                night_start_default = tariff_structure_cfg.get("night_start", "00:00")
                night_end_default = tariff_structure_cfg.get("night_end", "07:00")
            else:
                day_rate_default = 0.33
                night_rate_default = 0.15
                night_start_default = "00:00"
                night_end_default = "07:00"

            day_rate = st.number_input(f"Day rate ({currency}/kWh)", value=day_rate_default, step=0.01, min_value=0.0)
            night_rate = st.number_input(f"Night rate ({currency}/kWh)", value=night_rate_default, step=0.01, min_value=0.0)
            col_ns, col_ne = st.columns(2)
            with col_ns:
                night_start = st.text_input("Night start (HH:MM)", value=night_start_default)
            with col_ne:
                night_end = st.text_input("Night end (HH:MM)", value=night_end_default)

            tariff_structure_cfg = {
                "day_rate": day_rate,
                "night_rate": night_rate,
                "night_start": night_start,
                "night_end": night_end,
            }

        else:  # Custom bands
            # Build editable rules table
            rules_default = []
            if isinstance(tariff_structure_cfg, list) and tariff_structure_cfg:
                first_profile = tariff_structure_cfg[0]
                for r in first_profile.get("rules", []):
                    rules_default.append(
                        {
                            "name": r.get("name", ""),
                            "start": r.get("start", "00:00"),
                            "end": r.get("end", "00:00"),
                            "rate": r.get("rate", 0.35),
                        }
                    )
            if not rules_default:
                rules_default = [
                    {"name": "Off-peak", "start": "00:00", "end": "07:00", "rate": 0.15},
                    {"name": "Day", "start": "07:00", "end": "17:00", "rate": 0.30},
                    {"name": "Peak", "start": "17:00", "end": "20:00", "rate": 0.40},
                    {"name": "Evening", "start": "20:00", "end": "24:00", "rate": 0.30},
                ]

            rules_df = pd.DataFrame(rules_default)
            edited_rules = st.data_editor(
                rules_df,
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "start": st.column_config.TextColumn("Start (HH:MM)"),
                    "end": st.column_config.TextColumn("End (HH:MM)"),
                    "rate": st.column_config.NumberColumn(f"Rate ({currency}/kWh)", format="%.3f", step=0.01, min_value=0.0),
                },
            )

            # Build tariff structure list
            rules_list = []
            for _, row in edited_rules.iterrows():
                try:
                    rate_val = float(row.get("rate", 0.0))
                except Exception:
                    rate_val = 0.0
                rules_list.append(
                    {
                        "name": str(row.get("name", "")),
                        "start": str(row.get("start", "00:00")),
                        "end": str(row.get("end", "00:00")),
                        "rate": rate_val,
                    }
                )

            tariff_structure_cfg = [
                {
                    "valid_from": datetime.now().date().isoformat(),
                    "name": "Custom",
                    "rules": rules_list,
                }
            ]

        st.divider()

        # ------------------------------------------------------------------
        # 7. AI Context Inputs
        # ------------------------------------------------------------------
        ai_inputs: dict = {}
        for k, p in AI_CONTEXT_PROMPTS.items():
            existing = defaults["ai_context"].get(k, "")
            entered = st.text_area(
                p["label"],
                value=existing,
                placeholder=p.get("placeholder", ""),
                help=p["help"],
            )
            # If the user leaves the placeholder untouched (or empty), store as empty
            if not entered or entered.strip() == (p.get("placeholder", "").strip()):
                entered = ""
            ai_inputs[k] = entered

        # Refresh profile_name from session to avoid stale downloads when text was just edited
        profile_name = st.session_state.get("profile_name_input", profile_name)
        config_object = {
            "profile_name": profile_name,
            "created_at": datetime.now().isoformat(),
            "mapping": user_map,
            "units": user_units,
            "ai_context": ai_inputs,
            "rooms_per_zone": rooms_per_zone,
            "thresholds": thresholds_cfg,
            "physics_thresholds": physics_cfg,
            "tariff_structure": tariff_structure_cfg,
            "currency": currency,
            "therm_version": "2.0",
        }

    # ------------------------------------------------------------------
    # 8. Two-Step Actions: Save Configuration + Process Data (top bar)
    # ------------------------------------------------------------------
    with action_bar_top:
        c_btn1, c_btn2 = st.columns(2)

        with c_btn1:
            export_data = config_manager.export_config_for_sharing(config_object)
            export_data["rooms_per_zone"] = rooms_per_zone
            # Ensure updated profile name is preserved on export/download
            export_data["profile_name"] = profile_name
            st.download_button(
                label=" 1. Save Configuration",
                data=json.dumps(export_data, indent=2),
                file_name=f"therm_profile_{profile_name.replace(' ', '_')}.json",
                mime="application/json",
                type="secondary",
            )

        with c_btn2:
            if st.button(" 2. Process Uploaded Data", type="primary"):
                # Ensure at least one critical sensor is mapped
                if not any(k in user_map for k in REQUIRED_SENSORS):
                    st.error("Missing required sensors.")
                    return None
                else:
                    # Flag the main app to scroll to top on the next render
                    st.session_state["scroll_to_top"] = True
                    _log(f"render_configuration_interface done secs={time.time()-t_render_start:.3f} result=config_object")
                    return config_object

    _log(f"render_configuration_interface done secs={time.time()-t_render_start:.3f} result=None")
    return None


def render_config_download(config: dict) -> None:
    """
    Download helper for an already-built config object.

    Uses the same export logic as the main configuration interface,
    but works with an existing config passed in from the caller.
    """
    # Derive profile name from the config, with a sensible default
    profile_name = config.get("profile_name", "My Heat Pump")

    # Start from the canonical sharing export
    export_data = config_manager.export_config_for_sharing(config)

    # Ensure key fields are preserved
    export_data["profile_name"] = profile_name
    export_data["rooms_per_zone"] = config.get("rooms_per_zone", {})

    st.download_button(
        label=" 1. Save Configuration",
        data=json.dumps(export_data, indent=2),
        file_name=f"therm_profile_{profile_name.replace(' ', '_')}.json",
        mime="application/json",
        type="secondary",
        key=f"save_btn_{profile_name.replace(' ', '_')}",
    )


