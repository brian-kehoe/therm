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


def _log(msg: str) -> None:
    """Best-effort console logger for profiling (disabled by default)."""
    return


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
    Fast entity discovery with caching and timing debug info.
    """
    entity_cache = st.session_state.get("entity_cache", {})
    debug_scans = []
    found_entities = set()
    for file_obj in uploaded_files:
        file_obj.seek(0)
        sig = (getattr(file_obj, "name", None), getattr(file_obj, "size", None))

        if sig in entity_cache:
            found_entities.update(entity_cache[sig])
            debug_scans.append({
                "file": sig[0],
                "cached": True,
                "entities": len(entity_cache[sig]),
                "secs": 0.0,
            })
            file_obj.seek(0)
            continue

        t0 = time.time()
        try:
            df_head = pd.read_csv(file_obj, nrows=2)
            if 'entity_id' in df_head.columns:
                file_obj.seek(0)
                ents = _quick_entity_scan(file_obj)
                found_entities.update(ents)
                entity_cache[sig] = ents
                debug_scans.append({
                    "file": sig[0],
                    "cached": False,
                    "entities": len(ents),
                    "secs": time.time() - t0,
                })
            else:
                cols = [c for c in df_head.columns if c.lower() not in ['time', 'date', 'timestamp', 'last_changed', 'series', 'value']]
                found_entities.update(cols)
                entity_cache[sig] = cols
                debug_scans.append({
                    "file": sig[0],
                    "cached": False,
                    "entities": len(cols),
                    "secs": time.time() - t0,
                })
        except Exception:
            pass
        file_obj.seek(0)
    st.session_state["entity_cache"] = entity_cache
    st.session_state["entity_scan_debug"] = debug_scans
    return sorted(list(found_entities))

def render_sensor_row(label, internal_key, options, defaults, required=False, help_text=None):
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
    st.markdown("## ️ System Setup")

    col_load, col_name = st.columns([1, 2])

    # Defaults that can be overridden by a loaded profile
    defaults = {
        "mapping": {},
        "units": {},
        "ai_context": {},
        "profile_name": "My Heat Pump",
        "rooms_per_zone": {},
    }

    # Cached entity list (may be empty if skipping scan)
    available_entities = st.session_state.get("available_sensors", []) or []

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

                st.success(f"Loaded {defaults['profile_name']}")
                _log(f"profile_apply_defaults secs={time.time()-t_profile:.3f}")
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
    profile_entities = list((defaults.get("mapping") or {}).values())
    combined_entities = sorted(set(profile_entities + available_entities))
    options = ["None"] + combined_entities

    # --- Profile name ---
    with col_name:
        profile_name = st.text_input("Profile Name", value=defaults["profile_name"])

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
    # 4. Advanced / Environmental (alphabetical)
    # ------------------------------------------------------------------
    with st.expander("➕ Advanced / Environmental"):
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
            )
            if s != "None":
                user_map[key] = s
                if u:
                    user_units[key] = u

    st.divider()

    # ------------------------------------------------------------------
    # 5. AI Context Inputs
    # ------------------------------------------------------------------
    ai_inputs: dict = {}
    for k, p in AI_CONTEXT_PROMPTS.items():
        ai_inputs[k] = st.text_area(
            p["label"],
            value=defaults["ai_context"].get(k, ""),
            help=p["help"],
        )

    # ------------------------------------------------------------------
    # 6. Two-Step Actions: Save Configuration + Process Data
    # ------------------------------------------------------------------
    c_btn1, c_btn2 = st.columns(2)

    config_object = {
        "profile_name": profile_name,
        "created_at": datetime.now().isoformat(),
        "mapping": user_map,
        "units": user_units,
        "ai_context": ai_inputs,
        "rooms_per_zone": rooms_per_zone,
        "therm_version": "2.0",
    }

    with c_btn1:
        export_data = config_manager.export_config_for_sharing(config_object)
        export_data["rooms_per_zone"] = rooms_per_zone
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


