# mapping_ui.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from schema_defs import (
    REQUIRED_SENSORS, RECOMMENDED_SENSORS, OPTIONAL_SENSORS, 
    ZONE_SENSORS, ENVIRONMENTAL_SENSORS, AI_CONTEXT_PROMPTS, ROOM_SENSOR_PREFIX
)
import config_manager

def get_all_unique_entities(uploaded_files):
    found_entities = set()
    for file_obj in uploaded_files:
        file_obj.seek(0)
        try:
            df_head = pd.read_csv(file_obj, nrows=2)
            if 'entity_id' in df_head.columns:
                file_obj.seek(0)
                df_ent = pd.read_csv(file_obj, usecols=['entity_id'])
                found_entities.update(df_ent['entity_id'].dropna().unique())
            else:
                cols = [c for c in df_head.columns if c.lower() not in ['time', 'date', 'timestamp', 'last_changed', 'series']]
                found_entities.update(cols)
        except Exception:
            pass
        file_obj.seek(0)
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
    st.markdown("## 🛠️ System Setup")
    
    if "available_sensors" not in st.session_state:
        with st.spinner("Scanning files..."):
            st.session_state["available_sensors"] = get_all_unique_entities(uploaded_files)
    options = ["None"] + st.session_state.get("available_sensors", [])

    col_load, col_name = st.columns([1, 2])
    defaults = {"mapping": {}, "units": {}, "ai_context": {}, "profile_name": "My Heat Pump", "rooms_per_zone": {}}
    
    with col_load:
        uploaded_config = st.file_uploader("📂 Load Profile", type="json", key="cfg_up")
        if uploaded_config:
            try:
                loaded = json.load(uploaded_config)
                defaults.update(loaded)
                for k, v in defaults.get("mapping", {}).items():
                    if v in options: st.session_state[f"map_{k}"] = v
                for k, v in defaults.get("units", {}).items():
                    st.session_state[f"unit_{k}"] = v
                st.success(f"Loaded {defaults['profile_name']}")
            except: pass

    with col_name:
        profile_name = st.text_input("Profile Name", value=defaults["profile_name"])

    user_map = {}
    user_units = {}
    
    st.subheader("1. Critical Sensors")
    for key, d in REQUIRED_SENSORS.items():
        s, u = render_sensor_row(d['label'], key, options, defaults, True, d['description'])
        if s != "None": 
            user_map[key] = s
            if u: user_units[key] = u

    st.subheader("2. Recommended")
    for key, d in RECOMMENDED_SENSORS.items():
        s, u = render_sensor_row(d['label'], key, options, defaults, False, d['description'])
        if s != "None": 
            user_map[key] = s
            if u: user_units[key] = u

    st.subheader("3. Zones & Rooms")
    
    with st.expander("Step A: Map Room Sensors", expanded=True):
        mapped_rooms_labels = {} 
        r_cols = st.columns(2)
        for i in range(1, 9):
            r_key = f"{ROOM_SENSOR_PREFIX}{i}"
            with r_cols[(i-1)%2]:
                r_s, _ = render_sensor_row(f"Room Sensor {i}", r_key, options, defaults)
                if r_s != "None":
                    user_map[r_key] = r_s
                    mapped_rooms_labels[r_key] = r_s

    with st.expander("Step B: Configure Zones & Link Rooms", expanded=True):
        rooms_per_zone = {}
        friendly_room_options = {k: f"{v} ({k})" for k, v in mapped_rooms_labels.items()}
        room_keys_list = list(friendly_room_options.keys())
        
        for z_key, z_d in ZONE_SENSORS.items():
            st.markdown(f"**{z_d['label']}**")
            z_s, z_u = render_sensor_row("Zone Pump/Valve (Binary)", z_key, options, defaults, False, z_d['description'])
            
            if z_s != "None":
                user_map[z_key] = z_s
                if z_u: user_units[z_key] = z_u
                
                saved_links = defaults.get("rooms_per_zone", {}).get(z_key, [])
                valid_defaults = [r for r in saved_links if r in room_keys_list]
                
                selected_keys = st.multiselect(
                    f"Select rooms controlled by {z_d['label']}:",
                    options=room_keys_list,
                    default=valid_defaults,
                    format_func=lambda x: friendly_room_options[x],
                    key=f"link_{z_key}",
                    help="Select which room sensors belong to this zone."
                )
                rooms_per_zone[z_key] = selected_keys
            st.markdown("---")

    with st.expander("➕ Advanced / Environmental"):
        for key, d in {**OPTIONAL_SENSORS, **ENVIRONMENTAL_SENSORS}.items():
            s, u = render_sensor_row(d['label'], key, options, defaults, False, d['description'])
            if s != "None": 
                user_map[key] = s
                if u: user_units[key] = u

    st.divider()
    ai_inputs = {}
    for k, p in AI_CONTEXT_PROMPTS.items():
        ai_inputs[k] = st.text_area(p['label'], value=defaults["ai_context"].get(k, ""), help=p['help'])

    # --- TWO STEP ACTIONS ---
    c_btn1, c_btn2 = st.columns(2)
    
    config_object = {
        "profile_name": profile_name,
        "created_at": datetime.now().isoformat(),
        "mapping": user_map,
        "units": user_units,
        "ai_context": ai_inputs,
        "rooms_per_zone": rooms_per_zone,
        "therm_version": "2.0"
    }

    with c_btn1:
        # Step 1: Download
        export_data = config_manager.export_config_for_sharing(config_object)
        export_data["rooms_per_zone"] = rooms_per_zone
        
        st.download_button(
            label="💾 1. Save Configuration",
            data=json.dumps(export_data, indent=2),
            file_name=f"therm_profile_{profile_name.replace(' ', '_')}.json",
            mime="application/json",
            type="secondary"
        )

    with c_btn2:
        # Step 2: Process
        if st.button("🚀 2. Process Uploaded Data", type="primary"):
            if not any(k in user_map for k in REQUIRED_SENSORS):
                st.error("Missing required sensors.")
                return None
            else:
                return config_object
            
    return None

def render_config_download(config):
    export_data = config_manager.export_config_for_sharing(config)
    export_data["rooms_per_zone"] = config.get("rooms_per_zone", {})
    st.download_button("💾 Download Profile", json.dumps(export_data, indent=2), "therm_profile.json", "application/json")