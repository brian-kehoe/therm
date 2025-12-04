# processing.py
import pandas as pd
import numpy as np
import streamlit as st
from config import THRESHOLDS, SPECIFIC_HEAT_CAPACITY, PHYSICS_THRESHOLDS, NIGHT_HOURS, TARIFF_STRUCTURE
from utils import safe_div

# --- DYNAMIC HELPERS ---
def get_active_zone_columns(df):
    return [c for c in df.columns if c.startswith('Zone_') and len(c) <= 7]

def get_room_columns(df):
    return [c for c in df.columns if c.startswith('Room_')]

def get_friendly_name(internal_key, user_config):
    """
    Robust lookup for friendly names. 
    Handles cases where config might be None, empty, or malformed.
    """
    if not isinstance(user_config, dict):
        return str(internal_key)
    
    mapping = user_config.get("mapping")
    if not isinstance(mapping, dict):
        return str(internal_key)
        
    val = mapping.get(internal_key, internal_key)
    return str(val)

def calculate_physics_metrics(df):
    d = df.copy()
    # Delta T
    if 'DeltaT' not in d.columns:
        d['DeltaT'] = d['FlowTemp'] - d['ReturnTemp']

    return d

def apply_gatekeepers(df, user_config=None):
    d = df.copy()
    
    thresholds = PHYSICS_THRESHOLDS if isinstance(PHYSICS_THRESHOLDS, dict) else {}

    # 1. Physics
    d = calculate_physics_metrics(d)

    # 2. Activity
    power_min = thresholds.get('power_min', 50)
    d['is_active'] = (d['Power'] > power_min).astype(int)
    
    # 3. Detect DHW Mode
    is_dhw_mask = pd.Series(False, index=d.index)
    
    if 'DHW_Active' in d.columns:
         is_dhw_mask |= (d['DHW_Active'] > 0)

    if 'DHW_Mode' in d.columns:
         if pd.api.types.is_numeric_dtype(d['DHW_Mode']):
             is_dhw_mask |= (d['DHW_Mode'] > 0)
         else:
             is_dhw_mask |= d['DHW_Mode'].astype(str).str.lower().str.contains('on|active|hot|dhw', na=False)
         
    if 'ValveMode' in d.columns:
        is_dhw_mask |= d['ValveMode'].astype(str).str.lower().str.contains('hot|dhw', na=False)

    d['is_DHW'] = is_dhw_mask
    d['is_heating'] = (d['is_active'] == 1) & (~d['is_DHW'])

    # --- HEAT OUTPUT LOGIC ---
    has_flow = 'FlowRate' in d.columns
    has_heat_col = 'Heat' in d.columns
    has_heat_data = False

    if has_heat_col:
        # treat NaN as 0 when checking whether we have any signal
        heat_series = pd.to_numeric(d['Heat'], errors='coerce').fillna(0)
        has_heat_data = heat_series.abs().sum() > 0
        d['Heat'] = heat_series  # normalise type

    if not has_heat_data:
        if has_flow:
            # Only derive when we have hydraulics
            min_flow = thresholds.get("min_flow_rate_lpm", 0)
            min_freq = thresholds.get("min_freq_for_heat", 0)

            flow = pd.to_numeric(d['FlowRate'], errors='coerce').fillna(0)
            delta_t = pd.to_numeric(d.get('DeltaT', 0), errors='coerce').fillna(0)
            freq = pd.to_numeric(d.get('Freq', 0), errors='coerce').fillna(0)

            # Basic physics: Heat [W] = c * m_dot * ΔT
            heat_raw = SPECIFIC_HEAT_CAPACITY * flow * delta_t

            # Gatekeepers
            valid = (
                (flow >= min_flow) &
                (freq >= min_freq) &
                (delta_t.abs() >= thresholds.get("min_valid_delta_t", 0)) &
                (delta_t.abs() <= thresholds.get("max_valid_delta_t", 999))
            )

            d['Heat'] = 0.0
            d.loc[valid, 'Heat'] = heat_raw[valid]
            # No heat when compressor off
            d.loc[~d['is_active'].astype(bool), 'Heat'] = 0.0
        else:
            # No Heat sensor and no FlowRate → no energy channel
            d['Heat'] = 0.0

    # 4. Power & Heat Splits
    d['Power_Heating'] = np.where(d['is_heating'], d['Power'], 0)
    d['Power_DHW'] = np.where(d['is_DHW'], d['Power'], 0)
    d['Heat_Heating'] = np.where(d['is_heating'], d['Heat'], 0)
    d['Heat_DHW']     = np.where(d['is_DHW'],     d['Heat'], 0)
    
    d['COP_Real']  = safe_div(d['Heat'], d['Power'])
    d['COP_Graph'] = d['COP_Real'].clip(0, 6)

    # 5. Zone Config Strings
    zone_cols = get_active_zone_columns(d)
    
    if zone_cols:
        # Data Type Fix: Ensure all zone columns are numeric
        for z in zone_cols:
            if not pd.api.types.is_numeric_dtype(d[z]):
                st.write(f"⚠️ coercing non-numeric column {z} to numbers (Values: {d[z].unique()[:3]})")
                d[z] = d[z].astype(str).str.lower().replace({'on': 1, 'off': 0, 'true': 1, 'false': 0})
                d[z] = pd.to_numeric(d[z], errors='coerce').fillna(0)

        d['Active_Zones_Count'] = d[zone_cols].sum(axis=1)
        z_map = {z: get_friendly_name(z, user_config) for z in zone_cols}
        
        def get_zone_str(row):
            active = [str(z_map.get(z, z)) for z in zone_cols if row[z] > 0]
            return " + ".join(active) if active else "None"
        d['Zone_Config'] = d.apply(get_zone_str, axis=1)
    else:
        d['Active_Zones_Count'] = 0
        d['Zone_Config'] = "None"

    # 6. Immersion
    if 'Immersion_Mode' in d.columns:
        d['Immersion_Active'] = d['Immersion_Mode'] > 0
    elif 'Indoor_Power' in d.columns:
        d['Immersion_Active'] = d['Indoor_Power'] > 2500
    else:
        d['Immersion_Active'] = False

    d['Immersion_Power'] = np.where(d['Immersion_Active'], d['Indoor_Power'] if 'Indoor_Power' in d.columns else 3000, 0)

    # 7. Cost (CRASH FIX: Check type of TARIFF_STRUCTURE)
    d['hour'] = d.index.hour
    d['is_night_rate'] = d['hour'].isin(NIGHT_HOURS)
    
    if isinstance(TARIFF_STRUCTURE, dict):
        rate_day = TARIFF_STRUCTURE.get('day_rate', 0.35)
        rate_night = TARIFF_STRUCTURE.get('night_rate', 0.15)
    else:
        # Fallback if TARIFF_STRUCTURE is a list or legacy format
        rate_day = 0.35
        rate_night = 0.15
    
    d['Current_Rate'] = np.where(d['is_night_rate'], rate_night, rate_day)
    d['Cost_Inc'] = (d['Power'] / 1000 / 60) * d['Current_Rate']

    # --- HEAT OUTPUT LOGIC ---
    has_flow = 'FlowRate' in d.columns
    has_heat_col = 'Heat' in d.columns
    has_heat_data = False

    if has_heat_col:
        # treat NaN as 0 when checking whether we have any signal
        heat_series = pd.to_numeric(d['Heat'], errors='coerce').fillna(0)
        has_heat_data = heat_series.abs().sum() > 0
        d['Heat'] = heat_series  # normalise type

    if not has_heat_data:
        if has_flow:
            # Only derive when we have hydraulics
            min_flow = PHYSICS_THRESHOLDS.get("min_flow_rate_lpm", 0)
            min_freq = PHYSICS_THRESHOLDS.get("min_freq_for_heat", 0)

            flow = pd.to_numeric(d['FlowRate'], errors='coerce').fillna(0)
            delta_t = pd.to_numeric(d.get('DeltaT', 0), errors='coerce').fillna(0)
            freq = pd.to_numeric(d.get('Freq', 0), errors='coerce').fillna(0)

            # Basic physics: Heat [W] = c * m_dot * ΔT
            heat_raw = SPECIFIC_HEAT_CAPACITY * flow * delta_t

            # Gatekeepers
            valid = (
                (flow >= min_flow) &
                (freq >= min_freq) &
                (delta_t.abs() >= PHYSICS_THRESHOLDS.get("min_valid_delta_t", 0)) &
                (delta_t.abs() <= PHYSICS_THRESHOLDS.get("max_valid_delta_t", 999))
            )

            d['Heat'] = 0.0
            d.loc[valid, 'Heat'] = heat_raw[valid]
            # No heat when compressor off
            d.loc[~d['is_active'].astype(bool), 'Heat'] = 0.0
        else:
            # No Heat sensor and no FlowRate → no energy channel
            d['Heat'] = 0.0

    # --- existing splits + COP ---
    d['Heat_Heating'] = np.where(d['is_heating'], d['Heat'], 0)
    d['Heat_DHW']     = np.where(d['is_DHW'],     d['Heat'], 0)

    d['COP_Real']  = safe_div(d['Heat'], d['Power'])
    d['COP_Graph'] = d['COP_Real'].clip(0, 6)

    return d

def detect_runs(df, user_config=None):
    # Safety check on user_config type
    if isinstance(user_config, dict):
        rooms_per_zone = user_config.get("rooms_per_zone", {})
    else:
        rooms_per_zone = {}
    
    df['run_change'] = (
        (df['is_active'].diff().ne(0)) | 
        (df['is_DHW'].ne(df['is_DHW'].shift()))
    )
    df['run_id'] = df['run_change'].cumsum()
    
    runs = []
    active_groups = df[df['is_active'] == 1].groupby('run_id')
    
    zone_cols = get_active_zone_columns(df)
    room_cols = get_room_columns(df)

    for run_id, group in active_groups:
        if len(group) < 5: continue 
        
        run_type = "DHW" if group['is_DHW'].any() else "Heating"
        
        # Averages
        avg_outdoor = group['OutdoorTemp'].mean() if 'OutdoorTemp' in group.columns else 0
        avg_flow = group['FlowTemp'].mean()
        avg_flow_rate = group['FlowRate'].mean() if 'FlowRate' in group.columns else 0
        
        # Metrics
        heat_kwh = group['Heat'].sum() / 60000.0
        elec_kwh = group['Power'].sum() / 60000.0
        cop = safe_div(heat_kwh, elec_kwh)
        
        # Friendly Zones
        active_zones_list = []
        if run_type == "Heating" and zone_cols:
            for z in zone_cols:
                if (group[z].sum() / len(group)) > 0.5:
                    active_zones_list.append(z)
        
        friendly_zones = [str(get_friendly_name(z, user_config)) for z in active_zones_list]
        dominant_zones_str = " + ".join(friendly_zones) if friendly_zones else ("None" if run_type == "Heating" else "DHW")

        # Friendly Rooms
        relevant_rooms = []
        if run_type == "Heating" and rooms_per_zone and active_zones_list:
            for z in active_zones_list:
                relevant_rooms.extend(rooms_per_zone.get(z, []))
        relevant_rooms = list(set(relevant_rooms)) 
        
        # Room Deltas
        room_deltas = {}
        for r in room_cols:
            if len(group[r].dropna()) > 0:
                start_t = group[r].iloc[0]
                end_t = group[r].iloc[-1]
                friendly_r = get_friendly_name(r, user_config)
                room_deltas[friendly_r] = round(end_t - start_t, 2)

        runs.append({
            "id": int(run_id),
            "start": group.index[0],
            "end": group.index[-1],
            "duration_mins": len(group),
            "run_type": run_type,
            "avg_outdoor": round(avg_outdoor, 1),
            "avg_flow_temp": round(avg_flow, 1),
            "avg_dt": round(group['DeltaT'].mean(), 1),
            "avg_flow_rate": round(avg_flow_rate, 1),
            "run_cop": round(cop, 2),
            "heat_kwh": heat_kwh,
            "electricity_kwh": elec_kwh,
            "active_zones": dominant_zones_str,
            "dominant_zones": dominant_zones_str,
            "room_deltas": room_deltas,
            "relevant_rooms": relevant_rooms, 
            "immersion_kwh": group['Immersion_Power'].sum() / 60000.0,
            "immersion_mins": group['Immersion_Active'].sum()
        })
        
    return runs

def get_daily_stats(df):
    agg_dict = {
        'Power_Heating': ['sum'], 'Power_DHW': ['sum'], 
        'Heat_Heating': ['sum'], 'Heat_DHW': ['sum'],
        'Cost_Inc': ['sum'], 'is_active': ['sum'],
        'Immersion_Power': ['sum'],
        'is_DHW': ['sum'], 'is_heating': ['sum']
    }
    
    exclude = list(agg_dict.keys()) + ['last_changed', 'entity_id', 'state', 'Zone_Config']
    for col in df.columns:
        if col not in exclude:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = ['mean', 'min', 'max', 'count']
            else:
                agg_dict[col] = ['count']

    daily = df.resample('D').agg(agg_dict)
    daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily.columns]
    
    for core_col in ['Power_Heating', 'Power_DHW', 'Heat_Heating', 'Heat_DHW']:
        if core_col in daily.columns and f"{core_col}_sum" not in daily.columns:
            daily = daily.rename(columns={core_col: f"{core_col}_sum"})

    rename_map = {
        'Power_Heating_sum': 'Electricity_Heating_Wmin',
        'Power_DHW_sum': 'Electricity_DHW_Wmin',
        'Heat_Heating_sum': 'Heat_Heating_Wmin',
        'Heat_DHW_sum': 'Heat_DHW_Wmin',
        'Cost_Inc_sum': 'Daily_Cost_Euro',
        'is_active_sum': 'Active_Mins',
        'is_DHW_sum': 'DHW_Mins',
        'is_heating_sum': 'Heating_Mins', 
        'Immersion_Power_sum': 'Immersion_Wh',
        'OutdoorTemp_mean': 'Outdoor_Avg',
        'OutdoorTemp_min': 'Outdoor_Min',
        'OutdoorTemp_max': 'Outdoor_Max'
    }
    daily = daily.rename(columns=rename_map)
    
    daily['Electricity_Heating_kWh'] = daily.get('Electricity_Heating_Wmin', 0) / 60000.0
    daily['Electricity_DHW_kWh'] = daily.get('Electricity_DHW_Wmin', 0) / 60000.0
    daily['Heat_Heating_kWh'] = daily.get('Heat_Heating_Wmin', 0) / 60000.0
    daily['Heat_DHW_kWh'] = daily.get('Heat_DHW_Wmin', 0) / 60000.0
    daily['Immersion_kWh'] = daily.get('Immersion_Wh', 0) / 60000.0
    
    daily['Total_Electricity_kWh'] = daily['Electricity_Heating_kWh'] + daily['Electricity_DHW_kWh'] + daily['Immersion_kWh']
    daily['Total_Heat_kWh'] = daily['Heat_Heating_kWh'] + daily['Heat_DHW_kWh']
    
    daily['Global_SCOP'] = safe_div(daily['Total_Heat_kWh'], daily['Total_Electricity_kWh'])
    
    if 'Power' in df.columns:
        daily_counts = df['Power'].resample('D').count()
        daily['DQ_Score'] = (daily_counts / 1440 * 100).clip(0, 100)
    else:
        daily['DQ_Score'] = 0
    
    daily['DQ_Tier'] = np.where(daily['DQ_Score'] > 90, "Tier 1 (Gold)", "Tier 3 (Bronze)")

    return daily
