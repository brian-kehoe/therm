# processing.py
import pandas as pd
import numpy as np
from config import THRESHOLDS, SPECIFIC_HEAT_CAPACITY, PHYSICS_THRESHOLDS, NIGHT_HOURS, TARIFF_STRUCTURE
from utils import safe_div
import streamlit as st

# --- DYNAMIC HELPERS ---
def get_active_zone_columns(df):
    """Finds which Zone_X columns exist in the dataframe."""
    return [c for c in df.columns if c.startswith('Zone_') and len(c) <= 7]

def get_room_columns(df):
    """Finds Room_X columns."""
    return [c for c in df.columns if c.startswith('Room_')]

def get_friendly_name(internal_key, user_config):
    """Look up the friendly mapping for a key like 'Zone_1'."""
    if not user_config or "mapping" not in user_config:
        return internal_key
    return user_config["mapping"].get(internal_key, internal_key)

def calculate_physics_metrics(df):
    """Calculates DeltaT and Heat if missing."""
    d = df.copy()
    
    # Basic Validity Masks
    if 'Freq' in d.columns:
        is_compressor_on = d['Freq'] > 15
    else:
        is_compressor_on = d['Power'] > 50

    # 1. Delta T
    if 'DeltaT' not in d.columns:
        d['DeltaT'] = d['FlowTemp'] - d['ReturnTemp']
    
    # 2. Heat Output (if missing)
    if 'Heat' not in d.columns or d['Heat'].sum() == 0:
        d['Heat'] = d['FlowRate'] * 69.73 * d['DeltaT']
        d['Heat'] = d['Heat'].clip(lower=0)
        d.loc[~is_compressor_on, 'Heat'] = 0

    return d

def apply_gatekeepers(df, user_config=None):
    """Ensures data frame has necessary columns for analysis."""
    d = df.copy()
    
    # 1. Physics Engine
    d = calculate_physics_metrics(d)

    # 2. Determine Activity Status
    d['is_active'] = (d['Power'] > 50).astype(int)
    
    # 3. Detect DHW Mode - EXPANDED LOGIC
    # We check multiple possible sources for DHW status
    is_dhw_mask = pd.Series(False, index=d.index)
    
    # Check 1: Explicit DHW_Active binary
    if 'DHW_Active' in d.columns:
         is_dhw_mask |= (d['DHW_Active'] > 0)
         
    # Check 2: Valve Mode Text (common in Samsung/Midea)
    if 'ValveMode' in d.columns:
        is_dhw_mask |= d['ValveMode'].astype(str).str.lower().str.contains('hot|dhw', na=False)

    # Check 3: DHW Mode Value (specific to your config)
    if 'DHW_Mode' in d.columns:
         # Some sensors use 1/0, some use text. We check for non-zero or specific text.
         # Assuming numeric > 0 is active, or text contains 'hot'
         if pd.api.types.is_numeric_dtype(d['DHW_Mode']):
             is_dhw_mask |= (d['DHW_Mode'] > 0)
         else:
             is_dhw_mask |= d['DHW_Mode'].astype(str).str.lower().str.contains('on|active|hot', na=False)

    d['is_DHW'] = is_dhw_mask
    d['is_heating'] = (d['is_active'] == 1) & (~d['is_DHW'])

    # --- RESTORED COLUMNS FOR VIEW COMPATIBILITY ---
    d['Power_Clean'] = np.where(d['is_active'], d['Power'], 0)
    d['Heat_Clean'] = np.where(d['is_active'], d['Heat'], 0)
    # -----------------------------------------------

    d['Power_Heating'] = np.where(d['is_heating'], d['Power'], 0)
    d['Power_DHW'] = np.where(d['is_DHW'], d['Power'], 0)
    d['Heat_Heating'] = np.where(d['is_heating'], d['Heat'], 0)
    d['Heat_DHW'] = np.where(d['is_DHW'], d['Heat'], 0)
    
    d['COP_Real'] = safe_div(d['Heat'], d['Power'])
    d['COP_Graph'] = d['COP_Real'].clip(0, 6)

    # 5. Zones Active Count & Friendly Config String
    zone_cols = get_active_zone_columns(d)
    if zone_cols:
        d['Active_Zones_Count'] = d[zone_cols].sum(axis=1)
        
        # Create a string representation using FRIENDLY names
        # Pre-fetch friendly names to avoid repeated lookups
        z_map = {z: get_friendly_name(z, user_config) for z in zone_cols}
        
        def get_zone_str(row):
            active = []
            for z in zone_cols:
                if row[z] > 0:
                    # Use the friendly name if available, else the Zone ID
                    active.append(z_map.get(z, z)) 
            return " + ".join(active) if active else "None"
            
        d['Zone_Config'] = d.apply(get_zone_str, axis=1)
    else:
        d['Active_Zones_Count'] = 0
        d['Zone_Config'] = "None"

    # 6. Immersion Logic
    if 'Immersion_Mode' in d.columns:
        d['Immersion_Active'] = d['Immersion_Mode'] > 0
    elif 'Indoor_Power' in d.columns:
        d['Immersion_Active'] = d['Indoor_Power'] > 2500
    else:
        d['Immersion_Active'] = False

    if 'Indoor_Power' in d.columns:
        d['Immersion_Power'] = np.where(d['Immersion_Active'], d['Indoor_Power'], 0)
    else:
        d['Immersion_Power'] = np.where(d['Immersion_Active'], 3000, 0)

    # 7. Cost Calculation
    d['hour'] = d.index.hour
    d['is_night_rate'] = d['hour'].isin(NIGHT_HOURS)
    d['Current_Rate'] = np.where(d['is_night_rate'], 0.15, 0.35)
    d['Cost_Inc'] = (d['Power'] / 1000 / 60) * d['Current_Rate']

    return d

def detect_runs(df, user_config=None):
    """
    Segments data into discrete runs and links room sensors.
    """
    rooms_per_zone = user_config.get("rooms_per_zone", {}) if user_config else {}
    
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
        
        # Metrics
        heat_kwh = group['Heat'].sum() / 60000.0
        elec_kwh = group['Power'].sum() / 60000.0
        cop = safe_div(heat_kwh, elec_kwh)
        
        # Zones
        active_zones_list = []
        if run_type == "Heating" and zone_cols:
            for z in zone_cols:
                if (group[z].sum() / len(group)) > 0.5:
                    active_zones_list.append(z)
        
        # Create friendly string for dominant zones
        friendly_zones = []
        for z in active_zones_list:
            friendly_zones.append(get_friendly_name(z, user_config))
            
        dominant_zones_str = " + ".join(friendly_zones) if friendly_zones else ("None" if run_type == "Heating" else "DHW")

        # === ATTACH RELEVANT ROOMS ===
        relevant_rooms = []
        if run_type == "Heating" and rooms_per_zone and active_zones_list:
            for z in active_zones_list:
                # Add all rooms linked to this zone
                relevant_rooms.extend(rooms_per_zone.get(z, []))
        
        relevant_rooms = list(set(relevant_rooms)) # Dedupe

        # Room Response (Delta)
        # Store as { "Friendly Name": delta_value }
        room_deltas = {}
        for r in room_cols:
            if len(group[r].dropna()) > 0:
                start_t = group[r].iloc[0]
                end_t = group[r].iloc[-1]
                
                # GET FRIENDLY NAME HERE
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
            "avg_flow_rate": round(group['FlowRate'].mean(), 1),
            "run_cop": round(cop, 2),
            "heat_kwh": heat_kwh,
            "electricity_kwh": elec_kwh,
            "active_zones": dominant_zones_str,  # Now uses friendly names
            "dominant_zones": dominant_zones_str,
            "room_deltas": room_deltas,          # Now uses friendly names
            "relevant_rooms": relevant_rooms, 
            "immersion_kwh": group['Immersion_Power'].sum() / 60000.0,
            "immersion_mins": group['Immersion_Active'].sum()
        })
        
    return runs

def get_daily_stats(df):
    """Aggregates DataFrame into Daily totals."""
    
    # 1. Base Core Aggregations (Force MultiIndex with list)
    agg_dict = {
        'Power_Heating': ['sum'], 'Power_DHW': ['sum'], 
        'Heat_Heating': ['sum'], 'Heat_DHW': ['sum'],
        'Cost_Inc': ['sum'], 'is_active': ['sum'],
        'Immersion_Power': ['sum'],
        'is_DHW': ['sum'], 
        'is_heating': ['sum']
    }
    
    # 2. Dynamic Aggregations for ALL other columns
    exclude = list(agg_dict.keys()) + ['last_changed', 'entity_id', 'state', 'Zone_Config']
    for col in df.columns:
        if col not in exclude:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = ['mean', 'min', 'max', 'count']
            else:
                agg_dict[col] = ['count']

    # 3. Perform Resample
    daily = df.resample('D').agg(agg_dict)
    
    # 4. Flatten Columns
    daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily.columns]
    
    # 5. Fix suffixes (Safety check)
    for core_col in ['Power_Heating', 'Power_DHW', 'Heat_Heating', 'Heat_DHW']:
        if core_col in daily.columns and f"{core_col}_sum" not in daily.columns:
            daily = daily.rename(columns={core_col: f"{core_col}_sum"})

    # 6. Rename Core Columns for Views
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
    
    # 7. Calculate Energies (kWh)
    daily['Electricity_Heating_kWh'] = daily.get('Electricity_Heating_Wmin', 0) / 60000.0
    daily['Electricity_DHW_kWh'] = daily.get('Electricity_DHW_Wmin', 0) / 60000.0
    daily['Heat_Heating_kWh'] = daily.get('Heat_Heating_Wmin', 0) / 60000.0
    daily['Heat_DHW_kWh'] = daily.get('Heat_DHW_Wmin', 0) / 60000.0
    daily['Immersion_kWh'] = daily.get('Immersion_Wh', 0) / 60000.0
    
    daily['Total_Electricity_kWh'] = daily['Electricity_Heating_kWh'] + daily['Electricity_DHW_kWh'] + daily['Immersion_kWh']
    daily['Total_Heat_kWh'] = daily['Heat_Heating_kWh'] + daily['Heat_DHW_kWh']
    
    daily['Global_SCOP'] = safe_div(daily['Total_Heat_kWh'], daily['Total_Electricity_kWh'])
    
    # 8. Data Quality Scores
    if 'Power' in df.columns:
        daily_counts = df['Power'].resample('D').count()
        daily['DQ_Score'] = (daily_counts / 1440 * 100).clip(0, 100)
    else:
        daily['DQ_Score'] = 0
    
    daily['DQ_Tier'] = np.where(daily['DQ_Score'] > 90, "Tier 1 (Gold)", "Tier 3 (Bronze)")

    return daily