# processing.py
import pandas as pd
import numpy as np
from config import (THRESHOLDS, ENTITY_MAP, NIGHT_HOURS, MIN_HEAT_FREQ, MIN_POWER_W, 
                    TARIFF_STRUCTURE, ZONE_TO_ROOM_MAP, CONFIG_HISTORY, 
                    SPECIFIC_HEAT_CAPACITY, PHYSICS_THRESHOLDS)
from utils import safe_div, availability_pct

def calculate_physics_metrics(df):
    """
    Overwrites imported 'Heat' and 'DeltaT' columns with Python-calculated values.
    Returns NaN for DeltaT if invalid, so Data Quality correctly reports gaps.
    """
    d = df.copy()
    
    # --- LOGIC: COMPRESSOR STATUS ---
    freq_active = d['Freq'] > PHYSICS_THRESHOLDS['min_freq_for_heat']
    power_active = d['Power'] > MIN_POWER_W
    is_compressor_on = freq_active | power_active

    # 1. Calculate Raw Delta T (Flow - Return)
    raw_delta = d['FlowTemp'] - d['ReturnTemp']
    
    # 2. Apply Delta T Logic Mask
    valid_delta_mask = (
        is_compressor_on &
        (d['FlowRate'] > PHYSICS_THRESHOLDS['min_flow_rate_lpm']) &
        (raw_delta > PHYSICS_THRESHOLDS['min_valid_delta_t']) &
        (raw_delta <= PHYSICS_THRESHOLDS['max_valid_delta_t'])
    )
    
    # FIX: Use np.nan instead of 0.0 for invalid data so DQ counts it as missing
    d['DeltaT'] = np.where(valid_delta_mask, raw_delta, np.nan)
    
    # 3. Calculate Heat Output (Watts)
    defrost_active = d['Defrost'] == 1 if 'Defrost' in d.columns else False
    
    valid_heat_mask = (
        is_compressor_on &
        (d['FlowRate'] > PHYSICS_THRESHOLDS['min_flow_rate_lpm']) &
        (d['DeltaT'] > 0) & 
        (~defrost_active)
    )
    
    flow_lps = d['FlowRate'] / 60.0
    heat_watts = flow_lps * SPECIFIC_HEAT_CAPACITY * d['DeltaT'] * 1000.0
    
    # Keep Heat as 0.0 when off (because 0 watts is a valid "off" state)
    d['Heat'] = np.where(valid_heat_mask, heat_watts, 0.0)
    
    return d

def detect_hydraulic_interference(row_or_df):
    is_dhw = row_or_df['is_DHW']
    pump_cols = [c for c in ['Zone_UFH', 'Zone_DS', 'Zone_US'] if c in row_or_df]
    if pump_cols:
        if isinstance(row_or_df, pd.DataFrame):
            sensors_active = row_or_df[pump_cols].sum(axis=1) > 0
        else:
            sensors_active = sum(row_or_df[c] for c in pump_cols) > 0
    else:
        sensors_active = False 
        
    ghost_thresh = THRESHOLDS['ghost_power_threshold']
    if 'Immersion_Mode' in row_or_df:
        immersion_off = row_or_df['Immersion_Mode'] == 0
    else:
        immersion_off = True
    if 'Indoor_Power' in row_or_df:
        power_high = row_or_df['Indoor_Power'] > ghost_thresh
        power_reasonable = row_or_df['Indoor_Power'] < 2500 
        proxy_active = power_high & immersion_off & power_reasonable
    else:
        proxy_active = False
    return (is_dhw & sensors_active, is_dhw & proxy_active)

def apply_gatekeepers(df):
    d = df.copy()
    
    # 1. Ensure all expected columns exist (Gatekeeper 1)
    # UPDATED: Replaced HP_Binary_Status with Heat_Pump_Active
    expected = ['Freq', 'Power', 'Heat', 'DeltaT', 'FlowRate', 'FlowTemp', 'ReturnTemp',
                'OutdoorTemp', 'DHW_Temp', 'Indoor_Power', 'Immersion_Mode', 'Heat_Pump_Active',
                'Defrost', 'Zone_UFH', 'Zone_DS', 'Zone_US', 'Pump_Primary', 'Pump_Secondary']
    for col in expected:
        if col not in d.columns: d[col] = 0.0

    # --- NEW: INJECT PHYSICS ENGINE HERE ---
    d = calculate_physics_metrics(d)
    # ---------------------------------------

    d['Heat'] = d['Heat'].clip(lower=0, upper=9500)
    d['OutdoorTemp'] = d['OutdoorTemp'].clip(upper=30)
    
    if 'ValveMode' not in d.columns: d['ValveMode'] = 'Heating'
    if 'DHW_Mode' not in d.columns: d['DHW_Mode'] = 'Standard'

    # UPDATED: Logic now uses Heat_Pump_Active
    d['Heat_Pump_Active_Clean'] = d['Heat_Pump_Active'].fillna(0).astype(int)
    
    if d['Heat_Pump_Active_Clean'].sum() == 0 and 'Freq' in d.columns:
        d['Heat_Pump_Active_Clean'] = np.where(
            (d['Freq'] > MIN_HEAT_FREQ) | (abs(d['Power']) > MIN_POWER_W),
            1, 0
        ).astype(int)

    d['run_start_raw'] = (d['Heat_Pump_Active_Clean'].diff() == 1).astype(int)
    d['power_lagged'] = d['Power'].shift(1).fillna(0)
    false_start_mask = (d['run_start_raw'] == 1) & (d['power_lagged'] > MIN_POWER_W)
    d.loc[false_start_mask, 'Heat_Pump_Active_Clean'] = 0 
    d['run_start'] = np.where(false_start_mask, 0, d['run_start_raw'])
    d['Heat_Pump_Active'] = d['Heat_Pump_Active_Clean']

    # UPDATED: is_active depends on Heat_Pump_Active
    d['is_active'] = (d['Heat_Pump_Active'] == 1) | (abs(d['Power']) > MIN_POWER_W)
    d['is_active'] = d['is_active'].astype(int)
    d['is_heating'] = (d['Freq'] > MIN_HEAT_FREQ) | (abs(d['Power']) > MIN_POWER_W)
    
    is_active_lagged = d['is_active'].shift(1).fillna(0)
    is_active_leaded = d['is_active'].shift(-1).fillna(0)
    stick_mask = (d['is_active'] == 0) & (is_active_lagged == 1) & (is_active_leaded == 1)
    d.loc[stick_mask, 'is_active'] = 1

    d['Power_Clean'] = d.apply(lambda x: abs(x['Power']) if x['is_heating'] else 0, axis=1)
    d['Heat_Clean'] = d.apply(lambda x: x['Heat'] if x['is_heating'] else 0, axis=1)
    d['COP_Real'] = safe_div(d['Heat_Clean'], d['Power_Clean'])
    d['COP_Graph'] = d['COP_Real'].clip(0, 6)
    d['hour'] = d.index.hour
    d['is_night_rate'] = d['hour'].isin(NIGHT_HOURS)
    
    zone_cols = [c for c in ['Zone_UFH', 'Zone_DS', 'Zone_US'] if c in d.columns]
    d['Active_Zones_Count'] = d[zone_cols].fillna(0).sum(axis=1)
    if 'ValveMode' in d.columns:
        d['is_DHW'] = d['ValveMode'].astype(str).str.lower().str.contains('hot|dhw')
    else:
        d['is_DHW'] = False
        
    sensor_interf, power_proxy = detect_hydraulic_interference(d)
    d['Interference_Detected'] = sensor_interf
    d['Ghost_Power_Active'] = power_proxy

    d['Power_Heating'] = np.where(~d['is_DHW'], d['Power_Clean'], 0)
    d['Power_DHW'] = np.where(d['is_DHW'], d['Power_Clean'], 0)
    d['Heat_Heating'] = np.where(~d['is_DHW'], d['Heat_Clean'], 0)
    d['Heat_DHW'] = np.where(d['is_DHW'], d['Heat_Clean'], 0)
    d['Power_Night'] = np.where(d['is_night_rate'], d['Power_Clean'], 0)
    d['Power_Day'] = np.where(~d['is_night_rate'], d['Power_Clean'], 0)
    d['DeltaT_Active'] = d['DeltaT'].where(d['is_heating'])
    d['FlowRate_Active'] = d['FlowRate'].where(d['is_heating'])
    
    d['Current_Rate'] = np.nan
    sorted_tariffs = sorted(TARIFF_STRUCTURE, key=lambda x: x['valid_from'])
    for i, profile in enumerate(sorted_tariffs):
        start_date = pd.to_datetime(profile['valid_from']).tz_localize(d.index.tz)
        if i < len(sorted_tariffs) - 1:
            end_date = pd.to_datetime(sorted_tariffs[i+1]['valid_from']).tz_localize(d.index.tz)
        else:
            end_date = d.index.max() + pd.Timedelta(days=1)
        mask_date = (d.index >= start_date) & (d.index < end_date)
        if not mask_date.any(): continue
        for rule in profile['rules']:
            t_start = pd.to_datetime(rule['start']).time()
            t_end = pd.to_datetime(rule['end']).time()
            if t_start < t_end:
                rule_mask = (d.index.time >= t_start) & (d.index.time < t_end)
            else:
                rule_mask = (d.index.time >= t_start) | (d.index.time < t_end)
            final_mask = mask_date & rule_mask
            if final_mask.any():
                d.loc[final_mask, 'Current_Rate'] = rule['rate']
    d['Current_Rate'] = d['Current_Rate'].ffill().bfill()
    d['Cost_Inc'] = (d['Power_Clean'] / 1000 / 60) * d['Current_Rate']
    d['defrost_start'] = (d['Defrost'].fillna(0).diff() == 1).astype(int)
    
    def get_zone_str(row):
        active = []
        if 'Zone_UFH' in row and row['Zone_UFH'] > 0: active.append("UFH")
        if 'Zone_DS' in row and row['Zone_DS'] > 0: active.append("DS")
        if 'Zone_US' in row and row['Zone_US'] > 0: active.append("US")
        return "+".join(active) if active else "None"
    d['Zone_Config'] = d.apply(get_zone_str, axis=1)
    d['DHW_Active'] = d['is_DHW'].astype(int)
    return d

def detect_runs(df):
    df['run_change'] = (
        (df['is_active'].astype(int).diff().ne(0)) | 
        (df['ValveMode'].ne(df['ValveMode'].shift()) & df['is_active'])
    )
    df['run_id'] = df['run_change'].cumsum()
    runs = []
    
    active_runs_df = df[df['is_active'] == 1]
    
    for run_id, group in active_runs_df.groupby('run_id'):
        if len(group) < 5: continue 
        start_time = group.index[0]
        end_time = group.index[-1]
        mode_val = str(group['ValveMode'].mode(dropna=True).iloc[0]) if not group['ValveMode'].mode().empty else 'Heating'
        run_type = "DHW" if 'hot' in mode_val.lower() or 'dhw' in mode_val.lower() else "Heating"
        
        dhw_mode_val = None
        if run_type == "DHW" and 'DHW_Mode' in group.columns:
            m = group['DHW_Mode'].mode(dropna=True)
            if not m.empty: dhw_mode_val = m.iloc[0]

        dhw_temp_start = group['DHW_Temp'].iloc[0] if 'DHW_Temp' in group.columns else 0
        dhw_temp_end = group['DHW_Temp'].iloc[-1] if 'DHW_Temp' in group.columns else 0
        
        interference_detected = group['Interference_Detected'].any()
        power_ghost_detected = group['Ghost_Power_Active'].any()
        heating_pump_overlap_pct = 0.0
        if run_type == "DHW" and len(group) > 0:
            heating_pump_overlap_pct = (group['Interference_Detected'].sum() / len(group)) * 100

        immersion_active = False
        immersion_kwh = 0.0
        immersion_mins = 0
        if 'Immersion_Mode' in group.columns:
            imm_on = group['Immersion_Mode'] > 0
            immersion_active = imm_on.any()
            immersion_mins = imm_on.sum()
            if 'Indoor_Power' in group.columns:
                immersion_kwh = (group.loc[imm_on, 'Indoor_Power'].clip(lower=0).sum()) / 60000.0
        elif 'Indoor_Power' in group.columns:
            mask = group['Indoor_Power'] > 2500
            if mask.any():
                immersion_active = True
                immersion_mins = mask.sum()
                immersion_kwh = group.loc[mask, 'Indoor_Power'].clip(lower=0).sum() / 60000.0

        dominant_zone_str = group['Zone_Config'].mode()[0] if not group['Zone_Config'].empty else "None"
        room_deltas = {}
        if run_type == "Heating":
            zone_map = {'Zone_UFH': 'UFH', 'Zone_DS': 'DS', 'Zone_US': 'US'}
            active_z = [z for z, short in zone_map.items() if z in group.columns and group[z].sum() > 0]
            relevant_rooms = []
            if active_z:
                for z in active_z: relevant_rooms.extend(ZONE_TO_ROOM_MAP.get(z, []))
            else:
                if "US" in dominant_zone_str: relevant_rooms.extend(ZONE_TO_ROOM_MAP.get('Zone_US', []))
                if "DS" in dominant_zone_str: relevant_rooms.extend(ZONE_TO_ROOM_MAP.get('Zone_DS', []))
                if "UFH" in dominant_zone_str: relevant_rooms.extend(ZONE_TO_ROOM_MAP.get('Zone_UFH', []))
            for r_col in relevant_rooms:
                if r_col in group.columns:
                    room_deltas[r_col] = round(group[r_col].iloc[-1] - group[r_col].iloc[0], 2)

        runs.append({
            "id": int(run_id),
            "start": start_time,
            "end": end_time,
            "duration_mins": len(group),
            "run_type": run_type,
            "heating_during_dhw_detected": bool(interference_detected),
            "heating_during_dhw_pct": round(heating_pump_overlap_pct, 1),
            "ghost_pumping_power_detected": bool(power_ghost_detected) if run_type == "DHW" else False,
            "immersion_detected": bool(immersion_active),
            "immersion_kwh": round(immersion_kwh, 3),
            "immersion_mins": int(immersion_mins),          
            "dhw_temp_start": round(dhw_temp_start, 1),
            "dhw_temp_end": round(dhw_temp_end, 1),
            "dhw_rise": round(dhw_temp_end - dhw_temp_start, 1),
            "avg_dt": group['DeltaT'].mean(),
            "avg_flow": group['FlowRate'].mean() if 'FlowRate' in group.columns else 0,
            "run_cop": safe_div(group['Heat_Clean'].sum(), group['Power_Clean'].sum()),
            "heat_kwh": group['Heat_Clean'].sum() / 60000.0,
            "electricity_kwh": group['Power_Clean'].sum() / 60000.0,
            "dominant_zones": dominant_zone_str,
            "active_zones": dominant_zone_str,
            "room_deltas": room_deltas,
            "dhw_mode": dhw_mode_val,
            "quiet_mode_active": bool((group['Quiet_Mode'] > 0).any()) if 'Quiet_Mode' in group.columns else False
        })
    return runs

def get_config_for_date(date_index):
    history = sorted(CONFIG_HISTORY, key=lambda x: x["start"])
    tags = []
    notes = []
    for d in date_index:
        d_str = str(d.date())
        active_tag = "baseline_v1"
        active_note = ""
        for entry in history:
            if d_str >= entry["start"]:
                active_tag = entry["config_tag"]
                if d_str == entry["start"]: active_note = entry["change_note"]
            else: break
        tags.append(active_tag)
        notes.append(active_note)
    return tags, notes

def calculate_advanced_daily_metrics(day_df, date_label):
    stats = {}
    day_df = day_df.reset_index(drop=True)
    is_active_bool = day_df['is_active'].astype(bool)
    day_df['run_change'] = is_active_bool.astype(int).diff().ne(0)
    day_df['run_id'] = day_df['run_change'].cumsum()
    
    runs = []
    for _, group in day_df[is_active_bool].groupby('run_id'):
        runs.append({'duration': len(group), 'type': "DHW" if group['is_DHW'].any() else "Heating"})
        
    heating_runs = [r for r in runs if r['type'] == 'Heating']
    dhw_runs = [r for r in runs if r['type'] == 'DHW']
    
    stats['Short_Cycles_Count'] = sum(1 for r in heating_runs if r['duration'] < THRESHOLDS['short_cycle_min'])
    stats['Starts_Heating'] = len(heating_runs)
    stats['Starts_DHW'] = len(dhw_runs)
    stats['Cycling_Severity_Index'] = safe_div(stats['Short_Cycles_Count'], stats['Starts_Heating'])
    stats['Median_Run_Time_Mins'] = pd.Series([r['duration'] for r in heating_runs]).median() if heating_runs else 0

    heating_minutes = day_df[is_active_bool & (~day_df['is_DHW'])]
    total_heat_mins = len(heating_minutes)
    
    if total_heat_mins > 0:
        stats['Avg_Compressor_Hz'] = heating_minutes['Freq'].mean()
        stats['Median_Compressor_Hz'] = heating_minutes['Freq'].median()
        stats['Pct_Time_Hz_lt_25'] = (len(heating_minutes[heating_minutes['Freq'] < 25]) / total_heat_mins) * 100
        stats['Pct_Time_Hz_gt_45'] = (len(heating_minutes[heating_minutes['Freq'] > 45]) / total_heat_mins) * 100
        
        def strict_target(outdoor):
            t = 45 - 0.625 * (outdoor + 2)
            return max(22, min(43, t))
        
        targets = heating_minutes['OutdoorTemp'].apply(strict_target)
        is_limited = (targets > (heating_minutes['FlowTemp'] + THRESHOLDS['flow_limit_tolerance']))
        limit_blocks = (is_limited.astype(int).diff().ne(0)).cumsum()
        limit_duration = 0
        limit_events = 0
        for _, g in is_limited.groupby(limit_blocks):
            if g.iloc[0] and len(g) >= THRESHOLDS['flow_limit_min_duration']:
                limit_duration += len(g)
                limit_events += 1
                
        stats['Virtual_FTlim_Events'] = limit_events
        stats['Virtual_FTlim_Time_Mins'] = limit_duration
        stats['Pct_Heating_Time_Flow_Over_43C'] = (len(heating_minutes[heating_minutes['FlowTemp'] > 43]) / total_heat_mins) * 100
    else:
        stats.update({
            'Avg_Compressor_Hz': 0, 'Median_Compressor_Hz': 0,
            'Pct_Time_Hz_lt_25': 0, 'Pct_Time_Hz_gt_45': 0,
            'Virtual_FTlim_Events': 0, 'Virtual_FTlim_Time_Mins': 0,
            'Pct_Heating_Time_Flow_Over_43C': 0
        })

    avg_outdoor = day_df['OutdoorTemp'].mean()
    stats['HDD_18'] = max(0, THRESHOLDS['hdd_base_temp'] - avg_outdoor)
    stats['Buffer_Charge_Time_Mins'] = len(day_df[(day_df['is_active']) & (day_df['Active_Zones_Count'] == 0)])
    stats['Buffer_Discharge_Time_Mins'] = len(day_df[(~day_df['is_active']) & (day_df['Active_Zones_Count'] > 0)])
    return stats

def get_daily_stats(df):
    df['active_min'] = df['is_active'].astype(int)
    df['Run_UFH_Mins'] = np.where((df['is_active']) & (df['Zone_UFH'] > 0), 1, 0)
    df['Run_DS_Mins'] = np.where((df['is_active']) & (df['Zone_DS'] > 0), 1, 0)
    df['Run_US_Mins'] = np.where((df['is_active']) & (df['Zone_US'] > 0), 1, 0)

    def calc_target_flow(row):
        if row['is_DHW']: return 50.0
        t = 45 - 0.625 * (row['OutdoorTemp'] + 2)
        return max(22, min(43, t))

    df['Target_Flow'] = df.apply(calc_target_flow, axis=1)
    df['Target_Flow_Active'] = np.where(df['is_heating'], df['Target_Flow'], np.nan)
    df['Actual_Flow_Active'] = np.where(df['is_heating'], df['FlowTemp'], np.nan)
    df['Interference_Flag'] = df['Interference_Detected']

    immersion_mode = df['Immersion_Mode'] if 'Immersion_Mode' in df.columns else 0
    df['Immersion_Power_W'] = np.where(immersion_mode > 0, df.get('Indoor_Power', 0), 0)
    df['Immersion_Active_Min'] = (immersion_mode > 0).astype(int)
    df['DeltaT_DHW'] = df['DeltaT'].where(df['is_DHW'])
    df['ReturnTemp_DHW'] = df['ReturnTemp'].where(df['is_DHW'])
    df['Freq_DHW'] = df['Freq'].where(df['is_DHW'])

    agg_dict = {
        'Freq': 'count', 'Power_Heating': 'sum', 'Power_DHW': 'sum', 'Heat_Heating': 'sum', 'Heat_DHW': 'sum',
        'Power_Night': 'sum', 'Power_Day': 'sum', 'run_start': 'sum', 'defrost_start': 'sum', 'active_min': 'sum',
        'Run_UFH_Mins': 'sum', 'Run_DS_Mins': 'sum', 'Run_US_Mins': 'sum', 'Cost_Inc': 'sum',
        'Interference_Flag': 'sum', 'is_DHW': 'sum', 'Immersion_Power_W': 'sum', 'Immersion_Active_Min': 'sum',
        'DeltaT_DHW': 'mean', 'ReturnTemp_DHW': 'mean', 'Freq_DHW': 'mean',
    }
    if 'OutdoorTemp' in df.columns: agg_dict['OutdoorTemp'] = ['mean', 'min', 'max']
    if 'Solar_Rad' in df.columns: agg_dict['Solar_Rad'] = 'mean'
    if 'Wind_Speed' in df.columns: agg_dict['Wind_Speed'] = ['mean', 'max']
    if 'Outdoor_Humidity' in df.columns: agg_dict['Outdoor_Humidity'] = 'mean'
    if 'Target_Flow_Active' in df.columns: agg_dict['Target_Flow_Active'] = 'mean'
    if 'Actual_Flow_Active' in df.columns: agg_dict['Actual_Flow_Active'] = 'mean'
    if 'DeltaT_Active' in df.columns: agg_dict['DeltaT_Active'] = 'mean'
    if 'FlowRate_Active' in df.columns: agg_dict['FlowRate_Active'] = 'mean'

    for col in df.columns:
        if col in ENTITY_MAP.values():
            if col not in agg_dict: agg_dict[col] = 'count'
            elif isinstance(agg_dict[col], str) and agg_dict[col] != 'count': agg_dict[col] = [agg_dict[col], 'count']
            elif isinstance(agg_dict[col], list):
                if 'count' not in agg_dict[col]: agg_dict[col].append('count')
    
    for critical in ['Power', 'Heat', 'FlowTemp', 'ReturnTemp', 'FlowRate', 'Indoor_Power', 'OutdoorTemp']:
        if critical in df.columns and critical not in agg_dict: agg_dict[critical] = 'count'
        elif critical in df.columns and isinstance(agg_dict[critical], list):
             if 'count' not in agg_dict[critical]: agg_dict[critical].append('count')
        elif critical in df.columns and isinstance(agg_dict[critical], str) and agg_dict[critical] != 'count':
             agg_dict[critical] = [agg_dict[critical], 'count']

    daily = df.resample('D').agg(agg_dict)
    daily.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily.columns]
    
    daily = daily.rename(columns={
        'DeltaT_DHW_mean': 'Avg_DHW_DeltaT', 'ReturnTemp_DHW_mean': 'Avg_DHW_Return_Temp', 'Freq_DHW_mean': 'Avg_DHW_Compressor_Hz',
        'Freq_count': 'Recorded_Minutes', 'Power_Heating_sum': 'Electricity_Heating_Wmin', 'Power_DHW_sum': 'Electricity_DHW_Wmin',
        'Heat_Heating_sum': 'Heat_Heating_Wmin', 'Heat_DHW_sum': 'Heat_DHW_Wmin', 'Power_Night_sum': 'Electricity_Night_Wmin',
        'Power_Day_sum': 'Electricity_Day_Wmin', 'OutdoorTemp_mean': 'Outdoor_Avg', 'OutdoorTemp_min': 'Outdoor_Min',
        'OutdoorTemp_max': 'Outdoor_Max', 'Solar_Rad_mean': 'Solar_Avg', 'Wind_Speed_mean': 'Wind_Avg', 'Wind_Speed_max': 'Wind_Max',
        'Outdoor_Humidity_mean': 'Humidity_Avg', 'Target_Flow_Active_mean': 'Avg_Target_Flow', 'Actual_Flow_Active_mean': 'Avg_Actual_Flow',
        'DeltaT_Active_mean': 'Avg_DeltaT', 'FlowRate_Active_mean': 'Avg_FlowRate', 'run_start_sum': 'Starts', 'defrost_start_sum': 'Defrosts',
        'active_min_sum': 'Active_Mins', 'Cost_Inc_sum': 'Daily_Cost_Euro', 'Interference_Flag_sum': 'Heating_During_DHW_Mins',
        'is_DHW_sum': 'Total_DHW_Mins', 'FlowTemp_DHW_max': 'Max_DHW_Flow_Temp', 'Immersion_Power_W_sum': 'Immersion_Wh',
        'Immersion_Active_Min_sum': 'Immersion_Mins'
    })

    dq_mapping = {'Power': 'DQ_Power_Count', 'Heat': 'DQ_Heat_Count', 'FlowTemp': 'DQ_FlowTemp_Count', 'ReturnTemp': 'DQ_ReturnTemp_Count', 'FlowRate': 'DQ_FlowRate_Count'}
    for friendly, canonical in dq_mapping.items():
        col_raw = f"{friendly}_count"
        if col_raw in daily.columns: daily = daily.rename(columns={col_raw: canonical})
        alt = f"{friendly}_Count"
        if alt in daily.columns and canonical not in daily.columns: daily = daily.rename(columns={alt: canonical})

    for col in ['Electricity_Heating', 'Electricity_DHW', 'Heat_Heating', 'Heat_DHW', 'Electricity_Night', 'Electricity_Day']:
        daily[f"{col}_kWh"] = daily[f"{col}_Wmin"] / 60000.0
    daily['Immersion_kWh'] = daily.get('Immersion_Wh', 0) / 60000.0
    for col in ['Run_UFH', 'Run_DS', 'Run_US']:
        daily[f"{col}_Hours"] = daily.get(f"{col}_Mins_sum", 0) / 60.0

    daily['Total_Heat_kWh'] = daily['Heat_Heating_kWh'] + daily['Heat_DHW_kWh']
    daily['Total_Electricity_kWh'] = daily['Electricity_Heating_kWh'] + daily['Electricity_DHW_kWh']
    daily['Avg_Run_Time_Mins'] = safe_div(daily['Active_Mins'], daily['Starts'])
    daily['Global_SCOP'] = safe_div(daily['Total_Heat_kWh'], daily['Total_Electricity_kWh'])
    daily['DHW_SCOP'] = np.where(daily['Electricity_DHW_kWh'] > 0.01, daily['Heat_DHW_kWh'] / daily['Electricity_DHW_kWh'], 0)
    daily['DHW_Share_Of_Total_Elec'] = np.where(daily['Total_Electricity_kWh'] > 0, daily['Electricity_DHW_kWh'] / daily['Total_Electricity_kWh'], 0)
    daily['Night_Share_Of_Total_HP_Elec'] = np.where(daily['Total_Electricity_kWh'] > 0, daily['Electricity_Night_kWh'] / daily['Total_Electricity_kWh'], 0)
    daily['Cost_per_kWh_Heat'] = np.where(daily['Total_Heat_kWh'] > 0.1, daily['Daily_Cost_Euro'] / daily['Total_Heat_kWh'], 0)
    daily['Effective_Avg_Tariff'] = np.where(daily['Total_Electricity_kWh'] > 0.1, daily['Daily_Cost_Euro'] / daily['Total_Electricity_kWh'], 0)
    daily['DHW_Heating_Overlap_Share'] = safe_div(daily['Heating_During_DHW_Mins'], daily['Total_DHW_Mins'])
    
    tags, notes = get_config_for_date(daily.index)
    daily["Config_Tag"] = tags
    daily["Config_Change_Note"] = notes

    advanced_list = []
    for date_idx in daily.index:
        mask = (df.index >= date_idx) & (df.index < date_idx + pd.Timedelta(days=1))
        day_slice = df[mask]
        if not day_slice.empty:
            m = calculate_advanced_daily_metrics(day_slice, str(date_idx.date()))
            m['date'] = date_idx
            advanced_list.append(m)
            
    if advanced_list:
        adv_df = pd.DataFrame(advanced_list).set_index('date')
        daily = daily.join(adv_df, how='left')

    daily['Heating_kWh_per_HDD'] = safe_div(daily['Heat_Heating_kWh'], daily.get('HDD_18', 0))

    def assign_flags(row):
        flags = []
        if row.get('Cycling_Severity_Index', 0) > THRESHOLDS['short_cycling_ratio_high']: flags.append("short_cycling_high")
        if row.get('Pct_Heating_Time_Flow_Over_43C', 0) > THRESHOLDS['flow_over_43c_pct_high']: flags.append("flow_temp_too_high")
        if row.get('DHW_SCOP', 0) < THRESHOLDS['dhw_scop_low'] and row.get('Heat_DHW_kWh', 0) > 1: flags.append("dhw_inefficient")
        if row.get('Night_Share_Of_Total_HP_Elec', 0) < THRESHOLDS['night_share_elec_low'] and row.get('HDD_18', 0) > 5: flags.append("night_window_underused")
        if row.get('DHW_Heating_Overlap_Share', 0) > 0.15: flags.append("dhw_heating_interference")
        return flags if flags else ["normal_operation"]
    daily['Health_Flags'] = daily.apply(assign_flags, axis=1)
    
    def assign_dq_tier(row):
        opportunity_window = row.get('Recorded_Minutes', 0)
        if opportunity_window == 0:
            opportunity_window = max(row.get('DQ_Power_Count', 0), row.get('DQ_Heat_Count', 0), row.get('DQ_FlowTemp_Count', 0), 1)
        threshold = 0.90 * opportunity_window
        has_power = row.get('DQ_Power_Count', 0) >= threshold
        has_heat = row.get('DQ_Heat_Count', 0) >= threshold
        flow_ok = row.get('DQ_FlowTemp_Count', 0) >= threshold
        return_ok = row.get('DQ_ReturnTemp_Count', 0) >= threshold
        rate_ok = row.get('DQ_FlowRate_Count', 0) >= threshold
        has_hydraulics = flow_ok and return_ok and rate_ok
        
        if has_power and has_heat: return "Tier 1 (Gold)", 100
        elif has_power and has_hydraulics: return "Tier 2 (Silver)", 90
        elif has_power: return "Tier 3 (Bronze)", 50
        else: return "Critical (Missing Power)", 0

    dq_results = daily.apply(assign_dq_tier, axis=1, result_type='expand')
    daily['DQ_Tier'] = dq_results[0]
    daily['DQ_Score'] = dq_results[1]

    # Return ALL days (do not filter out zero-heat days)
    # This allows Data Quality to show full history even if heat physics failed
    return daily