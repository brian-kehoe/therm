# data_loader.py
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from config import ENTITY_MAP, SENSOR_ROLES, BASELINE_JSON_PATH
from baselines import build_sensor_baselines, analyze_sensor_reporting_patterns, smart_forward_fill, load_saved_heartbeat_baseline
from utils import _log_warn

# --- FALLBACK MAPPING ---
FALLBACK_OWM_MAP = {
    'sensor.openweathermap_wind_speed': 'Wind_Speed_OWM',
    'sensor.openweathermap_humidity': 'Outdoor_Humidity_OWM',
    'sensor.openweathermap_uv_index': 'UV_Index_OWM',
    'sensor.openweathermap_temperature': 'OutdoorTemp_OWM'
}

def _parse_grafana_timestamp(filename):
    match = re.search(r'-(\d{2}_\d{2}_\d{4} \d{2}_\d{2}_\d{2})\.csv$', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%d_%m_%Y %H_%M_%S')
        except ValueError:
            return None
    return None

def find_grafana_pairs(uploaded_files):
    numeric_files = {}
    state_files = {}
    for file_obj in uploaded_files:
        filename = file_obj['fileName']
        ts = _parse_grafana_timestamp(filename)
        if ts is None: continue
        indexed = {'name': filename, 'timestamp': ts, 'handle': file_obj['handle']}
        if "Numeric-data" in filename: numeric_files[ts] = indexed
        elif "State-data" in filename: state_files[ts] = indexed

    paired_files = []
    unmatched_files = [f for f in uploaded_files if _parse_grafana_timestamp(f['fileName']) is None]
    used_state_timestamps = set()
    for num_ts, num_file in numeric_files.items():
        match = None
        for state_ts, state_file in state_files.items():
            if state_file['timestamp'] in used_state_timestamps: continue
            if abs(num_ts - state_ts) <= timedelta(minutes=2):
                match = state_file
                break
        if match:
            paired_files.append((num_file['name'], match['name']))
            used_state_timestamps.add(match['timestamp'])
        else:
            unmatched_files.append({'fileName': num_file['name'], 'handle': num_file['handle']})
    for state_ts, state_file in state_files.items():
        if state_file['timestamp'] not in used_state_timestamps:
            unmatched_files.append({'fileName': state_file['name'], 'handle': state_file['handle']})
    return paired_files, unmatched_files

def _normalize_df_columns(df):
    cleaned_cols = []
    for c in df.columns:
        c = str(c).strip().strip('"').strip("'").replace('\r', '').replace('\n', '').replace('\ufeff', '')
        cleaned_cols.append(c)
    df.columns = cleaned_cols
    if 'Time' in df.columns: df = df.rename(columns={'Time': 'last_changed'})
    df.columns = [c.lower() for c in df.columns]
    if 'last_changed' not in df.columns:
        if 'time' in df.columns: df = df.rename(columns={'time': 'last_changed'})
        elif 'timestamp' in df.columns: df = df.rename(columns={'timestamp': 'last_changed'})
        elif len(df.columns) > 0: df = df.rename(columns={df.columns[0]: 'last_changed'})
    if 'state' not in df.columns and 'value' in df.columns:
        df = df.rename(columns={'value': 'state'})
    return df

def process_grafana_pair(numeric_file_handle, state_file_handle):
    numeric_file_handle.seek(0)
    df_num = pd.read_csv(numeric_file_handle)
    df_num = _normalize_df_columns(df_num)
    df_num = df_num.filter(items=['last_changed', 'state', 'entity_id'])

    state_file_handle.seek(0)
    df_state = pd.read_csv(state_file_handle)
    df_state = _normalize_df_columns(df_state)
    
    if 'state' in df_state.columns and 'value' in df_state.columns:
        df_state['state'] = df_state['state'].astype(str).replace('nan', np.nan).fillna(df_state['value'])
    elif 'value' in df_state.columns:
        df_state = df_state.rename(columns={'value': 'state'})
        
    df_state = df_state.filter(items=['last_changed', 'state', 'entity_id'])
    full_df = pd.concat([df_num, df_state], ignore_index=True)
    full_df['entity_id'] = full_df['entity_id'].astype(str).str.lower()
    return full_df

def load_and_clean_data(uploaded_files, progress_cb=None):
    uploaded_file_index = [{'fileName': f.name, 'handle': f} for f in uploaded_files]
    if progress_cb: progress_cb("Pairing files...", 10)
    paired_files, unmatched_files = find_grafana_pairs(uploaded_file_index)
    all_dfs = []
    total_files = len(paired_files) * 2 + len(unmatched_files)
    processed = 0

    # 1. Load Paired Grafana Files
    for num_file_name, state_file_name in paired_files:
        try:
            num_h = next(f['handle'] for f in uploaded_file_index if f['fileName'] == num_file_name)
            state_h = next(f['handle'] for f in uploaded_file_index if f['fileName'] == state_file_name)
            all_dfs.append(process_grafana_pair(num_h, state_h))
            processed += 2
            if progress_cb:
                pct = 10 + int((processed / max(total_files, 1)) * 25)
                progress_cb(f"Loading paired files ({processed}/{total_files})...", pct)
        except Exception: pass

    # 2. Load Unmatched Files (Includes OWM)
    for file_obj in unmatched_files:
        try:
            file_obj['handle'].seek(0)
            df = pd.read_csv(file_obj['handle'])
            df = _normalize_df_columns(df)
            
            # Basic check for Long Format (Home Assistant Export)
            if 'last_changed' in df.columns and 'state' in df.columns:
                if 'entity_id' in df.columns:
                    df['entity_id'] = df['entity_id'].astype(str).str.lower()
                    all_dfs.append(df.filter(items=['last_changed', 'state', 'entity_id']))
            processed += 1
            if progress_cb:
                pct = 10 + int((processed / max(total_files, 1)) * 25)
                progress_cb(f"Loading unmatched files ({processed}/{total_files})...", pct)
        except Exception: pass

    if not all_dfs: return None
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('last_changed')
    
    if progress_cb: progress_cb("Mapping entity IDs (Vectorized)...", 40)

    # 3. Entity Normalization (HIGH PERFORMANCE)
    ACTIVE_MAP = {**FALLBACK_OWM_MAP, **ENTITY_MAP}
    valid_keys = set(k.strip().lower() for k in ACTIVE_MAP.keys())
    
    # Get unique IDs only (small list, e.g., 50 items)
    unique_ids = full_df['entity_id'].unique()
    id_map = {}
    
    # Build map only for the unique items
    for raw_id in unique_ids:
        s_id = str(raw_id).strip().lower()
        
        # A. Direct Match
        if s_id in ACTIVE_MAP:
            id_map[raw_id] = s_id
            continue
            
        # B. Prefix Match
        found_prefix = False
        prefixes = ['sensor.', 'binary_sensor.', 'switch.']
        for prefix in prefixes:
            if prefix + s_id in ACTIVE_MAP:
                id_map[raw_id] = prefix + s_id
                found_prefix = True
                break
        
        # C. No Match? Keep Raw ID
        if not found_prefix:
            id_map[raw_id] = s_id

    # Apply mapping using fast vectorized replacement
    # This is much faster than .apply() or .map() for large N with small K categories
    full_df['entity_id'] = full_df['entity_id'].replace(id_map)

    # Identify Unmapped Entities
    final_unique = set(full_df['entity_id'].unique())
    unmapped_entities = sorted(list(final_unique - valid_keys))
    
    if not full_df.empty:
        # Robust Date Parsing
        full_df['last_changed'] = pd.to_datetime(full_df['last_changed'], errors='coerce', dayfirst=True, format='mixed', utc=True)
        full_df['last_changed'] = full_df['last_changed'].dt.tz_localize(None)
        full_df = full_df.dropna(subset=['last_changed'])

    # 4. Binary/Numeric Cleaning
    binary_mask = full_df['entity_id'].str.startswith('binary_') | full_df['entity_id'].str.contains('defrost')
    def clean_binary(x):
        s = str(x).lower()
        if s == 'on': return 1
        if s == 'off': return 0
        try: return float(x)
        except: return 0
    full_df.loc[binary_mask, 'state'] = full_df.loc[binary_mask, 'state'].apply(clean_binary)
    if full_df.empty: return None

    raw_history_df = full_df[['last_changed', 'entity_id']].copy()

    # 5. Load Baselines
    try:
        if not full_df.empty:
            month_values = full_df["last_changed"].dt.month.dropna()
            current_month = int(month_values.mode()[0]) if len(month_values) > 0 else datetime.now().month
        else:
            current_month = datetime.now().month
        saved_baselines, source_path = load_saved_heartbeat_baseline(BASELINE_JSON_PATH, current_month=current_month)
    except Exception as exc:
        _log_warn(f"Failed to load seasonal baseline: {exc}")
        saved_baselines, source_path = {}, None

    if progress_cb: progress_cb("Resampling & filling gaps...", 70)
    
    # Pivot (Creates Wide Format)
    df_pivot = full_df.pivot_table(index='last_changed', columns='entity_id', values='state', aggfunc='last')
    
    # 6. Resample
    df_res = df_pivot.resample('1min').last()

    runtime_baselines = build_sensor_baselines(full_df, SENSOR_ROLES)
    baselines = {**saved_baselines, **runtime_baselines}
    
    sensor_patterns = analyze_sensor_reporting_patterns(df_pivot, baselines=baselines)
    
    # Smart Forward Fill
    df_res = smart_forward_fill(df_res, sensor_patterns)
    
    # 7. Rename Entities
    clean_map = {k.strip().lower(): v for k, v in ACTIVE_MAP.items()}
    df_res = df_res.rename(columns=clean_map)

    # 8. Post-Processing & Types
    if 'Immersion_Mode' in df_res.columns:
        df_res['Immersion_Mode'] = df_res['Immersion_Mode'].apply(lambda x: 1 if str(x).lower() == 'on' else 0)
    if 'Quiet_Mode' in df_res.columns:
        df_res['Quiet_Mode'] = df_res['Quiet_Mode'].apply(lambda x: 1 if str(x).lower() == 'on' else 0)
    
    if 'DHW_Mode' in df_res.columns:
        def _norm_dhw_mode(val):
            if pd.isna(val): return np.nan
            s = str(val).strip().lower()
            if s in ("economic", "eco"): return "Economic"
            if s in ("standard", "std"): return "Standard"
            if s in ("power", "boost", "forced"): return "Power"
            return np.nan
        df_res['DHW_Mode'] = df_res['DHW_Mode'].apply(_norm_dhw_mode)

    # Final robust numeric conversion
    for col in [c for c in df_res.columns if c not in ['ValveMode', 'DHW_Mode']]:
        try:
            df_res[col] = pd.to_numeric(df_res[col], errors='coerce')
        except Exception: pass
        
    return {
        "df": df_res,
        "raw_history": raw_history_df,
        "baselines": baselines,
        "baseline_path": source_path,
        "patterns": sensor_patterns,
        "unmapped_entities": unmapped_entities
    }