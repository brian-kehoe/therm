# baselines.py
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timezone
from utils import _log_warn
from config import BASELINE_JSON_PATH, SENSOR_ROLES, CALC_VERSION

def build_sensor_baselines(history_df, sensor_roles, min_good_days=3):
    """
    Scans history to determine the 'normal' reporting frequency (samples/day)
    for each sensor.
    """
    baselines = {}
    if 'entity_id' not in history_df.columns or 'last_changed' not in history_df.columns:
        return baselines
        
    df_temp = history_df.copy()
    df_temp['date'] = df_temp['last_changed'].dt.date
    
    for entity_id, g in df_temp.groupby('entity_id'):
        role = sensor_roles.get(entity_id, 'unknown')
        if role == 'rare_event':
            baselines[entity_id] = {'role': role, 'has_baseline': False}
            continue
        if len(g) == 0: continue
        
        daily_stats = g.groupby('date')['last_changed'].count()
        best_minutes = daily_stats.max()
        if not np.isfinite(best_minutes) or best_minutes <= 0: continue
        
        # --- LOGIC UPDATE: Sparse Sensor Handling ---
        if role == 'weather_sparse':
            # For hourly sensors (OWM), even 12 samples (50%) is "good"
            good_min = max(1, best_minutes * 0.5)
        else:
            # For standard sensors, we expect 90% consistency
            good_min = best_minutes * 0.9
            
        good_days_mask = daily_stats >= good_min
        if good_days_mask.sum() < min_good_days:
            # Not enough consistent days to form a baseline
            baselines[entity_id] = {'role': role, 'has_baseline': False}
            continue
            
        # Calculate expected minutes based on the mode of "good days"
        # If multiple modes exist, take the max (safest bet for 'full' data)
        modes = daily_stats[good_days_mask].mode()
        if not modes.empty:
            expected_minutes = int(modes.max())
        else:
            expected_minutes = int(daily_stats[good_days_mask].mean())
            
        baselines[entity_id] = {
            'role': role,
            'expected_minutes': expected_minutes,
            'has_baseline': True
        }
        
    return baselines

def analyze_sensor_reporting_patterns(df, baselines=None):
    """
    Analyzes the current dataset to determine reporting intervals (gap thresholds)
    for filling data.
    """
    patterns = {}
    if df.empty: return patterns
    
    for entity_id in df.columns:
        # Drop NaNs to find actual update timestamps
        g = df[entity_id].dropna().index
        if len(g) < 2:
            patterns[entity_id] = {
                'normal_interval_sec': 60,
                'report_type': 'unknown',
                'gap_threshold_sec': 300
            }
            continue
            
        # Calculate intervals between updates
        diffs = g.to_series().diff().dt.total_seconds().dropna()
        if len(diffs) == 0: continue
        
        median_interval = diffs.median()
        q95_gap = diffs.quantile(0.95)
        
        # Check against baseline if available
        base = baselines.get(entity_id, {}) if baselines else {}
        role = base.get('role', SENSOR_ROLES.get(entity_id, 'unknown'))
        
        # Determine expected interval from baseline (1440 mins / expected_counts)
        baseline_interval = 60.0
        if base.get('expected_minutes', 0) > 0:
            baseline_interval = (24 * 60 * 60) / base['expected_minutes']
            
        # Smart fusion of observed vs baseline interval
        normal_interval = np.mean([median_interval, baseline_interval])
        # Use the larger of observed Q95 or implied baseline gap to prevent over-segmentation
        baseline_gap_95 = baseline_interval * 2.0 
        gap_95 = max(q95_gap, baseline_gap_95)

        # --- LOGIC UPDATE: Sparse Role Handling ---
        if role == 'weather_sparse':
            report_type = 'sparse'
            # Allow huge gaps (up to 2 hours) for OWM
            gap_threshold = max(normal_interval * 2.0, gap_95, 7200.0)
        elif role in ('core_periodic', 'room_temp', 'unknown'):
            report_type = 'periodic'
            gap_threshold = max(normal_interval * 6.0, gap_95, 600.0) 
        elif role == 'binary_state':
            report_type = 'on_change'
            gap_threshold = max(normal_interval * 4.0, gap_95, 1800.0)
        else:
            report_type = 'rare_event'
            gap_threshold = 3600.0

        patterns[entity_id] = {
            'normal_interval_sec': normal_interval,
            'report_type': report_type,
            'gap_threshold_sec': gap_threshold,
            'sample_count': int(len(g)),
            'role': role,
            'baseline_expected_minutes': base.get('expected_minutes')
        }
    return patterns

def smart_forward_fill(df_resampled, patterns):
    """
    Fills NaN gaps in the resampled DataFrame based on the calculated gap thresholds.
    """
    if df_resampled.empty: return df_resampled
    df_filled = df_resampled.copy()
    
    for col in df_filled.columns:
        pat = patterns.get(col)
        
        # Determine filling limit
        if pat is None:
            limit_minutes = 5
        else:
            # Convert gap threshold (seconds) to minutes for ffill limit
            gap_sec = pat.get('gap_threshold_sec', 300)
            limit_minutes = int(np.ceil(gap_sec / 60.0))

            # --- LOGIC UPDATE: Smart limits based on sensor reporting patterns ---
            if pat.get('report_type') == 'sparse':
                # OWM sensors: cap at 2 hours (120 mins)
                limit_minutes = min(limit_minutes, 120)
            elif pat.get('report_type') == 'periodic':
                # Core sensors: cap at 20 mins to avoid inventing data
                limit_minutes = min(limit_minutes, 20)
            elif pat.get('report_type') in ('on_change', 'rare_event'):
                # Binary state / rare event sensors: Trust the learned pattern
                # These report when state CHANGES, not periodically
                # Defrost might stay "off" for months, zones "off" for 20+ hours
                # Use the calculated gap threshold from actual data (95th percentile)
                # No arbitrary cap - the pattern analysis already provides safe limits
                pass  # Use limit_minutes as calculated from gap_threshold_sec
        
        # Apply forward fill with limit
        df_filled[col] = df_filled[col].ffill(limit=limit_minutes)

        # For binary state sensors: backfill leading NaNs with 0 (default "off" state)
        # These sensors report on state CHANGE - absence means default state
        if pat and pat.get('report_type') in ('on_change', 'rare_event'):
            # Only fill leading NaNs (before first real value)
            first_valid_idx = df_filled[col].first_valid_index()
            if first_valid_idx is not None:
                df_filled.loc[:first_valid_idx, col] = df_filled[col].loc[:first_valid_idx].fillna(0)

    return df_filled

def load_saved_heartbeat_baseline(json_path, current_month=None):
    """
    Loads the baseline JSON. If seasonal (list), picks the best match for the month.
    """
    if not os.path.exists(json_path):
        return {}, None
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 1. Simple dict format (Legacy)
        if isinstance(data, dict) and 'meta' not in data and 'baselines' not in data:
            return data, json_path
            
        # 2. Wrapped format (Standard)
        if 'baselines' in data and isinstance(data['baselines'], dict):
            return data['baselines'], json_path
            
        # 3. Seasonal List format (Advanced)
        if isinstance(data, list):
            # Find entry for current month
            if current_month is None: current_month = datetime.now().month
            
            best_entry = None
            for entry in data:
                if 'months' in entry and current_month in entry['months']:
                    best_entry = entry
                    break
            
            # Fallback to 'default' tag if no month match
            if not best_entry:
                for entry in data:
                    if entry.get('tag') == 'default':
                        best_entry = entry
                        break
            
            if best_entry:
                return best_entry.get('baselines', {}), f"{json_path} [{best_entry.get('tag')}]"
                
        return {}, json_path
        
    except Exception as e:
        _log_warn(f"Error loading baseline {json_path}: {e}")
        return {}, None

def build_offline_aware_seasonal_baseline(history_df, sensor_roles):
    """
    Wrapper to build baselines that handles offline days intelligently.
    """
    if history_df.empty: return {}
    
    # Simple check for "system active" days to filter out complete outages
    daily_counts = history_df.groupby(history_df['last_changed'].dt.date).size()
    busy_threshold = daily_counts.median() * 0.1
    active_days = daily_counts[daily_counts > busy_threshold].index
    
    # Filter history to only active days
    history_clean = history_df[history_df['last_changed'].dt.date.isin(active_days)]
    
    return build_sensor_baselines(history_clean, sensor_roles)

def save_heartbeat_baseline_to_json(baselines, tag, days_in_history=0):
    """
    Saves the baseline dictionary to a JSON file.
    """
    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calc_version": CALC_VERSION,
            "days_analyzed": int(days_in_history),
            "tag": tag or "manual_generation"
        },
        "baselines": baselines
    }
    
    with open(BASELINE_JSON_PATH, 'w') as f:
        json.dump(output, f, indent=2)
        
    return BASELINE_JSON_PATH