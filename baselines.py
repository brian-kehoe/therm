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
        
        good_min = best_minutes * 0.9
        good_days_mask = daily_stats >= good_min
        if good_days_mask.sum() < min_good_days:
            good_min = best_minutes * 0.7 
            good_days_mask = daily_stats >= good_min
            
        deltas = g['last_changed'].sort_values().diff().dt.total_seconds().dropna()
        if len(deltas) > 0:
             clean_deltas = deltas[deltas < deltas.quantile(0.95)]
             baseline_interval = float(clean_deltas.median()) if len(clean_deltas) > 0 else 60.0
             baseline_gap_95 = float(clean_deltas.quantile(0.95)) if len(clean_deltas) > 0 else 300.0
        else:
             baseline_interval = 60.0
             baseline_gap_95 = 300.0

        baselines[entity_id] = {
            'role': role,
            'has_baseline': True,
            'baseline_interval_sec': baseline_interval,
            'baseline_gap_95_sec': baseline_gap_95,
            'expected_minutes': float(best_minutes),
            'good_day_count': int(good_days_mask.sum())
        }
    return baselines

def build_offline_aware_seasonal_baseline(history_df, sensor_roles, min_coverage_fraction=0.25, min_valid_days=10, min_month_days=2):
    baselines = {}
    if history_df is None or history_df.empty: return baselines
    if "last_changed" not in history_df.columns or "entity_id" not in history_df.columns: return baselines

    df = history_df.copy()
    df["last_changed"] = pd.to_datetime(df["last_changed"], errors="coerce")
    df = df.dropna(subset=["last_changed"])
    df["entity_id"] = df["entity_id"].astype(str).str.lower()
    df["date"] = df["last_changed"].dt.date
    df["month"] = df["last_changed"].dt.month

    for entity_id, g in df.groupby("entity_id"):
        role = sensor_roles.get(entity_id, "unknown")
        if role == "rare_event":
            baselines[entity_id] = {"role": role, "has_baseline": False}
            continue

        g = g.sort_values("last_changed")
        if len(g) < 10: continue

        daily_counts = g.groupby("date")["last_changed"].agg(["count", "min", "max"]).rename(columns={"count": "samples"})
        if daily_counts["samples"].max() <= 0: continue
        first_seen_date = daily_counts.index.min()
        daily_counts = daily_counts[daily_counts["samples"] > 0]
        if daily_counts.empty: continue

        best_samples = daily_counts["samples"].max()
        min_samples_for_valid = best_samples * float(min_coverage_fraction)
        valid_days_idx = daily_counts[daily_counts["samples"] >= min_samples_for_valid].index
        if len(valid_days_idx) < min_valid_days:
            valid_days_idx = daily_counts.index
        if len(valid_days_idx) == 0: continue

        g["date"] = g["last_changed"].dt.date
        g["prev_date"] = g["date"].shift()
        g["delta_sec"] = g["last_changed"].diff().dt.total_seconds()

        mask_valid_delta = (g["date"] == g["prev_date"]) & (g["date"].isin(valid_days_idx))
        deltas_valid = g.loc[mask_valid_delta, "delta_sec"].dropna()
        if deltas_valid.empty:
            naive_deltas = g["last_changed"].diff().dt.total_seconds().dropna()
            if naive_deltas.empty: continue
            deltas_valid = naive_deltas

        clean_deltas_global = deltas_valid[deltas_valid < deltas_valid.quantile(0.99)]
        if clean_deltas_global.empty: clean_deltas_global = deltas_valid

        baseline_interval_sec = float(np.median(clean_deltas_global))
        baseline_gap_95_sec = float(clean_deltas_global.quantile(0.95))
        expected_minutes_global = float((best_samples * baseline_interval_sec) / 60.0)

        seasonal_months = {}
        valid_day_to_month = {d: d.month for d in valid_days_idx}

        for month_val in range(1, 13):
            month_days = [d for d in valid_days_idx if valid_day_to_month[d] == month_val]
            if len(month_days) < min_month_days: continue

            mask_month_delta = mask_valid_delta & g["date"].isin(month_days)
            deltas_month = g.loc[mask_month_delta, "delta_sec"].dropna()
            if len(deltas_month) < 10: continue

            clean_month = deltas_month[deltas_month < deltas_month.quantile(0.99)]
            if clean_month.empty: clean_month = deltas_month

            m_interval = float(np.median(clean_month))
            m_gap_95 = float(clean_month.quantile(0.95))
            m_best_samples = float(daily_counts.loc[month_days, "samples"].max())
            m_expected_minutes = float((m_best_samples * m_interval) / 60.0)

            seasonal_months[f"{month_val:02d}"] = {
                "baseline_interval_sec": m_interval,
                "baseline_gap_95_sec": m_gap_95,
                "expected_minutes": m_expected_minutes,
                "valid_day_count": int(len(month_days)),
            }

        baselines[entity_id] = {
            "role": role,
            "has_baseline": True,
            "baseline_interval_sec": baseline_interval_sec,
            "baseline_gap_95_sec": baseline_gap_95_sec,
            "expected_minutes": expected_minutes_global,
            "valid_day_count": int(len(valid_days_idx)),
            "first_seen_date": str(first_seen_date),
            "seasonal_months": seasonal_months,
        }
    return baselines

def save_heartbeat_baseline_to_json(baselines, path=None, days_in_history=None):
    if not baselines: raise ValueError("No baselines to save.")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    days_part = f"{int(days_in_history)}d" if days_in_history is not None else "NA"
    if path is None:
        filename = f"sensor_heartbeat_baseline_seasonal_{days_part}_{ts}.json"
        path = filename

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Offline-aware seasonal heartbeat baselines built from long-run history.",
            "calc_version": CALC_VERSION,
            "days_in_history": days_in_history,
        },
        "sensors": baselines,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

def load_saved_heartbeat_baseline(path=BASELINE_JSON_PATH, current_month=None):
    data = None
    source_path = None
    candidate_paths = []
    if path: candidate_paths.append(path)
    if not os.path.exists(path):
        matches = glob.glob("sensor_heartbeat_baseline_seasonal_*.json")
        matches = sorted(matches, key=lambda p: os.path.getmtime(p), reverse=True)
        candidate_paths.extend(matches)

    for p in candidate_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                source_path = p
                break
        except Exception as exc:
            _log_warn(f"Failed to load baseline JSON {p}: {exc}")
            continue

    if data is None: return {}, None

    sensors = data.get("sensors", data)
    result = {}
    month_key = f"{current_month:02d}" if current_month is not None else None

    for entity_id, entry in sensors.items():
        role = entry.get("role", "unknown")
        base_interval = entry.get("baseline_interval_sec")
        base_gap = entry.get("baseline_gap_95_sec")
        expected_minutes = entry.get("expected_minutes")
        seasonal = entry.get("seasonal_months", {})

        if month_key and seasonal and month_key in seasonal:
            m = seasonal[month_key]
            base_interval = m.get("baseline_interval_sec", base_interval)
            base_gap = m.get("baseline_gap_95_sec", base_gap)
            expected_minutes = m.get("expected_minutes", expected_minutes)

        if base_interval is None or base_gap is None: continue

        result[entity_id] = {
            "role": role,
            "has_baseline": entry.get("has_baseline", True),
            "baseline_interval_sec": float(base_interval),
            "baseline_gap_95_sec": float(base_gap),
            "expected_minutes": expected_minutes,
        }
    return result, source_path

def analyze_sensor_reporting_patterns(raw_df, time_col='last_changed', entity_col='entity_id', baselines=None):
    patterns = {}
    if time_col not in raw_df.columns or entity_col not in raw_df.columns: return patterns

    for entity_id, g in raw_df.groupby(entity_col):
        g = g.sort_values(time_col)
        if len(g) < 5: continue
        deltas = g[time_col].diff().dt.total_seconds().dropna()
        if len(deltas) == 0: continue
        median_interval = float(deltas.median())
        q95_gap = float(deltas.max())
        
        base = (baselines or {}).get(entity_id, {})
        role = base.get('role', SENSOR_ROLES.get(entity_id, 'unknown'))
        baseline_interval = base.get('baseline_interval_sec', median_interval)
        baseline_gap_95 = base.get('baseline_gap_95_sec', q95_gap)
        
        normal_interval = np.median([median_interval, baseline_interval])
        gap_95 = max(q95_gap, baseline_gap_95)

        if role in ('core_periodic', 'room_temp', 'unknown'):
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
    if df_resampled.empty: return df_resampled
    df_filled = df_resampled.copy()
    for col in df_filled.columns:
        pat = patterns.get(col)
        if pat is None:
            limit_minutes = 5
        else:
            limit_minutes = max(1, int(pat['gap_threshold_sec'] / 60.0))
            if pat['report_type'] == 'periodic':
                limit_minutes = min(limit_minutes, 60)
            elif pat['report_type'] == 'on_change':
                limit_minutes = min(limit_minutes, 180)
        df_filled[col] = df_filled[col].ffill(limit=limit_minutes)
    return df_filled