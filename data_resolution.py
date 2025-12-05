"""
data_resolution.py
------------------
Module for adaptive, sensor-specific resolution detection, retention estimation,
and confidence scoring for Heat Pump Analytics.

This module performs:
1. Automatic high-resolution window detection (up to ~30 days)
2. Per-sensor baseline frequency analysis from the high-resolution period
3. Detection of downsampled / degraded data outside the HR window
4. Confidence scoring (B1 model: High → Minimal)
5. Metadata export for processing + UI warning banners
"""

import numpy as np
import pandas as pd
from datetime import timedelta


# ================================================================
# Utility: Compute timestamp deltas safely
# ================================================================
def _compute_intervals(series: pd.Series):
    """Return array of time deltas (in seconds) between consecutive timestamps."""
    if series.empty:
        return np.array([])
    dt = series.diff().dropna()
    if dt.empty:
        return np.array([])
    return dt.dt.total_seconds().values


# ================================================================
# STEP 1 — Detect High-Resolution Window (HR window)
# ================================================================
def detect_high_res_window(df: pd.DataFrame, max_days: int = 30):
    """
    Automatically detect the highest-resolution window within the last N days.
    - Looks for dense timestamp regions
    - Window size dynamically adapts (3–10 days)

    Returns:
        (start_ts, end_ts)
    """
    if df.empty:
        return None, None

    df_sorted = df.sort_values("timestamp")
    end = df_sorted["timestamp"].iloc[-1]
    start_limit = end - timedelta(days=max_days)

    # restrict search
    recent = df_sorted[df_sorted["timestamp"] >= start_limit]
    if recent.empty:
        return None, None

    # Use sliding window sizes: 3 → 10 days
    window_sizes = [3, 5, 7, 10]
    best_score = -1
    best_range = (None, None)

    timestamps = recent["timestamp"].values

    for win in window_sizes:
        win_delta = np.timedelta64(win, "D")
        # 2-pointer method
        left = 0
        for right in range(len(timestamps)):
            while timestamps[right] - timestamps[left] > win_delta:
                left += 1
            count = right - left + 1
            if count <= 2:
                continue

            subset = recent.iloc[left : right + 1]
            intervals = _compute_intervals(subset["timestamp"])
            if len(intervals) == 0:
                continue

            density = count / win   # entries per day
            median_dt = np.median(intervals)
            score = density / median_dt  # higher density + lower dt = better resolution

            if score > best_score:
                best_score = score
                best_range = (subset["timestamp"].iloc[0], subset["timestamp"].iloc[-1])

    return best_range


# ================================================================
# STEP 2 — Determine per-sensor baseline frequency (from HR window)
# ================================================================
def compute_baseline_intervals(df: pd.DataFrame, hr_start, hr_end):
    """
    For each sensor column, compute:
    - baseline median interval (seconds)
    - classification: periodic_high / periodic_med / periodic_low /
                      event_based / hourly_native / irregular
    """
    if hr_start is None or hr_end is None:
        return {}

    baseline = {}
    hr_df = df[(df["timestamp"] >= hr_start) & (df["timestamp"] <= hr_end)]

    # Only continuous sensors (exclude non-value columns)
    value_cols = [c for c in df.columns if c not in ["timestamp", "entity_id"]]

    for col in value_cols:
        series = hr_df[hr_df[col].notna()][["timestamp", col]]
        if series.empty:
            baseline[col] = {
                "baseline_interval": None,
                "baseline_type": "no_data",
            }
            continue

        # Compute median dt
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            baseline[col] = {
                "baseline_interval": None,
                "baseline_type": "event_based",
            }
            continue

        m_dt = np.median(intervals)

        # Classify baseline type
        if m_dt <= 20:
            btype = "periodic_high"
        elif m_dt <= 90:
            btype = "periodic_medium"
        elif m_dt <= 300:
            btype = "periodic_low"
        elif abs(m_dt - 3600) < 120:  # aware of OWM hourly-native sensors
            btype = "hourly_native"
        else:
            # Could be event-based or irregular
            if len(series[col].unique()) <= 6:
                btype = "event_based"
            else:
                btype = "irregular"

        baseline[col] = {
            "baseline_interval": float(m_dt),
            "baseline_type": btype,
        }

    return baseline


# ================================================================
# STEP 3 — Compute degradation outside HR window
# ================================================================
def classify_resolution_vs_baseline(full_df, baseline):
    """
    For each sensor, compare full data intervals vs baseline intervals.
    Produces:
        resolution_status[col] = {baseline_interval, observed_interval, confidence, degraded}
    """

    resolution_map = {}
    value_cols = [c for c in full_df.columns if c not in ["timestamp", "entity_id"]]

    for col in value_cols:
        base_info = baseline.get(col, None)

        if base_info is None or base_info["baseline_interval"] is None:
            # No usable baseline → unknown
            resolution_map[col] = {
                "baseline_interval": None,
                "observed_interval": None,
                "confidence": "unknown",
                "degraded": False,
            }
            continue

        base_dt = base_info["baseline_interval"]
        base_type = base_info["baseline_type"]

        series = full_df[full_df[col].notna()][["timestamp", col]]
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            resolution_map[col] = {
                "baseline_interval": base_dt,
                "observed_interval": None,
                "confidence": "unknown",
                "degraded": False,
            }
            continue

        obs_dt = float(np.median(intervals))

        # B1 — Confidence scoring
        ratio = obs_dt / base_dt if base_dt > 0 else 999

        if ratio <= 1.5:
            conf = "high"
        elif ratio <= 3:
            conf = "medium"
        elif ratio <= 8:
            conf = "low"
        elif ratio <= 20:
            conf = "very_low"
        else:
            conf = "minimal"

        degraded = conf in ["low", "very_low", "minimal"]

        # If hourly but baseline is much faster → minimal
        if obs_dt >= 3000 and base_dt < 300:
            conf = "minimal"
            degraded = True

        resolution_map[col] = {
            "baseline_interval": base_dt,
            "observed_interval": obs_dt,
            "confidence": conf,
            "degraded": degraded,
        }

    return resolution_map


# ================================================================
# STEP 4 — Estimate retention window
# ================================================================
def estimate_retention_days(df):
    """
    Estimate how long high-resolution data is kept before downsampling.
    Looks for the earliest point where dt jumps significantly for multiple sensors.
    """
    if df.empty:
        return None

    df_sorted = df.sort_values("timestamp")
    ts = df_sorted["timestamp"]
    oldest = ts.iloc[0]
    newest = ts.iloc[-1]

    total_days = (newest - oldest).total_seconds() / 86400
    if total_days < 2:
        return total_days

    # Look for dt jumps across multiple sensors
    change_points = []

    for col in df.columns:
        if col in ["timestamp", "entity_id"]:
            continue
        series = df[df[col].notna()][["timestamp", col]]
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            continue
        # look for big jumps (≥ 1 hr)
        idx = np.where(intervals >= 3600)[0]
        if len(idx) > 0:
            change_points.append(series["timestamp"].iloc[idx[0]])

    if change_points:
        cutoff = min(change_points)
        return (newest - cutoff).total_seconds() / 86400

    return total_days


# ================================================================
# STEP 5 — Confidence for Heat & COP (B1 Propagation)
# ================================================================
def compute_global_confidence(res_map):
    """
    Determine global heat & COP confidence:
        heat_conf = min(FT_conf, RT_conf, Power_conf)
        cop_conf  = min(heat_conf, Power_conf)
    """

    def _get(col):
        if col not in res_map:
            return "unknown"
        return res_map[col]["confidence"]

    flow = _get("FlowTemp")
    ret = _get("ReturnTemp")
    power = _get("Power")

    # Simple priority order
    order = ["minimal", "very_low", "low", "medium", "high"]
    def score(c):
        if c not in order:
            return -1
        return order.index(c)

    heat_score = min(score(flow), score(ret), score(power))
    heat_conf = order[heat_score] if heat_score >= 0 else "unknown"

    cop_score = min(heat_score, score(power))
    cop_conf = order[cop_score] if cop_score >= 0 else "unknown"

    return {
        "heat_confidence": heat_conf,
        "cop_confidence": cop_conf,
    }


# ================================================================
# Main entry point used by data_loader
# ================================================================
def analyze_resolution(df: pd.DataFrame):
    """
    Unified API for:
        HR window detection
        baseline calculation
        resolution comparison
        retention estimation
        global confidence scoring

    Returns:
        {
          "hr_start": ...,
          "hr_end": ...,
          "baseline": {...},
          "resolution_map": {...},
          "retention_days": float,
          "global_confidence": {...}
        }
    """
    hr_start, hr_end = detect_high_res_window(df)

    baseline = compute_baseline_intervals(df, hr_start, hr_end)

    resolution_map = classify_resolution_vs_baseline(df, baseline)

    retention_days = estimate_retention_days(df)

    global_conf = compute_global_confidence(resolution_map)

    return {
        "hr_start": hr_start,
        "hr_end": hr_end,
        "baseline": baseline,
        "resolution_map": resolution_map,
        "retention_days": retention_days,
        "global_confidence": global_conf,
    }
