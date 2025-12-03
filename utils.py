# utils.py
import pandas as pd
import numpy as np

# Explicit exports
__all__ = ["safe_div", "availability_pct"]

# Lightweight logging helpers
LOG_VERBOSE = False

def _log_warn(msg):
    if LOG_VERBOSE:
        print(f"WARNING: {msg}")

def _log_error(msg):
    print(f"ERROR: {msg}")

def _log_info(msg):
    if LOG_VERBOSE:
        print(f"INFO: {msg}")

def safe_div(n, d, default=0.0):
    """Safe division helper handling both Scalars and Series."""
    # 1. Vectorized Path (pandas Series)
    if isinstance(n, pd.Series) or isinstance(d, pd.Series):
        result = n / d
        return result.replace([np.inf, -np.inf], default).fillna(default)
    
    # 2. Scalar Path
    if d == 0 or pd.isna(d):
        return default
    return n / d

def availability_pct(count_series: pd.Series, expected_series: pd.Series) -> pd.Series:
    """
    Compute availability % with guardrails:
      - Only days where expected > 0 and not NaN are scored.
      - Days where expected == 0 or is NaN → result is NaN (N/A).
      - Avoids misleading 0/0 or 100% when the reference window is missing.
    """
    count, expected = count_series.astype(float).align(expected_series.astype(float), join='outer')
    pct = pd.Series(np.nan, index=count.index, dtype=float)
    mask = (expected > 0) & expected.notna() & count.notna()
    if mask.any():
        pct.loc[mask] = (count.loc[mask] / expected.loc[mask]) * 100.0
    return pct.clip(lower=0, upper=100)
