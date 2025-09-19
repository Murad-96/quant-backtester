# src/qbt/signals.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Iterable

# ---------- Core cross-sectional transforms ----------

def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score per date (row-wise): (x - mean_t) / std_t.
    NaN-safe, returns NaN where std==0.
    """
    mu = df.mean(axis=1, skipna=True)
    sd = df.std(axis=1, ddof=1, skipna=True).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)

def cs_rank(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """
    Cross-sectional rank per date. pct=True gives [0,1] percentile ranks.
    """
    return df.rank(axis=1, pct=pct)

def cs_winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Cross-sectional winsorization per date at given quantiles (e.g., 1%/99%).
    """
    q_low = df.quantile(lower, axis=1, interpolation="linear")
    q_high = df.quantile(upper, axis=1, interpolation="linear")
    return df.clip(lower=q_low, upper=q_high, axis=0)

def clip_abs(df: pd.DataFrame, max_abs: float = 3.0) -> pd.DataFrame:
    """
    Symmetric hard clip at +/- max_abs.
    """
    return df.clip(lower=-max_abs, upper= max_abs)

# ---------- Neutralization (optional but very useful) ----------

def neutralize_by_group(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Demean signal within each group per date. 'groups' indexes columns (tickers) -> group label.
    Example groups: sectors or industries.
    """
    # Ensure group labels align to columns
    groups = groups.reindex(df.columns)
    out = df.copy()
    for g, cols in groups.groupby(groups).groups.items():
        sub = df.loc[:, cols]
        out.loc[:, cols] = sub.sub(sub.mean(axis=1, skipna=True), axis=0)
    return out

def neutralize_linear(df: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """
    Linear neutralization via single-factor OLS per date:
    residual = y - beta_hat * x, where x = exposure (e.g., market beta, size).
    Both df and exposure are (date x asset).
    """
    y, x = df.align(exposure, join="inner", axis=0)
    out = y.copy()
    # For each date: beta = cov(y,x)/var(x); residual = y - beta*x
    var_x = x.var(axis=1, ddof=1)
    cov_yx = (y * x).mean(axis=1) - y.mean(axis=1) * x.mean(axis=1)
    beta = cov_yx / var_x.replace(0, np.nan)
    out = y.sub(beta.values[:, None] * x.values, axis=0)
    return out

# ---------- Construction helpers ----------

def make_signal(
    feature: pd.DataFrame,
    standardize: str = "zscore",           # "zscore" | "rank" | "none"
    winsor: Optional[tuple[float, float]] = (0.01, 0.99),
    clip: Optional[float] = 3.0,
    group_labels: Optional[pd.Series] = None,
    neutralize_to: Optional[pd.DataFrame] = None,
    lag: int = 1                            # shift forward by 1 to trade next bar
) -> pd.DataFrame:
    """
    Turn a raw feature into a tradable signal:
      1) winsorize (optional)
      2) standardize cross-sectionally (zscore or rank)
      3) group-demean (optional)
      4) linear neutralize to an exposure (optional)
      5) clip (optional)
      6) lag to avoid look-ahead (default: 1 day)

    Returns: signal DataFrame aligned to feature (date x asset).
    """
    s = feature.copy()

    if winsor is not None:
        s = cs_winsorize(s, lower=winsor[0], upper=winsor[1])

    if standardize == "zscore":
        s = cs_zscore(s)
    elif standardize == "rank":
        s = cs_rank(s, pct=True) - 0.5   # center around 0
    elif standardize == "none":
        pass
    else:
        raise ValueError("standardize must be 'zscore' | 'rank' | 'none'")

    if group_labels is not None:
        s = neutralize_by_group(s, group_labels)

    if neutralize_to is not None:
        s = neutralize_linear(s, neutralize_to)

    if clip is not None:
        s = clip_abs(s, max_abs=clip)

    if lag:
        s = s.shift(lag)

    return s

def combine_signals(
    components: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    post_standardize: str = "zscore",      # apply a final standardization
    post_clip: Optional[float] = 3.0
) -> pd.DataFrame:
    """
    Combine multiple pre-made signals into one:
      combined = sum_i w_i * S_i (default equal weights),
      then (optionally) cross-sectionally standardize and clip.
    """
    if not components:
        raise ValueError("No components provided")

    # Align all signals
    aligned = list(components.values())
    idx = aligned[0].index
    cols = aligned[0].columns
    for df in aligned[1:]:
        df.index.equals(idx) and df.columns.equals(cols) or (_ for _ in ()).throw(
            ValueError("All signal components must share the same index/columns")
        )

    if weights is None:
        w = {k: 1.0 / len(components) for k in components}
    else:
        w = weights

    combined = sum(components[k] * float(w.get(k, 0.0)) for k in components)

    if post_standardize == "zscore":
        combined = cs_zscore(combined)
    elif post_standardize == "rank":
        combined = cs_rank(combined, pct=True) - 0.5
    elif post_standardize == "none":
        pass
    else:
        raise ValueError("post_standardize must be 'zscore' | 'rank' | 'none'")

    if post_clip is not None:
        combined = clip_abs(combined, max_abs=post_clip)

    return combined

def lag_signal(signal: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Shift the signal forward in time to represent trade-at-next-bar logic.
    """
    return signal.shift(periods)

def mask_signal(signal: pd.DataFrame, valid: pd.DataFrame | pd.Series | np.ndarray) -> pd.DataFrame:
    """
    Elementwise mask to zero-out assets that are not tradable/valid on a given date.
    E.g., illiquid names, IPOs without min history, etc.
    """
    return signal.where(valid, other=np.nan)
