import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Simple Moving Average (SMA).
    
    Parameters
    ----------
    close : pd.DataFrame
        Wide (date x ticker) close prices.
    window : int
        Rolling window length in trading days.
    
    Returns
    -------
    pd.DataFrame
        SMA values with same shape as input.
    """
    return close.rolling(window).mean()


def ema(close: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Exponential Moving Average (EMA).
    """
    return close.ewm(span=span, adjust=False).mean()


def ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Time-series z-score (per asset).
    
    For each column (ticker), compute:
        z_t = (x_t - mean_{t-window+1..t}) / std_{t-window+1..t}
    """
    mean = df.rolling(window).mean()
    std = df.rolling(window).std(ddof=1)
    return (df - mean) / std


def realized_vol(close: pd.DataFrame, window: int = 20, ann_factor: int = 252) -> pd.DataFrame:
    """
    Realized volatility (annualized).
    
    vol_t = std(returns_{t-window+1..t}) * sqrt(ann_factor)
    """
    rets = close.pct_change()
    vol = rets.rolling(window).std(ddof=1) * np.sqrt(ann_factor)
    return vol


def returns(close: pd.DataFrame, period: int = 1, method: str = "simple") -> pd.DataFrame:
    """
    Compute returns from close prices.
    
    Parameters
    ----------
    close : pd.DataFrame
    period : int
        Return horizon (in days).
    method : {"simple","log"}
    
    Returns
    -------
    pd.DataFrame
        Returns with same shape as input.
    """
    if method == "simple":
        return close.pct_change(periods=period)
    elif method == "log":
        return np.log(close / close.shift(period))
    else:
        raise ValueError("method must be 'simple' or 'log'")

# ---------------------------------------------------------------------
# Research staples
# ---------------------------------------------------------------------

def momentum(close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Cross-asset momentum: price_t / price_{t-lookback} - 1.
    """
    return close / close.shift(lookback) - 1.0


def mean_reversion_spread(close: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    Mean-reversion spread: SMA(short) - SMA(long).
    """
    return sma(close, short) - sma(close, long)


def mean_reversion_z(close: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    Cross-sectional z-score of SMA(short) - SMA(long).
    
    At each date, z-score across tickers (row-wise).
    """
    spread = mean_reversion_spread(close, short, long)
    mu = spread.mean(axis=1, skipna=True)
    sd = spread.std(axis=1, ddof=1, skipna=True).replace(0, np.nan)
    return spread.sub(mu, axis=0).div(sd, axis=0)


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling average dollar volume = (close * volume).
    
    Useful as a liquidity filter.
    """
    dv = close * volume
    return dv.rolling(window).mean()

# ---------------------------------------------------------------------
# Optional utilities
# ---------------------------------------------------------------------

def rolling_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling correlation between two DataFrames (matched shape).
    """
    return x.rolling(window).corr(y)


def rolling_beta(returns_x: pd.DataFrame, returns_mkt: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling beta of each asset's returns vs. a market return series.
    
    Parameters
    ----------
    returns_x : pd.DataFrame
        Wide (date x ticker) returns of assets.
    returns_mkt : pd.Series
        Market returns (date index).
    """
    cov = returns_x.rolling(window).cov(returns_mkt)
    var = returns_mkt.rolling(window).var()
    return cov.div(var, axis=0)
