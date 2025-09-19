"""
qbt.metrics
===========
Performance and risk metrics for quantitative backtesting.

This module provides comprehensive performance analytics including:
- Return calculations (total, annualized, rolling)
- Drawdown analysis (maximum, average, duration)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Turnover and trading analytics
- Risk metrics (volatility, VaR, beta)

Canonical data format: Time series (pd.Series) for returns, prices, and portfolio values.
Wide DataFrames (date × ticker) for multi-asset analytics.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
import warnings


# ---------------------------- #
#      RETURN CALCULATIONS     #
# ---------------------------- #

def calculate_returns(
    prices_or_values: pd.Series,
    method: str = "simple",
    periods: int = 1,
    fill_method: Optional[str] = "ffill"
) -> pd.Series:
    """
    Calculate returns from price or portfolio value series.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    method : {"simple", "log"}, default "simple"
        Return calculation method:
        - "simple": (P_t / P_{t-1}) - 1
        - "log": ln(P_t / P_{t-1})
    periods : int, default 1
        Number of periods for return calculation (1 = daily, 21 = monthly, etc.)
    fill_method : str, optional, default "ffill"
        Method to fill NaN values before calculation
        
    Returns
    -------
    pd.Series
        Time series of returns
        
    Examples
    --------
    >>> prices = pd.Series([100, 101, 99, 102], index=pd.date_range('2023-01-01', periods=4))
    >>> returns = calculate_returns(prices)
    >>> print(returns)
    """
    # Handle NaN values
    if fill_method is not None:
        clean_series = prices_or_values.fillna(method=fill_method)
    else:
        clean_series = prices_or_values.copy()
    
    if method == "simple":
        returns = clean_series.pct_change(periods=periods)
    elif method == "log":
        returns = np.log(clean_series / clean_series.shift(periods))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns


def total_return(
    prices_or_values: pd.Series,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None
) -> float:
    """
    Calculate total return over a period.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    start_date : str or pd.Timestamp, optional
        Start date for calculation. If None, uses first available date.
    end_date : str or pd.Timestamp, optional
        End date for calculation. If None, uses last available date.
        
    Returns
    -------
    float
        Total return as decimal (e.g., 0.15 = 15% return)
    """
    # Filter by date range if specified
    if start_date is not None or end_date is not None:
        mask = pd.Series(True, index=prices_or_values.index)
        if start_date is not None:
            mask &= (prices_or_values.index >= pd.to_datetime(start_date))
        if end_date is not None:
            mask &= (prices_or_values.index <= pd.to_datetime(end_date))
        series = prices_or_values.loc[mask]
    else:
        series = prices_or_values
    
    if len(series) < 2:
        return 0.0
    
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    
    if start_value <= 0:
        warnings.warn("Start value is zero or negative, cannot calculate return")
        return np.nan
    
    return (end_value / start_value) - 1.0


def annualized_return(
    prices_or_values: pd.Series,
    periods_per_year: int = 252,
    method: str = "compound"
) -> float:
    """
    Calculate annualized return.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    periods_per_year : int, default 252
        Number of periods per year (252 for daily, 12 for monthly, etc.)
    method : {"compound", "simple"}, default "compound"
        Annualization method:
        - "compound": (final/initial)^(periods_per_year/total_periods) - 1
        - "simple": total_return * periods_per_year / total_periods
        
    Returns
    -------
    float
        Annualized return as decimal
    """
    total_ret = total_return(prices_or_values)
    
    if pd.isna(total_ret):
        return np.nan
    
    total_periods = len(prices_or_values) - 1
    
    if total_periods <= 0:
        return 0.0
    
    if method == "compound":
        return (1 + total_ret) ** (periods_per_year / total_periods) - 1
    elif method == "simple":
        return total_ret * periods_per_year / total_periods
    else:
        raise ValueError("method must be 'compound' or 'simple'")


def rolling_returns(
    prices_or_values: pd.Series,
    window: int = 21,
    min_periods: Optional[int] = None,
    annualized: bool = False,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling returns over a specified window.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    window : int, default 21
        Rolling window size in periods
    min_periods : int, optional
        Minimum number of observations required. If None, uses window size.
    annualized : bool, default False
        Whether to annualize the rolling returns
    periods_per_year : int, default 252
        Periods per year for annualization
        
    Returns
    -------
    pd.Series
        Rolling returns
    """
    if min_periods is None:
        min_periods = window
    
    # Calculate rolling total returns
    rolling_rets = prices_or_values.rolling(
        window=window, 
        min_periods=min_periods
    ).apply(lambda x: total_return(x), raw=False)
    
    if annualized:
        rolling_rets = (1 + rolling_rets) ** (periods_per_year / window) - 1
    
    return rolling_rets


# ---------------------------- #
#     DRAWDOWN CALCULATIONS    #
# ---------------------------- #

def calculate_drawdowns(
    prices_or_values: pd.Series,
    method: str = "percent"
) -> pd.Series:
    """
    Calculate drawdowns from peak values.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    method : {"percent", "dollar"}, default "percent"
        Drawdown calculation method:
        - "percent": (current - peak) / peak
        - "dollar": current - peak
        
    Returns
    -------
    pd.Series
        Time series of drawdowns (negative values indicate drawdown)
    """
    cumulative_max = prices_or_values.expanding().max()
    
    if method == "percent":
        drawdowns = (prices_or_values - cumulative_max) / cumulative_max
    elif method == "dollar":
        drawdowns = prices_or_values - cumulative_max
    else:
        raise ValueError("method must be 'percent' or 'dollar'")
    
    return drawdowns


def max_drawdown(prices_or_values: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
        
    Returns
    -------
    float
        Maximum drawdown as negative decimal (e.g., -0.15 = -15% drawdown)
    """
    drawdowns = calculate_drawdowns(prices_or_values, method="percent")
    return drawdowns.min()


def drawdown_statistics(
    prices_or_values: pd.Series
) -> Dict[str, float]:
    """
    Calculate comprehensive drawdown statistics.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
        
    Returns
    -------
    dict
        Dictionary containing:
        - max_drawdown: Maximum drawdown
        - avg_drawdown: Average drawdown when in drawdown
        - drawdown_duration_avg: Average drawdown duration in periods
        - drawdown_duration_max: Maximum drawdown duration in periods
        - recovery_time: Time to recover from max drawdown
        - current_drawdown: Current drawdown level
    """
    drawdowns = calculate_drawdowns(prices_or_values, method="percent")
    
    # Basic statistics
    max_dd = drawdowns.min()
    
    # Only consider periods with actual drawdown
    in_drawdown = drawdowns < 0
    avg_dd = drawdowns[in_drawdown].mean() if in_drawdown.any() else 0.0
    
    # Drawdown duration analysis
    dd_periods = (drawdowns < 0).astype(int)
    dd_groups = (dd_periods != dd_periods.shift()).cumsum()
    dd_durations = dd_periods.groupby(dd_groups).sum()
    dd_durations = dd_durations[dd_durations > 0]  # Only actual drawdown periods
    
    avg_duration = dd_durations.mean() if len(dd_durations) > 0 else 0.0
    max_duration = dd_durations.max() if len(dd_durations) > 0 else 0.0
    
    # Recovery time for maximum drawdown
    max_dd_idx = drawdowns.idxmin()
    recovery_idx = drawdowns[max_dd_idx:].ge(0).idxmax() if max_dd_idx in drawdowns.index else None
    
    if recovery_idx and recovery_idx != max_dd_idx:
        recovery_periods = (drawdowns.index.get_loc(recovery_idx) - 
                          drawdowns.index.get_loc(max_dd_idx))
    else:
        recovery_periods = np.nan
    
    # Current drawdown
    current_dd = drawdowns.iloc[-1]
    
    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "drawdown_duration_avg": avg_duration,
        "drawdown_duration_max": max_duration,
        "recovery_time": recovery_periods,
        "current_drawdown": current_dd
    }


# ---------------------------- #
#    RISK-ADJUSTED METRICS     #
# ---------------------------- #

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Time series of returns
    risk_free_rate : float, default 0.0
        Risk-free rate (annualized)
    periods_per_year : int, default 252
        Number of periods per year for annualization
        
    Returns
    -------
    float
        Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate annualized metrics
    mean_excess = excess_returns.mean() * periods_per_year
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    if volatility == 0:
        return np.nan if mean_excess == 0 else np.inf * np.sign(mean_excess)
    
    return mean_excess / volatility


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation version of Sharpe).
    
    Parameters
    ----------
    returns : pd.Series
        Time series of returns
    risk_free_rate : float, default 0.0
        Risk-free rate (annualized)
    periods_per_year : int, default 252
        Number of periods per year for annualization
        
    Returns
    -------
    float
        Sortino ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation (only negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    
    # Calculate annualized excess return
    mean_excess = excess_returns.mean() * periods_per_year
    
    if downside_deviation == 0:
        return np.nan if mean_excess == 0 else np.inf * np.sign(mean_excess)
    
    return mean_excess / downside_deviation


def calmar_ratio(
    prices_or_values: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of prices or portfolio values
    periods_per_year : int, default 252
        Number of periods per year for annualization
        
    Returns
    -------
    float
        Calmar ratio
    """
    ann_return = annualized_return(prices_or_values, periods_per_year)
    max_dd = abs(max_drawdown(prices_or_values))
    
    if max_dd == 0:
        return np.nan if ann_return == 0 else np.inf * np.sign(ann_return)
    
    return ann_return / max_dd


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio (active return / tracking error).
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int, default 252
        Number of periods per year for annualization
        
    Returns
    -------
    float
        Information ratio
    """
    # Align series
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
    
    if len(aligned_returns) == 0:
        return np.nan
    
    # Calculate active returns
    active_returns = aligned_returns - aligned_benchmark
    
    # Calculate tracking error (annualized standard deviation of active returns)
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    # Calculate annualized active return
    active_return = active_returns.mean() * periods_per_year
    
    if tracking_error == 0:
        return np.nan if active_return == 0 else np.inf * np.sign(active_return)
    
    return active_return / tracking_error


# ---------------------------- #
#     TURNOVER CALCULATIONS    #
# ---------------------------- #

def calculate_turnover(
    positions: pd.DataFrame,
    method: str = "sum_abs_changes"
) -> pd.Series:
    """
    Calculate portfolio turnover from position weights.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Wide format (date × ticker) position weights
    method : {"sum_abs_changes", "average_abs_changes"}, default "sum_abs_changes"
        Turnover calculation method:
        - "sum_abs_changes": Sum of absolute changes in weights
        - "average_abs_changes": Average absolute change per position
        
    Returns
    -------
    pd.Series
        Daily turnover values
    """
    # Calculate position changes
    position_changes = positions.diff()
    
    if method == "sum_abs_changes":
        turnover = position_changes.abs().sum(axis=1)
    elif method == "average_abs_changes":
        # Average of absolute changes for active positions
        num_active = (positions != 0).sum(axis=1)
        turnover = position_changes.abs().sum(axis=1) / num_active.replace(0, np.nan)
    else:
        raise ValueError("method must be 'sum_abs_changes' or 'average_abs_changes'")
    
    return turnover


def turnover_statistics(
    positions: pd.DataFrame,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive turnover statistics.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Wide format (date × ticker) position weights
    periods_per_year : int, default 252
        Number of periods per year for annualization
        
    Returns
    -------
    dict
        Dictionary containing turnover statistics:
        - daily_turnover_mean: Average daily turnover
        - daily_turnover_std: Standard deviation of daily turnover
        - annual_turnover: Annualized turnover
        - turnover_per_trade: Average turnover per rebalancing event
    """
    daily_turnover = calculate_turnover(positions)
    
    # Remove first day (NaN from diff)
    daily_turnover = daily_turnover.dropna()
    
    if len(daily_turnover) == 0:
        return {
            "daily_turnover_mean": 0.0,
            "daily_turnover_std": 0.0,
            "annual_turnover": 0.0,
            "turnover_per_trade": 0.0
        }
    
    # Basic statistics
    mean_daily = daily_turnover.mean()
    std_daily = daily_turnover.std()
    
    # Annualized turnover
    annual_turnover = mean_daily * periods_per_year
    
    # Turnover per trade (only on days with actual trading)
    trading_days = daily_turnover[daily_turnover > 1e-6]  # Filter out tiny changes
    avg_turnover_per_trade = trading_days.mean() if len(trading_days) > 0 else 0.0
    
    return {
        "daily_turnover_mean": mean_daily,
        "daily_turnover_std": std_daily,
        "annual_turnover": annual_turnover,
        "turnover_per_trade": avg_turnover_per_trade
    }


def calculate_trades_from_positions(
    positions: pd.DataFrame,
    prices: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate trade amounts and trade values from position changes.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Wide format (date × ticker) position weights
    prices : pd.DataFrame
        Wide format (date × ticker) prices
        
    Returns
    -------
    tuple
        (trade_weights, trade_values)
        trade_weights : Position weight changes per date/ticker
        trade_values : Total trade value per date (sum of absolute trades)
    """
    # Align inputs
    positions, prices = positions.align(prices, join='inner')
    
    # Calculate weight changes
    trade_weights = positions.diff()
    
    # Calculate dollar trade values (assuming unit portfolio value)
    # This gives relative trade values
    trade_values = trade_weights.abs().sum(axis=1)
    
    return trade_weights, trade_values


# ---------------------------- #
#     COMPREHENSIVE METRICS   #
# ---------------------------- #

def performance_metrics(
    prices_or_values: pd.Series,
    returns: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    prices_or_values : pd.Series
        Time series of portfolio values or prices
    returns : pd.Series, optional
        Pre-calculated returns. If None, calculated from prices_or_values.
    benchmark_returns : pd.Series, optional
        Benchmark returns for relative metrics
    risk_free_rate : float, default 0.0
        Risk-free rate for Sharpe/Sortino calculations
    periods_per_year : int, default 252
        Number of periods per year
        
    Returns
    -------
    dict
        Comprehensive performance metrics including:
        - Return metrics (total, annualized, volatility)
        - Risk metrics (Sharpe, Sortino, Calmar ratios)
        - Drawdown statistics
        - Relative performance (if benchmark provided)
    """
    # Calculate returns if not provided
    if returns is None:
        returns = calculate_returns(prices_or_values)
    
    # Basic return metrics
    total_ret = total_return(prices_or_values)
    ann_return = annualized_return(prices_or_values, periods_per_year)
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calmar_ratio(prices_or_values, periods_per_year)
    
    # Drawdown metrics
    dd_stats = drawdown_statistics(prices_or_values)
    
    # Basic statistics
    best_day = returns.max()
    worst_day = returns.min()
    positive_days = (returns > 0).sum()
    negative_days = (returns < 0).sum()
    win_rate = positive_days / (positive_days + negative_days) if len(returns) > 0 else 0.0
    
    # Compile results
    metrics = {
        # Return metrics
        "total_return": total_ret,
        "annualized_return": ann_return,
        "volatility": volatility,
        
        # Risk-adjusted metrics
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        
        # Drawdown metrics
        "max_drawdown": dd_stats["max_drawdown"],
        "avg_drawdown": dd_stats["avg_drawdown"],
        "current_drawdown": dd_stats["current_drawdown"],
        "max_drawdown_duration": dd_stats["drawdown_duration_max"],
        "recovery_time": dd_stats["recovery_time"],
        
        # Distribution metrics
        "best_day": best_day,
        "worst_day": worst_day,
        "win_rate": win_rate,
        "skewness": returns.skew() if len(returns) > 0 else np.nan,
        "kurtosis": returns.kurtosis() if len(returns) > 0 else np.nan,
        
        # Period information
        "start_date": prices_or_values.index[0],
        "end_date": prices_or_values.index[-1],
        "total_periods": len(prices_or_values),
        "years": len(prices_or_values) / periods_per_year
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        info_ratio = information_ratio(returns, benchmark_returns, periods_per_year)
        
        # Calculate beta
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        if len(aligned_returns) > 1:
            beta = np.cov(aligned_returns, aligned_benchmark)[0, 1] / np.var(aligned_benchmark)
            alpha = (aligned_returns.mean() - beta * aligned_benchmark.mean()) * periods_per_year
        else:
            beta = np.nan
            alpha = np.nan
        
        metrics.update({
            "information_ratio": info_ratio,
            "beta": beta,
            "alpha": alpha
        })
    
    return metrics


def portfolio_metrics(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio metrics including turnover.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Wide format (date × ticker) position weights
    prices : pd.DataFrame
        Wide format (date × ticker) prices
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    risk_free_rate : float, default 0.0
        Risk-free rate
    periods_per_year : int, default 252
        Periods per year
        
    Returns
    -------
    dict
        Complete portfolio performance and turnover metrics
    """
    # Align inputs
    positions, prices = positions.align(prices, join='inner')
    
    # Calculate portfolio returns
    returns = (positions.shift(1) * prices.pct_change()).sum(axis=1).dropna()
    portfolio_values = (1 + returns).cumprod()
    
    # Performance metrics
    perf_metrics = performance_metrics(
        portfolio_values, 
        returns, 
        benchmark_returns, 
        risk_free_rate, 
        periods_per_year
    )
    
    # Turnover metrics
    turnover_metrics = turnover_statistics(positions, periods_per_year)
    
    # Position statistics
    num_positions_avg = (positions != 0).sum(axis=1).mean()
    gross_leverage_avg = positions.abs().sum(axis=1).mean()
    net_leverage_avg = positions.sum(axis=1).mean()
    
    # Combine all metrics
    all_metrics = {
        **perf_metrics,
        **turnover_metrics,
        "avg_num_positions": num_positions_avg,
        "avg_gross_leverage": gross_leverage_avg,
        "avg_net_leverage": net_leverage_avg
    }
    
    return all_metrics
