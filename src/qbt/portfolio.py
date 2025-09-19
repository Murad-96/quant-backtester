"""
qbt.portfolio
=============
Portfolio construction utilities including position sizing, transaction costs, 
and leverage management for the quant backtesting pipeline.

This module provides tools for:
- Position sizing (equal weight, volatility-adjusted, signal-weighted)
- Transaction cost modeling (linear, market impact, bid-ask spreads)
- Leverage management and risk controls
- Portfolio rebalancing utilities

Canonical data format: Wide DataFrames (date × ticker) for prices, signals, and positions.
"""

from __future__ import annotations
from typing import Optional, Literal, Dict, Tuple
import numpy as np
import pandas as pd
import warnings


# ---------------------------- #
#        POSITION SIZING       #
# ---------------------------- #

def equal_weight_positions(
    signals: pd.DataFrame,
    *,
    leverage: float = 1.0,
    max_positions: Optional[int] = None,
    min_signal: Optional[float] = None
) -> pd.DataFrame:
    """
    Equal-weight position sizing based on signals.
    
    Parameters
    ----------
    signals : pd.DataFrame
        Wide format (date × ticker) signals. Non-zero values indicate desired exposure.
    leverage : float, default 1.0
        Target gross leverage (sum of absolute weights).
    max_positions : int, optional
        Maximum number of positions per date. Uses top absolute signals if specified.
    min_signal : float, optional
        Minimum absolute signal threshold for position entry.
        
    Returns
    -------
    pd.DataFrame
        Position weights (date × ticker). Weights sum to target leverage.
        
    Examples
    --------
    >>> signals = pd.DataFrame({'A': [1.0, -0.5, 0.0], 'B': [0.8, 1.2, -0.3]})
    >>> equal_weight_positions(signals, leverage=1.0)
    """
    positions = signals.copy()
    
    # Apply minimum signal threshold
    if min_signal is not None:
        positions = positions.where(np.abs(positions) >= min_signal, 0.0)
    
    # Convert to binary signals (maintain sign)
    positions = np.sign(positions)
    
    # Apply max positions constraint per date
    if max_positions is not None:
        for date in positions.index:
            row = positions.loc[date]
            abs_signals = np.abs(signals.loc[date])
            
            # Get top N positions by absolute signal strength
            if (row != 0).sum() > max_positions:
                top_tickers = abs_signals.nlargest(max_positions).index
                mask = pd.Series(False, index=row.index)
                mask[top_tickers] = True
                positions.loc[date] = positions.loc[date].where(mask, 0.0)
    
    # Scale to target leverage per date
    for date in positions.index:
        row = positions.loc[date]
        gross_exposure = np.abs(row).sum()
        if gross_exposure > 0:
            positions.loc[date] = row * (leverage / gross_exposure)
    
    return positions


def volatility_adjusted_positions(
    signals: pd.DataFrame,
    volatilities: pd.DataFrame,
    *,
    leverage: float = 1.0,
    target_vol: Optional[float] = None,
    max_weight: float = 0.10,
    min_signal: Optional[float] = None
) -> pd.DataFrame:
    """
    Volatility-adjusted position sizing (inverse volatility weighting).
    
    Parameters
    ----------
    signals : pd.DataFrame
        Wide format (date × ticker) signals.
    volatilities : pd.DataFrame
        Wide format (date × ticker) realized volatilities (annualized).
    leverage : float, default 1.0
        Target gross leverage.
    target_vol : float, optional
        Target portfolio volatility. If specified, scales overall leverage.
    max_weight : float, default 0.10
        Maximum absolute weight per asset.
    min_signal : float, optional
        Minimum absolute signal threshold.
        
    Returns
    -------
    pd.DataFrame
        Volatility-adjusted position weights.
    """
    # Align inputs
    signals, vols = signals.align(volatilities, join='inner')
    
    # Apply signal threshold
    if min_signal is not None:
        signals = signals.where(np.abs(signals) >= min_signal, 0.0)
    
    # Compute inverse volatility weights
    # Use median vol to handle NaN/zero volatilities
    med_vol = vols.median(axis=1)
    inv_vol = 1.0 / vols.where(vols > 0, med_vol.values[:, None])
    
    # Raw positions = signal * inverse volatility
    raw_positions = signals * inv_vol
    
    # Apply maximum weight constraint
    positions = raw_positions.clip(lower=-max_weight, upper=max_weight)
    
    # Scale to target leverage per date
    for date in positions.index:
        row = positions.loc[date]
        gross_exposure = np.abs(row).sum()
        if gross_exposure > 0:
            positions.loc[date] = row * (leverage / gross_exposure)
    
    # Optional: scale for target portfolio volatility
    if target_vol is not None:
        portfolio_vol = estimate_portfolio_volatility(positions, vols)
        vol_scalar = target_vol / portfolio_vol.replace(0, np.nan)
        positions = positions.multiply(vol_scalar, axis=0)
    
    return positions


def signal_weighted_positions(
    signals: pd.DataFrame,
    *,
    leverage: float = 1.0,
    normalize_method: Literal["sum", "norm", "max"] = "sum",
    max_weight: float = 0.20,
    min_signal: Optional[float] = None
) -> pd.DataFrame:
    """
    Signal-weighted position sizing (weights proportional to signal strength).
    
    Parameters
    ----------
    signals : pd.DataFrame
        Wide format (date × ticker) signals.
    leverage : float, default 1.0
        Target gross leverage.
    normalize_method : {"sum", "norm", "max"}, default "sum"
        How to normalize signal weights:
        - "sum": divide by sum of absolute signals
        - "norm": divide by L2 norm of signals  
        - "max": divide by maximum absolute signal
    max_weight : float, default 0.20
        Maximum absolute weight per asset.
    min_signal : float, optional
        Minimum absolute signal threshold.
        
    Returns
    -------
    pd.DataFrame
        Signal-weighted position weights.
    """
    positions = signals.copy()
    
    # Apply signal threshold
    if min_signal is not None:
        positions = positions.where(np.abs(positions) >= min_signal, 0.0)
    
    # Normalize weights per date
    for date in positions.index:
        row = positions.loc[date]
        
        if normalize_method == "sum":
            normalizer = np.abs(row).sum()
        elif normalize_method == "norm":
            normalizer = np.linalg.norm(row)
        elif normalize_method == "max":
            normalizer = np.abs(row).max()
        else:
            raise ValueError("normalize_method must be 'sum', 'norm', or 'max'")
        
        if normalizer > 0:
            positions.loc[date] = row / normalizer
    
    # Apply maximum weight constraint
    positions = positions.clip(lower=-max_weight, upper=max_weight)
    
    # Rescale to target leverage
    for date in positions.index:
        row = positions.loc[date]
        gross_exposure = np.abs(row).sum()
        if gross_exposure > 0:
            positions.loc[date] = row * (leverage / gross_exposure)
    
    return positions


def risk_parity_positions(
    signals: pd.DataFrame,
    volatilities: pd.DataFrame,
    correlations: Optional[pd.DataFrame] = None,
    *,
    leverage: float = 1.0,
    max_weight: float = 0.15,
    min_signal: Optional[float] = None
) -> pd.DataFrame:
    """
    Risk parity position sizing (equal risk contribution).
    
    Simplified implementation assuming equal risk contribution target.
    For more sophisticated optimization, consider using cvxpy or similar.
    
    Parameters
    ----------
    signals : pd.DataFrame
        Wide format (date × ticker) signals.
    volatilities : pd.DataFrame
        Wide format (date × ticker) volatilities.
    correlations : pd.DataFrame, optional
        Asset correlation matrix. If None, assumes zero correlations.
    leverage : float, default 1.0
        Target gross leverage.
    max_weight : float, default 0.15
        Maximum absolute weight per asset.
    min_signal : float, optional
        Minimum absolute signal threshold.
        
    Returns
    -------
    pd.DataFrame
        Risk parity weighted positions.
    """
    # Align inputs
    signals, vols = signals.align(volatilities, join='inner')
    
    # Apply signal threshold
    if min_signal is not None:
        signals = signals.where(np.abs(signals) >= min_signal, 0.0)
    
    # Simple risk parity: weight ∝ 1/vol (ignoring correlations for simplicity)
    inv_vol = 1.0 / vols.where(vols > 0, vols.median(axis=1).values[:, None])
    
    # Apply signal direction
    risk_weights = np.sign(signals) * inv_vol
    
    # Normalize to equal risk contribution (approximate)
    for date in risk_weights.index:
        row = risk_weights.loc[date]
        active_mask = row != 0
        if active_mask.sum() > 0:
            # Equal risk budget per active position
            n_active = active_mask.sum()
            risk_budget_per_asset = 1.0 / n_active
            
            # Approximate weight: risk_budget / volatility
            vol_row = vols.loc[date]
            target_weights = np.sign(row) * risk_budget_per_asset / vol_row
            risk_weights.loc[date] = target_weights.where(active_mask, 0.0)
    
    # Apply weight constraints
    positions = risk_weights.clip(lower=-max_weight, upper=max_weight)
    
    # Scale to target leverage
    for date in positions.index:
        row = positions.loc[date]
        gross_exposure = np.abs(row).sum()
        if gross_exposure > 0:
            positions.loc[date] = row * (leverage / gross_exposure)
    
    return positions


# ---------------------------- #
#      TRANSACTION COSTS       #
# ---------------------------- #

def linear_transaction_costs(
    trades: pd.DataFrame,
    cost_rate: float = 0.001,
    prices: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Linear transaction cost model: cost = |trade_value| * cost_rate.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Wide format (date × ticker) trade amounts (in shares or dollars).
    cost_rate : float, default 0.001
        Linear cost rate (e.g., 0.001 = 10 basis points).
    prices : pd.DataFrame, optional
        Prices for computing dollar trade values. If None, assumes trades are in dollars.
        
    Returns
    -------
    pd.DataFrame
        Transaction costs per asset per date.
    """
    if prices is not None:
        # Convert share trades to dollar trades
        dollar_trades = trades * prices
    else:
        dollar_trades = trades
    
    costs = np.abs(dollar_trades) * cost_rate
    return costs


def market_impact_costs(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    temporary_impact: float = 0.1,
    permanent_impact: float = 0.05,
    participation_rate: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Market impact cost model based on trade size relative to volume.
    
    Cost = (temporary + permanent) * impact_factor
    Impact factor = (trade_size / daily_volume) ^ 0.5
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade amounts in shares.
    prices : pd.DataFrame
        Prices for computing dollar impact.
    volumes : pd.DataFrame
        Daily trading volumes.
    temporary_impact : float, default 0.1
        Temporary impact coefficient (in basis points per sqrt(participation)).
    permanent_impact : float, default 0.05
        Permanent impact coefficient.
    participation_rate : pd.DataFrame, optional
        Custom participation rates. If None, computed from trades/volumes.
        
    Returns
    -------
    pd.DataFrame
        Market impact costs per asset per date.
    """
    # Align all inputs
    trades, prices, volumes = trades.align(prices, join='inner')[0], \
                             trades.align(prices, join='inner')[1], \
                             trades.align(volumes, join='inner')[1]
    
    # Compute participation rate
    if participation_rate is None:
        participation = np.abs(trades) / volumes.where(volumes > 0, np.inf)
        participation = participation.clip(upper=0.5)  # Cap at 50% of volume
    else:
        participation = participation_rate
    
    # Impact factor (square root law)
    impact_factor = np.sqrt(participation)
    
    # Total impact rate (in basis points)
    total_impact_bps = (temporary_impact + permanent_impact) * impact_factor
    
    # Convert to dollar costs
    dollar_trades = np.abs(trades) * prices
    costs = dollar_trades * (total_impact_bps / 10000)  # Convert bps to decimal
    
    return costs


def bid_ask_spread_costs(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    spreads: pd.DataFrame
) -> pd.DataFrame:
    """
    Bid-ask spread transaction costs.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade amounts in shares.
    prices : pd.DataFrame
        Mid prices.
    spreads : pd.DataFrame
        Bid-ask spreads (as fraction of mid price).
        
    Returns
    -------
    pd.DataFrame
        Bid-ask spread costs.
    """
    # Cost = 0.5 * spread * |trade_value|
    dollar_trades = np.abs(trades) * prices
    costs = 0.5 * spreads * dollar_trades
    return costs


def total_transaction_costs(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    spreads: Optional[pd.DataFrame] = None,
    *,
    linear_rate: float = 0.0005,
    impact_temp: float = 0.1,
    impact_perm: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute total transaction costs combining multiple cost components.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade amounts in shares.
    prices : pd.DataFrame
        Asset prices.
    volumes : pd.DataFrame, optional
        Trading volumes for market impact calculation.
    spreads : pd.DataFrame, optional
        Bid-ask spreads for spread cost calculation.
    linear_rate : float, default 0.0005
        Linear cost rate (5 basis points).
    impact_temp : float, default 0.1
        Temporary market impact coefficient.
    impact_perm : float, default 0.05
        Permanent market impact coefficient.
        
    Returns
    -------
    tuple
        (total_costs, cost_breakdown)
        total_costs : Total transaction costs per asset per date
        cost_breakdown : Dictionary of individual cost components
    """
    cost_components = {}
    
    # Linear costs (always included)
    cost_components['linear'] = linear_transaction_costs(trades, linear_rate, prices)
    
    # Market impact costs (if volume data available)
    if volumes is not None:
        cost_components['market_impact'] = market_impact_costs(
            trades, prices, volumes,
            temporary_impact=impact_temp,
            permanent_impact=impact_perm
        )
    
    # Spread costs (if spread data available)
    if spreads is not None:
        cost_components['bid_ask'] = bid_ask_spread_costs(trades, prices, spreads)
    
    # Sum all components
    total_costs = sum(cost_components.values())
    
    return total_costs, cost_components


# ---------------------------- #
#       LEVERAGE MANAGEMENT    #
# ---------------------------- #

def compute_leverage(positions: pd.DataFrame) -> pd.Series:
    """
    Compute gross leverage (sum of absolute weights) per date.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position weights (date × ticker).
        
    Returns
    -------
    pd.Series
        Gross leverage per date.
    """
    return np.abs(positions).sum(axis=1)


def apply_leverage_constraint(
    positions: pd.DataFrame,
    max_leverage: float = 1.0,
    method: Literal["scale", "truncate"] = "scale"
) -> pd.DataFrame:
    """
    Apply leverage constraints to position weights.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position weights (date × ticker).
    max_leverage : float, default 1.0
        Maximum allowed gross leverage.
    method : {"scale", "truncate"}, default "scale"
        Constraint method:
        - "scale": proportionally scale all positions
        - "truncate": keep largest positions until leverage limit
        
    Returns
    -------
    pd.DataFrame
        Leverage-constrained positions.
    """
    constrained = positions.copy()
    current_leverage = compute_leverage(positions)
    
    for date in positions.index:
        if current_leverage[date] > max_leverage:
            row = positions.loc[date]
            
            if method == "scale":
                # Scale proportionally
                scale_factor = max_leverage / current_leverage[date]
                constrained.loc[date] = row * scale_factor
                
            elif method == "truncate":
                # Keep largest positions by absolute value
                abs_weights = np.abs(row)
                sorted_idx = abs_weights.sort_values(ascending=False).index
                
                cumulative_leverage = 0.0
                truncated_row = pd.Series(0.0, index=row.index)
                
                for ticker in sorted_idx:
                    new_leverage = cumulative_leverage + abs_weights[ticker]
                    if new_leverage <= max_leverage:
                        truncated_row[ticker] = row[ticker]
                        cumulative_leverage = new_leverage
                    else:
                        # Partial allocation for the last asset
                        remaining_capacity = max_leverage - cumulative_leverage
                        if remaining_capacity > 0:
                            truncated_row[ticker] = row[ticker] * (remaining_capacity / abs_weights[ticker])
                        break
                
                constrained.loc[date] = truncated_row
            
            else:
                raise ValueError("method must be 'scale' or 'truncate'")
    
    return constrained


def estimate_portfolio_volatility(
    positions: pd.DataFrame,
    asset_volatilities: pd.DataFrame,
    correlations: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Estimate portfolio volatility from positions and asset volatilities.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position weights (date × ticker).
    asset_volatilities : pd.DataFrame
        Asset volatilities (date × ticker).
    correlations : pd.DataFrame, optional
        Asset correlation matrix. If None, assumes zero correlations.
        
    Returns
    -------
    pd.Series
        Estimated portfolio volatility per date.
    """
    # Align inputs
    positions, vols = positions.align(asset_volatilities, join='inner')
    
    if correlations is None:
        # Simple case: portfolio_vol = sqrt(sum(w_i^2 * vol_i^2))
        portfolio_var = (positions**2 * vols**2).sum(axis=1)
    else:
        # Full covariance case (simplified - assumes constant correlation matrix)
        portfolio_var = pd.Series(index=positions.index, dtype=float)
        
        for date in positions.index:
            w = positions.loc[date].values.reshape(-1, 1)
            vol_diag = np.diag(vols.loc[date].values)
            cov_matrix = vol_diag @ correlations.values @ vol_diag
            portfolio_var[date] = (w.T @ cov_matrix @ w)[0, 0]
    
    return np.sqrt(portfolio_var.clip(lower=0))


def apply_volatility_target(
    positions: pd.DataFrame,
    asset_volatilities: pd.DataFrame,
    target_volatility: float = 0.15,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Scale positions to target a specific portfolio volatility.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position weights (date × ticker).
    asset_volatilities : pd.DataFrame
        Asset volatilities (date × ticker).
    target_volatility : float, default 0.15
        Target portfolio volatility (15% annualized).
    lookback : int, default 20
        Lookback period for volatility estimation.
        
    Returns
    -------
    pd.DataFrame
        Volatility-targeted positions.
    """
    # Estimate rolling portfolio volatility
    estimated_vol = estimate_portfolio_volatility(positions, asset_volatilities)
    smoothed_vol = estimated_vol.rolling(lookback, min_periods=1).mean()
    
    # Compute scaling factor
    vol_scalar = target_volatility / smoothed_vol.replace(0, np.nan)
    vol_scalar = vol_scalar.clip(lower=0.1, upper=5.0)  # Reasonable bounds
    
    # Apply scaling
    scaled_positions = positions.multiply(vol_scalar, axis=0)
    
    return scaled_positions


# ---------------------------- #
#     PORTFOLIO UTILITIES      #
# ---------------------------- #

def compute_trades_from_positions(
    target_positions: pd.DataFrame,
    current_positions: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute required trades from target and current positions.
    
    Parameters
    ----------
    target_positions : pd.DataFrame
        Target position weights.
    current_positions : pd.DataFrame, optional
        Current position weights. If None, assumes starting from zero.
        
    Returns
    -------
    pd.DataFrame
        Required trades (target - current).
    """
    if current_positions is None:
        return target_positions.fillna(0)
    
    # Align and compute difference
    target, current = target_positions.align(current_positions, join='outer', fill_value=0)
    trades = target - current
    
    return trades


def compute_turnover(trades: pd.DataFrame) -> pd.Series:
    """
    Compute portfolio turnover (sum of absolute trades) per date.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade amounts (date × ticker).
        
    Returns
    -------
    pd.Series
        Portfolio turnover per date.
    """
    return np.abs(trades).sum(axis=1)


def rebalance_schedule(
    positions: pd.DataFrame,
    frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly",
    offset: int = 0
) -> pd.DataFrame:
    """
    Create a rebalancing schedule by sampling positions at specified frequency.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Position weights (date × ticker).
    frequency : {"daily", "weekly", "monthly", "quarterly"}, default "monthly"
        Rebalancing frequency.
    offset : int, default 0
        Day offset for rebalancing (e.g., first Monday of month).
        
    Returns
    -------
    pd.DataFrame
        Rebalanced positions with constant weights between rebalancing dates.
    """
    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS",  # Month start
        "quarterly": "QS"  # Quarter start
    }
    
    if frequency not in freq_map:
        raise ValueError(f"frequency must be one of {list(freq_map.keys())}")
    
    # Sample positions at rebalancing dates
    rebal_freq = freq_map[frequency]
    rebal_positions = positions.resample(rebal_freq).first()
    
    # Forward fill to create constant weights between rebalancing
    full_schedule = rebal_positions.reindex(positions.index).ffill()
    
    return full_schedule
