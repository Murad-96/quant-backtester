"""
qbt.plotting
============
Visualization utilities for quantitative backtesting results.

This module provides plotting functions for:
- Portfolio equity curves and performance visualization
- Factor Information Coefficient (IC) analysis 
- Portfolio turnover and trading activity
- Risk metrics and drawdown analysis
- Signal distribution and correlation analysis

Key functions:
- plot_equity_curve(): Portfolio value and returns over time
- plot_factor_ic(): Information coefficient analysis
- plot_turnover(): Portfolio turnover and trading frequency
- plot_performance_dashboard(): Comprehensive performance overview
"""

from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import warnings

# Import metrics for calculations
from . import metrics


# ---------------------------- #
#        UTILITY FUNCTIONS     #
# ---------------------------- #

def _setup_date_axis(ax: Axes, dates: pd.DatetimeIndex, rotate_labels: bool = True) -> None:
    """
    Setup date formatting for time series plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    dates : pd.DatetimeIndex
        Date index for determining formatting
    rotate_labels : bool, default True
        Whether to rotate date labels
    """
    # Determine appropriate date formatting based on date range
    date_range = (dates.max() - dates.min()).days
    
    if date_range <= 60:  # 2 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    elif date_range <= 365:  # 1 year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif date_range <= 1825:  # 5 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:  # > 5 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    if rotate_labels:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def _format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _add_performance_text(ax: Axes, metrics_dict: Dict[str, float], 
                         position: str = 'upper left') -> None:
    """
    Add performance metrics text box to a plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add text to
    metrics_dict : dict
        Dictionary of metrics to display
    position : str, default 'upper left'
        Position for the text box
    """
    # Format metrics text
    text_lines = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'ratio' in key.lower():
                text_lines.append(f"{key}: {_format_percentage(value) if abs(value) < 10 else f'{value:.2f}'}")
            else:
                text_lines.append(f"{key}: {value:.3f}")
        else:
            text_lines.append(f"{key}: {value}")
    
    text_str = '\n'.join(text_lines)
    
    # Position mapping
    pos_map = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02)
    }
    
    x, y = pos_map.get(position, (0.02, 0.98))
    ha = 'left' if x < 0.5 else 'right'
    va = 'top' if y > 0.5 else 'bottom'
    
    ax.text(x, y, text_str, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment=va, horizontalalignment=ha)


# ---------------------------- #
#        EQUITY CURVE          #
# ---------------------------- #

def plot_equity_curve(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve",
    show_drawdown: bool = True,
    show_rolling_sharpe: bool = True,
    rolling_window: int = 252,
    figsize: Tuple[int, int] = (12, 8),
    show_metrics: bool = True
) -> Figure:
    """
    Plot portfolio equity curve with optional benchmark comparison.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of portfolio values
    benchmark_values : pd.Series, optional
        Time series of benchmark values for comparison
    title : str, default "Portfolio Equity Curve"
        Plot title
    show_drawdown : bool, default True
        Whether to show drawdown subplot
    show_rolling_sharpe : bool, default True
        Whether to show rolling Sharpe ratio subplot
    rolling_window : int, default 252
        Window for rolling calculations (252 = 1 year for daily data)
    figsize : tuple, default (12, 8)
        Figure size in inches
    show_metrics : bool, default True
        Whether to show performance metrics text box
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate number of subplots
    n_subplots = 1 + int(show_drawdown) + int(show_rolling_sharpe)
    
    # Create figure
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, 
                            gridspec_kw={'height_ratios': [3] + [1] * (n_subplots - 1)})
    if n_subplots == 1:
        axes = [axes]
    
    ax_main = axes[0]
    
    # Normalize to start at 100
    portfolio_norm = portfolio_values / portfolio_values.iloc[0] * 100
    
    # Plot portfolio equity curve
    ax_main.plot(portfolio_norm.index, portfolio_norm.values, 
                linewidth=2, label='Portfolio', color='blue')
    
    # Plot benchmark if provided
    if benchmark_values is not None:
        benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100
        ax_main.plot(benchmark_norm.index, benchmark_norm.values,
                    linewidth=1.5, label='Benchmark', color='gray', alpha=0.7)
    
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Portfolio Value (Base = 100)', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    
    # Format x-axis
    _setup_date_axis(ax_main, portfolio_values.index)
    
    # Add performance metrics
    if show_metrics:
        portfolio_returns = portfolio_values.pct_change().dropna()
        perf_metrics = {
            'Total Return': metrics.total_return(portfolio_values),
            'Ann. Return': metrics.annualized_return(portfolio_values),
            'Volatility': portfolio_returns.std() * np.sqrt(252),
            'Sharpe Ratio': metrics.sharpe_ratio(portfolio_returns),
            'Max Drawdown': metrics.max_drawdown(portfolio_values)
        }
        _add_performance_text(ax_main, perf_metrics, 'upper left')
    
    # Drawdown subplot
    subplot_idx = 1
    if show_drawdown:
        ax_dd = axes[subplot_idx]
        drawdowns = metrics.calculate_drawdowns(portfolio_values)
        
        ax_dd.fill_between(drawdowns.index, drawdowns.values, 0, 
                          color='red', alpha=0.3, label='Drawdown')
        ax_dd.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
        
        ax_dd.set_ylabel('Drawdown', fontsize=11)
        ax_dd.grid(True, alpha=0.3)
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        _setup_date_axis(ax_dd, portfolio_values.index)
        
        subplot_idx += 1
    
    # Rolling Sharpe ratio subplot
    if show_rolling_sharpe:
        ax_sharpe = axes[subplot_idx]
        returns = portfolio_values.pct_change().dropna()
        
        # Calculate rolling Sharpe ratio
        rolling_returns = returns.rolling(rolling_window, min_periods=int(rolling_window/2))
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        
        ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe.values, 
                      color='green', linewidth=1.5, label=f'{rolling_window//252}Y Rolling Sharpe')
        ax_sharpe.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_sharpe.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='Sharpe = 1')
        
        ax_sharpe.set_ylabel('Rolling Sharpe', fontsize=11)
        ax_sharpe.set_xlabel('Date', fontsize=11)
        ax_sharpe.grid(True, alpha=0.3)
        ax_sharpe.legend()
        _setup_date_axis(ax_sharpe, portfolio_values.index)
    
    plt.tight_layout()
    return fig


# ---------------------------- #
#         FACTOR IC            #
# ---------------------------- #

def plot_factor_ic(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    title: str = "Factor Information Coefficient Analysis",
    ic_method: str = "pearson",
    rolling_window: int = 252,
    figsize: Tuple[int, int] = (14, 10),
    max_factors: int = 6
) -> Figure:
    """
    Plot factor Information Coefficient (IC) analysis.
    
    The IC measures the correlation between factor signals and subsequent returns,
    indicating the predictive power of each factor.
    
    Parameters
    ----------
    signals : pd.DataFrame
        Wide format (date × ticker) factor signals
    returns : pd.DataFrame
        Wide format (date × ticker) forward returns (should be shifted)
    title : str, default "Factor Information Coefficient Analysis"
        Plot title
    ic_method : {"pearson", "spearman"}, default "pearson"
        Correlation method for IC calculation
    rolling_window : int, default 252
        Window for rolling IC calculation
    figsize : tuple, default (14, 10)
        Figure size in inches
    max_factors : int, default 6
        Maximum number of factors to plot (for readability)
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Align data and handle forward returns
    signals_aligned, returns_aligned = signals.align(returns, join='inner')
    
    if len(signals_aligned) == 0:
        raise ValueError("No overlapping dates between signals and returns")
    
    # Calculate IC for each factor (assuming signals represent different factors over time)
    # We'll calculate cross-sectional IC per time period
    ic_series_list = []
    factor_names = []
    
    # If signals has multiple columns (multiple stocks), calculate cross-sectional IC
    if signals_aligned.shape[1] > 1:
        ic_ts = []
        for date in signals_aligned.index:
            signal_cross_section = signals_aligned.loc[date].dropna()
            return_cross_section = returns_aligned.loc[date].dropna()
            
            # Align the cross-sections
            aligned_signal, aligned_return = signal_cross_section.align(return_cross_section, join='inner')
            
            if len(aligned_signal) > 3:  # Need minimum observations
                if ic_method == "pearson":
                    ic_value = aligned_signal.corr(aligned_return)
                elif ic_method == "spearman":
                    ic_value = aligned_signal.corr(aligned_return, method='spearman')
                else:
                    raise ValueError("ic_method must be 'pearson' or 'spearman'")
                ic_ts.append(ic_value)
            else:
                ic_ts.append(np.nan)
        
        ic_series = pd.Series(ic_ts, index=signals_aligned.index)
        ic_series_list = [ic_series]
        factor_names = ['Cross-Sectional IC']
    
    else:
        # If single column, treat each rolling window as a separate analysis
        warnings.warn("Single column signals detected. Consider providing multi-asset signals for cross-sectional IC analysis.")
        ic_series = pd.Series(index=signals_aligned.index, dtype=float)
        ic_series[:] = np.nan
        ic_series_list = [ic_series]
        factor_names = ['Time-Series IC']
    
    # Limit number of factors for readability
    ic_series_list = ic_series_list[:max_factors]
    factor_names = factor_names[:max_factors]
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot 1: IC Time Series
    ax1 = axes[0]
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            ax1.plot(ic_series.index, ic_series.values, 
                    label=name, alpha=0.7, linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='IC = 0.05')
    ax1.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5, label='IC = -0.05')
    ax1.set_title('Information Coefficient Time Series', fontweight='bold')
    ax1.set_ylabel('IC')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    _setup_date_axis(ax1, signals_aligned.index)
    
    # Plot 2: Rolling IC
    ax2 = axes[1]
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            rolling_ic = ic_series.rolling(rolling_window, min_periods=int(rolling_window/4)).mean()
            ax2.plot(rolling_ic.index, rolling_ic.values, 
                    label=f'{name} ({rolling_window//21}M Rolling)', linewidth=2)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title(f'Rolling IC ({rolling_window} periods)', fontweight='bold')
    ax2.set_ylabel('Rolling IC')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _setup_date_axis(ax2, signals_aligned.index)
    
    # Plot 3: IC Distribution
    ax3 = axes[2]
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            ic_clean = ic_series.dropna()
            if len(ic_clean) > 0:
                ax3.hist(ic_clean.values, bins=30, alpha=0.6, 
                        label=f'{name} (μ={ic_clean.mean():.3f})', density=True)
    
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('IC Distribution', fontweight='bold')
    ax3.set_xlabel('IC Value')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: IC Statistics
    ax4 = axes[3]
    ic_stats = []
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            ic_clean = ic_series.dropna()
            if len(ic_clean) > 0:
                stats = {
                    'Mean IC': ic_clean.mean(),
                    'IC Std': ic_clean.std(),
                    'IC IR': ic_clean.mean() / ic_clean.std() if ic_clean.std() > 0 else 0,
                    'Hit Rate': (ic_clean > 0).mean()
                }
                ic_stats.append((name, stats))
    
    # Create bar plot of IC statistics
    if ic_stats:
        stat_names = list(ic_stats[0][1].keys())
        x = np.arange(len(stat_names))
        width = 0.8 / len(ic_stats)
        
        for i, (factor_name, stats) in enumerate(ic_stats):
            values = list(stats.values())
            ax4.bar(x + i * width, values, width, label=factor_name, alpha=0.7)
        
        ax4.set_title('IC Summary Statistics', fontweight='bold')
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x + width * (len(ic_stats) - 1) / 2)
        ax4.set_xticklabels(stat_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 5: IC Autocorrelation
    ax5 = axes[4]
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            ic_clean = ic_series.dropna()
            if len(ic_clean) > 20:
                # Calculate autocorrelation for lags 1-20
                lags = range(1, min(21, len(ic_clean)//4))
                autocorr = [ic_clean.autocorr(lag=lag) for lag in lags]
                ax5.plot(lags, autocorr, marker='o', label=name, alpha=0.7)
    
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('IC Autocorrelation', fontweight='bold')
    ax5.set_xlabel('Lag (periods)')
    ax5.set_ylabel('Autocorrelation')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: IC Decay Analysis
    ax6 = axes[5]
    # This would show how IC decays over different forward return horizons
    # For now, show cumulative IC
    for i, (ic_series, name) in enumerate(zip(ic_series_list, factor_names)):
        if not ic_series.isna().all():
            cumulative_ic = ic_series.fillna(0).cumsum()
            ax6.plot(cumulative_ic.index, cumulative_ic.values, 
                    label=f'{name} Cumulative', linewidth=2)
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_title('Cumulative IC', fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Cumulative IC')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    _setup_date_axis(ax6, signals_aligned.index)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


# ---------------------------- #
#         TURNOVER             #
# ---------------------------- #

def plot_turnover(
    positions: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    title: str = "Portfolio Turnover Analysis",
    rolling_window: int = 21,
    figsize: Tuple[int, int] = (12, 8),
    show_distribution: bool = True
) -> Figure:
    """
    Plot portfolio turnover analysis.
    
    Parameters
    ----------
    positions : pd.DataFrame
        Wide format (date × ticker) position weights
    prices : pd.DataFrame, optional
        Wide format (date × ticker) prices for trade value calculations
    title : str, default "Portfolio Turnover Analysis"
        Plot title
    rolling_window : int, default 21
        Window for rolling turnover calculations
    figsize : tuple, default (12, 8)
        Figure size in inches
    show_distribution : bool, default True
        Whether to show turnover distribution subplot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate turnover
    turnover_daily = metrics.calculate_turnover(positions)
    turnover_daily = turnover_daily.dropna()
    
    if len(turnover_daily) == 0:
        raise ValueError("No turnover data available")
    
    # Calculate rolling averages
    turnover_rolling = turnover_daily.rolling(rolling_window, min_periods=1).mean()
    
    # Calculate trade values if prices provided
    trade_values = None
    if prices is not None:
        trade_weights, trade_vals = metrics.calculate_trades_from_positions(positions, prices)
        trade_values = trade_vals.dropna()
    
    # Create subplots
    n_plots = 3 if show_distribution else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize,
                            gridspec_kw={'height_ratios': [2, 1, 1] if show_distribution else [2, 1]})
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Turnover Time Series
    ax1 = axes[0]
    
    # Daily turnover
    ax1.plot(turnover_daily.index, turnover_daily.values, 
            color='lightblue', alpha=0.6, linewidth=0.8, label='Daily Turnover')
    
    # Rolling average
    ax1.plot(turnover_rolling.index, turnover_rolling.values,
            color='blue', linewidth=2, label=f'{rolling_window}-Day Rolling Avg')
    
    # Highlight high turnover periods
    high_turnover_threshold = turnover_daily.quantile(0.95)
    high_turnover_mask = turnover_daily > high_turnover_threshold
    if high_turnover_mask.any():
        ax1.scatter(turnover_daily[high_turnover_mask].index, 
                   turnover_daily[high_turnover_mask].values,
                   color='red', alpha=0.7, s=10, label='High Turnover Days')
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Turnover', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    _setup_date_axis(ax1, turnover_daily.index)
    
    # Add turnover statistics
    turnover_stats = {
        'Mean Daily': turnover_daily.mean(),
        'Std Daily': turnover_daily.std(),
        'Annual': turnover_daily.mean() * 252,
        '95th Pct': turnover_daily.quantile(0.95)
    }
    _add_performance_text(ax1, turnover_stats, 'upper right')
    
    # Plot 2: Number of Position Changes
    ax2 = axes[1]
    
    # Calculate number of position changes per day
    position_changes = (positions.diff() != 0).sum(axis=1)
    position_changes = position_changes.dropna()
    
    # Plot as bar chart
    ax2.bar(position_changes.index, position_changes.values, 
           width=1, alpha=0.6, color='green', label='Position Changes')
    
    # Rolling average
    changes_rolling = position_changes.rolling(rolling_window, min_periods=1).mean()
    ax2.plot(changes_rolling.index, changes_rolling.values,
            color='darkgreen', linewidth=2, label=f'{rolling_window}-Day Rolling Avg')
    
    ax2.set_ylabel('# Position Changes', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _setup_date_axis(ax2, position_changes.index)
    
    # Plot 3: Distribution (if requested)
    if show_distribution:
        ax3 = axes[2]
        
        # Turnover distribution
        ax3.hist(turnover_daily.values, bins=50, alpha=0.7, color='skyblue', 
                density=True, label='Daily Turnover')
        
        # Add statistics lines
        mean_turnover = turnover_daily.mean()
        median_turnover = turnover_daily.median()
        ax3.axvline(mean_turnover, color='red', linestyle='--', 
                   label=f'Mean: {_format_percentage(mean_turnover)}')
        ax3.axvline(median_turnover, color='orange', linestyle='--',
                   label=f'Median: {_format_percentage(median_turnover)}')
        
        ax3.set_xlabel('Daily Turnover', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    plt.tight_layout()
    return fig


# ---------------------------- #
#    PERFORMANCE DASHBOARD     #
# ---------------------------- #

def plot_performance_dashboard(
    portfolio_values: pd.Series,
    positions: Optional[pd.DataFrame] = None,
    benchmark_values: Optional[pd.Series] = None,
    title: str = "Portfolio Performance Dashboard",
    figsize: Tuple[int, int] = (16, 12)
) -> Figure:
    """
    Create a comprehensive performance dashboard.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of portfolio values
    positions : pd.DataFrame, optional
        Wide format (date × ticker) position weights for turnover analysis
    benchmark_values : pd.Series, optional
        Time series of benchmark values for comparison
    title : str, default "Portfolio Performance Dashboard"
        Dashboard title
    figsize : tuple, default (16, 12)
        Figure size in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The created dashboard figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Calculate returns and metrics
    returns = portfolio_values.pct_change().dropna()
    
    # 1. Equity Curve (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    portfolio_norm = portfolio_values / portfolio_values.iloc[0] * 100
    ax1.plot(portfolio_norm.index, portfolio_norm.values, linewidth=2, color='blue')
    
    if benchmark_values is not None:
        benchmark_norm = benchmark_values / benchmark_values.iloc[0] * 100
        ax1.plot(benchmark_norm.index, benchmark_norm.values, 
                linewidth=1.5, color='gray', alpha=0.7, label='Benchmark')
        ax1.legend()
    
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_ylabel('Value (Base = 100)')
    ax1.grid(True, alpha=0.3)
    _setup_date_axis(ax1, portfolio_values.index)
    
    # 2. Drawdown (top right, spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    drawdowns = metrics.calculate_drawdowns(portfolio_values)
    ax2.fill_between(drawdowns.index, drawdowns.values, 0, 
                    color='red', alpha=0.3)
    ax2.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
    ax2.set_title('Drawdown', fontweight='bold')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    _setup_date_axis(ax2, portfolio_values.index)
    
    # 3. Monthly Returns Heatmap (middle left, spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_returns) > 0:
        # Create pivot table for heatmap
        months_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        if len(months_df) > 1:
            pivot_table = months_df.pivot(index='Year', columns='Month', values='Return')
            
            # Plot heatmap
            im = ax3.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto',
                           vmin=-0.1, vmax=0.1)
            
            # Set ticks and labels
            ax3.set_xticks(range(12))
            ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax3.set_yticks(range(len(pivot_table.index)))
            ax3.set_yticklabels(pivot_table.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('Monthly Return')
            
            # Add text annotations
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    if not np.isnan(pivot_table.iloc[i, j]):
                        text = f'{pivot_table.iloc[i, j]:.1%}'
                        ax3.text(j, i, text, ha='center', va='center',
                               color='white' if abs(pivot_table.iloc[i, j]) > 0.05 else 'black',
                               fontsize=8)
    
    ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
    
    # 4. Return Distribution (middle right, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.hist(returns.values, bins=50, alpha=0.7, color='lightblue', density=True)
    ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {_format_percentage(returns.mean())}')
    ax4.axvline(returns.median(), color='orange', linestyle='--', label=f'Median: {_format_percentage(returns.median())}')
    ax4.set_title('Daily Returns Distribution', fontweight='bold')
    ax4.set_xlabel('Daily Return')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # 5. Performance Metrics (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    perf_metrics = metrics.performance_metrics(portfolio_values, returns)
    
    metrics_text = [
        f"Total Return: {_format_percentage(perf_metrics['total_return'])}",
        f"Ann. Return: {_format_percentage(perf_metrics['annualized_return'])}",
        f"Volatility: {_format_percentage(perf_metrics['volatility'])}",
        f"Sharpe Ratio: {perf_metrics['sharpe_ratio']:.2f}",
        f"Max Drawdown: {_format_percentage(perf_metrics['max_drawdown'])}",
        f"Calmar Ratio: {perf_metrics['calmar_ratio']:.2f}",
        f"Best Day: {_format_percentage(perf_metrics['best_day'])}",
        f"Worst Day: {_format_percentage(perf_metrics['worst_day'])}",
        f"Win Rate: {_format_percentage(perf_metrics['win_rate'])}"
    ]
    
    ax5.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax5.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax5.set_title('Performance Metrics', fontweight='bold')
    ax5.axis('off')
    
    # 6. Rolling Metrics (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    rolling_sharpe = returns.rolling(252, min_periods=126).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_vol = returns.rolling(252, min_periods=126).std() * np.sqrt(252)
    
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    color='green', linewidth=1.5, label='1Y Sharpe')
    line2 = ax6_twin.plot(rolling_vol.index, rolling_vol.values,
                         color='purple', linewidth=1.5, label='1Y Volatility')
    
    ax6.set_title('Rolling Metrics (1Y)', fontweight='bold')
    ax6.set_ylabel('Sharpe Ratio', color='green')
    ax6_twin.set_ylabel('Volatility', color='purple')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    _setup_date_axis(ax6, returns.index, rotate_labels=False)
    
    # 7. Turnover (bottom right, if positions provided)
    ax7 = fig.add_subplot(gs[2, 2])
    if positions is not None:
        turnover = metrics.calculate_turnover(positions).dropna()
        if len(turnover) > 0:
            turnover_rolling = turnover.rolling(21, min_periods=1).mean()
            ax7.plot(turnover_rolling.index, turnover_rolling.values, 
                    color='brown', linewidth=1.5)
            ax7.set_title('21D Avg Turnover', fontweight='bold')
            ax7.set_ylabel('Turnover')
            ax7.grid(True, alpha=0.3)
            ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            _setup_date_axis(ax7, turnover.index, rotate_labels=False)
        else:
            ax7.text(0.5, 0.5, 'No Turnover Data', ha='center', va='center', 
                    transform=ax7.transAxes)
            ax7.set_title('Turnover', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'No Position Data', ha='center', va='center',
                transform=ax7.transAxes)
        ax7.set_title('Turnover', fontweight='bold')
    
    # 8. Additional metrics (bottom far right)
    ax8 = fig.add_subplot(gs[2, 3])
    
    if positions is not None:
        # Position statistics
        num_positions = (positions != 0).sum(axis=1)
        leverage = positions.abs().sum(axis=1)
        
        pos_stats_text = [
            f"Avg Positions: {num_positions.mean():.1f}",
            f"Max Positions: {num_positions.max():.0f}",
            f"Avg Leverage: {leverage.mean():.2f}",
            f"Max Leverage: {leverage.max():.2f}"
        ]
        
        if len(turnover) > 0:
            pos_stats_text.extend([
                f"Avg Daily TO: {_format_percentage(turnover.mean())}",
                f"Annual TO: {_format_percentage(turnover.mean() * 252)}"
            ])
    else:
        pos_stats_text = ["No position data available"]
    
    ax8.text(0.05, 0.95, '\n'.join(pos_stats_text), transform=ax8.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax8.set_title('Portfolio Stats', fontweight='bold')
    ax8.axis('off')
    
    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    return fig


# ---------------------------- #
#      CONVENIENCE FUNCTIONS   #
# ---------------------------- #

def quick_plot_backtest_results(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (16, 10)
) -> Figure:
    """
    Quick plotting function for backtest results dictionary.
    
    Parameters
    ----------
    results : dict
        Backtest results dictionary from BacktestEngine.run_backtest()
    figsize : tuple, default (16, 10)
        Figure size in inches
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if 'performance_history' not in results:
        raise ValueError("Results dictionary must contain 'performance_history'")
    
    perf_df = results['performance_history']
    portfolio_values = perf_df['portfolio_value']
    
    # Create dashboard
    fig = plot_performance_dashboard(
        portfolio_values=portfolio_values,
        title=f"Backtest Results ({results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')})",
        figsize=figsize
    )
    
    return fig


def save_plots(
    figures: Union[Figure, List[Figure]], 
    filenames: Union[str, List[str]], 
    dpi: int = 300,
    format: str = 'png'
) -> None:
    """
    Save one or more figures to files.
    
    Parameters
    ----------
    figures : Figure or list of Figure
        Matplotlib figure(s) to save
    filenames : str or list of str
        Filename(s) for saving
    dpi : int, default 300
        Resolution for saved figures
    format : str, default 'png'
        File format ('png', 'pdf', 'svg', etc.)
    """
    if not isinstance(figures, list):
        figures = [figures]
    if not isinstance(filenames, list):
        filenames = [filenames]
    
    if len(figures) != len(filenames):
        raise ValueError("Number of figures must match number of filenames")
    
    for fig, filename in zip(figures, filenames):
        # Add extension if not provided
        if '.' not in filename:
            filename = f"{filename}.{format}"
        
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved figure: {filename}")
