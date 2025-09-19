"""
qbt
===
Quant research and backtesting toolkit.
"""

__version__ = "0.1.0"

# Import commonly used functions at package level
from .data import load_prices_long_csv, pivot_close, merge_wide
from . import portfolio
from . import plotting
from .backtest import run_backtest, BacktestEngine
from .metrics import (
    calculate_returns, total_return, annualized_return, 
    max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
    calculate_turnover, performance_metrics, portfolio_metrics
)
#from .features import momentum, mean_reversion_z, realized_vol
#from .signals import combine_and_standardize

__all__ = [
    "load_prices_long_csv",
    "pivot_close", 
    "merge_wide",
    "run_backtest",
    "BacktestEngine",
    "calculate_returns",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "calculate_turnover",
    "performance_metrics",
    "portfolio_metrics",
    "momentum",
    "mean_reversion_z",
    "realized_vol",
    "combine_and_standardize",
]
