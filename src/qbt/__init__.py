"""
qbt
===
Quant research and backtesting toolkit.
"""

__version__ = "0.1.0"

# Import commonly used functions at package level
from .data import load_prices_long_csv, pivot_close, merge_wide
#from .features import momentum, mean_reversion_z, realized_vol
#from .signals import combine_and_standardize
#from .backtest import run_backtest

__all__ = [
    "load_prices_long_csv",
    "pivot_close",
    "merge_wide",
    "momentum",
    "mean_reversion_z",
    "realized_vol",
    "combine_and_standardize",
    "run_backtest",
]
