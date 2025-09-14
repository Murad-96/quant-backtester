"""
qbt.data
========
Data loading, normalization, alignment, and resampling utilities for the quant
research pipeline.

Canonical internal schema (LONG format):
    date, ticker, open, high, low, close, volume

Notes
-----
- Column *order* in input files does NOT matter; we map by names.
- For single-ticker vendor exports (e.g., "Date, Price, Open, High, Low, Vol., Change %"),
  use `load_single_ticker_csv(...)` to normalize into the canonical long schema.
- Downstream modules (features, signals, backtest) generally expect a *wide* panel
  of close prices: DataFrame indexed by date with tickers as columns. Use `pivot_close`.
"""

from __future__ import annotations
from typing import Iterable, Literal, Optional, Tuple, List
from functools import reduce
import re
import numpy as np
import pandas as pd


# ---------------------------- #
# Helper parsing & validation  #
# ---------------------------- #

_NUMERIC_OHLC = ("open", "high", "low", "close", "volume")


def _coerce_numeric_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_volume_token(v) -> float:
    """
    Parse volume strings like '163.25M', '103.7K', '1.2B', '-' into a float.
    Returns NaN if cannot parse.
    """
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "—", "NaN", "nan"}:
        return np.nan

    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([KMB]?)", s, flags=re.IGNORECASE)
    if not m:
        # fallback to numeric coerce
        try:
            return float(s)
        except Exception:
            return np.nan

    num = float(m.group(1))
    suf = m.group(2).upper()
    if suf == "K":
        num *= 1e3
    elif suf == "M":
        num *= 1e6
    elif suf == "B":
        num *= 1e9
    return num


def _normalize_column_names(df: pd.DataFrame, columns_map: Optional[dict]) -> pd.DataFrame:
    if columns_map:
        df = df.rename(columns=columns_map)
    # Additionally: strip, lower, remove trailing dots for robustness
    df = df.rename(columns=lambda c: str(c).strip().lower().rstrip("."))
    return df


# ---------------------------- #
#           LOADERS            #
# ---------------------------- #

def load_prices_long_csv(
    path: str,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    columns_map: Optional[dict] = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Load a LONG-format OHLCV CSV and return a canonical DataFrame with:
    ['date','ticker','open','high','low','close','volume'] (sorted).

    Parameters
    ----------
    path : str
        CSV file path.
    date_col : str, default 'date'
        Name of the date column in the CSV (before optional renaming).
    ticker_col : str, default 'ticker'
        Name of the ticker column in the CSV (before optional renaming).
    columns_map : dict, optional
        Mapping to rename input columns into canonical names (e.g., {'Date':'date'}).
    dayfirst : bool, default False
        Pass to pd.to_datetime for dd/mm vs mm/dd disambiguation.

    Returns
    -------
    pd.DataFrame (LONG)
        Columns: date (datetime64[ns]), ticker (str), open/high/low/close/volume (float)

    Raises
    ------
    KeyError if required columns are missing after renaming.
    """
    df = pd.read_csv(path)
    df = _normalize_column_names(df, columns_map)

    # Validate required columns
    required = {date_col, ticker_col, "open", "high", "low", "close", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    # Parse and coerce
    df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    df["ticker"] = df[ticker_col].astype(str)
    df = _coerce_numeric_cols(df, _NUMERIC_OHLC)
    df = df.dropna(subset=["date", "ticker"]).copy()

    # Keep only canonical cols, sort, drop dups (keep last)
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["date", "ticker"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    return df


def load_prices_parquet(
    path: str,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Load a LONG-format OHLCV parquet file. Same schema as CSV loader.
    """
    df = pd.read_parquet(path)
    df = _normalize_column_names(df, None)

    required = {date_col, ticker_col, "open", "high", "low", "close", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in parquet: {missing}")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    df["ticker"] = df[ticker_col].astype(str)
    df = _coerce_numeric_cols(df, _NUMERIC_OHLC)
    df = df.dropna(subset=["date", "ticker"]).copy()
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    return df


def load_single_ticker_csv(
    path: str,
    ticker: str,
    *,
    columns_map: Optional[dict] = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """
    Normalize a single-ticker vendor export (e.g., columns like:
    'Date','Price','Open','High','Low','Vol.','Change %') into canonical LONG schema.

    Parameters
    ----------
    path : str
        CSV file path.
    ticker : str
        Ticker symbol to assign to all rows.
    columns_map : dict, optional
        Override default vendor→canonical mapping. Example:
        {'Date':'date','Close':'close','Price':'close','Open':'open',...}
        If provided, keys are matched AFTER lowercasing/stripping/trimming dots.
    dayfirst : bool, default False
        Date parsing disambiguation.

    Returns
    -------
    pd.DataFrame (LONG)
        ['date','ticker','open','high','low','close','volume']
    """
    df = pd.read_csv(path)
    # Default map for common vendor exports (case/periods insensitive)
    default_map = {
        "date": "date",
        "price": "close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "vol": "volume",
        "vol.": "volume",
        "volume": "volume",
    }
    # Normalize and apply map
    df = _normalize_column_names(df, None)
    if columns_map:
        # Normalize user-provided map too
        columns_map_norm = {str(k).strip().lower().rstrip("."): v for k, v in columns_map.items()}
        df = df.rename(columns=columns_map_norm)
    df = df.rename(columns={k: v for k, v in default_map.items() if k in df.columns})

    # Validate presence of required fields (no ticker yet)
    need = {"date", "open", "high", "low", "close", "volume"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for single-ticker CSV: {missing}")

    # Parse date & numbers
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=dayfirst)
    # Parse volume tokens like 163.25M
    df["volume"] = df["volume"].apply(_parse_volume_token)

    # Coerce ohlc numerics
    df = _coerce_numeric_cols(df, ("open", "high", "low", "close"))
    df["ticker"] = str(ticker)

    out = df[["date", "ticker", "open", "high", "low", "close", "volume"]].dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


# ---------------------------- #
#        PIVOT / ALIGN         #
# ---------------------------- #

def pivot_close(df_long: pd.DataFrame, *, value_col: str = "close") -> pd.DataFrame:
    """
    Pivot a LONG OHLCV DataFrame into a WIDE price panel (date × ticker).

    Parameters
    ----------
    df_long : pd.DataFrame
        LONG-format canonical schema.
    value_col : str, default 'close'
        Which field to pivot as values (usually 'close').

    Returns
    -------
    pd.DataFrame (WIDE)
        index: date (ascending)
        columns: tickers
        values: selected price field
    """
    required = {"date", "ticker", value_col}
    missing = [c for c in required if c not in df_long.columns]
    if missing:
        raise KeyError(f"Missing required columns in LONG df: {missing}")

    # Ensure uniqueness per (date,ticker); keep last if duplicates
    df = df_long.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    wide = df.pivot(index="date", columns="ticker", values=value_col).sort_index()
    return wide


def align_prices(
    *dfs: pd.DataFrame,
    how: Literal["inner", "outer", "left", "right"] = "inner",
) -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple WIDE DataFrames on the **intersection or union** of dates/columns.

    Returns a tuple of aligned DataFrames in the same order.
    """
    if not dfs:
        return tuple()
    # Align index
    idx = dfs[0].index
    cols = dfs[0].columns
    for df in dfs[1:]:
        idx = idx.join(df.index, how=how)
        cols = cols.join(df.columns, how=how)
    aligned = []
    for df in dfs:
        a = df.reindex(index=idx, columns=cols)
        aligned.append(a)
    return tuple(aligned)


def forward_fill_missing(prices: pd.DataFrame, *, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Forward-fill missing values per column (common for holiday gaps, partial listings).
    """
    return prices.sort_index().ffill(limit=limit)


def merge_wide(dfs: List[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    """
    Merge multiple wide-format DataFrames into one.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames in wide format (e.g., columns are tickers, 
        index is datetime).
    how : str, default 'outer'
        Type of join to use: 'outer', 'inner', 'left', 'right'.

    Returns
    -------
    pd.DataFrame
        Single merged DataFrame with aligned indices and combined columns.

    Examples
    --------
    >>> df1 = pd.DataFrame({"AAPL": [1, 2]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    >>> df2 = pd.DataFrame({"MSFT": [5, 6]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    >>> merge_wide([df1, df2])
                AAPL  MSFT
    2020-01-01   1.0   NaN
    2020-01-02   2.0   5.0
    2020-01-03   NaN   6.0
    """
    if not dfs:
        return pd.DataFrame()

    return reduce(lambda left, right: left.join(right, how=how), dfs)


# ---------------------------- #
#        RESAMPLING/RETURNS    #
# ---------------------------- #

def resample_to_frequency(
    prices: pd.DataFrame,
    *,
    freq: str = "W-FRI",
    how: Literal["last", "first", "mean", "median"] = "last",
) -> pd.DataFrame:
    """
    Resample a WIDE price panel (e.g., daily → weekly). For prices, 'last' is typical.

    Parameters
    ----------
    prices : pd.DataFrame
        WIDE panel of prices.
    freq : str
        Pandas offset alias (e.g., 'W-FRI', 'M', 'Q').
    how : {'last','first','mean','median'}
        Aggregation method within each resample window.
    """
    agg = {"last": "last", "first": "first", "mean": "mean", "median": "median"}[how]
    return prices.resample(freq).agg(agg)


def to_returns(
    prices: pd.DataFrame,
    *,
    method: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """
    Convert prices to returns (column-wise).

    Parameters
    ----------
    method : {'simple','log'}
        simple:  p[t]/p[t-1] - 1
        log:     ln(p[t]) - ln(p[t-1])
    """
    prices = prices.sort_index()
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices).diff()
    else:
        raise ValueError("method must be 'simple' or 'log'")


# ---------------------------- #
#        UNIVERSE FILTERS      #
# ---------------------------- #

def filter_universe(prices: pd.DataFrame, *, min_history: int = 252) -> pd.DataFrame:
    """
    Keep only tickers with at least `min_history` non-NaN observations.
    """
    ok = prices.notna().sum(axis=0) >= int(min_history)
    return prices.loc[:, ok]


def drop_inactive(prices: pd.DataFrame, *, max_na_frac: float = 0.10) -> pd.DataFrame:
    """
    Drop columns (tickers) whose fraction of NaN exceeds `max_na_frac`.
    """
    frac = prices.isna().mean(axis=0)
    keep = frac <= float(max_na_frac)
    return prices.loc[:, keep]
