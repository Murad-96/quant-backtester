import numpy as np
import pandas as pd

import qbt.features as F  # if your package is 'src/qbt', ensure PYTHONPATH includes 'src'

# ---------- helpers ----------
def _toy_prices(n_days=100, n_assets=4, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # random walk price paths
    rets = rng.normal(0, 0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=cols)

def _toy_volume(n_days=100, n_assets=4, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    vols = rng.integers(5_000_000, 20_000_000, size=(n_days, n_assets))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(vols, index=dates, columns=cols)

# ---------- tests ----------

def test_sma_shape_and_nans():
    close = _toy_prices()
    out = F.sma(close, window=10)
    assert out.shape == close.shape
    assert out.iloc[:9].isna().all().all()  # first window-1 rows NaN

def test_ema_shape():
    close = _toy_prices()
    out = F.ema(close, span=10)
    assert out.shape == close.shape
    # EMA shouldn't be all NaN
    assert out.notna().sum().sum() > 0

def test_ts_zscore_head_nans_and_scale():
    close = _toy_prices()
    z = F.ts_zscore(close, window=20)
    assert z.shape == close.shape
    # first 19 rows should be NaN
    assert z.iloc[:19].isna().all().all()
    # later rows should have mean ~ 0 per column (allow small tolerance)
    mean_late = z.iloc[30:].mean()
    assert np.all(np.isfinite(mean_late))
    assert (mean_late.abs() < 0.2).all()  # loose sanity bound

def test_returns_simple_vs_log_shapes():
    close = _toy_prices()
    r_simple = F.returns(close, period=5, method="simple")
    r_log = F.returns(close, period=5, method="log")
    assert r_simple.shape == close.shape == r_log.shape
    # simple and log returns are related but not equal; check they’re finite after the lag
    assert np.isfinite(r_simple.iloc[10:].to_numpy()).mean() > 0.95
    assert np.isfinite(r_log.iloc[10:].to_numpy()).mean() > 0.95

def test_momentum_equivalence_to_simple_returns():
    close = _toy_prices()
    L = 15
    mom = F.momentum(close, lookback=L)
    rL = F.returns(close, period=L, method="simple")
    # momentum = future/lag - 1 == simple returns over L
    # allow NaNs near head; compare where both finite
    both = mom.notna() & rL.notna()
    diff = (mom[both] - rL[both]).abs().to_numpy()
    assert np.nanmax(diff) < 1e-12

def test_mean_reversion_spread_basics():
    close = _toy_prices()
    spr = F.mean_reversion_spread(close, short=5, long=20)
    assert spr.shape == close.shape
    # first (long-1) rows should be NaN (due to the long SMA)
    assert spr.iloc[:19].isna().all().all()

def test_mean_reversion_z_rowwise_standardization():
    close = _toy_prices()
    z = F.mean_reversion_z(close, short=5, long=20)
    assert z.shape == close.shape
    # By construction, each row has mean ~ 0 (ignoring NaNs)
    row_means = z.mean(axis=1, skipna=True)
    assert row_means.iloc[30:].abs().mean() < 0.05

def test_realized_vol_positive_and_window_nans():
    close = _toy_prices()
    vol = F.realized_vol(close, window=20, ann_factor=252)
    assert vol.shape == close.shape
    # first 20 rows should have NaNs (std needs full window)
    assert vol.iloc[:20].isna().all().all()
    # later values should be non-negative
    assert (vol.iloc[40:].min().min() >= 0)

def test_dollar_volume_roll_mean():
    close = _toy_prices()
    volume = _toy_volume()
    dv = F.dollar_volume(close, volume, window=10)
    assert dv.shape == close.shape
    assert dv.iloc[:9].isna().all().all()

def test_rolling_corr_shapes():
    x = _toy_prices()
    y = _toy_prices(seed=99)
    corr = F.rolling_corr(x, y, window=15)
    assert corr.shape == x.shape
    # First window-1 rows should be NaN
    assert corr.iloc[:14].isna().all().all()

def test_rolling_beta_against_market_series():
    # asset returns vs market returns (Series)
    ret_x = F.returns(_toy_prices(), period=1, method="simple").dropna()
    mkt = ret_x.mean(axis=1)  # fake "market" return series
    beta = F.rolling_beta(ret_x, mkt, window=20)
    assert beta.shape == ret_x.shape
    # first window rows NaN (since cov/var need window)
    assert beta.iloc[:20].isna().all().all()
    # later rows mostly finite
    assert beta.iloc[40:].notna().mean().mean() > 0.8
