# Quantitative Backtesting Toolkit (QBT)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A professional-grade quantitative research and backtesting framework designed for institutional-quality systematic trading strategies. Built with event-driven architecture, comprehensive risk management, and production-ready code patterns.

## 🎯 Key Features

### Core Architecture
- **Event-Driven Backtesting Engine**: Realistic simulation with proper event sequencing (market data → signals → orders → fills)
- **Modular Design**: Cleanly separated components for data handling, signal generation, portfolio construction, and performance analytics
- **Production-Ready**: Type hints, comprehensive documentation, and institutional-grade code quality

### Quantitative Capabilities
- **Advanced Signal Generation**: Momentum, mean reversion, volatility-based indicators with cross-sectional normalization
- **Sophisticated Portfolio Construction**: Multiple position sizing methods (equal weight, volatility-adjusted, signal-weighted, risk parity)
- **Transaction Cost Modeling**: Linear costs, market impact (√law), and bid-ask spread integration
- **Comprehensive Risk Analytics**: Sharpe ratio, max drawdown, VaR, expected shortfall, factor exposures

### Professional Features
- **Performance Attribution**: Detailed trade-level analytics and portfolio decomposition
- **Flexible Rebalancing**: Configurable frequencies (daily, weekly, monthly, quarterly) with proper event timing
- **Data Pipeline**: Robust data loading, cleaning, and feature engineering workflows
- **Visualization Suite**: Professional-quality charts for equity curves, factor analysis, and risk metrics

![alt text](https://github.com/Murad-96/quant-backtester/blob/main/test_dashboard.png?raw=true)

## 🏗️ Architecture Overview

```
qbt/
├── backtest.py      # Event-driven backtesting engine
├── portfolio.py     # Position sizing and risk management
├── signals.py       # Technical indicators and signal processing
├── metrics.py       # Performance and risk analytics
├── data.py          # Data loading and preprocessing utilities
├── features.py      # Feature engineering toolkit
└── plotting.py      # Visualization and charting
```

The framework follows a clean separation of concerns:

1. **Data Layer**: Handles price/volume ingestion and preprocessing
2. **Signal Layer**: Generates predictive features and trading signals  
3. **Portfolio Layer**: Constructs positions with risk controls
4. **Execution Layer**: Simulates realistic trading with costs
5. **Analytics Layer**: Measures performance and risk

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/quant-backtester.git
cd quant-backtester
pip install -r requirements.txt
```

### Basic Usage

```python
import qbt
import pandas as pd

# Load price data
prices = qbt.merge_wide({
    'AAPL': qbt.load_prices_long_csv('data/raw/AAPL.csv'),
    'MSFT': qbt.load_prices_long_csv('data/raw/MSFT.csv'),
    'TSLA': qbt.load_prices_long_csv('data/raw/TSLA.csv')
}, 'close')

# Generate momentum signals
signals = qbt.momentum(prices, lookback=20)
signals = qbt.cross_sectional_normalize(signals, method='zscore')

# Run backtest with event-driven engine
results = qbt.run_backtest(
    prices=prices,
    signals=signals,
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1_000_000,
    position_sizing='equal_weight',
    leverage=1.0,
    commission_rate=0.001,
    rebalance_frequency='monthly'
)

# Analyze performance
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")

# Generate professional charts
qbt.plotting.plot_performance_dashboard(results)
```

## 📊 Advanced Features

### Event-Driven Backtesting
The core engine implements proper event sequencing for realistic simulation:

```python
# Initialize backtest engine
engine = qbt.BacktestEngine(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1_000_000,
    commission_rate=0.001,
    market_impact=0.0001
)

# Configure strategy
engine.set_market_data(prices, volumes)
engine.set_strategy_components(
    signal_generator=my_signal_function,
    position_sizer=qbt.portfolio.volatility_adjusted_positions,
    rebalance_frequency='monthly'
)

# Run with full event log
results = engine.run_backtest()
```

### Sophisticated Position Sizing

```python
# Volatility-adjusted positions
positions = qbt.portfolio.volatility_adjusted_positions(
    signals=signals,
    volatilities=realized_vol,
    leverage=1.5,
    target_vol=0.15,
    max_weight=0.10
)

# Risk parity allocation
positions = qbt.portfolio.risk_parity_positions(
    signals=signals,
    volatilities=volatilities,
    correlations=correlation_matrix,
    leverage=1.0
)
```

### Transaction Cost Analysis

```python
# Comprehensive cost modeling
total_costs, cost_breakdown = qbt.portfolio.total_transaction_costs(
    trades=trade_data,
    prices=price_data,
    volumes=volume_data,
    spreads=spread_data,
    linear_rate=0.0005,      # 5 bps
    impact_temp=0.1,         # Temporary impact
    impact_perm=0.05         # Permanent impact
)
```

### Performance Analytics

```python
# Comprehensive performance metrics
metrics = qbt.portfolio_metrics(
    returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02
)

# Factor exposure analysis
exposures = qbt.factor_exposures(
    returns=portfolio_returns,
    factor_returns=factor_data,
    window=252
)
```

## 📈 Research Pipeline

The framework includes a complete research workflow demonstrated in Jupyter notebooks:

1. **Data Ingestion** (`01_data_ingestion.ipynb`): Load and clean market data
2. **Feature Engineering** (`02_feature_engineering_demo.ipynb`): Generate predictive signals
3. **Signal Research** (`03_signal_research.ipynb`): Analyze signal quality and decay
4. **Research Pipeline** (`04_research_pipeline.ipynb`): Complete strategy development workflow

Each notebook demonstrates production-ready practices for institutional quantitative research.

## 🔬 Technical Implementation

### Data Structures
- **Wide Format**: Primary data format (date × ticker) for efficient vectorized operations
- **Type Safety**: Comprehensive type hints for IDE support and error prevention
- **Memory Efficiency**: Optimized pandas operations with minimal data copying

### Performance Optimizations
- **Vectorized Operations**: NumPy/pandas optimizations for signal calculations
- **Event Queue**: Priority queue implementation for proper event ordering
- **Lazy Evaluation**: Computed metrics only when requested

### Code Quality
- **Modular Architecture**: Clean separation of concerns with minimal coupling
- **Documentation**: Comprehensive docstrings with parameter types and examples
- **Error Handling**: Graceful handling of edge cases and data quality issues

## 📋 Requirements

### Core Dependencies
```
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.5.0
scipy >= 1.7.0
```

### Development Environment
- Python 3.8+
- Jupyter Lab for research notebooks
- Git for version control

## 🎓 Educational Value

This project demonstrates proficiency in:

- **Quantitative Finance**: Signal generation, portfolio theory, risk management
- **Software Engineering**: Clean architecture, type safety, modularity
- **Data Science**: Time series analysis, statistical modeling, visualization
- **Production Systems**: Event-driven design, transaction cost modeling, performance optimization

## 📚 Documentation

Comprehensive documentation is available in the source code with detailed docstrings for all public functions. Key modules include:

- `backtest.py`: Event-driven backtesting engine with proper sequencing
- `portfolio.py`: Position sizing algorithms and risk management tools
- `signals.py`: Technical indicators and signal processing utilities
- `metrics.py`: Performance analytics and risk measurement
- `plotting.py`: Professional visualization suite

## 🔮 Future Enhancements

- [ ] Options and derivatives support
- [ ] Alternative data integration (news, earnings, etc.)
- [ ] Machine learning signal generation
- [ ] Live trading integration
- [ ] Multi-asset class support (equities, futures, FX)
- [ ] Real-time risk monitoring

## 🤝 Contributing

This project follows professional software development practices. Contributions should include:
- Type hints for all functions
- Comprehensive unit tests
- Updated documentation
- Performance benchmarks

## 📄 License

MIT License - see LICENSE file for details.

---

**Built for quantitative researchers who demand institutional-grade tools and production-ready code quality.**
