"""
qbt.backtest
============
Event-driven backtesting engine for quantitative trading strategies.

This module provides:
- Event-driven architecture for realistic backtesting
- Portfolio state management and position tracking
- Integration with signals, portfolio construction, and data modules
- Performance analytics and risk metrics

Key components:
- Event system: Market data, signals, orders, fills
- BacktestEngine: Main event loop coordinator
- Portfolio state tracking with transaction costs
- Performance and risk analytics
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
from queue import PriorityQueue
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import warnings

from . import data, signals, portfolio


# ---------------------------- #
#          EVENT SYSTEM        #
# ---------------------------- #

@dataclass
class Event(ABC):
    """Base class for all backtest events."""
    timestamp: datetime
    priority: int = 0  # Lower numbers = higher priority
    
    def __lt__(self, other: 'Event') -> bool:
        """Priority queue ordering: timestamp first, then priority."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


@dataclass 
class MarketEvent(Event):
    """Market data update event."""
    prices: Dict[str, float]  # ticker -> price
    volumes: Optional[Dict[str, float]] = None
    priority: int = field(default=1, init=False)


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    signals: Dict[str, float]  # ticker -> signal strength
    priority: int = field(default=2, init=False)


@dataclass
class OrderEvent(Event):
    """Order placement event."""
    ticker: str
    quantity: float  # Positive = buy, negative = sell
    order_type: str = "market"  # "market" or "limit"
    limit_price: Optional[float] = None
    priority: int = field(default=3, init=False)


@dataclass
class FillEvent(Event):
    """Order execution/fill event."""
    ticker: str
    quantity: float  # Actual filled quantity
    fill_price: float
    commission: float = 0.0
    priority: int = field(default=4, init=False)


class EventQueue:
    """Priority queue for managing backtest events."""
    
    def __init__(self):
        self._queue = PriorityQueue()
        self._event_count = 0
    
    def put(self, event: Event) -> None:
        """Add event to queue."""
        # Add sequence number to ensure FIFO for same timestamp/priority
        self._queue.put((event, self._event_count))
        self._event_count += 1
    
    def get(self) -> Optional[Event]:
        """Get next event from queue."""
        if self.empty():
            return None
        event, _ = self._queue.get()
        return event
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()


# ---------------------------- #
#      PORTFOLIO STATE         #
# ---------------------------- #

@dataclass
class PortfolioState:
    """Current portfolio state tracking."""
    cash: float = 1000000.0  # Starting cash
    positions: Dict[str, float] = field(default_factory=dict)  # ticker -> shares
    prices: Dict[str, float] = field(default_factory=dict)  # ticker -> current price
    
    def get_position_value(self, ticker: str) -> float:
        """Get dollar value of position in ticker."""
        shares = self.positions.get(ticker, 0.0)
        price = self.prices.get(ticker, 0.0)
        return shares * price
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        position_value = sum(self.get_position_value(ticker) for ticker in self.positions)
        return self.cash + position_value
    
    def get_gross_exposure(self) -> float:
        """Get gross exposure (sum of absolute position values)."""
        return sum(abs(self.get_position_value(ticker)) for ticker in self.positions)
    
    def get_net_exposure(self) -> float:
        """Get net exposure (sum of signed position values)."""
        return sum(self.get_position_value(ticker) for ticker in self.positions)
    
    def update_prices(self, new_prices: Dict[str, float]) -> None:
        """Update current price information."""
        self.prices.update(new_prices)
    
    def execute_trade(self, ticker: str, quantity: float, price: float, commission: float = 0.0) -> None:
        """Execute a trade and update portfolio state."""
        # Update position
        current_shares = self.positions.get(ticker, 0.0)
        self.positions[ticker] = current_shares + quantity
        
        # Update cash (negative quantity = buy, positive = sell in terms of cash flow)
        cash_flow = -quantity * price - commission
        self.cash += cash_flow
        
        # Clean up zero positions
        if abs(self.positions[ticker]) < 1e-8:
            del self.positions[ticker]


# ---------------------------- #
#     BACKTEST ENGINE          #
# ---------------------------- #

class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    The engine coordinates between:
    - Market data feed
    - Signal generation
    - Portfolio construction
    - Order execution
    - Performance tracking
    """
    
    def __init__(
        self,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,
        market_impact: float = 0.0001,
    ):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        start_date : str, datetime, or date
            Backtest start date
        end_date : str, datetime, or date  
            Backtest end date
        initial_capital : float, default 1000000.0
            Starting capital in dollars
        commission_rate : float, default 0.001
            Commission rate (0.001 = 0.1%)
        market_impact : float, default 0.0001
            Market impact rate (0.0001 = 1 basis point)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.market_impact = market_impact
        
        # Event system
        self.event_queue = EventQueue()
        self.current_time = None
        
        # Portfolio state
        self.portfolio_state = PortfolioState(cash=initial_capital)
        
        # Data storage
        self.market_data: Optional[pd.DataFrame] = None
        self.signal_data: Optional[pd.DataFrame] = None
        self.volume_data: Optional[pd.DataFrame] = None
        
        # Strategy components
        self.signal_generator: Optional[Callable] = None
        self.position_sizer: Optional[Callable] = None
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.order_history: List[Dict] = []
        
        # Execution tracking
        self._last_signal_time = None
        self._rebalance_frequency = "monthly"  # daily, weekly, monthly, quarterly
    
    def set_market_data(self, prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> None:
        """
        Set market data for backtesting.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Wide format price data (date × ticker)
        volumes : pd.DataFrame, optional
            Wide format volume data (date × ticker)
        """
        # Filter to backtest date range
        mask = (prices.index >= self.start_date) & (prices.index <= self.end_date)
        self.market_data = prices.loc[mask].copy()
        
        if volumes is not None:
            self.volume_data = volumes.loc[mask].copy()
        
        # Initialize portfolio prices
        if len(self.market_data) > 0:
            first_prices = self.market_data.iloc[0].dropna().to_dict()
            self.portfolio_state.update_prices(first_prices)
    
    def set_signal_data(self, signals: pd.DataFrame) -> None:
        """
        Set pre-computed signal data.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Wide format signal data (date × ticker)
        """
        # Filter to backtest date range  
        mask = (signals.index >= self.start_date) & (signals.index <= self.end_date)
        self.signal_data = signals.loc[mask].copy()
    
    def set_strategy_components(
        self,
        signal_generator: Optional[Callable] = None,
        position_sizer: Optional[Callable] = None,
        rebalance_frequency: str = "monthly"
    ) -> None:
        """
        Set strategy components for dynamic signal generation.
        
        Parameters
        ----------
        signal_generator : callable, optional
            Function that takes (prices, current_time) and returns signals
        position_sizer : callable, optional
            Function that takes (signals, **kwargs) and returns target positions
        rebalance_frequency : str, default "monthly"
            Rebalancing frequency: daily, weekly, monthly, quarterly
        """
        self.signal_generator = signal_generator
        self.position_sizer = position_sizer
        self._rebalance_frequency = rebalance_frequency
    
    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if portfolio should rebalance based on frequency."""
        if self._last_signal_time is None:
            return True
        
        if self._rebalance_frequency == "daily":
            return True
        elif self._rebalance_frequency == "weekly":
            return current_time.weekday() == 0  # Monday
        elif self._rebalance_frequency == "monthly":
            return current_time.day <= 7 and current_time.weekday() == 0  # First Monday
        elif self._rebalance_frequency == "quarterly":
            return current_time.month in [1, 4, 7, 10] and current_time.day <= 7 and current_time.weekday() == 0
        
        return False
    
    def _generate_market_events(self) -> None:
        """Generate market data events for entire backtest period."""
        if self.market_data is None:
            raise ValueError("Market data not set. Call set_market_data() first.")
        
        for timestamp, price_row in self.market_data.iterrows():
            # Skip rows with all NaN prices
            valid_prices = price_row.dropna()
            if len(valid_prices) == 0:
                continue
            
            # Create market event
            price_dict = valid_prices.to_dict()
            volume_dict = None
            
            if self.volume_data is not None and timestamp in self.volume_data.index:
                volume_row = self.volume_data.loc[timestamp]
                volume_dict = volume_row.dropna().to_dict()
            
            market_event = MarketEvent(
                timestamp=timestamp,
                prices=price_dict,
                volumes=volume_dict
            )
            self.event_queue.put(market_event)
    
    def _generate_signal_events(self) -> None:
        """Generate signal events based on signal data or signal generator."""
        if self.signal_data is not None:
            # Use pre-computed signals
            for timestamp, signal_row in self.signal_data.iterrows():
                valid_signals = signal_row.dropna()
                if len(valid_signals) == 0:
                    continue
                
                signal_event = SignalEvent(
                    timestamp=timestamp,
                    signals=valid_signals.to_dict()
                )
                self.event_queue.put(signal_event)
        
        elif self.signal_generator is not None:
            # Generate signals dynamically
            for timestamp in self.market_data.index:
                if self._should_rebalance(timestamp):
                    # Get historical data up to current time
                    hist_data = self.market_data.loc[:timestamp]
                    
                    # Generate signals
                    signals = self.signal_generator(hist_data, timestamp)
                    
                    if signals is not None and len(signals) > 0:
                        if isinstance(signals, pd.Series):
                            signals = signals.dropna().to_dict()
                        
                        signal_event = SignalEvent(
                            timestamp=timestamp,
                            signals=signals
                        )
                        self.event_queue.put(signal_event)
                        self._last_signal_time = timestamp
    
    def _handle_market_event(self, event: MarketEvent) -> None:
        """Process market data event."""
        self.current_time = event.timestamp
        
        # Update portfolio prices
        self.portfolio_state.update_prices(event.prices)
        
        # Record portfolio performance
        portfolio_value = self.portfolio_state.get_total_portfolio_value()
        gross_exposure = self.portfolio_state.get_gross_exposure()
        net_exposure = self.portfolio_state.get_net_exposure()
        
        perf_record = {
            'timestamp': self.current_time,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio_state.cash,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'leverage': gross_exposure / portfolio_value if portfolio_value > 0 else 0.0,
            'num_positions': len(self.portfolio_state.positions)
        }
        
        # Add individual position values
        for ticker, shares in self.portfolio_state.positions.items():
            if ticker in event.prices:
                perf_record[f'position_{ticker}'] = shares * event.prices[ticker]
        
        self.performance_history.append(perf_record)
    
    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Process signal event and generate orders."""
        if self.position_sizer is None:
            # Default equal-weight position sizing
            signals_df = pd.DataFrame([event.signals], index=[event.timestamp])
            target_positions = portfolio.equal_weight_positions(
                signals_df, 
                leverage=1.0,
                max_positions=None,
                min_signal=0.01
            )
            target_weights = target_positions.iloc[0].dropna()
        else:
            # Use custom position sizer
            signals_df = pd.DataFrame([event.signals], index=[event.timestamp])
            target_positions = self.position_sizer(signals_df)
            target_weights = target_positions.iloc[0].dropna()
        
        # Convert target weights to target dollar amounts
        portfolio_value = self.portfolio_state.get_total_portfolio_value()
        target_dollars = target_weights * portfolio_value
        
        # Generate orders for each ticker
        for ticker, target_dollar_amount in target_dollars.items():
            if ticker not in self.portfolio_state.prices:
                continue
            
            current_price = self.portfolio_state.prices[ticker]
            target_shares = target_dollar_amount / current_price
            current_shares = self.portfolio_state.positions.get(ticker, 0.0)
            
            # Calculate required trade
            trade_shares = target_shares - current_shares
            
            # Only place order if trade is significant
            if abs(trade_shares) > 0.001:  # Minimum trade threshold
                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    quantity=trade_shares,
                    order_type="market"
                )
                self.event_queue.put(order_event)
    
    def _handle_order_event(self, event: OrderEvent) -> None:
        """Process order event and generate fill."""
        # Get current price
        if event.ticker not in self.portfolio_state.prices:
            warnings.warn(f"No price available for {event.ticker} at {event.timestamp}")
            return
        
        current_price = self.portfolio_state.prices[event.ticker]
        
        # Calculate execution price with market impact
        impact_factor = 1.0 + (self.market_impact * np.sign(event.quantity))
        execution_price = current_price * impact_factor
        
        # Calculate commission
        trade_value = abs(event.quantity * execution_price)
        commission = trade_value * self.commission_rate
        
        # Record order
        order_record = {
            'timestamp': event.timestamp,
            'ticker': event.ticker,
            'quantity': event.quantity,
            'order_type': event.order_type,
            'limit_price': event.limit_price
        }
        self.order_history.append(order_record)
        
        # Generate fill event
        fill_event = FillEvent(
            timestamp=event.timestamp,
            ticker=event.ticker,
            quantity=event.quantity,
            fill_price=execution_price,
            commission=commission
        )
        self.event_queue.put(fill_event)
    
    def _handle_fill_event(self, event: FillEvent) -> None:
        """Process fill event and update portfolio."""
        # Execute trade in portfolio
        self.portfolio_state.execute_trade(
            ticker=event.ticker,
            quantity=event.quantity,
            price=event.fill_price,
            commission=event.commission
        )
        
        # Record trade
        trade_record = {
            'timestamp': event.timestamp,
            'ticker': event.ticker,
            'quantity': event.quantity,
            'fill_price': event.fill_price,
            'commission': event.commission,
            'trade_value': abs(event.quantity * event.fill_price)
        }
        self.trade_history.append(trade_record)
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtest.
        
        Returns
        -------
        dict
            Backtest results including performance metrics and history
        """
        # Validate setup
        if self.market_data is None:
            raise ValueError("Market data not set. Call set_market_data() first.")
        
        if self.signal_data is None and self.signal_generator is None:
            raise ValueError("Either signal_data or signal_generator must be provided.")
        
        # Clear previous state
        self.event_queue = EventQueue()
        self.portfolio_state = PortfolioState(cash=self.initial_capital)
        self.performance_history = []
        self.trade_history = []
        self.order_history = []
        self._last_signal_time = None
        
        # Generate all events
        print("Generating market events...")
        self._generate_market_events()
        print("Generating signal events...")
        self._generate_signal_events()
        
        # Main event loop
        print(f"Starting backtest from {self.start_date} to {self.end_date}")
        event_count = 0
        
        while not self.event_queue.empty():
            event = self.event_queue.get()
            
            if isinstance(event, MarketEvent):
                self._handle_market_event(event)
            elif isinstance(event, SignalEvent):
                self._handle_signal_event(event)
            elif isinstance(event, OrderEvent):
                self._handle_order_event(event)
            elif isinstance(event, FillEvent):
                self._handle_fill_event(event)
            
            event_count += 1
            if event_count % 1000 == 0:
                print(f"Processed {event_count} events...")
        
        print(f"Backtest complete. Processed {event_count} events.")
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        return results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.performance_history:
            return {"error": "No performance history available"}
        
        # Convert to DataFrame
        perf_df = pd.DataFrame(self.performance_history)
        perf_df.set_index('timestamp', inplace=True)
        
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        
        # Basic performance metrics
        start_value = self.initial_capital
        end_value = perf_df['portfolio_value'].iloc[-1]
        total_return = (end_value / start_value) - 1.0
        
        # Calculate daily returns
        daily_returns = perf_df['portfolio_value'].pct_change().dropna()
        
        # Risk metrics
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Drawdown analysis
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trading statistics
        num_trades = len(trades_df)
        total_commission = trades_df['commission'].sum() if num_trades > 0 else 0.0
        avg_trade_size = trades_df['trade_value'].mean() if num_trades > 0 else 0.0
        
        # Turnover calculation
        if len(perf_df) > 1:
            gross_exposures = perf_df['gross_exposure']
            portfolio_values = perf_df['portfolio_value']
            avg_turnover = (gross_exposures.diff().abs() / portfolio_values).mean()
        else:
            avg_turnover = 0.0
        
        results = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_value': end_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'total_commission': total_commission,
            'avg_trade_size': avg_trade_size,
            'avg_turnover': avg_turnover,
            'performance_history': perf_df,
            'trade_history': trades_df,
            'order_history': pd.DataFrame(self.order_history) if self.order_history else pd.DataFrame()
        }
        
        return results


# ---------------------------- #
#        CONVENIENCE API       #
# ---------------------------- #

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    *,
    initial_capital: float = 1000000.0,
    position_sizing: str = "equal_weight",
    leverage: float = 1.0,
    max_positions: Optional[int] = None,
    commission_rate: float = 0.001,
    market_impact: float = 0.0001,
    rebalance_frequency: str = "monthly",
    **position_sizing_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run a complete backtest.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide format price data (date × ticker)
    signals : pd.DataFrame  
        Wide format signal data (date × ticker)
    start_date, end_date : str, datetime, or date
        Backtest period
    initial_capital : float, default 1000000.0
        Starting capital
    position_sizing : str, default "equal_weight"
        Position sizing method: "equal_weight", "signal_weight", "volatility_adjusted"
    leverage : float, default 1.0
        Target leverage
    max_positions : int, optional
        Maximum number of positions
    commission_rate : float, default 0.001
        Commission rate (0.1%)
    market_impact : float, default 0.0001  
        Market impact rate (1 basis point)
    rebalance_frequency : str, default "monthly"
        Rebalancing frequency
    **position_sizing_kwargs
        Additional arguments for position sizing function
        
    Returns
    -------
    dict
        Backtest results and performance metrics
    """
    # Initialize backtest engine
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        market_impact=market_impact
    )
    
    # Set market data
    engine.set_market_data(prices)
    
    # Set up position sizing
    if position_sizing == "equal_weight":
        position_sizer = lambda sigs: portfolio.equal_weight_positions(
            sigs, leverage=leverage, max_positions=max_positions, **position_sizing_kwargs
        )
    elif position_sizing == "signal_weight":
        position_sizer = lambda sigs: portfolio.signal_weighted_positions(
            sigs, leverage=leverage, max_weight=0.20, **position_sizing_kwargs
        )
    elif position_sizing == "volatility_adjusted":
        # Would need volatility data - simplified for now
        position_sizer = lambda sigs: portfolio.equal_weight_positions(
            sigs, leverage=leverage, max_positions=max_positions, **position_sizing_kwargs
        )
    else:
        raise ValueError(f"Unknown position_sizing method: {position_sizing}")
    
    # Set strategy components
    engine.set_strategy_components(
        position_sizer=position_sizer,
        rebalance_frequency=rebalance_frequency
    )
    
    # Set signal data
    engine.set_signal_data(signals)
    
    # Run backtest
    results = engine.run_backtest()
    
    return results
