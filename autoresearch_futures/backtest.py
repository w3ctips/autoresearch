"""
Backtest engine for futures strategy evaluation.
Calculates performance metrics with trading costs.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from autoresearch_futures.config import SCORE_WEIGHTS, BacktestConfig

# Re-export for convenience
__all__ = [
    "BacktestResult",
    "calc_score",
    "calc_commission",
    "calc_slippage",
    "calc_total_cost",
    "run_backtest",
    "run_multi_backtest",
    "SCORE_WEIGHTS",
]


@dataclass
class BacktestResult:
    """Container for backtest results and metrics."""
    # Return metrics
    net_return: float          # Net return after costs
    annual_return: float       # Annualized return

    # Risk metrics
    max_drawdown: float        # Maximum drawdown
    volatility: float          # Return volatility
    var_95: float              # 95% Value at Risk

    # Risk-adjusted metrics
    sharpe_ratio: float        # Sharpe ratio
    calmar_ratio: float        # Calmar ratio
    sortino_ratio: float       # Sortino ratio

    # Trade statistics
    total_trades: int          # Total number of trades
    win_rate: float            # Winning trade percentage
    profit_factor: float       # Gross profit / Gross loss
    avg_holding_bars: float    # Average holding period in bars

    # Signal quality
    signal_precision: float    # Precision of signals
    signal_recall: float       # Recall of profitable opportunities


def calc_score(result: BacktestResult, weights: Dict[str, float]) -> float:
    """
    Calculate weighted composite score.

    Score = w1*sharpe + w2*net_return + w3*win_rate + w4*precision - w5*drawdown
    """
    return (
        weights["sharpe"] * result.sharpe_ratio +
        weights["net_return"] * result.net_return +
        weights["win_rate"] * result.win_rate +
        weights["precision"] * result.signal_precision -
        weights["drawdown"] * result.max_drawdown
    )


def calc_commission(trade_value: float, commission_rate: float) -> float:
    """
    Calculate commission cost.

    Args:
        trade_value: Total trade value in yuan
        commission_rate: Commission rate (e.g., 0.0001 = 万分之一)

    Returns:
        Commission cost in yuan
    """
    return trade_value * commission_rate


def calc_slippage(volume: int, tick_size: float, slippage_ticks: int = 1) -> float:
    """
    Calculate slippage cost.

    Assumes slippage affects both entry and exit.

    Args:
        volume: Number of contracts
        tick_size: Minimum price tick
        slippage_ticks: Number of ticks for slippage

    Returns:
        Slippage cost in yuan (per contract multiplier = 1)
    """
    # Slippage affects both entry and exit, and is per tick
    return volume * tick_size * 2 * slippage_ticks


def calc_total_cost(
    trade_value: float,
    volume: int,
    tick_size: float,
    commission_rate: float = None,
    slippage_ticks: int = None,
) -> float:
    """
    Calculate total trading cost.

    Args:
        trade_value: Total trade value
        volume: Number of contracts
        tick_size: Price tick size
        commission_rate: Commission rate (default from config)
        slippage_ticks: Slippage in ticks (default from config)

    Returns:
        Total trading cost in yuan
    """
    if commission_rate is None:
        commission_rate = BacktestConfig.commission_rate
    if slippage_ticks is None:
        slippage_ticks = BacktestConfig.slippage_ticks

    commission = calc_commission(trade_value, commission_rate)
    slippage = calc_slippage(volume, tick_size, slippage_ticks)

    return commission + slippage


def run_backtest(
    signals: pd.Series,
    data: pd.DataFrame,
    tick_size: float = 1.0,
    contract_multiplier: int = 10,
    initial_capital: float = None,
    commission_rate: float = None,
    slippage_ticks: int = None,
) -> BacktestResult:
    """
    Run backtest on a single symbol.

    Args:
        signals: Series of signals (-1, 0, 1) indexed by bar
        data: DataFrame with OHLCV data
        tick_size: Minimum price tick
        contract_multiplier: Contract size multiplier
        initial_capital: Starting capital (default from config)
        commission_rate: Commission rate (default from config)
        slippage_ticks: Slippage ticks (default from config)

    Returns:
        BacktestResult with performance metrics
    """
    if initial_capital is None:
        initial_capital = BacktestConfig.initial_capital
    if commission_rate is None:
        commission_rate = BacktestConfig.commission_rate
    if slippage_ticks is None:
        slippage_ticks = BacktestConfig.slippage_ticks

    # Track position and trades
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]
    capital = initial_capital

    # Iterate through bars
    for i in range(len(data)):
        signal = signals.iloc[i] if i < len(signals) else 0
        close_price = data["close"].iloc[i]

        # Open or close position based on signal
        if signal == 1 and position == 0:  # Buy
            position = 1
            entry_price = close_price
            # Deduct cost
            trade_value = close_price * contract_multiplier
            cost = calc_total_cost(trade_value, 1, tick_size, commission_rate, slippage_ticks)
            capital -= cost

        elif signal == -1 and position == 1:  # Sell (close long)
            pnl = (close_price - entry_price) * contract_multiplier
            trade_value = close_price * contract_multiplier
            cost = calc_total_cost(trade_value, 1, tick_size, commission_rate, slippage_ticks)
            net_pnl = pnl - cost
            capital += net_pnl

            trades.append({
                "entry_price": entry_price,
                "exit_price": close_price,
                "pnl": pnl,
                "net_pnl": net_pnl,
            })

            position = 0
            entry_price = 0

        # Update equity
        if position == 1:
            unrealized_pnl = (close_price - entry_price) * contract_multiplier
            equity_curve.append(capital + unrealized_pnl)
        else:
            equity_curve.append(capital)

    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    # Basic metrics
    net_return = (capital - initial_capital) / initial_capital
    total_trades = len(trades) * 2  # Each round trip = 2 trades

    # Win rate
    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Sharpe ratio (assuming 252 trading days, 16 hours/day, 4 bars/hour)
    trading_bars_per_year = 252 * 16 * 4
    annual_return = net_return * (trading_bars_per_year / len(data))
    volatility = returns.std() * np.sqrt(trading_bars_per_year) if len(returns) > 0 else 0
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_bars_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # VaR (95%)
    var_95 = abs(np.percentile(returns, 5)) if len(returns) > 0 else 0

    # Average holding bars
    avg_holding_bars = len(data) / len(trades) if trades else 0

    # Signal quality (simplified)
    signal_bars = (signals != 0).sum()
    signal_precision = win_rate  # Simplified
    signal_recall = len(trades) / max(signal_bars / 2, 1) if signal_bars > 0 else 0

    return BacktestResult(
        net_return=net_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        var_95=var_95,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_holding_bars=avg_holding_bars,
        signal_precision=signal_precision,
        signal_recall=signal_recall,
    )


def run_multi_backtest(
    signals_dict: dict,
    data_dict: dict,
    tick_sizes: dict,
    contract_multipliers: dict,
    **kwargs,
) -> dict:
    """
    Run backtest on multiple symbols.

    Args:
        signals_dict: Dict mapping symbol to signal Series
        data_dict: Dict mapping symbol to price DataFrame
        tick_sizes: Dict mapping symbol to tick size
        contract_multipliers: Dict mapping symbol to contract multiplier
        **kwargs: Additional arguments passed to run_backtest

    Returns:
        Dict mapping symbol to BacktestResult
    """
    results = {}
    for symbol in signals_dict:
        if symbol not in data_dict:
            continue
        results[symbol] = run_backtest(
            signals=signals_dict[symbol],
            data=data_dict[symbol],
            tick_size=tick_sizes.get(symbol, 1.0),
            contract_multiplier=contract_multipliers.get(symbol, 10),
            **kwargs,
        )
    return results