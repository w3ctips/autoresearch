"""
Backtest engine for futures strategy evaluation.
Calculates performance metrics with trading costs.
"""
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