"""
Backtest engine for futures strategy evaluation.
Calculates performance metrics with trading costs.
"""
from dataclasses import dataclass
from typing import Dict

from autoresearch_futures.config import SCORE_WEIGHTS

# Re-export for convenience
__all__ = ["BacktestResult", "calc_score", "SCORE_WEIGHTS"]


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