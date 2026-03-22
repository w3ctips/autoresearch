"""
Evolution main loop for futures strategy research.
Orchestrates signal generation, backtesting, and strategy selection.
"""
import os
from typing import Dict, List, Optional
import pandas as pd

from autoresearch_futures.config import DEFAULT_PARAMS, THEORY_WEIGHTS, SCORE_WEIGHTS
from autoresearch_futures.signals import momentum_signals, smc_signals, linear_signals
from autoresearch_futures.ensemble import ensemble_signals
from autoresearch_futures.backtest import run_multi_backtest, calc_score, BacktestResult


def generate_all_signals(
    data_dict: Dict[str, pd.DataFrame],
    params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Generate signals from all theories for all symbols."""
    if params is None:
        params = DEFAULT_PARAMS

    all_signals = {}
    for symbol, df in data_dict.items():
        smc = smc_signals(df, params.get("smc"))
        momentum = momentum_signals(df, params.get("momentum"))
        linear = linear_signals(df, params.get("linear"))

        combined = ensemble_signals(
            {"smc": smc["signal"], "momentum": momentum["signal"], "linear": linear["signal"]},
            THEORY_WEIGHTS,
        )
        all_signals[symbol] = {
            "smc": smc, "momentum": momentum, "linear": linear,
            "signal": combined["signal"], "confidence": combined["confidence"],
        }
    return all_signals


def aggregate_scores(results: Dict[str, BacktestResult], weights: Optional[Dict] = None) -> float:
    """Calculate aggregate score across multiple symbols."""
    if weights is None:
        weights = SCORE_WEIGHTS
    scores = [calc_score(r, weights) for r in results.values()]
    return sum(scores) / len(scores) if scores else 0.0


def log_results(
    split_id: str, commit: str, score: float, results: Dict[str, BacktestResult],
    description: str, filepath: str = "results.tsv",
) -> None:
    """Log evolution results to TSV file."""
    avg_sharpe = sum(r.sharpe_ratio for r in results.values()) / len(results)
    avg_return = sum(r.net_return for r in results.values()) / len(results)
    avg_win_rate = sum(r.win_rate for r in results.values()) / len(results)
    max_dd = max(r.max_drawdown for r in results.values())

    write_header = not os.path.exists(filepath)
    with open(filepath, "a") as f:
        if write_header:
            f.write("split_id\tcommit\tscore\tsharpe\tnet_return\twin_rate\tmax_dd\tdescription\n")
        f.write(f"{split_id}\t{commit}\t{score:.6f}\t{avg_sharpe:.4f}\t{avg_return:.4f}\t{avg_win_rate:.4f}\t{max_dd:.4f}\t{description}\n")


def run_evolution_step(
    data_dict: Dict[str, pd.DataFrame], params: Dict,
    tick_sizes: Dict[str, float], contract_multipliers: Dict[str, int],
) -> tuple:
    """Run a single evolution step. Returns (score, results_dict, signals_dict)."""
    signals_dict = generate_all_signals(data_dict, params)
    backtest_signals = {symbol: signals_dict[symbol]["signal"] for symbol in signals_dict}
    results = run_multi_backtest(backtest_signals, data_dict, tick_sizes, contract_multipliers)
    score = aggregate_scores(results)
    return score, results, signals_dict