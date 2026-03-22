"""
Multi-theory ensemble module.
Combines signals from SMC, momentum, and linear theories.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""
    total = sum(weights.values())
    if total == 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def simple_vote(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
    threshold: float = 0.3,  # 降低阈值，使单个理论也能产生信号
) -> pd.Series:
    """
    Combine signals using weighted voting.

    Args:
        signals: Dict of signal Series (values: -1, 0, 1)
        weights: Dict of theory weights
        threshold: Minimum weighted sum to generate a signal

    Returns:
        Combined signal Series
    """
    weights = normalize_weights(weights)
    weighted_sum = pd.Series(0.0, index=next(iter(signals.values())).index)
    for theory, signal in signals.items():
        weighted_sum += signal * weights.get(theory, 0)

    result = pd.Series(0, index=weighted_sum.index)
    result[weighted_sum >= threshold] = 1
    result[weighted_sum <= -threshold] = -1

    return result


def consensus_filter(signals: Dict[str, pd.Series]) -> pd.Series:
    """Only generate signal when all theories agree."""
    signal_list = list(signals.values())
    result = pd.Series(0, index=signal_list[0].index)

    for i in range(len(result)):
        values = [s.iloc[i] for s in signal_list]
        if all(v == 1 for v in values):
            result.iloc[i] = 1
        elif all(v == -1 for v in values):
            result.iloc[i] = -1

    return result


def any_signal(signals: Dict[str, pd.Series]) -> pd.Series:
    """
    Generate signal when any theory has a signal.
    Less restrictive than consensus, more signals.
    """
    signal_list = list(signals.values())
    result = pd.Series(0, index=signal_list[0].index)

    for i in range(len(result)):
        values = [s.iloc[i] for s in signal_list]
        buy_signals = sum(1 for v in values if v == 1)
        sell_signals = sum(1 for v in values if v == -1)

        if buy_signals > sell_signals and buy_signals > 0:
            result.iloc[i] = 1
        elif sell_signals > buy_signals and sell_signals > 0:
            result.iloc[i] = -1

    return result


def calc_confidence(
    signals_at_point: Dict[str, int],
    weights: Dict[str, float],
) -> float:
    """Calculate confidence based on signal agreement."""
    weights = normalize_weights(weights)
    total_weight = sum(abs(v) * weights.get(k, 0) for k, v in signals_at_point.items())
    non_zero_signals = [v for v in signals_at_point.values() if v != 0]
    if len(non_zero_signals) == 0:
        return 0.0

    direction = non_zero_signals[0]
    agreement = all(s == direction for s in non_zero_signals)

    if agreement:
        return min(total_weight, 1.0)
    else:
        return total_weight * 0.5


def ensemble_signals(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
    mode: str = "vote",
) -> Dict:
    """
    Combine signals from multiple theories.

    Args:
        signals: Dict of signal Series from each theory
        weights: Dict of theory weights
        mode: Combination mode
            - "vote": Weighted voting (default)
            - "consensus": Require all theories to agree
            - "any": Any theory with signal

    Returns:
        Dict with combined 'signal' and 'confidence'
    """
    if mode == "vote":
        combined = simple_vote(signals, weights)
    elif mode == "consensus":
        combined = consensus_filter(signals)
    elif mode == "any":
        combined = any_signal(signals)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    confidence = pd.Series(0.0, index=combined.index)
    for i in range(len(combined)):
        signals_at_i = {k: v.iloc[i] for k, v in signals.items()}
        confidence.iloc[i] = calc_confidence(signals_at_i, weights)

    return {
        "signal": combined,
        "confidence": confidence,
    }