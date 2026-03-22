"""
Signal generation module for futures trading strategies.
Implements SMC, momentum, and linear extrapolation signals.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


# Default parameters for each theory
DEFAULT_MOMENTUM_PARAMS = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}

DEFAULT_SMC_PARAMS = {
    "ob_lookback": 20,
    "fvg_min_size": 0.002,
    "sweep_threshold": 0.01,
}

DEFAULT_LINEAR_PARAMS = {
    "regression_period": 20,
    "band_std": 2.0,
    "breakout_confirm": 3,
}


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """Calculate MACD indicator."""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def momentum_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate momentum-based trading signals.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for momentum indicators

    Returns:
        Dict with RSI, MACD, velocity, and combined signal
    """
    if params is None:
        params = DEFAULT_MOMENTUM_PARAMS

    close = df["close"]

    # RSI
    rsi = calc_rsi(close, params["rsi_period"])

    # MACD
    macd, macd_signal, macd_hist = calc_macd(
        close,
        params["macd_fast"],
        params["macd_slow"],
        params["macd_signal"],
    )

    # Price velocity (rate of change)
    velocity = close.pct_change(params["rsi_period"]) * 100

    # Generate signals
    signal = pd.Series(0, index=df.index)

    # RSI signals
    rsi_buy = rsi < params["rsi_oversold"]
    rsi_sell = rsi > params["rsi_overbought"]

    # MACD signals
    macd_buy = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    macd_sell = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))

    # Combine signals
    signal[rsi_buy | macd_buy] = 1
    signal[rsi_sell | macd_sell] = -1

    return {
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "velocity": velocity,
        "signal": signal,
    }