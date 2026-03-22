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


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect Order Blocks (OB).
    An OB is the last opposite candle before a strong move.
    """
    ob = pd.Series(0, index=df.index)

    for i in range(lookback, len(df)):
        # Check for strong move in next few bars
        future_high = df["high"].iloc[i:i+5].max()
        future_low = df["low"].iloc[i:i+5].min()
        current_close = df["close"].iloc[i]

        # Strong bullish move
        if future_high > current_close * 1.01:
            if df["close"].iloc[i] < df["open"].iloc[i]:
                ob.iloc[i] = 1

        # Strong bearish move
        if future_low < current_close * 0.99:
            if df["close"].iloc[i] > df["open"].iloc[i]:
                ob.iloc[i] = -1

    return ob


def detect_fvg(
    df: pd.DataFrame,
    min_size: float = 0.002,
) -> pd.Series:
    """
    Detect Fair Value Gaps (FVG).
    A FVG occurs when there's a gap between candles.
    """
    fvg = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        # Bullish FVG
        if df["high"].iloc[i-2] < df["low"].iloc[i]:
            gap_size = (df["low"].iloc[i] - df["high"].iloc[i-2]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = 1

        # Bearish FVG
        if df["low"].iloc[i-2] > df["high"].iloc[i]:
            gap_size = (df["low"].iloc[i-2] - df["high"].iloc[i]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = -1

    return fvg


def detect_liquidity_sweep(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Detect liquidity sweeps.
    A sweep occurs when price breaks a swing high/low then reverses.
    """
    sweep = pd.Series(0, index=df.index)

    for i in range(5, len(df) - 5):
        recent_high = df["high"].iloc[i-5:i].max()
        if df["high"].iloc[i] > recent_high:
            if df["close"].iloc[i] < recent_high * (1 - threshold):
                sweep.iloc[i] = -1

        recent_low = df["low"].iloc[i-5:i].min()
        if df["low"].iloc[i] < recent_low:
            if df["close"].iloc[i] > recent_low * (1 + threshold):
                sweep.iloc[i] = 1

    return sweep


def smc_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate SMC (Smart Money Concept) trading signals.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for SMC detection

    Returns:
        Dict with OB, FVG, liquidity sweep, and combined signal
    """
    if params is None:
        params = DEFAULT_SMC_PARAMS

    # Detect patterns
    order_block = detect_order_blocks(df, params["ob_lookback"])
    fvg = detect_fvg(df, params["fvg_min_size"])
    sweep = detect_liquidity_sweep(df, params["sweep_threshold"])

    # Generate combined signal
    signal = pd.Series(0, index=df.index)

    bullish = ((order_block == 1) & (fvg == 1)) | (sweep == 1)
    bearish = ((order_block == -1) & (fvg == -1)) | (sweep == -1)

    signal[bullish] = 1
    signal[bearish] = -1

    return {
        "order_block": order_block,
        "fvg": fvg,
        "liquidity_sweep": sweep,
        "signal": signal,
    }