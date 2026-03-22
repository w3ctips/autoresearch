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

    # 需要足够的未来数据进行检测
    for i in range(lookback, len(df) - 5):
        # Check for strong move in next 5 bars
        future_high = df["high"].iloc[i:i+5].max()
        future_low = df["low"].iloc[i:i+5].min()
        current_close = df["close"].iloc[i]

        # Strong bullish move (1% gain)
        if future_high > current_close * 1.005:
            # Bearish candle before bullish move
            if df["close"].iloc[i] < df["open"].iloc[i]:
                ob.iloc[i] = 1

        # Strong bearish move (1% drop)
        if future_low < current_close * 0.995:
            # Bullish candle before bearish move
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
        # Bullish FVG: gap between candle i-2's high and candle i's low
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            gap_size = (df["low"].iloc[i] - df["high"].iloc[i-2]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = 1

        # Bearish FVG: gap between candle i-2's low and candle i's high
        if df["high"].iloc[i] < df["low"].iloc[i-2]:
            gap_size = (df["low"].iloc[i-2] - df["high"].iloc[i]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = -1

    return fvg


def detect_liquidity_sweep(
    df: pd.DataFrame,
    threshold: float = 0.005,  # 降低阈值
) -> pd.Series:
    """
    Detect liquidity sweeps.
    A sweep occurs when price breaks a swing high/low then reverses.
    """
    sweep = pd.Series(0, index=df.index)

    for i in range(10, len(df) - 5):  # 需要前后数据
        # Check for sweep above recent high
        recent_high = df["high"].iloc[i-10:i].max()
        if df["high"].iloc[i] > recent_high:
            # Price broke high but closed below it (reversal)
            if df["close"].iloc[i] < recent_high * (1 - threshold):
                sweep.iloc[i] = -1  # Bearish sweep

        # Check for sweep below recent low
        recent_low = df["low"].iloc[i-10:i].min()
        if df["low"].iloc[i] < recent_low:
            # Price broke low but closed above it (reversal)
            if df["close"].iloc[i] > recent_low * (1 + threshold):
                sweep.iloc[i] = 1  # Bullish sweep

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

    # Generate combined signal - 使用更宽松的条件
    signal = pd.Series(0, index=df.index)

    # Bullish: OB bullish OR FVG bullish OR sweep bullish
    bullish = (order_block == 1) | (fvg == 1) | (sweep == 1)

    # Bearish: OB bearish OR FVG bearish OR sweep bearish
    bearish = (order_block == -1) | (fvg == -1) | (sweep == -1)

    signal[bullish] = 1
    signal[bearish] = -1

    return {
        "order_block": order_block,
        "fvg": fvg,
        "liquidity_sweep": sweep,
        "signal": signal,
    }


def calc_regression_band(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict:
    """
    Calculate linear regression channel.

    Returns dict with regression line, upper and lower bands.
    """
    regression = close.rolling(window=period).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] * (len(x) - 1) + x.iloc[0],
        raw=False,
    )

    residuals = close - regression
    std = residuals.rolling(window=period).std()

    upper = regression + num_std * std
    lower = regression - num_std * std

    return {
        "regression": regression,
        "upper": upper,
        "lower": lower,
        "std": std,
    }


def detect_trend(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    Detect trend direction using linear regression slope.
    """
    close = df["close"]
    slopes = close.rolling(window=period).apply(
        lambda x: stats.linregress(range(len(x)), x)[0],
        raw=False,
    )

    normalized_slope = slopes / close * 100

    trend = pd.Series(0, index=df.index)
    trend[normalized_slope > 0.01] = 1
    trend[normalized_slope < -0.01] = -1

    return trend


def detect_breakout(
    df: pd.DataFrame,
    band: dict,
    confirm_bars: int = 3,
) -> pd.Series:
    """
    Detect breakouts from regression channel.
    """
    close = df["close"]
    breakout = pd.Series(0, index=df.index)

    above_upper = close > band["upper"]
    for i in range(confirm_bars, len(df)):
        if above_upper.iloc[i-confirm_bars:i].all():
            breakout.iloc[i] = 1

    below_lower = close < band["lower"]
    for i in range(confirm_bars, len(df)):
        if below_lower.iloc[i-confirm_bars:i].all():
            breakout.iloc[i] = -1

    return breakout


def linear_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate linear extrapolation trading signals.

    Uses linear regression channels and trend analysis.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for linear signals

    Returns:
        Dict with trend, regression band, and combined signal
    """
    if params is None:
        params = DEFAULT_LINEAR_PARAMS

    close = df["close"]

    band = calc_regression_band(
        close,
        params["regression_period"],
        params["band_std"],
    )

    trend = detect_trend(df, params["regression_period"])
    breakout = detect_breakout(df, band, params["breakout_confirm"])

    signal = pd.Series(0, index=df.index)

    buy = (trend == 1) & (close <= band["regression"])
    buy = buy | (breakout == 1)

    sell = (trend == -1) & (close >= band["regression"])
    sell = sell | (breakout == -1)

    signal[buy] = 1
    signal[sell] = -1

    return {
        "trend": trend,
        "regression_band": band,
        "breakout": breakout,
        "signal": signal,
    }