"""
Real-time monitoring module for futures trading signals.
Polls K-line data during trading hours and generates signals.
"""
import time
import schedule
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Callable
import pandas as pd

from autoresearch_futures.config import DEFAULT_PARAMS, THEORY_WEIGHTS, Config
from autoresearch_futures.signals import momentum_signals, smc_signals, linear_signals
from autoresearch_futures.ensemble import ensemble_signals
from autoresearch_futures.notify import push_signal, SignalEvent, NotifyConfig


# 中国期货交易时间段
TRADING_HOURS = {
    "day": {
        "start": dt_time(9, 0),
        "end": dt_time(11, 30),
    },
    "afternoon": {
        "start": dt_time(13, 30),
        "end": dt_time(15, 0),
    },
    "night": {
        "start": dt_time(21, 0),
        "end": dt_time(23, 0),  # 部分品种到次日02:30
    },
}

# 夜盘品种及其收盘时间
NIGHT_SESSION_END = {
    # 上期所
    "cu": dt_time(1, 0),  # 铜
    "al": dt_time(1, 0),  # 铝
    "zn": dt_time(1, 0),  # 锌
    "pb": dt_time(1, 0),  # 铅
    "ni": dt_time(1, 0),  # 镍
    "sn": dt_time(1, 0),  # 锡
    "au": dt_time(2, 30), # 黄金
    "ag": dt_time(2, 30), # 白银
    "rb": dt_time(23, 0), # 螺纹钢
    "hc": dt_time(23, 0), # 热卷
    "ss": dt_time(23, 0), # 不锈钢
    "fu": dt_time(23, 0), # 燃油
    "bu": dt_time(23, 0), # 沥青
    "ru": dt_time(23, 0), # 橡胶
    "sp": dt_time(23, 0), # 纸浆
    # 大商所
    "i": dt_time(23, 0),  # 铁矿石
    "j": dt_time(23, 0),  # 焦炭
    "jm": dt_time(23, 0), # 焦煤
    "p": dt_time(23, 0),  # 棕榈油
    "y": dt_time(23, 0),  # 豆油
    "m": dt_time(23, 0),  # 豆粕
    "c": dt_time(23, 0),  # 玉米
    "l": dt_time(23, 0),  # 塑料
    "v": dt_time(23, 0),  # PVC
    "pp": dt_time(23, 0), # 聚丙烯
    "eb": dt_time(23, 0), # 苯乙烯
    "eg": dt_time(23, 0), # 乙二醇
    "pg": dt_time(23, 0), # 液化石油气
    # 郑商所
    "cf": dt_time(23, 0), # 棉花
    "sr": dt_time(23, 0), # 白糖
    "ta": dt_time(23, 0), # PTA
    "ma": dt_time(23, 0), # 甲醇
    "fg": dt_time(23, 0), # 玻璃
    "sa": dt_time(23, 0), # 纯碱
}


def is_trading_time(symbol: str = None) -> bool:
    """
    Check if current time is within trading hours.

    Args:
        symbol: Optional symbol to check specific night session end time

    Returns:
        True if within trading hours
    """
    now = datetime.now()
    current_time = now.time()

    # 日盘: 9:00 - 11:30
    if TRADING_HOURS["day"]["start"] <= current_time <= TRADING_HOURS["day"]["end"]:
        return True

    # 下午盘: 13:30 - 15:00
    if TRADING_HOURS["afternoon"]["start"] <= current_time <= TRADING_HOURS["afternoon"]["end"]:
        return True

    # 夜盘: 21:00 开始
    if current_time >= TRADING_HOURS["night"]["start"]:
        return True

    # 夜盘结束检查 (跨日)
    if symbol and symbol.lower() in NIGHT_SESSION_END:
        end_time = NIGHT_SESSION_END[symbol.lower()]
        if current_time <= end_time:
            return True
    elif current_time <= dt_time(1, 0):  # 默认夜盘到1:00
        return True

    return False


def get_next_trading_time() -> Optional[datetime]:
    """
    Get the next trading session start time.

    Returns:
        Datetime of next trading session, or None if currently trading
    """
    now = datetime.now()
    current_time = now.time()

    # 如果当前在交易时间
    if is_trading_time():
        return None

    # 日盘开始
    day_start = datetime.combine(now.date(), TRADING_HOURS["day"]["start"])

    # 下午盘开始
    afternoon_start = datetime.combine(now.date(), TRADING_HOURS["afternoon"]["start"])

    # 夜盘开始
    night_start = datetime.combine(now.date(), TRADING_HOURS["night"]["start"])

    # 判断下一个交易时间
    if current_time < TRADING_HOURS["day"]["start"]:
        return day_start
    elif current_time < TRADING_HOURS["afternoon"]["start"]:
        return afternoon_start
    elif current_time < TRADING_HOURS["night"]["start"]:
        return night_start
    else:
        # 夜盘结束后，下一个交易日日盘
        from datetime import timedelta
        next_day = now + timedelta(days=1)
        return datetime.combine(next_day.date(), TRADING_HOURS["day"]["start"])


class RealtimeMonitor:
    """
    Real-time K-line monitor for futures trading signals.
    """

    def __init__(
        self,
        symbols: List[str],
        params: Dict = None,
        weights: Dict = None,
        poll_interval_minutes: int = 15,
        on_signal: Optional[Callable] = None,
        notify_config: Optional[NotifyConfig] = None,
    ):
        """
        Initialize the monitor.

        Args:
            symbols: List of futures symbols to monitor
            params: Signal parameters (default: DEFAULT_PARAMS)
            weights: Theory weights (default: THEORY_WEIGHTS)
            poll_interval_minutes: K-line polling interval
            on_signal: Callback function for signal events
            notify_config: Notification configuration for push notifications
        """
        self.symbols = symbols
        self.params = params or DEFAULT_PARAMS
        self.weights = weights or THEORY_WEIGHTS
        self.poll_interval = poll_interval_minutes
        self.on_signal = on_signal
        self.notify_config = notify_config or NotifyConfig()

        self._running = False
        self._last_signals: Dict[str, int] = {}  # Track last signal per symbol
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch latest K-line data for a symbol.

        Args:
            symbol: Futures symbol

        Returns:
            DataFrame with latest data
        """
        try:
            from autoresearch_futures.prepare import download_minute_data

            df = download_minute_data(symbol, period="15")
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Futures symbol
            df: K-line DataFrame

        Returns:
            Dict with signal info
        """
        # Generate signals from each theory
        smc = smc_signals(df, self.params.get("smc"))
        momentum = momentum_signals(df, self.params.get("momentum"))
        linear = linear_signals(df, self.params.get("linear"))

        # Ensemble
        combined = ensemble_signals(
            {
                "smc": smc["signal"],
                "momentum": momentum["signal"],
                "linear": linear["signal"],
            },
            self.weights,
        )

        signal = combined["signal"]
        confidence = combined["confidence"]

        # Get latest signal
        latest_signal = int(signal.iloc[-1]) if len(signal) > 0 else 0
        latest_confidence = float(confidence.iloc[-1]) if len(confidence) > 0 else 0.0

        # Get supporting indicators
        latest_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        latest_time = df["datetime"].iloc[-1] if len(df) > 0 else None

        return {
            "symbol": symbol,
            "signal": latest_signal,
            "confidence": latest_confidence,
            "price": latest_price,
            "time": latest_time,
            "smc_signal": int(smc["signal"].iloc[-1]) if len(smc["signal"]) > 0 else 0,
            "momentum_signal": int(momentum["signal"].iloc[-1]) if len(momentum["signal"]) > 0 else 0,
            "linear_signal": int(linear["signal"].iloc[-1]) if len(linear["signal"]) > 0 else 0,
        }

    def check_and_notify(self, signal_info: Dict) -> Optional[SignalEvent]:
        """
        Check if signal changed and send notification.

        Args:
            signal_info: Signal information dict

        Returns:
            SignalEvent if signal changed, None otherwise
        """
        symbol = signal_info["symbol"]
        signal = signal_info["signal"]
        last_signal = self._last_signals.get(symbol, 0)

        # Signal changed?
        if signal != last_signal and signal != 0:
            # Build sources list
            sources = []
            if signal_info["smc_signal"] != 0:
                sources.append(f"SMC ({'做多' if signal_info['smc_signal'] == 1 else '做空'})")
            if signal_info["momentum_signal"] != 0:
                sources.append(f"动能 ({'做多' if signal_info['momentum_signal'] == 1 else '做空'})")
            if signal_info["linear_signal"] != 0:
                sources.append(f"线性 ({'做多' if signal_info['linear_signal'] == 1 else '做空'})")

            event = SignalEvent(
                symbol=symbol,
                direction=signal,  # 1 for buy, -1 for sell
                timestamp=signal_info["time"] if isinstance(signal_info["time"], datetime) else datetime.now(),
                price=signal_info["price"],
                confidence=signal_info["confidence"],
                sources=sources,
            )

            # Update last signal
            self._last_signals[symbol] = signal

            return event

        return None

    def poll_once(self) -> Dict[str, Dict]:
        """
        Execute one polling cycle.

        Returns:
            Dict mapping symbol to signal info
        """
        results = {}

        for symbol in self.symbols:
            # Check if trading time for this symbol
            if not is_trading_time(symbol):
                continue

            # Fetch data
            df = self.fetch_latest_data(symbol)
            if df is None or df.empty:
                continue

            # Generate signal
            signal_info = self.generate_signal(symbol, df)
            results[symbol] = signal_info

            # Check for signal change
            event = self.check_and_notify(signal_info)

            if event:
                # Call callback if provided
                if self.on_signal:
                    self.on_signal(event)

                # Push notification
                push_signal(event, self.notify_config)

                # Print to console
                direction_text = "做多" if event.direction == 1 else "做空"
                print(f"\n{'='*50}")
                print(f"信号触发: {event.symbol}")
                print(f"方向: {direction_text}")
                print(f"价格: {event.price:.2f}")
                print(f"置信度: {event.confidence:.2f}")
                print(f"时间: {event.timestamp}")
                print(f"{'='*50}\n")

        return results

    def run(self, continuous: bool = True):
        """
        Start the monitoring loop.

        Args:
            continuous: If True, run continuously; if False, run once
        """
        print(f"启动实时监控...")
        print(f"监控品种: {', '.join(self.symbols)}")
        print(f"轮询间隔: {self.poll_interval} 分钟")
        print(f"最优参数: {self.params}")
        print("-" * 50)

        if not continuous:
            # Single poll
            self.poll_once()
            return

        # Schedule polling
        schedule.every(self.poll_interval).minutes.do(self.poll_once)

        self._running = True
        while self._running:
            # Check if any market is open
            any_trading = any(is_trading_time(s) for s in self.symbols)

            if any_trading:
                schedule.run_pending()
            else:
                next_time = get_next_trading_time()
                if next_time:
                    wait_seconds = (next_time - datetime.now()).total_seconds()
                    if wait_seconds > 0:
                        print(f"非交易时间，等待至 {next_time.strftime('%Y-%m-%d %H:%M')}")
                        time.sleep(min(wait_seconds, 60))  # Check every minute

            time.sleep(1)

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        print("监控已停止")


def run_monitor(
    symbols: List[str],
    params: Dict = None,
    weights: Dict = None,
    poll_interval: int = 15,
    continuous: bool = True,
    notify_config: Optional[NotifyConfig] = None,
):
    """
    Convenience function to run the monitor.

    Args:
        symbols: List of futures symbols
        params: Signal parameters
        weights: Theory weights
        poll_interval: Polling interval in minutes
        continuous: Run continuously or once
        notify_config: Notification configuration
    """
    monitor = RealtimeMonitor(
        symbols=symbols,
        params=params,
        weights=weights,
        poll_interval_minutes=poll_interval,
        notify_config=notify_config,
    )
    monitor.run(continuous=continuous)


if __name__ == "__main__":
    # Example: Run monitor with optimized parameters
    OPTIMIZED_PARAMS = {
        "smc": {
            "ob_lookback": 25,
            "fvg_min_size": 0.001,
            "sweep_threshold": 0.004,
            "timeframe": "1h",
        },
        "momentum": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "timeframe": "15min",
        },
        "linear": {
            "regression_period": 25,
            "band_std": 2.0,
            "breakout_confirm": 2,
            "timeframe": "30min",
        },
    }

    run_monitor(
        symbols=["rb", "i"],
        params=OPTIMIZED_PARAMS,
        continuous=True,
    )