"""
Global configuration for autoresearch-futures.
All tunable parameters are defined here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data preparation configuration."""
    base_timeframe: str = "15min"
    synthetic_timeframes: List[str] = field(default_factory=lambda: ["30min", "1h", "2h", "4h"])

    # Walk-Forward split parameters
    train_window_months: int = 12
    embargo_weeks: int = 2
    valid_window_months: int = 1
    locked_predict_months: int = 6

    # Data directory
    cache_dir: str = "~/.cache/autoresearch-futures"


@dataclass
class BacktestConfig:
    """Backtest engine configuration."""
    commission_rate: float = 0.0001  # 万分之一
    slippage_ticks: int = 1

    # Initial capital
    initial_capital: float = 1_000_000.0


@dataclass
class NotifyConfig:
    """Signal notification configuration."""
    # WeChat (企业微信机器人)
    wechat_enabled: bool = False
    wechat_webhook: str = ""

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Email
    email_enabled: bool = False
    email_smtp: str = ""
    email_sender: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)

    # Cooldown
    signal_cooldown_minutes: int = 30


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)


# Signal parameters for each theory (agent can modify these)
DEFAULT_PARAMS = {
    "smc": {
        "ob_lookback": 20,
        "fvg_min_size": 0.002,
        "sweep_threshold": 0.01,
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
        "regression_period": 20,
        "band_std": 2.0,
        "breakout_confirm": 3,
        "timeframe": "30min",
    },
}

# Theory weights for ensemble (should sum to 1.0)
THEORY_WEIGHTS = {
    "smc": 0.35,
    "momentum": 0.35,
    "linear": 0.30,
}

# Score weights for backtest evaluation (should sum to 1.0)
SCORE_WEIGHTS = {
    "sharpe": 0.25,
    "net_return": 0.30,
    "win_rate": 0.15,
    "precision": 0.15,
    "drawdown": 0.15,
}