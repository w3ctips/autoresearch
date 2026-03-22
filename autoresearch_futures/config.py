"""
Global configuration for autoresearch-futures.
All tunable parameters are defined here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data preparation configuration."""
    # 数据源类型: "minute" (15分钟线，推荐) 或 "daily" (日线)
    # 注意: 分钟数据只能获取最近约1000条（约2.5个月）
    data_type: str = "minute"

    # 分钟数据的基础周期
    base_timeframe: str = "15min"

    # 可合成的周期 (从15分钟合成)
    synthetic_timeframes: List[str] = field(default_factory=lambda: ["30min", "1h", "2h", "4h"])

    # Walk-Forward split parameters
    # 注意: 分钟数据时间范围有限，需要调整窗口大小
    # 适合分钟数据的配置: 1个月训练 + 1周embargo + 2周验证
    train_window_months: int = 1      # 训练窗口 (分钟数据推荐1个月)
    embargo_weeks: int = 1            # 隔离期
    valid_window_months: int = 0      # 验证窗口月份，用周数代替
    valid_window_weeks: int = 2       # 验证窗口周数
    locked_predict_weeks: int = 2     # 锁定预测集 (分钟数据推荐2周)

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
# 适用于15分钟高频策略
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