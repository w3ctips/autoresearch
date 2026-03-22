"""Tests for config module."""
import pytest
from autoresearch_futures.config import (
    Config,
    DataConfig,
    BacktestConfig,
    NotifyConfig,
    DEFAULT_PARAMS,
    THEORY_WEIGHTS,
    SCORE_WEIGHTS,
)


class TestConfig:
    def test_config_has_data_config(self):
        """Config should have data configuration."""
        config = Config()
        assert hasattr(config, 'data')
        assert isinstance(config.data, DataConfig)

    def test_data_config_defaults(self):
        """DataConfig should have correct defaults."""
        data_config = DataConfig()
        assert data_config.base_timeframe == "15min"
        assert data_config.synthetic_timeframes == ["30min", "1h", "2h", "4h"]
        assert data_config.train_window_months == 12
        assert data_config.embargo_weeks == 2
        assert data_config.valid_window_months == 1
        assert data_config.locked_predict_months == 6

    def test_backtest_config_defaults(self):
        """BacktestConfig should have correct defaults."""
        bt_config = BacktestConfig()
        assert bt_config.commission_rate == 0.0001
        assert bt_config.slippage_ticks == 1

    def test_notify_config_defaults(self):
        """NotifyConfig should have disabled channels by default."""
        notify_config = NotifyConfig()
        assert notify_config.wechat_enabled is False
        assert notify_config.telegram_enabled is False
        assert notify_config.email_enabled is False

    def test_default_params_structure(self):
        """DEFAULT_PARAMS should have all three theories."""
        assert "smc" in DEFAULT_PARAMS
        assert "momentum" in DEFAULT_PARAMS
        assert "linear" in DEFAULT_PARAMS

    def test_theory_weights_sum_to_one(self):
        """THEORY_WEIGHTS should sum to 1.0."""
        total = sum(THEORY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_score_weights_structure(self):
        """SCORE_WEIGHTS should have all required keys."""
        required_keys = ["sharpe", "net_return", "win_rate", "precision", "drawdown"]
        for key in required_keys:
            assert key in SCORE_WEIGHTS