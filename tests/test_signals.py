"""Tests for signals module."""
import pytest
import pandas as pd
import numpy as np
from autoresearch_futures.signals import (
    momentum_signals,
    DEFAULT_MOMENTUM_PARAMS,
)


class TestMomentumSignals:
    @pytest.fixture
    def sample_data(self):
        """Create sample price data with trends."""
        dates = pd.date_range("2024-01-01 09:00", periods=200, freq="15min")
        # Create trending data
        prices = 3600 + np.cumsum(np.sin(np.linspace(0, 4*np.pi, 200)) * 20)
        df = pd.DataFrame({
            "datetime": dates,
            "open": prices,
            "high": prices + 5,
            "low": prices - 5,
            "close": prices,
            "volume": [10000] * 200,
        })
        return df

    def test_momentum_signals_returns_dict(self, sample_data):
        """momentum_signals should return a dict with expected keys."""
        result = momentum_signals(sample_data, DEFAULT_MOMENTUM_PARAMS)
        assert isinstance(result, dict)
        assert "rsi" in result
        assert "macd" in result
        assert "signal" in result

    def test_momentum_signal_range(self, sample_data):
        """Signal values should be in [-1, 0, 1]."""
        result = momentum_signals(sample_data, DEFAULT_MOMENTUM_PARAMS)
        signal = result["signal"]
        assert signal.min() >= -1
        assert signal.max() <= 1

    def test_rsi_calculation(self, sample_data):
        """RSI should be calculated correctly."""
        result = momentum_signals(sample_data, DEFAULT_MOMENTUM_PARAMS)
        rsi = result["rsi"]
        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_macd_calculation(self, sample_data):
        """MACD should have expected structure."""
        result = momentum_signals(sample_data, DEFAULT_MOMENTUM_PARAMS)
        macd = result["macd"]
        macd_signal = result["macd_signal"]
        # MACD and signal should have similar length
        assert len(macd.dropna()) > 0
        assert len(macd_signal.dropna()) > 0