"""Tests for backtest module."""
import pytest
import pandas as pd
import numpy as np
from autoresearch_futures.backtest import (
    BacktestResult,
    calc_score,
    calc_commission,
    calc_slippage,
    calc_total_cost,
    SCORE_WEIGHTS,
    run_backtest,
    run_multi_backtest,
)


class TestBacktestResult:
    def test_backtest_result_creation(self):
        """BacktestResult should be creatable with all fields."""
        result = BacktestResult(
            net_return=0.15,
            annual_return=0.18,
            max_drawdown=0.08,
            volatility=0.12,
            var_95=0.02,
            sharpe_ratio=1.5,
            calmar_ratio=2.25,
            sortino_ratio=2.0,
            total_trades=100,
            win_rate=0.55,
            profit_factor=1.8,
            avg_holding_bars=4.5,
            signal_precision=0.60,
            signal_recall=0.45,
        )
        assert result.net_return == 0.15
        assert result.sharpe_ratio == 1.5

    def test_calc_score_positive(self):
        """calc_score should return positive for good metrics."""
        result = BacktestResult(
            net_return=0.15,
            max_drawdown=0.08,
            sharpe_ratio=1.5,
            win_rate=0.55,
            signal_precision=0.60,
            annual_return=0.18,
            volatility=0.12,
            var_95=0.02,
            calmar_ratio=2.25,
            sortino_ratio=2.0,
            total_trades=100,
            profit_factor=1.8,
            avg_holding_bars=4.5,
            signal_recall=0.45,
        )
        score = calc_score(result, SCORE_WEIGHTS)
        assert score > 0

    def test_calc_score_with_high_drawdown(self):
        """calc_score should penalize high drawdown."""
        result_good = BacktestResult(
            net_return=0.15, max_drawdown=0.05, sharpe_ratio=1.5,
            win_rate=0.55, signal_precision=0.60, annual_return=0.18,
            volatility=0.12, var_95=0.02, calmar_ratio=2.25, sortino_ratio=2.0,
            total_trades=100, profit_factor=1.8, avg_holding_bars=4.5, signal_recall=0.45,
        )
        result_bad = BacktestResult(
            net_return=0.15, max_drawdown=0.20, sharpe_ratio=1.5,
            win_rate=0.55, signal_precision=0.60, annual_return=0.18,
            volatility=0.12, var_95=0.02, calmar_ratio=0.75, sortino_ratio=2.0,
            total_trades=100, profit_factor=1.8, avg_holding_bars=4.5, signal_recall=0.45,
        )
        score_good = calc_score(result_good, SCORE_WEIGHTS)
        score_bad = calc_score(result_bad, SCORE_WEIGHTS)
        assert score_good > score_bad


class TestTradeCost:
    def test_calc_commission(self):
        """calc_commission should calculate commission based on trade value."""
        trade_value = 1_000_000  # 100万
        commission = calc_commission(trade_value, 0.0001)
        assert commission == 100  # 万分之一 = 100元

    def test_calc_slippage(self):
        """calc_slippage should calculate slippage based on ticks."""
        volume = 10  # 10手
        tick_size = 1.0  # 最小变动价位
        slippage = calc_slippage(volume, tick_size, slippage_ticks=1)
        # Slippage = volume * tick_size * 2 (entry + exit) * slippage_ticks
        assert slippage == 20  # 10 * 1 * 2

    def test_calc_total_cost(self):
        """calc_total_cost should sum commission and slippage."""
        cost = calc_total_cost(
            trade_value=1_000_000,
            volume=10,
            tick_size=1.0,
            commission_rate=0.0001,
            slippage_ticks=1,
        )
        assert cost == 120  # 100 (commission) + 20 (slippage)


class TestRunBacktest:
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2024-01-01 09:00", periods=100, freq="15min")
        prices = 3600 + np.cumsum(np.random.randn(100) * 5)
        df = pd.DataFrame({
            "datetime": dates,
            "open": prices,
            "high": prices + 10,
            "low": prices - 10,
            "close": prices,
            "volume": [10000] * 100,
        })
        return df

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        # Simple: buy at bar 10, sell at bar 30
        signals = pd.Series(0, index=range(100))
        signals.iloc[10] = 1   # Buy
        signals.iloc[30] = -1  # Sell
        return signals

    def test_run_backtest_returns_result(self, sample_price_data, sample_signals):
        """run_backtest should return BacktestResult."""
        result = run_backtest(
            signals=sample_signals,
            data=sample_price_data,
            tick_size=1.0,
            contract_multiplier=10,
        )
        assert isinstance(result, BacktestResult)

    def test_run_backtest_counts_trades(self, sample_price_data, sample_signals):
        """run_backtest should correctly count trades."""
        result = run_backtest(
            signals=sample_signals,
            data=sample_price_data,
            tick_size=1.0,
            contract_multiplier=10,
        )
        # 1 buy + 1 sell = 1 round trip = 2 trades
        assert result.total_trades >= 2

    def test_run_multi_backtest(self, sample_price_data, sample_signals):
        """run_multi_backtest should handle multiple symbols."""
        signals_dict = {
            "rb": sample_signals,
            "i": sample_signals,
        }
        data_dict = {
            "rb": sample_price_data,
            "i": sample_price_data,
        }
        results = run_multi_backtest(
            signals_dict=signals_dict,
            data_dict=data_dict,
            tick_sizes={"rb": 1.0, "i": 0.5},
            contract_multipliers={"rb": 10, "i": 100},
        )
        assert "rb" in results
        assert "i" in results
        assert isinstance(results["rb"], BacktestResult)