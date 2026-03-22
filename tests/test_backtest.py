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