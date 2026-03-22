"""Tests for evolve module."""
import pytest
import pandas as pd
import numpy as np
from autoresearch_futures.evolve import (
    generate_all_signals,
    aggregate_scores,
    log_results,
    run_evolution_step,
)
from autoresearch_futures.config import DEFAULT_PARAMS, SCORE_WEIGHTS


class TestEvolve:
    @pytest.fixture
    def sample_data(self):
        """Create sample data dict."""
        dates = pd.date_range("2024-01-01 09:00", periods=500, freq="15min")
        prices = 3600 + np.cumsum(np.random.randn(500) * 3)
        df = pd.DataFrame({
            "datetime": dates,
            "open": prices, "high": prices + 5, "low": prices - 5,
            "close": prices, "volume": [100000] * 500,
        })
        return {"rb": df, "i": df.copy()}

    def test_generate_all_signals(self, sample_data):
        """generate_all_signals should generate signals for all symbols."""
        signals = generate_all_signals(sample_data, DEFAULT_PARAMS)
        assert "rb" in signals
        assert "i" in signals
        assert "signal" in signals["rb"]

    def test_aggregate_scores(self):
        """aggregate_scores should calculate average score."""
        from autoresearch_futures.backtest import BacktestResult, calc_score

        results = {
            "rb": BacktestResult(
                net_return=0.15, max_drawdown=0.08, sharpe_ratio=1.5,
                win_rate=0.55, signal_precision=0.60, annual_return=0.18,
                volatility=0.12, var_95=0.02, calmar_ratio=2.25, sortino_ratio=2.0,
                total_trades=100, profit_factor=1.8, avg_holding_bars=4.5, signal_recall=0.45,
            ),
            "i": BacktestResult(
                net_return=0.10, max_drawdown=0.05, sharpe_ratio=1.2,
                win_rate=0.50, signal_precision=0.55, annual_return=0.12,
                volatility=0.10, var_95=0.015, calmar_ratio=2.4, sortino_ratio=1.8,
                total_trades=80, profit_factor=1.6, avg_holding_bars=5.0, signal_recall=0.40,
            ),
        }
        score = aggregate_scores(results, SCORE_WEIGHTS)
        assert isinstance(score, float)

    def test_log_results(self, tmp_path):
        """log_results should write to TSV file."""
        from autoresearch_futures.backtest import BacktestResult

        results = {
            "rb": BacktestResult(
                net_return=0.15, max_drawdown=0.08, sharpe_ratio=1.5,
                win_rate=0.55, signal_precision=0.60, annual_return=0.18,
                volatility=0.12, var_95=0.02, calmar_ratio=2.25, sortino_ratio=2.0,
                total_trades=100, profit_factor=1.8, avg_holding_bars=4.5, signal_recall=0.45,
            ),
        }
        tsv_path = str(tmp_path / "results.tsv")
        log_results(split_id="001", commit="abc123", score=0.85,
                   results=results, description="test run", filepath=tsv_path)
        assert (tmp_path / "results.tsv").exists()

    def test_run_evolution_step(self, sample_data):
        """run_evolution_step should return score, results, and signals."""
        tick_sizes = {"rb": 1.0, "i": 0.5}
        contract_multipliers = {"rb": 10, "i": 100}

        score, results, signals_dict = run_evolution_step(
            sample_data, DEFAULT_PARAMS, tick_sizes, contract_multipliers
        )

        assert isinstance(score, float)
        assert "rb" in results
        assert "i" in results
        assert "rb" in signals_dict