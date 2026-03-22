"""Tests for ensemble module."""
import pytest
import pandas as pd
import numpy as np
from autoresearch_futures.ensemble import (
    ensemble_signals,
    simple_vote,
    consensus_filter,
    normalize_weights,
    calc_confidence,
)


class TestEnsemble:
    @pytest.fixture
    def sample_signals(self):
        """Create sample signals from three theories."""
        n = 100
        return {
            "smc": pd.Series(np.random.choice([-1, 0, 1], size=n)),
            "momentum": pd.Series(np.random.choice([-1, 0, 1], size=n)),
            "linear": pd.Series(np.random.choice([-1, 0, 1], size=n)),
        }

    def test_normalize_weights(self):
        """normalize_weights should sum to 1."""
        weights = {"smc": 0.4, "momentum": 0.4, "linear": 0.3}
        normalized = normalize_weights(weights)
        assert abs(sum(normalized.values()) - 1.0) < 0.001

    def test_simple_vote(self, sample_signals):
        """simple_vote should combine signals with weights."""
        weights = {"smc": 0.35, "momentum": 0.35, "linear": 0.30}
        result = simple_vote(sample_signals, weights)
        assert len(result) == 100
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_consensus_filter(self, sample_signals):
        """consensus_filter should require agreement."""
        result = consensus_filter(sample_signals)
        for i in range(len(result)):
            if result.iloc[i] != 0:
                assert sample_signals["smc"].iloc[i] == result.iloc[i]
                assert sample_signals["momentum"].iloc[i] == result.iloc[i]
                assert sample_signals["linear"].iloc[i] == result.iloc[i]

    def test_ensemble_signals(self, sample_signals):
        """ensemble_signals should return final signal with confidence."""
        weights = {"smc": 0.35, "momentum": 0.35, "linear": 0.30}
        result = ensemble_signals(sample_signals, weights)
        assert "signal" in result
        assert "confidence" in result
        assert len(result["signal"]) == 100
        assert len(result["confidence"]) == 100

    def test_calc_confidence(self, sample_signals):
        """calc_confidence should return value between 0 and 1."""
        weights = {"smc": 0.35, "momentum": 0.35, "linear": 0.30}
        for i in range(len(sample_signals["smc"])):
            signals_at_i = {k: v.iloc[i] for k, v in sample_signals.items()}
            conf = calc_confidence(signals_at_i, weights)
            assert 0 <= conf <= 1