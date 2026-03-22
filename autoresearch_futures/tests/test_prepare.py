"""Tests for prepare module."""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import functions to test
from autoresearch_futures.prepare import (
    CACHE_DIR,
    get_futures_list,
    download_contract,
    synthesize_timeframe,
    synthesize_all_timeframes,
    generate_walk_forward_splits,
    get_locked_predict_dates,
    filter_symbols_by_age,
    load_data,
    list_available_symbols,
    save_raw_data,
    load_raw_data,
    save_synthetic_data,
    load_synthetic_data,
    save_splits,
    load_splits,
)


class TestPrepareDownload:
    def test_cache_dir_expanded(self):
        """CACHE_DIR should be expanded from ~."""
        assert "~" not in CACHE_DIR
        assert CACHE_DIR.endswith("autoresearch-futures")

    @patch("akshare.futures_main_sina")
    def test_get_futures_list_returns_list(self, mock_akshare):
        """get_futures_list should return a list of symbols."""
        mock_akshare.return_value = pd.DataFrame({"symbol": ["rb", "i", "hc", "j"]})
        result = get_futures_list()
        assert isinstance(result, list)
        assert "rb" in result

    @patch("akshare.futures_zh_minute_sina")
    def test_download_contract_returns_dataframe(self, mock_akshare):
        """download_contract should return a DataFrame."""
        mock_akshare.return_value = pd.DataFrame({
            "日期": ["2024-01-01 09:00", "2024-01-01 09:15"],
            "开盘价": [3600, 3650],
            "最高价": [3700, 3750],
            "最低价": [3550, 3600],
            "收盘价": [3650, 3700],
            "成交量": [100000, 120000],
        })
        result = download_contract("rb")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("akshare.futures_zh_minute_sina")
    def test_download_contract_standardizes_columns(self, mock_akshare):
        """download_contract should standardize column names."""
        mock_akshare.return_value = pd.DataFrame({
            "日期": ["2024-01-01 09:00"],
            "开盘价": [3600],
            "最高价": [3700],
            "最低价": [3550],
            "收盘价": [3650],
            "成交量": [100000],
        })
        result = download_contract("rb")
        assert "datetime" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns


class TestTimeframeSynthesis:
    def test_synthesize_timeframe_30min(self):
        """synthesize_timeframe should correctly aggregate 15min to 30min."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=8, freq="15min"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [105, 106, 107, 108, 109, 110, 111, 112],
            "low": [98, 99, 100, 101, 102, 103, 104, 105],
            "close": [103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        })
        result = synthesize_timeframe(df_15min, "30min")
        assert len(result) == 4
        # First 30min bar: open=100, high=max(105,106)=106, low=min(98,99)=98, close=104, volume=2100
        assert result.iloc[0]["open"] == 100
        assert result.iloc[0]["high"] == 106
        assert result.iloc[0]["low"] == 98
        assert result.iloc[0]["close"] == 104
        assert result.iloc[0]["volume"] == 2100

    def test_synthesize_timeframe_1h(self):
        """synthesize_timeframe should correctly aggregate 15min to 1h."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=4, freq="15min"),
            "open": [100, 101, 102, 103],
            "high": [105, 106, 107, 108],
            "low": [98, 99, 100, 101],
            "close": [103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300],
        })
        result = synthesize_timeframe(df_15min, "1h")
        assert len(result) == 1
        assert result.iloc[0]["open"] == 100
        assert result.iloc[0]["high"] == 108
        assert result.iloc[0]["low"] == 98
        assert result.iloc[0]["close"] == 106
        assert result.iloc[0]["volume"] == 4600

    def test_synthesize_all_timeframes(self):
        """synthesize_all_timeframes should create all configured timeframes."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=100, freq="15min"),
            "open": [100] * 100, "high": [105] * 100, "low": [98] * 100,
            "close": [103] * 100, "volume": [1000] * 100,
        })
        results = synthesize_all_timeframes(df_15min)
        assert "30min" in results
        assert "1h" in results
        assert "2h" in results
        assert "4h" in results

    def test_synthesize_timeframe_invalid(self):
        """synthesize_timeframe should raise error for invalid timeframe."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=4, freq="15min"),
            "open": [100] * 4, "high": [105] * 4, "low": [98] * 4,
            "close": [103] * 4, "volume": [1000] * 4,
        })
        with pytest.raises(ValueError, match="Unknown target timeframe"):
            synthesize_timeframe(df_15min, "invalid_tf")


class TestWalkForwardSplits:
    def test_generate_walk_forward_splits(self):
        """generate_walk_forward_splits should create correct split indices."""
        splits = generate_walk_forward_splits("2022-01-01", "2024-12-31")
        assert len(splits) > 0
        split = splits[0]
        assert "split_id" in split
        assert "train_start" in split
        assert "train_end" in split
        assert "embargo_end" in split
        assert "valid_start" in split
        assert "valid_end" in split

    def test_split_has_embargo_gap(self):
        """Each split should have embargo gap between train and valid."""
        splits = generate_walk_forward_splits("2022-01-01", "2024-12-31")
        for split in splits:
            train_end = pd.to_datetime(split["train_end"])
            embargo_end = pd.to_datetime(split["embargo_end"])
            gap_days = (embargo_end - train_end).days
            # 2 weeks = 14 days, allow some flexibility for calendar
            assert 10 <= gap_days <= 18, f"Gap is {gap_days} days, expected 10-18"

    def test_split_valid_follows_embargo(self):
        """Validation period should start at embargo_end."""
        splits = generate_walk_forward_splits("2022-01-01", "2024-12-31")
        for split in splits:
            embargo_end = split["embargo_end"]
            valid_start = split["valid_start"]
            assert embargo_end == valid_start

    def test_locked_prediction_set(self):
        """get_locked_predict_dates should return correct range."""
        locked_start, locked_end = get_locked_predict_dates("2024-12-31")
        # 6 months before Dec 31 is June 30
        assert locked_start == "2024-06-30"
        assert locked_end == "2024-12-31"

    def test_locked_prediction_set_custom(self):
        """get_locked_predict_dates should work with different end dates."""
        locked_start, locked_end = get_locked_predict_dates("2024-06-30")
        # 6 months before June 30 is Dec 30 (same day-of-month logic)
        assert locked_start == "2023-12-30"
        assert locked_end == "2024-06-30"

    def test_filter_symbols_by_age(self):
        """filter_symbols_by_age should exclude new symbols."""
        symbol_start_dates = {
            "rb": "2020-01-01", "i": "2020-06-01", "new_symbol": "2024-01-01",
        }
        valid = filter_symbols_by_age(symbol_start_dates, "2024-12-31", 18)
        assert "rb" in valid
        assert "i" in valid
        assert "new_symbol" not in valid

    def test_filter_symbols_by_age_edge_case(self):
        """filter_symbols_by_age should include symbols exactly at min age."""
        symbol_start_dates = {
            "exact_18": "2023-06-30",  # Exactly 18 months before 2024-12-31
        }
        valid = filter_symbols_by_age(symbol_start_dates, "2024-12-31", 18)
        # Should include because age is exactly 18 months
        assert "exact_18" in valid


class TestDataStorage:
    def test_save_and_load_raw_data(self, tmp_path):
        """save_raw_data and load_raw_data should work together."""
        # Override CACHE_DIR for testing
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=10, freq="15min"),
                "open": [100] * 10, "high": [105] * 10, "low": [98] * 10,
                "close": [103] * 10, "volume": [1000] * 10,
            })
            save_raw_data("test_symbol", df)
            loaded = load_raw_data("test_symbol")
            assert loaded is not None
            assert len(loaded) == 10
            assert list(loaded.columns) == list(df.columns)
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_save_and_load_synthetic_data(self, tmp_path):
        """save_synthetic_data and load_synthetic_data should work together."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=5, freq="30min"),
                "open": [100] * 5, "high": [105] * 5, "low": [98] * 5,
                "close": [103] * 5, "volume": [2000] * 5,
            })
            save_synthetic_data("test_symbol", "30min", df)
            loaded = load_synthetic_data("test_symbol", "30min")
            assert loaded is not None
            assert len(loaded) == 5
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_save_and_load_splits(self, tmp_path):
        """save_splits and load_splits should work together."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            splits = [
                {"split_id": 0, "train_start": "2022-01-01", "train_end": "2022-12-31"},
                {"split_id": 1, "train_start": "2022-02-01", "train_end": "2023-01-31"},
            ]
            save_splits(splits)
            loaded = load_splits()
            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0]["split_id"] == 0
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_load_splits_missing_file(self, tmp_path):
        """load_splits should return None for missing file."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            loaded = load_splits()
            assert loaded is None
        finally:
            prep_module.CACHE_DIR = original_cache_dir


class TestDataLoader:
    def test_load_data_raw(self, tmp_path):
        """load_data should load raw 15min data when timeframe is 15min."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=10, freq="15min"),
                "open": [100] * 10, "high": [105] * 10, "low": [98] * 10,
                "close": [103] * 10, "volume": [1000] * 10,
            })
            save_raw_data("test", df)
            loaded = load_data("test", "15min")
            assert loaded is not None
            assert len(loaded) == 10
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_load_data_synthetic(self, tmp_path):
        """load_data should load synthetic data for other timeframes."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=5, freq="30min"),
                "open": [100] * 5, "high": [105] * 5, "low": [98] * 5,
                "close": [103] * 5, "volume": [2000] * 5,
            })
            save_synthetic_data("test", "30min", df)
            loaded = load_data("test", "30min")
            assert loaded is not None
            assert len(loaded) == 5
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_list_available_symbols(self, tmp_path):
        """list_available_symbols should return symbols with cached data."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=10, freq="15min"),
                "open": [100] * 10, "high": [105] * 10, "low": [98] * 10,
                "close": [103] * 10, "volume": [1000] * 10,
            })
            save_raw_data("rb", df)
            save_raw_data("i", df)
            symbols = list_available_symbols()
            assert "rb" in symbols
            assert "i" in symbols
        finally:
            prep_module.CACHE_DIR = original_cache_dir

    def test_list_available_symbols_empty(self, tmp_path):
        """list_available_symbols should return empty list if no data."""
        import autoresearch_futures.prepare as prep_module
        original_cache_dir = prep_module.CACHE_DIR
        prep_module.CACHE_DIR = str(tmp_path)

        try:
            symbols = list_available_symbols()
            assert symbols == []
        finally:
            prep_module.CACHE_DIR = original_cache_dir