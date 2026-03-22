# 中国期货自进化策略系统实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 autoresearch 改造为期货策略自主研究系统，实现多理论信号进化与实盘推送。

**Architecture:** 混合架构，单文件进化入口 + 模块化信号库。数据层 → 回测层 → 信号层 → 推送层 → 进化主循环。

**Tech Stack:** Python 3.10+, akshare, pandas, numpy, requests

---

## Chunk 1: 项目基础设施与配置

### Task 1: 创建项目配置模块

**Files:**
- Create: `autoresearch_futures/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test for config module**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures'"

- [ ] **Step 3: Create package structure**

```bash
mkdir -p autoresearch_futures
touch autoresearch_futures/__init__.py
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 4: Write the config module implementation**

```python
# autoresearch_futures/config.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch_futures/__init__.py autoresearch_futures/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add config module with data/backtest/notify configurations"
```

### Task 2: 创建 pyproject.toml 依赖配置

**Files:**
- Create: `autoresearch_futures/pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
# autoresearch_futures/pyproject.toml
[project]
name = "autoresearch-futures"
version = "0.1.0"
description = "Self-evolving futures strategy research system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "akshare>=1.12.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=14.0.0",
    "requests>=2.31.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch_futures/pyproject.toml
git commit -m "feat: add pyproject.toml with dependencies"
```

---

## Chunk 2: 数据准备模块

### Task 3: 数据下载功能

**Files:**
- Create: `autoresearch_futures/prepare.py`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test for data download**

```python
# tests/test_prepare.py
"""Tests for prepare module."""
import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from autoresearch_futures.prepare import (
    get_futures_list,
    download_contract,
    download_all_contracts,
    CACHE_DIR,
)


class TestPrepareDownload:
    def test_cache_dir_expanded(self):
        """CACHE_DIR should be expanded from ~."""
        assert "~" not in CACHE_DIR
        assert CACHE_DIR.endswith("autoresearch-futures")

    @patch("akshare.futures_main_sina")
    def test_get_futures_list_returns_list(self, mock_akshare):
        """get_futures_list should return a list of symbols."""
        mock_akshare.return_value = pd.DataFrame({
            "symbol": ["rb", "i", "hc", "j"]
        })
        result = get_futures_list()
        assert isinstance(result, list)
        assert "rb" in result

    @patch("akshare.futures_main_sina")
    def test_download_contract_returns_dataframe(self, mock_akshare):
        """download_contract should return a DataFrame."""
        mock_akshare.return_value = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "open": [3600, 3650],
            "high": [3700, 3750],
            "low": [3550, 3600],
            "close": [3650, 3700],
            "volume": [100000, 120000],
        })
        result = download_contract("rb")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("autoresearch_futures.prepare.download_contract")
    def test_download_all_contracts_creates_files(self, mock_download, tmp_path):
        """download_all_contracts should save parquet files."""
        # Setup mock
        mock_download.return_value = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="15min"),
            "open": [3600] * 10,
            "high": [3700] * 10,
            "low": [3550] * 10,
            "close": [3650] * 10,
            "volume": [100000] * 10,
        })

        # Use tmp_path as cache dir
        with patch("autoresearch_futures.prepare.CACHE_DIR", str(tmp_path)):
            with patch("autoresearch_futures.prepare.get_futures_list", return_value=["rb"]):
                download_all_contracts(symbols=["rb"])

        # Check file was created
        raw_dir = tmp_path / "raw"
        assert raw_dir.exists()
        assert (raw_dir / "rb.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.prepare'"

- [ ] **Step 3: Write the prepare module (download functions)**

```python
# autoresearch_futures/prepare.py
"""
Data preparation for autoresearch-futures.
Downloads futures data from akshare, synthesizes timeframes, and splits datasets.
"""
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import akshare as ak

from autoresearch_futures.config import DataConfig

# Expanded cache directory
CACHE_DIR = os.path.expanduser(DataConfig.cache_dir)
RAW_DIR = os.path.join(CACHE_DIR, "raw")
SYNTHETIC_DIR = os.path.join(CACHE_DIR, "synthetic")
SPLITS_DIR = os.path.join(CACHE_DIR, "splits")


def ensure_dirs():
    """Ensure all cache directories exist."""
    for d in [RAW_DIR, SYNTHETIC_DIR, SPLITS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_futures_list() -> List[str]:
    """
    Get list of main futures contracts from akshare.
    Returns list of symbol codes (e.g., ['rb', 'i', 'hc', ...]).
    """
    try:
        df = ak.futures_main_sina()
        # Extract unique symbols
        symbols = df["symbol"].str.extract(r"^([A-Za-z]+)")[0].unique().tolist()
        return sorted(symbols)
    except Exception as e:
        print(f"Error getting futures list: {e}")
        return []


def download_contract(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download 15min data for a single futures contract.

    Args:
        symbol: Futures symbol (e.g., 'rb' for 螺纹钢)
        start_date: Start date in YYYY-MM-DD format (optional)

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    try:
        # Get main contract data
        df = ak.futures_main_sina(symbol=symbol)

        # Rename columns
        df = df.rename(columns={
            "日期": "datetime",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
        })

        # Ensure datetime is proper format
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Filter by start date if provided
        if start_date:
            df = df[df["datetime"] >= pd.to_datetime(start_date)]

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def download_all_contracts(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Download data for all specified futures contracts.

    Args:
        symbols: List of symbols to download (None = all)
        start_date: Start date for data
        force: Redownload even if file exists
    """
    ensure_dirs()

    if symbols is None:
        symbols = get_futures_list()

    print(f"Downloading {len(symbols)} contracts...")

    for i, symbol in enumerate(symbols):
        filepath = os.path.join(RAW_DIR, f"{symbol}.parquet")

        # Skip if exists and not forcing
        if os.path.exists(filepath) and not force:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: already exists")
            continue

        df = download_contract(symbol, start_date)

        if len(df) > 0:
            df.to_parquet(filepath, index=False)
            print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(df)} rows saved")
        else:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: FAILED (no data)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/prepare.py tests/test_prepare.py
git commit -m "feat: add data download functions for futures contracts"
```

### Task 4: K线周期合成功能

**Files:**
- Modify: `autoresearch_futures/prepare.py`
- Modify: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test for timeframe synthesis**

```python
# Add to tests/test_prepare.py

class TestTimeframeSynthesis:
    def test_synthesize_timeframe_30min(self):
        """synthesize_timeframe should correctly aggregate 15min to 30min."""
        # Create 15min data
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=8, freq="15min"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [105, 106, 107, 108, 109, 110, 111, 112],
            "low": [98, 99, 100, 101, 102, 103, 104, 105],
            "close": [103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        })

        result = synthesize_timeframe(df_15min, "30min")

        assert len(result) == 4  # 8 * 15min = 4 * 30min
        assert result.iloc[0]["open"] == 100  # First bar's open
        assert result.iloc[0]["high"] == 107  # Max of first 2 bars
        assert result.iloc[0]["low"] == 98    # Min of first 2 bars
        assert result.iloc[0]["close"] == 104 # Second bar's close
        assert result.iloc[0]["volume"] == 2100  # Sum of volumes

    def test_synthesize_timeframe_1h(self):
        """synthesize_timeframe should correctly aggregate 15min to 1h."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=12, freq="15min"),
            "open": list(range(100, 112)),
            "high": list(range(105, 117)),
            "low": list(range(98, 110)),
            "close": list(range(103, 115)),
            "volume": [1000] * 12,
        })

        result = synthesize_timeframe(df_15min, "1h")

        assert len(result) == 3  # 12 * 15min = 3 * 1h
        assert result.iloc[0]["volume"] == 4000  # 4 bars * 1000

    def test_synthesize_all_timeframes(self):
        """synthesize_all_timeframes should create all configured timeframes."""
        df_15min = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=100, freq="15min"),
            "open": [100] * 100,
            "high": [105] * 100,
            "low": [98] * 100,
            "close": [103] * 100,
            "volume": [1000] * 100,
        })

        results = synthesize_all_timeframes(df_15min)

        assert "30min" in results
        assert "1h" in results
        assert "2h" in results
        assert "4h" in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py::TestTimeframeSynthesis -v`
Expected: FAIL with "NameError: name 'synthesize_timeframe' is not defined"

- [ ] **Step 3: Implement timeframe synthesis**

```python
# Add to autoresearch_futures/prepare.py

def synthesize_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Synthesize larger timeframes from 15min data.

    Args:
        df: DataFrame with 15min data
        target_tf: Target timeframe ('30min', '1h', '2h', '4h')

    Returns:
        DataFrame aggregated to target timeframe
    """
    # Map timeframe string to pandas frequency
    tf_map = {
        "30min": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
    }

    if target_tf not in tf_map:
        raise ValueError(f"Unsupported timeframe: {target_tf}")

    # Set datetime as index for resampling
    df = df.copy()
    df = df.set_index("datetime")

    # Resample and aggregate
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    result = df.resample(tf_map[target_tf]).agg(agg_dict).dropna()

    # Reset index
    result = result.reset_index()

    return result


def synthesize_all_timeframes(
    df_15min: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
) -> dict:
    """
    Synthesize all timeframes from 15min data.

    Args:
        df_15min: DataFrame with 15min data
        timeframes: List of timeframes to synthesize (None = default from config)

    Returns:
        Dict mapping timeframe to DataFrame
    """
    if timeframes is None:
        timeframes = DataConfig.synthetic_timeframes

    results = {}
    for tf in timeframes:
        results[tf] = synthesize_timeframe(df_15min, tf)

    return results


def save_synthetic_data(symbol: str, df_15min: pd.DataFrame) -> None:
    """
    Save all synthetic timeframes for a symbol.
    """
    ensure_dirs()

    timeframes = synthesize_all_timeframes(df_15min)

    for tf, df in timeframes.items():
        filepath = os.path.join(SYNTHETIC_DIR, f"{symbol}_{tf}.parquet")
        df.to_parquet(filepath, index=False)
        print(f"  Saved {symbol} {tf}: {len(df)} bars")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py::TestTimeframeSynthesis -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/prepare.py tests/test_prepare.py
git commit -m "feat: add timeframe synthesis for 30min/1h/2h/4h"
```

### Task 5: 数据集划分功能

**Files:**
- Modify: `autoresearch_futures/prepare.py`
- Modify: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test for walk-forward splits**

```python
# Add to tests/test_prepare.py

class TestWalkForwardSplits:
    def test_generate_walk_forward_splits(self):
        """generate_walk_forward_splits should create correct split indices."""
        # Date range: 2022-01-01 to 2024-12-31 (3 years)
        start_date = "2022-01-01"
        end_date = "2024-12-31"

        splits = generate_walk_forward_splits(start_date, end_date)

        # Should have multiple splits
        assert len(splits) > 0

        # Each split should have required fields
        split = splits[0]
        assert "split_id" in split
        assert "train_start" in split
        assert "train_end" in split
        assert "embargo_end" in split
        assert "valid_end" in split

    def test_split_has_embargo_gap(self):
        """Each split should have embargo gap between train and valid."""
        splits = generate_walk_forward_splits("2022-01-01", "2024-12-31")

        for split in splits:
            train_end = pd.to_datetime(split["train_end"])
            embargo_end = pd.to_datetime(split["embargo_end"])

            # Embargo should be ~2 weeks
            gap_days = (embargo_end - train_end).days
            assert 10 <= gap_days <= 18  # Approximately 2 weeks

    def test_locked_prediction_set(self):
        """get_locked_predict_dates should return correct range."""
        end_date = "2024-12-31"
        locked_start, locked_end = get_locked_predict_dates(end_date)

        # Should be last 6 months
        assert locked_start == "2024-07-01"
        assert locked_end == "2024-12-31"

    def test_filter_symbols_by_age(self):
        """filter_symbols_by_age should exclude new symbols."""
        # Mock data availability
        symbol_start_dates = {
            "rb": "2020-01-01",   # Old enough
            "i": "2020-06-01",    # Old enough
            "new_symbol": "2024-01-01",  # Too new (needs 18 months)
        }

        current_date = "2024-12-31"
        min_age_months = 18

        valid = filter_symbols_by_age(symbol_start_dates, current_date, min_age_months)

        assert "rb" in valid
        assert "i" in valid
        assert "new_symbol" not in valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py::TestWalkForwardSplits -v`
Expected: FAIL with "NameError: name 'generate_walk_forward_splits' is not defined"

- [ ] **Step 3: Implement walk-forward split functions**

```python
# Add to autoresearch_futures/prepare.py
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def generate_walk_forward_splits(
    start_date: str,
    end_date: str,
    train_months: int = None,
    embargo_weeks: int = None,
    valid_months: int = None,
    step_months: int = 1,
) -> List[dict]:
    """
    Generate walk-forward split indices.

    Args:
        start_date: Start date of available data (YYYY-MM-DD)
        end_date: End date of available data (YYYY-MM-DD)
        train_months: Training window size in months
        embargo_weeks: Embargo period in weeks
        valid_months: Validation window size in months
        step_months: Step size for rolling forward

    Returns:
        List of split dictionaries with train/embargo/valid boundaries
    """
    if train_months is None:
        train_months = DataConfig.train_window_months
    if embargo_weeks is None:
        embargo_weeks = DataConfig.embargo_weeks
    if valid_months is None:
        valid_months = DataConfig.valid_window_months

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    splits = []
    split_id = 1

    # Calculate minimum data needed for first split
    min_required = relativedelta(months=train_months + valid_months)
    min_required_date = start + min_required

    # Don't include locked prediction set
    locked_start = end - relativedelta(months=DataConfig.locked_predict_months)

    # Generate splits
    current_train_end = start + relativedelta(months=train_months)

    while current_train_end + timedelta(weeks=embargo_weeks) + relativedelta(months=valid_months) <= locked_start:
        train_start = current_train_end - relativedelta(months=train_months)
        train_end = current_train_end
        embargo_end = train_end + timedelta(weeks=embargo_weeks)
        valid_end = embargo_end + relativedelta(months=valid_months)

        splits.append({
            "split_id": f"{split_id:03d}",
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "embargo_end": embargo_end.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
        })

        split_id += 1
        current_train_end += relativedelta(months=step_months)

    return splits


def get_locked_predict_dates(end_date: str) -> tuple:
    """
    Get the date range for locked prediction set.

    Args:
        end_date: End date of available data

    Returns:
        Tuple of (start_date, end_date) for locked prediction set
    """
    end = pd.to_datetime(end_date)
    start = end - relativedelta(months=DataConfig.locked_predict_months) + timedelta(days=1)

    # Align to start of month
    start = start.replace(day=1)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def filter_symbols_by_age(
    symbol_start_dates: dict,
    current_date: str,
    min_age_months: int = 18,
) -> List[str]:
    """
    Filter out symbols that don't have enough history.

    Args:
        symbol_start_dates: Dict mapping symbol to its start date
        current_date: Current date
        min_age_months: Minimum required age in months

    Returns:
        List of valid symbols
    """
    current = pd.to_datetime(current_date)
    min_start = current - relativedelta(months=min_age_months)

    valid_symbols = []
    for symbol, start_date in symbol_start_dates.items():
        if pd.to_datetime(start_date) <= min_start:
            valid_symbols.append(symbol)

    return sorted(valid_symbols)


def save_splits(splits: List[dict], filename: str = "walk_forward.json") -> None:
    """Save splits to JSON file."""
    ensure_dirs()
    filepath = os.path.join(SPLITS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved {len(splits)} splits to {filepath}")


def load_splits(filename: str = "walk_forward.json") -> List[dict]:
    """Load splits from JSON file."""
    filepath = os.path.join(SPLITS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Splits file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py::TestWalkForwardSplits -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/prepare.py tests/test_prepare.py
git commit -m "feat: add walk-forward split generation with embargo period"
```

### Task 6: 数据加载器

**Files:**
- Modify: `autoresearch_futures/prepare.py`
- Modify: `tests/test_prepare.py`

- [ ] **Step 1: Write the failing test for data loader**

```python
# Add to tests/test_prepare.py

class TestDataLoader:
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data for testing."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 09:00", periods=1000, freq="15min"),
            "open": [3600.0] * 1000,
            "high": [3700.0] * 1000,
            "low": [3500.0] * 1000,
            "close": [3650.0] * 1000,
            "volume": [100000] * 1000,
        })
        return df

    def test_load_raw_data(self, sample_data, tmp_path):
        """load_data should load raw 15min data."""
        # Save sample data
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sample_data.to_parquet(raw_dir / "rb.parquet", index=False)

        with patch("autoresearch_futures.prepare.RAW_DIR", str(raw_dir)):
            result = load_data("rb")

        assert len(result) == 1000
        assert "datetime" in result.columns
        assert "close" in result.columns

    def test_load_data_with_timeframe(self, sample_data, tmp_path):
        """load_data should load specific timeframe."""
        raw_dir = tmp_path / "raw"
        synthetic_dir = tmp_path / "synthetic"
        raw_dir.mkdir()
        synthetic_dir.mkdir()

        # Save 15min data
        sample_data.to_parquet(raw_dir / "rb.parquet", index=False)

        # Save synthetic 1h data
        df_1h = synthesize_timeframe(sample_data, "1h")
        df_1h.to_parquet(synthetic_dir / "rb_1h.parquet", index=False)

        with patch("autoresearch_futures.prepare.RAW_DIR", str(raw_dir)):
            with patch("autoresearch_futures.prepare.SYNTHETIC_DIR", str(synthetic_dir)):
                result_15min = load_data("rb", timeframe="15min")
                result_1h = load_data("rb", timeframe="1h")

        assert len(result_15min) == 1000
        assert len(result_1h) == 250  # 1000 / 4

    def test_load_split_data(self, sample_data, tmp_path):
        """load_split_data should filter by date range."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sample_data.to_parquet(raw_dir / "rb.parquet", index=False)

        split = {
            "train_start": "2024-01-01",
            "train_end": "2024-01-05",
        }

        with patch("autoresearch_futures.prepare.RAW_DIR", str(raw_dir)):
            result = load_split_data("rb", split, "train")

        # Should filter to date range
        assert result["datetime"].min() >= pd.to_datetime("2024-01-01")
        assert result["datetime"].max() <= pd.to_datetime("2024-01-05 23:59")

    def test_list_available_symbols(self, sample_data, tmp_path):
        """list_available_symbols should return list of downloaded symbols."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        for symbol in ["rb", "i", "hc"]:
            sample_data.to_parquet(raw_dir / f"{symbol}.parquet", index=False)

        with patch("autoresearch_futures.prepare.RAW_DIR", str(raw_dir)):
            symbols = list_available_symbols()

        assert set(symbols) == {"rb", "i", "hc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare.py::TestDataLoader -v`
Expected: FAIL with "NameError: name 'load_data' is not defined"

- [ ] **Step 3: Implement data loader functions**

```python
# Add to autoresearch_futures/prepare.py

def load_data(symbol: str, timeframe: str = "15min") -> pd.DataFrame:
    """
    Load data for a symbol at specified timeframe.

    Args:
        symbol: Futures symbol
        timeframe: Timeframe ('15min', '30min', '1h', '2h', '4h')

    Returns:
        DataFrame with OHLCV data
    """
    if timeframe == "15min":
        filepath = os.path.join(RAW_DIR, f"{symbol}.parquet")
    else:
        filepath = os.path.join(SYNTHETIC_DIR, f"{symbol}_{timeframe}.parquet")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data not found: {filepath}")

    return pd.read_parquet(filepath)


def load_split_data(
    symbol: str,
    split: dict,
    split_type: str,
    timeframe: str = "15min",
) -> pd.DataFrame:
    """
    Load data for a specific split (train/valid/locked).

    Args:
        symbol: Futures symbol
        split: Split dictionary with date boundaries
        split_type: 'train', 'valid', or 'locked'
        timeframe: Data timeframe

    Returns:
        Filtered DataFrame
    """
    df = load_data(symbol, timeframe)

    if split_type == "train":
        start = pd.to_datetime(split["train_start"])
        end = pd.to_datetime(split["train_end"])
    elif split_type == "valid":
        start = pd.to_datetime(split["embargo_end"])
        end = pd.to_datetime(split["valid_end"])
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    # Filter by date range
    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    return df[mask].reset_index(drop=True)


def load_locked_predict_data(
    symbol: str,
    end_date: str,
    timeframe: str = "15min",
) -> pd.DataFrame:
    """
    Load locked prediction set data.

    Args:
        symbol: Futures symbol
        end_date: End date of available data
        timeframe: Data timeframe

    Returns:
        Filtered DataFrame
    """
    df = load_data(symbol, timeframe)
    start_date, _ = get_locked_predict_dates(end_date)

    mask = df["datetime"] >= pd.to_datetime(start_date)
    return df[mask].reset_index(drop=True)


def list_available_symbols() -> List[str]:
    """
    List all symbols with downloaded data.

    Returns:
        List of symbol codes
    """
    if not os.path.exists(RAW_DIR):
        return []

    files = os.listdir(RAW_DIR)
    symbols = [f.replace(".parquet", "") for f in files if f.endswith(".parquet")]
    return sorted(symbols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare.py::TestDataLoader -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/prepare.py tests/test_prepare.py
git commit -m "feat: add data loader with split filtering"
```
---

## Chunk 3: 回测引擎模块

### Task 7: BacktestResult 数据类

**Files:**
- Create: `autoresearch_futures/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test for BacktestResult**

```python
# tests/test_backtest.py
"""Tests for backtest module."""
import pytest
import pandas as pd
import numpy as np
from autoresearch_futures.backtest import (
    BacktestResult,
    calc_score,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.backtest'"

- [ ] **Step 3: Implement BacktestResult**

```python
# autoresearch_futures/backtest.py
"""
Backtest engine for futures strategy evaluation.
Calculates performance metrics with trading costs.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from autoresearch_futures.config import BacktestConfig, SCORE_WEIGHTS


@dataclass
class BacktestResult:
    """Container for backtest results and metrics."""
    # Return metrics
    net_return: float          # Net return after costs
    annual_return: float       # Annualized return

    # Risk metrics
    max_drawdown: float        # Maximum drawdown
    volatility: float          # Return volatility
    var_95: float              # 95% Value at Risk

    # Risk-adjusted metrics
    sharpe_ratio: float        # Sharpe ratio
    calmar_ratio: float        # Calmar ratio
    sortino_ratio: float       # Sortino ratio

    # Trade statistics
    total_trades: int          # Total number of trades
    win_rate: float            # Winning trade percentage
    profit_factor: float       # Gross profit / Gross loss
    avg_holding_bars: float    # Average holding period in bars

    # Signal quality
    signal_precision: float    # Precision of signals
    signal_recall: float       # Recall of profitable opportunities


def calc_score(result: BacktestResult, weights: Dict[str, float]) -> float:
    """
    Calculate weighted composite score.

    Score = w1*sharpe + w2*net_return + w3*win_rate + w4*precision - w5*drawdown
    """
    return (
        weights["sharpe"] * result.sharpe_ratio +
        weights["net_return"] * result.net_return +
        weights["win_rate"] * result.win_rate +
        weights["precision"] * result.signal_precision -
        weights["drawdown"] * result.max_drawdown
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py::TestBacktestResult -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/backtest.py tests/test_backtest.py
git commit -m "feat: add BacktestResult dataclass and score calculation"
```

### Task 8: 交易成本计算

**Files:**
- Modify: `autoresearch_futures/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test for trade cost**

```python
# Add to tests/test_backtest.py

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py::TestTradeCost -v`
Expected: FAIL with "NameError: name 'calc_commission' is not defined"

- [ ] **Step 3: Implement trade cost functions**

```python
# Add to autoresearch_futures/backtest.py

def calc_commission(trade_value: float, commission_rate: float) -> float:
    """
    Calculate commission cost.

    Args:
        trade_value: Total trade value in yuan
        commission_rate: Commission rate (e.g., 0.0001 = 万分之一)

    Returns:
        Commission cost in yuan
    """
    return trade_value * commission_rate


def calc_slippage(volume: int, tick_size: float, slippage_ticks: int = 1) -> float:
    """
    Calculate slippage cost.

    Assumes slippage affects both entry and exit.

    Args:
        volume: Number of contracts
        tick_size: Minimum price tick
        slippage_ticks: Number of ticks for slippage

    Returns:
        Slippage cost in yuan (per contract multiplier = 1)
    """
    # Slippage affects both entry and exit, and is per tick
    return volume * tick_size * 2 * slippage_ticks


def calc_total_cost(
    trade_value: float,
    volume: int,
    tick_size: float,
    commission_rate: float = None,
    slippage_ticks: int = None,
) -> float:
    """
    Calculate total trading cost.

    Args:
        trade_value: Total trade value
        volume: Number of contracts
        tick_size: Price tick size
        commission_rate: Commission rate (default from config)
        slippage_ticks: Slippage in ticks (default from config)

    Returns:
        Total trading cost in yuan
    """
    if commission_rate is None:
        commission_rate = BacktestConfig.commission_rate
    if slippage_ticks is None:
        slippage_ticks = BacktestConfig.slippage_ticks

    commission = calc_commission(trade_value, commission_rate)
    slippage = calc_slippage(volume, tick_size, slippage_ticks)

    return commission + slippage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py::TestTradeCost -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/backtest.py tests/test_backtest.py
git commit -m "feat: add trade cost calculation (commission + slippage)"
```

### Task 9: 核心回测函数

**Files:**
- Modify: `autoresearch_futures/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test for run_backtest**

```python
# Add to tests/test_backtest.py

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py::TestRunBacktest -v`
Expected: FAIL with "NameError: name 'run_backtest' is not defined"

- [ ] **Step 3: Implement run_backtest function**

```python
# Add to autoresearch_futures/backtest.py

def run_backtest(
    signals: pd.Series,
    data: pd.DataFrame,
    tick_size: float = 1.0,
    contract_multiplier: int = 10,
    initial_capital: float = None,
    commission_rate: float = None,
    slippage_ticks: int = None,
) -> BacktestResult:
    """
    Run backtest on a single symbol.

    Args:
        signals: Series of signals (-1, 0, 1) indexed by bar
        data: DataFrame with OHLCV data
        tick_size: Minimum price tick
        contract_multiplier: Contract size multiplier
        initial_capital: Starting capital (default from config)
        commission_rate: Commission rate (default from config)
        slippage_ticks: Slippage ticks (default from config)

    Returns:
        BacktestResult with performance metrics
    """
    if initial_capital is None:
        initial_capital = BacktestConfig.initial_capital
    if commission_rate is None:
        commission_rate = BacktestConfig.commission_rate
    if slippage_ticks is None:
        slippage_ticks = BacktestConfig.slippage_ticks

    # Track position and trades
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]
    capital = initial_capital

    # Iterate through bars
    for i in range(len(data)):
        signal = signals.iloc[i] if i < len(signals) else 0
        close_price = data["close"].iloc[i]

        # Open or close position based on signal
        if signal == 1 and position == 0:  # Buy
            position = 1
            entry_price = close_price
            # Deduct cost
            trade_value = close_price * contract_multiplier
            cost = calc_total_cost(trade_value, 1, tick_size, commission_rate, slippage_ticks)
            capital -= cost

        elif signal == -1 and position == 1:  # Sell (close long)
            pnl = (close_price - entry_price) * contract_multiplier
            trade_value = close_price * contract_multiplier
            cost = calc_total_cost(trade_value, 1, tick_size, commission_rate, slippage_ticks)
            net_pnl = pnl - cost
            capital += net_pnl

            trades.append({
                "entry_price": entry_price,
                "exit_price": close_price,
                "pnl": pnl,
                "net_pnl": net_pnl,
            })

            position = 0
            entry_price = 0

        # Update equity
        if position == 1:
            unrealized_pnl = (close_price - entry_price) * contract_multiplier
            equity_curve.append(capital + unrealized_pnl)
        else:
            equity_curve.append(capital)

    # Calculate metrics
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    # Basic metrics
    net_return = (capital - initial_capital) / initial_capital
    total_trades = len(trades) * 2  # Each round trip = 2 trades

    # Win rate
    winning_trades = [t for t in trades if t["net_pnl"] > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    # Sharpe ratio (assuming 252 trading days, 16 hours/day, 4 bars/hour)
    trading_bars_per_year = 252 * 16 * 4
    annual_return = net_return * (trading_bars_per_year / len(data))
    volatility = returns.std() * np.sqrt(trading_bars_per_year) if len(returns) > 0 else 0
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_bars_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # VaR (95%)
    var_95 = abs(np.percentile(returns, 5)) if len(returns) > 0 else 0

    # Average holding bars
    avg_holding_bars = len(data) / len(trades) if trades else 0

    # Signal quality (simplified)
    signal_bars = (signals != 0).sum()
    signal_precision = win_rate  # Simplified
    signal_recall = len(trades) / max(signal_bars / 2, 1) if signal_bars > 0 else 0

    return BacktestResult(
        net_return=net_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        var_95=var_95,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_holding_bars=avg_holding_bars,
        signal_precision=signal_precision,
        signal_recall=signal_recall,
    )


def run_multi_backtest(
    signals_dict: dict,
    data_dict: dict,
    tick_sizes: dict,
    contract_multipliers: dict,
    **kwargs,
) -> dict:
    """
    Run backtest on multiple symbols.

    Args:
        signals_dict: Dict mapping symbol to signal Series
        data_dict: Dict mapping symbol to price DataFrame
        tick_sizes: Dict mapping symbol to tick size
        contract_multipliers: Dict mapping symbol to contract multiplier
        **kwargs: Additional arguments passed to run_backtest

    Returns:
        Dict mapping symbol to BacktestResult
    """
    results = {}
    for symbol in signals_dict:
        if symbol not in data_dict:
            continue
        results[symbol] = run_backtest(
            signals=signals_dict[symbol],
            data=data_dict[symbol],
            tick_size=tick_sizes.get(symbol, 1.0),
            contract_multiplier=contract_multipliers.get(symbol, 10),
            **kwargs,
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py::TestRunBacktest -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/backtest.py tests/test_backtest.py
git commit -m "feat: add core backtest functions with performance metrics"
```

---

## Chunk 4: 信号规则模块

### Task 10: 动能信号

**Files:**
- Create: `autoresearch_futures/signals.py`
- Test: `tests/test_signals.py`

- [ ] **Step 1: Write the failing test for momentum signals**

```python
# tests/test_signals.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.signals'"

- [ ] **Step 3: Implement momentum signals**

```python
# autoresearch_futures/signals.py
"""
Signal generation module for futures trading strategies.
Implements SMC, momentum, and linear extrapolation signals.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


# Default parameters for each theory
DEFAULT_MOMENTUM_PARAMS = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}

DEFAULT_SMC_PARAMS = {
    "ob_lookback": 20,
    "fvg_min_size": 0.002,
    "sweep_threshold": 0.01,
}

DEFAULT_LINEAR_PARAMS = {
    "regression_period": 20,
    "band_std": 2.0,
    "breakout_confirm": 3,
}


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """Calculate MACD indicator."""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def momentum_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate momentum-based trading signals.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for momentum indicators

    Returns:
        Dict with RSI, MACD, velocity, and combined signal
    """
    if params is None:
        params = DEFAULT_MOMENTUM_PARAMS

    close = df["close"]

    # RSI
    rsi = calc_rsi(close, params["rsi_period"])

    # MACD
    macd, macd_signal, macd_hist = calc_macd(
        close,
        params["macd_fast"],
        params["macd_slow"],
        params["macd_signal"],
    )

    # Price velocity (rate of change)
    velocity = close.pct_change(params["rsi_period"]) * 100

    # Generate signals
    signal = pd.Series(0, index=df.index)

    # RSI signals
    rsi_buy = rsi < params["rsi_oversold"]
    rsi_sell = rsi > params["rsi_overbought"]

    # MACD signals
    macd_buy = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    macd_sell = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))

    # Combine signals
    signal[rsi_buy | macd_buy] = 1
    signal[rsi_sell | macd_sell] = -1

    return {
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "velocity": velocity,
        "signal": signal,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals.py::TestMomentumSignals -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/signals.py tests/test_signals.py
git commit -m "feat: add momentum signal generation (RSI, MACD)"
```

### Task 11: SMC信号

**Files:**
- Modify: `autoresearch_futures/signals.py`
- Modify: `tests/test_signals.py`

- [ ] **Step 1: Write the failing test for SMC signals**

```python
# Add to tests/test_signals.py

class TestSMCSignals:
    @pytest.fixture
    def sample_data(self):
        """Create sample price data with SMC patterns."""
        dates = pd.date_range("2024-01-01 09:00", periods=200, freq="15min")
        # Create data with potential OB and FVG patterns
        prices = 3600 + np.cumsum(np.random.randn(200) * 3)
        df = pd.DataFrame({
            "datetime": dates,
            "open": prices,
            "high": prices + np.abs(np.random.randn(200) * 5),
            "low": prices - np.abs(np.random.randn(200) * 5),
            "close": prices + np.random.randn(200) * 2,
            "volume": [10000] * 200,
        })
        return df

    def test_smc_signals_returns_dict(self, sample_data):
        """smc_signals should return a dict with expected keys."""
        result = smc_signals(sample_data, DEFAULT_SMC_PARAMS)
        assert isinstance(result, dict)
        assert "order_block" in result
        assert "fvg" in result
        assert "signal" in result

    def test_order_block_detection(self, sample_data):
        """Order blocks should be detected."""
        result = smc_signals(sample_data, DEFAULT_SMC_PARAMS)
        ob = result["order_block"]
        # OB should be boolean or binary
        assert ob.dtype in [bool, np.int64, np.float64]

    def test_fvg_detection(self, sample_data):
        """FVG should be detected."""
        result = smc_signals(sample_data, DEFAULT_SMC_PARAMS)
        fvg = result["fvg"]
        # FVG should be boolean or binary
        assert fvg.dtype in [bool, np.int64, np.float64]

    def test_smc_signal_range(self, sample_data):
        """SMC signal should be in [-1, 0, 1]."""
        result = smc_signals(sample_data, DEFAULT_SMC_PARAMS)
        signal = result["signal"]
        assert signal.min() >= -1
        assert signal.max() <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals.py::TestSMCSignals -v`
Expected: FAIL with "NameError: name 'smc_signals' is not defined"

- [ ] **Step 3: Implement SMC signals**

```python
# Add to autoresearch_futures/signals.py

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect Order Blocks (OB).
    An OB is the last opposite candle before a strong move.

    Bullish OB: Last bearish candle before strong bullish move
    Bearish OB: Last bullish candle before strong bearish move
    """
    ob = pd.Series(0, index=df.index)

    for i in range(lookback, len(df)):
        # Check for strong move in next few bars
        future_high = df["high"].iloc[i:i+5].max()
        future_low = df["low"].iloc[i:i+5].min()
        current_close = df["close"].iloc[i]

        # Strong bullish move
        if future_high > current_close * 1.01:  # 1% move up
            # Check if this was a bearish candle
            if df["close"].iloc[i] < df["open"].iloc[i]:
                ob.iloc[i] = 1

        # Strong bearish move
        if future_low < current_close * 0.99:  # 1% move down
            # Check if this was a bullish candle
            if df["close"].iloc[i] > df["open"].iloc[i]:
                ob.iloc[i] = -1

    return ob


def detect_fvg(
    df: pd.DataFrame,
    min_size: float = 0.002,
) -> pd.Series:
    """
    Detect Fair Value Gaps (FVG).
    A FVG occurs when there's a gap between candles.

    Bullish FVG: Candle 1 high < Candle 3 low
    Bearish FVG: Candle 1 low > Candle 3 high
    """
    fvg = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        # Bullish FVG
        if df["high"].iloc[i-2] < df["low"].iloc[i]:
            gap_size = (df["low"].iloc[i] - df["high"].iloc[i-2]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = 1

        # Bearish FVG
        if df["low"].iloc[i-2] > df["high"].iloc[i]:
            gap_size = (df["low"].iloc[i-2] - df["high"].iloc[i]) / df["close"].iloc[i-1]
            if gap_size >= min_size:
                fvg.iloc[i-1] = -1

    return fvg


def detect_liquidity_sweep(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Detect liquidity sweeps.
    A sweep occurs when price breaks a swing high/low then reverses.
    """
    sweep = pd.Series(0, index=df.index)

    # Find swing highs and lows
    for i in range(5, len(df) - 5):
        # Check for sweep of recent high
        recent_high = df["high"].iloc[i-5:i].max()
        if df["high"].iloc[i] > recent_high:
            # Check if we reversed (close below recent high)
            if df["close"].iloc[i] < recent_high * (1 - threshold):
                sweep.iloc[i] = -1

        # Check for sweep of recent low
        recent_low = df["low"].iloc[i-5:i].min()
        if df["low"].iloc[i] < recent_low:
            # Check if we reversed (close above recent low)
            if df["close"].iloc[i] > recent_low * (1 + threshold):
                sweep.iloc[i] = 1

    return sweep


def smc_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate SMC (Smart Money Concept) trading signals.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for SMC detection

    Returns:
        Dict with OB, FVG, liquidity sweep, and combined signal
    """
    if params is None:
        params = DEFAULT_SMC_PARAMS

    # Detect patterns
    order_block = detect_order_blocks(df, params["ob_lookback"])
    fvg = detect_fvg(df, params["fvg_min_size"])
    sweep = detect_liquidity_sweep(df, params["sweep_threshold"])

    # Generate combined signal
    signal = pd.Series(0, index=df.index)

    # Bullish: OB + FVG or sweep
    bullish = ((order_block == 1) & (fvg == 1)) | (sweep == 1)
    # Bearish: OB + FVG or sweep
    bearish = ((order_block == -1) & (fvg == -1)) | (sweep == -1)

    signal[bullish] = 1
    signal[bearish] = -1

    return {
        "order_block": order_block,
        "fvg": fvg,
        "liquidity_sweep": sweep,
        "signal": signal,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals.py::TestSMCSignals -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/signals.py tests/test_signals.py
git commit -m "feat: add SMC signal generation (OB, FVG, liquidity sweep)"
```

### Task 12: 线性推演信号

**Files:**
- Modify: `autoresearch_futures/signals.py`
- Modify: `tests/test_signals.py`

- [ ] **Step 1: Write the failing test for linear signals**

```python
# Add to tests/test_signals.py

class TestLinearSignals:
    @pytest.fixture
    def sample_data(self):
        """Create sample price data with trend."""
        dates = pd.date_range("2024-01-01 09:00", periods=200, freq="15min")
        # Create trending data
        trend = np.linspace(0, 100, 200)
        noise = np.random.randn(200) * 5
        prices = 3600 + trend + noise
        df = pd.DataFrame({
            "datetime": dates,
            "open": prices,
            "high": prices + 5,
            "low": prices - 5,
            "close": prices,
            "volume": [10000] * 200,
        })
        return df

    def test_linear_signals_returns_dict(self, sample_data):
        """linear_signals should return a dict with expected keys."""
        result = linear_signals(sample_data, DEFAULT_LINEAR_PARAMS)
        assert isinstance(result, dict)
        assert "trend" in result
        assert "regression_band" in result
        assert "signal" in result

    def test_trend_detection(self, sample_data):
        """Trend should be detected."""
        result = linear_signals(sample_data, DEFAULT_LINEAR_PARAMS)
        trend = result["trend"]
        # Trend should be -1, 0, or 1
        assert set(trend.dropna().unique()).issubset({-1, 0, 1})

    def test_regression_band(self, sample_data):
        """Regression band should be calculated."""
        result = linear_signals(sample_data, DEFAULT_LINEAR_PARAMS)
        band = result["regression_band"]
        # Band should have upper and lower
        assert "upper" in band
        assert "lower" in band

    def test_linear_signal_range(self, sample_data):
        """Linear signal should be in [-1, 0, 1]."""
        result = linear_signals(sample_data, DEFAULT_LINEAR_PARAMS)
        signal = result["signal"]
        assert signal.min() >= -1
        assert signal.max() <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signals.py::TestLinearSignals -v`
Expected: FAIL with "NameError: name 'linear_signals' is not defined"

- [ ] **Step 3: Implement linear signals**

```python
# Add to autoresearch_futures/signals.py

def calc_regression_band(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict:
    """
    Calculate linear regression channel.

    Returns dict with regression line, upper and lower bands.
    """
    # Rolling linear regression
    regression = close.rolling(window=period).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] * (len(x) - 1) + x.iloc[0],
        raw=False,
    )

    # Calculate residuals for bands
    residuals = close - regression
    std = residuals.rolling(window=period).std()

    upper = regression + num_std * std
    lower = regression - num_std * std

    return {
        "regression": regression,
        "upper": upper,
        "lower": lower,
        "std": std,
    }


def detect_trend(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    Detect trend direction using linear regression slope.
    """
    close = df["close"]
    slopes = close.rolling(window=period).apply(
        lambda x: stats.linregress(range(len(x)), x)[0],
        raw=False,
    )

    # Normalize slope
    normalized_slope = slopes / close * 100

    trend = pd.Series(0, index=df.index)
    trend[normalized_slope > 0.01] = 1   # Uptrend
    trend[normalized_slope < -0.01] = -1  # Downtrend

    return trend


def detect_breakout(
    df: pd.DataFrame,
    band: dict,
    confirm_bars: int = 3,
) -> pd.Series:
    """
    Detect breakouts from regression channel.
    """
    close = df["close"]
    breakout = pd.Series(0, index=df.index)

    # Breakout above upper band
    above_upper = close > band["upper"]
    # Confirm with multiple bars
    for i in range(confirm_bars, len(df)):
        if above_upper.iloc[i-confirm_bars:i].all():
            breakout.iloc[i] = 1

    # Breakout below lower band
    below_lower = close < band["lower"]
    for i in range(confirm_bars, len(df)):
        if below_lower.iloc[i-confirm_bars:i].all():
            breakout.iloc[i] = -1

    return breakout


def linear_signals(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Generate linear extrapolation trading signals.

    Uses linear regression channels and trend analysis.

    Args:
        df: DataFrame with OHLCV data
        params: Parameters for linear signals

    Returns:
        Dict with trend, regression band, and combined signal
    """
    if params is None:
        params = DEFAULT_LINEAR_PARAMS

    close = df["close"]

    # Calculate regression band
    band = calc_regression_band(
        close,
        params["regression_period"],
        params["band_std"],
    )

    # Detect trend
    trend = detect_trend(df, params["regression_period"])

    # Detect breakout
    breakout = detect_breakout(df, band, params["breakout_confirm"])

    # Generate combined signal
    signal = pd.Series(0, index=df.index)

    # Buy: Uptrend + price near lower band (mean reversion in trend)
    buy = (trend == 1) & (close <= band["regression"])
    # Or: Breakout above upper band
    buy = buy | (breakout == 1)

    # Sell: Downtrend + price near upper band
    sell = (trend == -1) & (close >= band["regression"])
    # Or: Breakout below lower band
    sell = sell | (breakout == -1)

    signal[buy] = 1
    signal[sell] = -1

    return {
        "trend": trend,
        "regression_band": band,
        "breakout": breakout,
        "signal": signal,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signals.py::TestLinearSignals -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/signals.py tests/test_signals.py
git commit -m "feat: add linear extrapolation signals (trend, regression band, breakout)"
```

---

## Chunk 5: 多理论集成模块

### Task 13: 信号集成函数

**Files:**
- Create: `autoresearch_futures/ensemble.py`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write the failing test for ensemble**

```python
# tests/test_ensemble.py
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
        # Should only have signal when all theories agree
        for i in range(len(result)):
            if result.iloc[i] != 0:
                # All signals should be same direction
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ensemble.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.ensemble'"

- [ ] **Step 3: Implement ensemble module**

```python
# autoresearch_futures/ensemble.py
"""
Multi-theory ensemble module.
Combines signals from SMC, momentum, and linear theories.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1."""
    total = sum(weights.values())
    if total == 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def simple_vote(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
    threshold: float = 0.5,
) -> pd.Series:
    """
    Combine signals using weighted voting.

    Args:
        signals: Dict mapping theory name to signal Series
        weights: Dict mapping theory name to weight
        threshold: Minimum weighted sum to generate signal

    Returns:
        Combined signal Series
    """
    weights = normalize_weights(weights)

    # Calculate weighted sum
    weighted_sum = pd.Series(0.0, index=next(iter(signals.values())).index)
    for theory, signal in signals.items():
        weighted_sum += signal * weights.get(theory, 0)

    # Apply threshold
    result = pd.Series(0, index=weighted_sum.index)
    result[weighted_sum >= threshold] = 1
    result[weighted_sum <= -threshold] = -1

    return result


def consensus_filter(signals: Dict[str, pd.Series]) -> pd.Series:
    """
    Only generate signal when all theories agree.

    Args:
        signals: Dict mapping theory name to signal Series

    Returns:
        Combined signal Series (only non-zero when all agree)
    """
    signal_list = list(signals.values())

    # Check for unanimous agreement
    all_bullish = all(s == 1 for s in signal_list)
    all_bearish = all(s == -1 for s in signal_list)

    result = pd.Series(0, index=signal_list[0].index)
    result[all_bullish] = 1
    result[all_bearish] = -1

    return result


def calc_confidence(
    signals_at_point: Dict[str, int],
    weights: Dict[str, float],
) -> float:
    """
    Calculate confidence based on signal agreement.

    Confidence is higher when more theories agree and have higher weights.

    Args:
        signals_at_point: Dict mapping theory to signal value at a point
        weights: Dict mapping theory to weight

    Returns:
        Confidence value between 0 and 1
    """
    weights = normalize_weights(weights)

    # Sum of absolute weighted signals
    total_weight = sum(abs(v) * weights.get(k, 0) for k, v in signals_at_point.items())

    # Direction agreement
    non_zero_signals = [v for v in signals_at_point.values() if v != 0]
    if len(non_zero_signals) == 0:
        return 0.0

    # Check if all non-zero signals agree
    direction = non_zero_signals[0]
    agreement = all(s == direction for s in non_zero_signals)

    if agreement:
        # Confidence = weighted agreement
        return min(total_weight, 1.0)
    else:
        # Conflicting signals reduce confidence
        return total_weight * 0.5


def ensemble_signals(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
    mode: str = "vote",
) -> Dict:
    """
    Combine signals from multiple theories.

    Args:
        signals: Dict mapping theory name to signal Series
        weights: Dict mapping theory name to weight
        mode: Combination mode ('vote' or 'consensus')

    Returns:
        Dict with combined signal and confidence Series
    """
    if mode == "vote":
        combined = simple_vote(signals, weights)
    elif mode == "consensus":
        combined = consensus_filter(signals)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Calculate confidence for each point
    confidence = pd.Series(0.0, index=combined.index)
    for i in range(len(combined)):
        signals_at_i = {k: v.iloc[i] for k, v in signals.items()}
        confidence.iloc[i] = calc_confidence(signals_at_i, weights)

    return {
        "signal": combined,
        "confidence": confidence,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ensemble.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/ensemble.py tests/test_ensemble.py
git commit -m "feat: add multi-theory ensemble with voting and consensus modes"
```

---

## Chunk 6: 信号推送模块

### Task 14: 推送功能

**Files:**
- Create: `autoresearch_futures/notify.py`
- Test: `tests/test_notify.py`

- [ ] **Step 1: Write the failing test for notify**

```python
# tests/test_notify.py
"""Tests for notify module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from autoresearch_futures.notify import (
    format_signal_message,
    push_signal,
    should_push_signal,
    NotifyConfig,
)


class TestNotify:
    @pytest.fixture
    def config(self):
        """Create test config."""
        return NotifyConfig(
            wechat_enabled=True,
            wechat_webhook="https://example.com/webhook",
            telegram_enabled=False,
            email_enabled=False,
        )

    @pytest.fixture
    def signal_event(self):
        """Create test signal event."""
        from dataclasses import dataclass
        @dataclass
        class SignalEvent:
            symbol: str
            direction: int
            timestamp: datetime
            price: float
            confidence: float
            sources: list
            suggested_stop: float
            suggested_target: float

        return SignalEvent(
            symbol="rb2405",
            direction=1,
            timestamp=datetime(2024, 3, 23, 14, 30),
            price=3650.0,
            confidence=0.75,
            sources=["SMC: Order Block 支撑", "动能: RSI 超卖反弹"],
            suggested_stop=3620.0,
            suggested_target=3720.0,
        )

    def test_format_signal_message(self, signal_event):
        """format_signal_message should format correctly."""
        msg = format_signal_message(signal_event)
        assert "rb2405" in msg
        assert "做多" in msg
        assert "3650" in msg
        assert "75%" in msg

    @patch("requests.post")
    def test_push_signal_wechat(self, mock_post, signal_event, config):
        """push_signal should call wechat webhook."""
        mock_post.return_value = MagicMock(status_code=200)

        push_signal(signal_event, config)

        # Should have called webhook
        assert mock_post.called
        call_args = mock_post.call_args
        assert config.wechat_webhook in call_args[0]

    def test_should_push_signal_cooldown(self):
        """should_push_signal should respect cooldown."""
        # First signal should be allowed
        assert should_push_signal("rb", 1) is True

        # Record the signal
        from autoresearch_futures.notify import record_signal
        record_signal("rb", 1, datetime.now())

        # Immediate second signal should be blocked
        assert should_push_signal("rb", 1) is False

        # After cooldown should be allowed
        from autoresearch_futures.notify import SIGNAL_COOLDOWN_MINUTES
        future_time = datetime.now() + timedelta(minutes=SIGNAL_COOLDOWN_MINUTES + 1)
        assert should_push_signal("rb", 1, current_time=future_time) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_notify.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.notify'"

- [ ] **Step 3: Implement notify module**

```python
# autoresearch_futures/notify.py
"""
Signal notification module.
Pushes trading signals via WeChat, Telegram, and Email.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from autoresearch_futures.config import NotifyConfig

# Signal cooldown
SIGNAL_COOLDOWN_MINUTES = 30

# In-memory store for last signal times (would be persisted in production)
_last_signals: Dict[str, datetime] = {}


@dataclass
class SignalEvent:
    """Container for a trading signal event."""
    symbol: str
    direction: int  # -1, 0, 1
    timestamp: datetime
    price: float
    confidence: float
    sources: List[str]
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None


def format_signal_message(signal: SignalEvent) -> str:
    """Format signal as human-readable message."""
    direction_text = "做多" if signal.direction == 1 else "做空"

    sources_text = "\n".join(f"- {s}" for s in signal.sources)

    msg = f"""【期货信号提醒】
品种: {signal.symbol}
方向: {direction_text}
时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
价格: {signal.price:.2f}
置信度: {signal.confidence:.0%}

理论依据:
{sources_text}
"""
    if signal.suggested_stop:
        msg += f"\n建议止损: {signal.suggested_stop:.2f}"
    if signal.suggested_target:
        msg += f"\n建议止盈: {signal.suggested_target:.2f}"

    return msg


def push_wechat(message: str, webhook: str) -> bool:
    """Push message to WeChat (企业微信机器人)."""
    try:
        response = requests.post(
            webhook,
            json={"msgtype": "text", "text": {"content": message}},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"WeChat push failed: {e}")
        return False


def push_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """Push message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram push failed: {e}")
        return False


def push_email(message: str, config: NotifyConfig) -> bool:
    """Push message via email."""
    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = "期货信号提醒"
        msg["From"] = config.email_sender
        msg["To"] = ", ".join(config.email_recipients)

        with smtplib.SMTP(config.email_smtp, 587) as server:
            server.starttls()
            server.login(config.email_sender, config.email_password)
            server.sendmail(
                config.email_sender,
                config.email_recipients,
                msg.as_string(),
            )
        return True
    except Exception as e:
        print(f"Email push failed: {e}")
        return False


def record_signal(symbol: str, direction: int, timestamp: datetime) -> None:
    """Record signal time for cooldown tracking."""
    key = f"{symbol}_{direction}"
    _last_signals[key] = timestamp


def should_push_signal(
    symbol: str,
    direction: int,
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if signal should be pushed (respecting cooldown)."""
    if current_time is None:
        current_time = datetime.now()

    key = f"{symbol}_{direction}"
    last_time = _last_signals.get(key)

    if last_time is None:
        return True

    elapsed = current_time - last_time
    return elapsed.total_seconds() > SIGNAL_COOLDOWN_MINUTES * 60


def push_signal(signal: SignalEvent, config: NotifyConfig) -> Dict[str, bool]:
    """
    Push signal to all enabled channels.

    Returns:
        Dict mapping channel name to success status
    """
    message = format_signal_message(signal)
    results = {}

    if config.wechat_enabled and config.wechat_webhook:
        results["wechat"] = push_wechat(message, config.wechat_webhook)

    if config.telegram_enabled and config.telegram_bot_token:
        results["telegram"] = push_telegram(
            message,
            config.telegram_bot_token,
            config.telegram_chat_id,
        )

    if config.email_enabled:
        results["email"] = push_email(message, config)

    # Record signal time
    if any(results.values()):
        record_signal(signal.symbol, signal.direction, signal.timestamp)

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_notify.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/notify.py tests/test_notify.py
git commit -m "feat: add signal notification via WeChat/Telegram/Email"
```

---

## Chunk 7: 进化主循环模块

### Task 15: 进化主循环

**Files:**
- Create: `autoresearch_futures/evolve.py`
- Test: `tests/test_evolve.py`

- [ ] **Step 1: Write the failing test for evolve**

```python
# tests/test_evolve.py
"""Tests for evolve module."""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from autoresearch_futures.evolve import (
    generate_all_signals,
    aggregate_scores,
    log_results,
    get_current_split,
    advance_split,
)


class TestEvolve:
    @pytest.fixture
    def sample_data(self):
        """Create sample data dict."""
        dates = pd.date_range("2024-01-01 09:00", periods=500, freq="15min")
        df = pd.DataFrame({
            "datetime": dates,
            "open": [3600.0] * 500,
            "high": [3700.0] * 500,
            "low": [3500.0] * 500,
            "close": [3650.0] * 500,
            "volume": [100000] * 500,
        })
        return {"rb": df, "i": df.copy()}

    def test_generate_all_signals(self, sample_data):
        """generate_all_signals should generate signals for all symbols."""
        from autoresearch_futures.config import DEFAULT_PARAMS
        signals = generate_all_signals(sample_data, DEFAULT_PARAMS)
        assert "rb" in signals
        assert "i" in signals
        # Each should have combined signal
        assert "signal" in signals["rb"]

    def test_aggregate_scores(self):
        """aggregate_scores should calculate average score."""
        from autoresearch_futures.backtest import BacktestResult, calc_score
        from autoresearch_futures.config import SCORE_WEIGHTS

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
        # Should be average of individual scores
        score_rb = calc_score(results["rb"], SCORE_WEIGHTS)
        score_i = calc_score(results["i"], SCORE_WEIGHTS)
        expected = (score_rb + score_i) / 2
        assert abs(score - expected) < 0.001

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

        tsv_path = tmp_path / "results.tsv"
        log_results(
            split_id="001",
            commit="abc123",
            score=0.85,
            results=results,
            description="test run",
            filepath=str(tsv_path),
        )

        assert tsv_path.exists()
        content = tsv_path.read_text()
        assert "001" in content
        assert "abc123" in content
        assert "test run" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'autoresearch_futures.evolve'"

- [ ] **Step 3: Implement evolve module**

```python
# autoresearch_futures/evolve.py
"""
Evolution main loop for futures strategy research.
Orchestrates signal generation, backtesting, and strategy selection.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from autoresearch_futures.config import (
    DEFAULT_PARAMS,
    THEORY_WEIGHTS,
    SCORE_WEIGHTS,
)
from autoresearch_futures.signals import (
    momentum_signals,
    smc_signals,
    linear_signals,
)
from autoresearch_futures.ensemble import ensemble_signals
from autoresearch_futures.backtest import (
    run_multi_backtest,
    calc_score,
    BacktestResult,
)


# Current split state (would be persisted in production)
_current_split_index = 0
_splits: List[dict] = []


def generate_all_signals(
    data_dict: Dict[str, pd.DataFrame],
    params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Generate signals from all theories for all symbols.

    Args:
        data_dict: Dict mapping symbol to price DataFrame
        params: Signal parameters

    Returns:
        Dict mapping symbol to signal dict
    """
    if params is None:
        params = DEFAULT_PARAMS

    all_signals = {}

    for symbol, df in data_dict.items():
        # Generate signals from each theory
        smc = smc_signals(df, params.get("smc"))
        momentum = momentum_signals(df, params.get("momentum"))
        linear = linear_signals(df, params.get("linear"))

        # Ensemble
        combined = ensemble_signals(
            {
                "smc": smc["signal"],
                "momentum": momentum["signal"],
                "linear": linear["signal"],
            },
            THEORY_WEIGHTS,
        )

        all_signals[symbol] = {
            "smc": smc,
            "momentum": momentum,
            "linear": linear,
            "signal": combined["signal"],
            "confidence": combined["confidence"],
        }

    return all_signals


def aggregate_scores(
    results: Dict[str, BacktestResult],
    weights: Optional[Dict] = None,
) -> float:
    """
    Calculate aggregate score across multiple symbols.

    Args:
        results: Dict mapping symbol to BacktestResult
        weights: Score weights

    Returns:
        Average score across symbols
    """
    if weights is None:
        weights = SCORE_WEIGHTS

    scores = [calc_score(r, weights) for r in results.values()]
    return sum(scores) / len(scores) if scores else 0.0


def log_results(
    split_id: str,
    commit: str,
    score: float,
    results: Dict[str, BacktestResult],
    description: str,
    filepath: str = "results.tsv",
) -> None:
    """
    Log evolution results to TSV file.

    Args:
        split_id: Split identifier
        commit: Git commit hash
        score: Aggregate score
        results: Backtest results dict
        description: Experiment description
        filepath: Path to TSV file
    """
    # Calculate aggregate metrics
    avg_sharpe = sum(r.sharpe_ratio for r in results.values()) / len(results)
    avg_return = sum(r.net_return for r in results.values()) / len(results)
    avg_win_rate = sum(r.win_rate for r in results.values()) / len(results)
    max_dd = max(r.max_drawdown for r in results.values())

    # Check if file exists (write header if not)
    write_header = not os.path.exists(filepath)

    with open(filepath, "a") as f:
        if write_header:
            f.write("split_id\tcommit\tscore\tsharpe\tnet_return\twin_rate\tmax_dd\tdescription\n")

        f.write(f"{split_id}\t{commit}\t{score:.6f}\t{avg_sharpe:.4f}\t{avg_return:.4f}\t{avg_win_rate:.4f}\t{max_dd:.4f}\t{description}\n")


def get_current_split() -> dict:
    """Get the current split for evolution."""
    global _current_split_index, _splits
    if _current_split_index < len(_splits):
        return _splits[_current_split_index]
    return {}


def advance_split() -> bool:
    """Advance to the next split. Returns False if no more splits."""
    global _current_split_index, _splits
    _current_split_index += 1
    return _current_split_index < len(_splits)


def set_splits(splits: List[dict]) -> None:
    """Set the splits for evolution."""
    global _splits, _current_split_index
    _splits = splits
    _current_split_index = 0


def run_evolution_step(
    data_dict: Dict[str, pd.DataFrame],
    params: Dict,
    tick_sizes: Dict[str, float],
    contract_multipliers: Dict[str, int],
) -> tuple:
    """
    Run a single evolution step.

    Returns:
        Tuple of (score, results_dict, signals_dict)
    """
    # Generate signals
    signals_dict = generate_all_signals(data_dict, params)

    # Extract combined signals for backtest
    backtest_signals = {
        symbol: signals_dict[symbol]["signal"]
        for symbol in signals_dict
    }

    # Run backtest
    results = run_multi_backtest(
        backtest_signals,
        data_dict,
        tick_sizes,
        contract_multipliers,
    )

    # Calculate aggregate score
    score = aggregate_scores(results)

    return score, results, signals_dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolve.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch_futures/evolve.py tests/test_evolve.py
git commit -m "feat: add evolution main loop with signal generation and scoring"
```

---

## Chunk 8: Agent指令文件与入口脚本

### Task 16: 创建 program.md

**Files:**
- Create: `program.md`

- [ ] **Step 1: Write program.md**

```markdown
# autoresearch-futures

这是期货策略自主研究实验。

## Setup

1. 确认分支: `git checkout -b autoresearch/run-$(date +%Y%m%d)`
2. 读取文件:
   - `prepare.py` — 数据准备（固定）
   - `signals.py` — 信号规则（可修改）
   - `evolve.py` — 进化循环（可修改）
   - `backtest.py` — 回测引擎（固定）
3. 验证数据: `ls ~/.cache/autoresearch-futures/raw/`
   - 如无数据，运行: `python -m autoresearch_futures.prepare`
4. 确认配置: 检查 `config.py` 中的参数

## 进化规则

**可以修改:**
- `signals.py` — 信号规则逻辑、参数阈值
- `evolve.py` 中的 `DEFAULT_PARAMS` — 各理论参数
- `evolve.py` 中的 `THEORY_WEIGHTS` — 理论权重

**禁止修改:**
- `prepare.py` — 数据准备逻辑固定
- `backtest.py` — 回测引擎固定
- `config.py` 中的数据划分参数（train_window_months 等）

**目标: 最大化验证集上的综合得分**

得分公式:
```
score = 0.25*夏普 + 0.30*扣费收益 + 0.15*胜率 + 0.15*精确率 - 0.15*最大回撤
```

## 实验循环

```
LOOP:
1. 修改 signals.py 或 evolve.py 中的参数/规则
2. git commit -m "experiment: 描述改动"
3. 运行: python -m autoresearch_futures.evolve > run.log 2>&1
4. 提取结果: grep "^score:" run.log
5. 记录到 results.tsv
6. 如果得分提高: 保留提交
7. 如果得分降低: git reset --hard HEAD~1
8. 继续下一个实验
```

## 结果日志格式

results.tsv (TSV格式，不要提交到git):
```
split_id	commit	score	sharpe	net_return	win_rate	max_dd	description
001	abc123	0.850000	1.2400	0.1520	0.5800	0.0800	baseline
002	def456	0.920000	1.4500	0.1780	0.6200	0.0700	increase RSI period to 21
```

## 进化操作类型

| 操作 | 文件 | 示例 |
|------|------|------|
| 参数调整 | evolve.py | `"rsi_period": 14 → 21` |
| 阈值调整 | signals.py | `rsi_overbought: 70 → 75` |
| 信号组合 | signals.py | `signal = smc_signal & momentum_signal` |
| 周期切换 | evolve.py | `"timeframe": "1h" → "2h"` |
| 集成权重 | evolve.py | `"smc": 0.35 → 0.40` |

## 注意事项

- 简单优于复杂：相同表现时选择更简单的规则
- 永不停止：直到人工中断
- 不要在同一窗口上反复测试（过拟合风险）
- 记录每个实验的想法和结果
```

- [ ] **Step 2: Commit**

```bash
git add program.md
git commit -m "docs: add program.md with agent instructions for autonomous research"
```

### Task 17: 创建主入口脚本

**Files:**
- Create: `autoresearch_futures/__main__.py`

- [ ] **Step 1: Write __main__.py**

```python
# autoresearch_futures/__main__.py
"""
Main entry point for autoresearch-futures.
Run with: python -m autoresearch_futures [command]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch-futures: Self-evolving futures strategy research"
    )
    parser.add_argument(
        "command",
        choices=["prepare", "evolve", "test"],
        help="Command to run",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload data",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        from autoresearch_futures.prepare import (
            download_all_contracts,
            ensure_dirs,
            save_splits,
            generate_walk_forward_splits,
            list_available_symbols,
        )

        ensure_dirs()
        print("Downloading futures data...")
        download_all_contracts(symbols=args.symbols, force=args.force)

        # Generate splits
        symbols = list_available_symbols()
        if symbols:
            # Use first symbol to get date range
            import pandas as pd
            from autoresearch_futures.prepare import load_data
            df = load_data(symbols[0])
            start = df["datetime"].min().strftime("%Y-%m-%d")
            end = df["datetime"].max().strftime("%Y-%m-%d")

            splits = generate_walk_forward_splits(start, end)
            save_splits(splits)
            print(f"Generated {len(splits)} walk-forward splits")

        print("Done!")

    elif args.command == "evolve":
        from autoresearch_futures.evolve import (
            set_splits,
            run_evolution_step,
            log_results,
            get_current_split,
            advance_split,
        )
        from autoresearch_futures.prepare import (
            load_splits,
            load_split_data,
            list_available_symbols,
        )
        from autoresearch_futures.config import DEFAULT_PARAMS
        import subprocess

        # Load splits
        try:
            splits = load_splits()
            set_splits(splits)
        except FileNotFoundError:
            print("No splits found. Run 'python -m autoresearch_futures prepare' first.")
            sys.exit(1)

        # Get symbols
        symbols = args.symbols or list_available_symbols()
        if not symbols:
            print("No symbols available. Run 'python -m autoresearch_futures prepare' first.")
            sys.exit(1)

        # TODO: Load tick_sizes and contract_multipliers from config
        tick_sizes = {s: 1.0 for s in symbols}
        contract_multipliers = {s: 10 for s in symbols}

        # Get current commit
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()

        # Run evolution on current split
        split = get_current_split()
        if not split:
            print("No more splits to process.")
            sys.exit(0)

        print(f"Running evolution on split {split['split_id']}...")

        # Load data for this split
        data_dict = {}
        for symbol in symbols:
            data_dict[symbol] = load_split_data(symbol, split, "train")

        # Run evolution step
        score, results, signals = run_evolution_step(
            data_dict,
            DEFAULT_PARAMS,
            tick_sizes,
            contract_multipliers,
        )

        # Output results
        print("---")
        print(f"score:    {score:.6f}")
        for symbol, result in results.items():
            print(f"{symbol}:")
            print(f"  sharpe:   {result.sharpe_ratio:.4f}")
            print(f"  return:   {result.net_return:.4f}")
            print(f"  win_rate: {result.win_rate:.4f}")
            print(f"  max_dd:   {result.max_drawdown:.4f}")

        # Log results
        log_results(
            split_id=split["split_id"],
            commit=commit,
            score=score,
            results=results,
            description="experiment run",
        )

    elif args.command == "test":
        # Run tests
        import subprocess
        result = subprocess.run(
            ["pytest", "tests/", "-v"],
            cwd="/Users/maxclaw/Documents/github/autoresearch",
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch_futures/__main__.py
git commit -m "feat: add main entry point for CLI commands"
```

### Task 18: 更新 pyproject.toml

**Files:**
- Modify: `autoresearch_futures/pyproject.toml`

- [ ] **Step 1: Update pyproject.toml with scripts**

```toml
# autoresearch_futures/pyproject.toml
[project]
name = "autoresearch-futures"
version = "0.1.0"
description = "Self-evolving futures strategy research system"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "akshare>=1.12.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=14.0.0",
    "requests>=2.31.0",
    "scipy>=1.11.0",
    "python-dateutil>=2.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]

[project.scripts]
autoresearch-futures = "autoresearch_futures.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short"
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch_futures/pyproject.toml
git commit -m "chore: update pyproject.toml with CLI script and all dependencies"
```

---

## Summary

**实现计划完成，共 18 个任务，按依赖顺序：**

| Phase | Tasks | Description |
|-------|-------|-------------|
| Chunk 1 | 1-2 | 项目配置与基础设施 |
| Chunk 2 | 3-6 | 数据准备模块 |
| Chunk 3 | 7-9 | 回测引擎模块 |
| Chunk 4 | 10-12 | 信号规则模块 |
| Chunk 5 | 13 | 多理论集成模块 |
| Chunk 6 | 14 | 信号推送模块 |
| Chunk 7 | 15 | 进化主循环模块 |
| Chunk 8 | 16-18 | Agent指令与入口脚本 |

**执行命令：**
```bash
# 安装依赖
cd autoresearch_futures && pip install -e ".[dev]"

# 数据准备
python -m autoresearch_futures prepare

# 运行进化
python -m autoresearch_futures evolve

# 运行测试
python -m autoresearch_futures test
# 或
pytest tests/ -v
```
