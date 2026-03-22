"""
Data preparation module for autoresearch-futures.

Handles:
- Data download from akshare
- K-line timeframe synthesis
- Walk-Forward split generation
- Data loading utilities
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import akshare as ak

from autoresearch_futures.config import DataConfig, Config


# Global config instance
_config = Config()
_data_config = _config.data

# Cache directory (expanded from ~)
CACHE_DIR = os.path.expanduser(_data_config.cache_dir)


# =============================================================================
# Data Download Functions
# =============================================================================

def get_futures_list() -> List[str]:
    """
    Get list of futures symbols from akshare.

    Returns:
        List of futures symbol codes (e.g., ['rb', 'i', 'hc', 'j'])
    """
    try:
        df = ak.futures_main_sina()
        # Extract unique symbols from the returned data
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique().tolist()
        elif '代码' in df.columns:
            symbols = df['代码'].unique().tolist()
        else:
            # Try to extract from first column
            symbols = df.iloc[:, 0].unique().tolist()
        return sorted(symbols)
    except Exception as e:
        print(f"Warning: Could not fetch futures list: {e}")
        # Return common futures symbols as fallback
        return ['rb', 'i', 'hc', 'j', 'jm', 'cu', 'al', 'zn', 'au', 'ag']


def download_contract(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download historical data for a single futures contract.

    Args:
        symbol: Futures symbol code (e.g., 'rb' for rebar)
        start_date: Optional start date filter (YYYY-MM-DD)

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    try:
        # Try to get 15-minute data
        df = ak.futures_zh_minute_sina(symbol=symbol, period="15")

        # Rename columns to standardized names
        column_map = {
            '日期': 'datetime',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            'date': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Ensure datetime column exists and is datetime type
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            # Create datetime from index if needed
            df = df.reset_index()
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            df = df.rename(columns={df.columns[0]: 'datetime'})

        # Filter by start date if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['datetime'] >= start_dt]

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def download_all_contracts(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    save: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple futures symbols.

    Args:
        symbols: List of symbols to download. If None, uses get_futures_list()
        start_date: Optional start date filter
        save: Whether to save data to cache

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if symbols is None:
        symbols = get_futures_list()

    results = {}
    for symbol in symbols:
        df = download_contract(symbol, start_date)
        if not df.empty:
            results[symbol] = df
            if save:
                save_raw_data(symbol, df)
        print(f"Downloaded {symbol}: {len(df)} rows")

    return results


def save_raw_data(symbol: str, df: pd.DataFrame) -> None:
    """Save raw downloaded data to cache."""
    cache_path = Path(CACHE_DIR) / "raw" / f"{symbol}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def load_raw_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load raw downloaded data from cache."""
    cache_path = Path(CACHE_DIR) / "raw" / f"{symbol}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


# =============================================================================
# K-line Timeframe Synthesis
# =============================================================================

def synthesize_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Convert 15min K-lines to a higher timeframe using OHLCV aggregation.

    Args:
        df: DataFrame with 15min data (must have datetime column)
        target_tf: Target timeframe ('30min', '1h', '2h', '4h')

    Returns:
        DataFrame with aggregated OHLCV data
    """
    if df.empty:
        return df.copy()

    # Make a copy and ensure datetime is set as index
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # Map target timeframe to pandas resample rule
    tf_map = {
        '30min': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
    }

    if target_tf not in tf_map:
        raise ValueError(f"Unknown target timeframe: {target_tf}")

    rule = tf_map[target_tf]

    # Aggregate OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    resampled = df.resample(rule).agg(agg_dict).dropna()

    # Reset datetime as column
    resampled = resampled.reset_index()

    return resampled


def synthesize_all_timeframes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Synthesize all configured timeframes from 15min base data.

    Args:
        df: DataFrame with 15min data

    Returns:
        Dictionary mapping timeframe name to DataFrame
    """
    results = {}
    for tf in _data_config.synthetic_timeframes:
        results[tf] = synthesize_timeframe(df, tf)
    return results


def save_synthetic_data(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Save synthesized timeframe data to cache."""
    cache_path = Path(CACHE_DIR) / "synthetic" / symbol / f"{timeframe}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def load_synthetic_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load synthesized timeframe data from cache."""
    cache_path = Path(CACHE_DIR) / "synthetic" / symbol / f"{timeframe}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


# =============================================================================
# Walk-Forward Split Generation
# =============================================================================

def generate_walk_forward_splits(
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Generate walk-forward split indices for backtesting.

    Pattern: 12 months train + 2 weeks embargo + 1 month validation

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        List of split dictionaries with:
        - split_id: integer
        - train_start, train_end: training period
        - embargo_end: end of embargo gap
        - valid_start, valid_end: validation period
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    train_months = _data_config.train_window_months
    embargo_weeks = _data_config.embargo_weeks
    valid_months = _data_config.valid_window_months

    splits = []
    split_id = 0

    current_start = start

    while True:
        # Training period: train_months from current_start
        train_end = current_start + relativedelta(months=train_months)

        # Embargo period: embargo_weeks gap
        embargo_end = train_end + relativedelta(weeks=embargo_weeks)

        # Validation period: valid_months after embargo
        valid_end = embargo_end + relativedelta(months=valid_months)

        # Stop if validation end exceeds end_date
        if valid_end > end:
            break

        splits.append({
            'split_id': split_id,
            'train_start': current_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'embargo_end': embargo_end.strftime('%Y-%m-%d'),
            'valid_start': embargo_end.strftime('%Y-%m-%d'),
            'valid_end': valid_end.strftime('%Y-%m-%d'),
        })

        # Move forward by valid_months for next split
        current_start = current_start + relativedelta(months=valid_months)
        split_id += 1

    return splits


def get_locked_predict_dates(end_date: str) -> Tuple[str, str]:
    """
    Get locked prediction set dates (last N months before end_date).

    Args:
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date) for locked prediction set
    """
    end = pd.to_datetime(end_date)
    start = end - relativedelta(months=_data_config.locked_predict_months)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def filter_symbols_by_age(
    symbol_start_dates: Dict[str, str],
    current_date: str,
    min_age_months: int = 18
) -> List[str]:
    """
    Filter symbols that have existed for at least min_age_months.

    Args:
        symbol_start_dates: Dictionary mapping symbol to its start date
        current_date: Current date string (YYYY-MM-DD)
        min_age_months: Minimum age in months

    Returns:
        List of valid symbols
    """
    current = pd.to_datetime(current_date)
    valid = []

    for symbol, start_date in symbol_start_dates.items():
        start = pd.to_datetime(start_date)
        age_months = (current.year - start.year) * 12 + (current.month - start.month)
        if age_months >= min_age_months:
            valid.append(symbol)

    return valid


def save_splits(splits: List[Dict], filename: str = "splits.json") -> None:
    """Save splits to cache directory."""
    cache_path = Path(CACHE_DIR) / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(splits, f, indent=2)


def load_splits(filename: str = "splits.json") -> Optional[List[Dict]]:
    """Load splits from cache directory."""
    cache_path = Path(CACHE_DIR) / filename
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# Data Loader Functions
# =============================================================================

def load_data(symbol: str, timeframe: str = "15min") -> Optional[pd.DataFrame]:
    """
    Load data for a symbol at a specific timeframe.

    Args:
        symbol: Futures symbol
        timeframe: Timeframe ('15min', '30min', '1h', '2h', '4h')

    Returns:
        DataFrame or None if not found
    """
    if timeframe == "15min":
        # Load raw 15min data
        df = load_raw_data(symbol)
    else:
        # Load synthesized data
        df = load_synthetic_data(symbol, timeframe)

    if df is not None and not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def load_split_data(
    symbol: str,
    split: Dict,
    split_type: str = "train"
) -> Optional[pd.DataFrame]:
    """
    Load data for a symbol within a specific split period.

    Args:
        symbol: Futures symbol
        split: Split dictionary with train_start, train_end, etc.
        split_type: 'train', 'valid', or 'embargo'

    Returns:
        DataFrame filtered to the split period
    """
    df = load_raw_data(symbol)
    if df is None or df.empty:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])

    if split_type == "train":
        start = pd.to_datetime(split['train_start'])
        end = pd.to_datetime(split['train_end'])
    elif split_type == "valid":
        start = pd.to_datetime(split['valid_start'])
        end = pd.to_datetime(split['valid_end'])
    elif split_type == "embargo":
        start = pd.to_datetime(split['train_end'])
        end = pd.to_datetime(split['embargo_end'])
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    return df[(df['datetime'] >= start) & (df['datetime'] < end)].copy()


def load_locked_predict_data(
    symbol: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Load data for locked prediction set.

    Args:
        symbol: Futures symbol
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame for the locked prediction period
    """
    df = load_raw_data(symbol)
    if df is None or df.empty:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])
    start, end = get_locked_predict_dates(end_date)

    return df[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()


def list_available_symbols() -> List[str]:
    """
    List all symbols with downloaded data in cache.

    Returns:
        List of available symbol codes
    """
    raw_path = Path(CACHE_DIR) / "raw"
    if not raw_path.exists():
        return []

    symbols = []
    for f in raw_path.glob("*.parquet"):
        symbols.append(f.stem)

    return sorted(symbols)


def get_symbol_start_date(symbol: str) -> Optional[str]:
    """
    Get the start date (earliest date) for a symbol's data.

    Args:
        symbol: Futures symbol

    Returns:
        Start date string or None if no data
    """
    df = load_data(symbol)
    if df is None or df.empty:
        return None

    return df['datetime'].min().strftime('%Y-%m-%d')