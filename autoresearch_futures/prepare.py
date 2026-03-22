"""
Data preparation module for autoresearch-futures.

Handles:
- Data download from akshare
- K-line timeframe synthesis
- Walk-Forward split generation
- Data loading utilities

Reference: https://akshare.akfamily.xyz/data/futures/futures.html
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

# 期货品种代码映射 (小写 -> 大写，用于接口调用)
# 主力合约格式: 大写品种代码 + "0"，如 "RB0"
FUTURES_SYMBOLS = {
    # 上期所 SHFE
    'rb': 'RB',   # 螺纹钢
    'hc': 'HC',   # 热卷
    'cu': 'CU',   # 铜
    'al': 'AL',   # 铝
    'zn': 'ZN',   # 锌
    'au': 'AU',   # 黄金
    'ag': 'AG',   # 白银
    'ni': 'NI',   # 镍
    'sn': 'SN',   # 锡
    'pb': 'PB',   # 铅
    'ss': 'SS',   # 不锈钢
    'wr': 'WR',   # 线材
    'sp': 'SP',   # 纸浆
    'fu': 'FU',   # 燃油
    'bu': 'BU',   # 沥青
    'ru': 'RU',   # 橡胶
    'nr': 'NR',   # 20号胶
    'ao': 'AO',   # 氧化铝
    'br': 'BR',   # 丁二烯橡胶
    'ec': 'EC',   # 集运指数(欧线)
    # 大商所 DCE
    'i': 'I',     # 铁矿石
    'j': 'J',     # 焦炭
    'jm': 'JM',   # 焦煤
    'v': 'V',     # PVC
    'l': 'L',     # 塑料
    'pp': 'PP',   # 聚丙烯
    'eb': 'EB',   # 苯乙烯
    'eg': 'EG',   # 乙二醇
    'pg': 'PG',   # 液化石油气
    'p': 'P',     # 棕榈油
    'y': 'Y',     # 豆油
    'a': 'A',     # 豆一
    'b': 'B',     # 豆二
    'm': 'M',     # 豆粕
    'c': 'C',     # 玉米
    'cs': 'CS',   # 淀粉
    'jd': 'JD',   # 鸡蛋
    'rr': 'RR',   # 粳米
    'fb': 'FB',   # 纤维板
    'bb': 'BB',   # 胶合板
    'lh': 'LH',   # 生猪
    # 郑商所 CZCE
    'cf': 'CF',   # 棉花
    'sr': 'SR',   # 白糖
    'ta': 'TA',   # PTA
    'ma': 'MA',   # 甲醇
    'fg': 'FG',   # 玻璃
    'sa': 'SA',   # 纯碱
    'ur': 'UR',   # 尿素
    'oi': 'OI',   # 菜油
    'rm': 'RM',   # 菜粕
    'rs': 'RS',   # 菜籽
    'pm': 'PM',   # 普麦
    'wh': 'WH',   # 强麦
    'ri': 'RI',   # 早籼稻
    'lr': 'LR',   # 晚籼稻
    'jr': 'JR',   # 粳稻
    'ap': 'AP',   # 苹果
    'cj': 'CJ',   # 红枣
    'pk': 'PK',   # 花生
    'sf': 'SF',   # 硅铁
    'sm': 'SM',   # 锰铁
    'zc': 'ZC',   # 动力煤
    'pf': 'PF',   # 短纤
    'sh': 'SH',   # 烧碱
    'px': 'PX',   # 对二甲苯
    # 广期所 GFEX
    'si': 'SI',   # 工业硅
    'lc': 'LC',   # 碳酸锂
}


# =============================================================================
# Data Download Functions
# =============================================================================

def get_futures_list() -> List[str]:
    """
    Get list of futures symbols.

    Returns:
        List of futures symbol codes (e.g., ['rb', 'i', 'hc', 'j'])
    """
    return sorted(FUTURES_SYMBOLS.keys())


def get_main_contract_symbol(symbol: str) -> str:
    """
    Convert symbol to main contract format for akshare API.

    Args:
        symbol: Futures symbol code (e.g., 'rb')

    Returns:
        Main contract symbol (e.g., 'RB0')
    """
    upper = FUTURES_SYMBOLS.get(symbol.lower(), symbol.upper())
    return f"{upper}0"


def download_minute_data(symbol: str, period: str = "15") -> pd.DataFrame:
    """
    Download minute-level historical data for a futures symbol.

    Note: This API only returns approximately the last 1000 bars.

    Args:
        symbol: Futures symbol code (e.g., 'rb' for rebar)
        period: Time period ('1', '5', '15', '30', '60') minutes

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume, hold
    """
    try:
        main_symbol = get_main_contract_symbol(symbol)
        df = ak.futures_zh_minute_sina(symbol=main_symbol, period=period)

        if df is None or df.empty:
            return pd.DataFrame()

        # 列名已经是标准格式: datetime, open, high, low, close, volume, hold
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Error downloading minute data for {symbol}: {e}")
        return pd.DataFrame()


def download_daily_data(symbol: str) -> pd.DataFrame:
    """
    Download daily historical data for a futures main contract.

    Args:
        symbol: Futures symbol code (e.g., 'rb' for rebar)

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    try:
        main_symbol = get_main_contract_symbol(symbol)
        df = ak.futures_main_sina(symbol=main_symbol)

        if df is None or df.empty:
            return pd.DataFrame()

        # 重命名中文列名
        column_map = {
            '日期': 'datetime',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '持仓量': 'open_interest',
            '动态结算价': 'settle',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 选择需要的列
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in required_cols if c in df.columns]

        return df[available_cols]

    except Exception as e:
        print(f"Error downloading daily data for {symbol}: {e}")
        return pd.DataFrame()


def download_contract(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download historical data for a single futures contract.

    Data type is determined by DataConfig.data_type:
    - "daily": Download daily K-line data (recommended for strategy research)
    - "minute": Download 15-minute K-line data (limited to ~1000 bars)

    Args:
        symbol: Futures symbol code (e.g., 'rb' for rebar)
        start_date: Optional start date filter (YYYY-MM-DD)

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    # 根据配置选择数据类型
    if _data_config.data_type == "minute":
        df = download_minute_data(symbol, period="15")
    else:
        df = download_daily_data(symbol)

    if df.empty:
        return pd.DataFrame()

    # 过滤日期
    if start_date and 'datetime' in df.columns:
        start_dt = pd.to_datetime(start_date)
        df = df[df['datetime'] >= start_dt]

    return df


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
        print(f"Downloading {symbol}...")
        df = download_contract(symbol, start_date)
        if not df.empty:
            results[symbol] = df
            if save:
                save_raw_data(symbol, df)
            print(f"  Downloaded {symbol}: {len(df)} rows")
        else:
            print(f"  No data for {symbol}")

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
    Convert K-lines to a higher timeframe using OHLCV aggregation.

    For daily base data:
        - 'weekly': Weekly aggregation
        - 'monthly': Monthly aggregation

    For minute base data:
        - '30min', '1h', '2h', '4h': Time-based aggregation
        - 'daily': Day-based aggregation

    Args:
        df: DataFrame with base data (must have datetime column)
        target_tf: Target timeframe

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
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',
    }

    if target_tf not in tf_map:
        raise ValueError(f"Unknown target timeframe: {target_tf}. "
                        f"Available: {list(tf_map.keys())}")

    rule = tf_map[target_tf]

    # Aggregate OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    # Include hold/open_interest if present
    if 'hold' in df.columns:
        agg_dict['hold'] = 'last'
    if 'open_interest' in df.columns:
        agg_dict['open_interest'] = 'last'

    resampled = df.resample(rule).agg(agg_dict).dropna()

    # Reset datetime as column
    resampled = resampled.reset_index()

    return resampled


def synthesize_all_timeframes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Synthesize all configured timeframes from base data.

    Args:
        df: DataFrame with base data

    Returns:
        Dictionary mapping timeframe name to DataFrame
    """
    results = {}
    for tf in _data_config.synthetic_timeframes:
        try:
            results[tf] = synthesize_timeframe(df, tf)
        except Exception as e:
            print(f"Warning: Could not synthesize {tf}: {e}")
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

    Pattern: train + embargo + validation
    - 分钟数据模式 (推荐): 1个月训练 + 1周embargo + 2周验证
    - 日线数据模式: 12个月训练 + 2周embargo + 1个月验证

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

    # 支持周数或月数作为验证窗口
    valid_months = getattr(_data_config, 'valid_window_months', 0)
    valid_weeks = getattr(_data_config, 'valid_window_weeks', 2)

    # 确定滚动步长
    if valid_months > 0:
        roll_delta = relativedelta(months=valid_months)
    else:
        roll_delta = relativedelta(weeks=valid_weeks)

    splits = []
    split_id = 0

    current_start = start

    while True:
        # Training period: train_months from current_start
        train_end = current_start + relativedelta(months=train_months)

        # Embargo period: embargo_weeks gap
        embargo_end = train_end + relativedelta(weeks=embargo_weeks)

        # Validation period: valid_months or valid_weeks after embargo
        if valid_months > 0:
            valid_end = embargo_end + relativedelta(months=valid_months)
        else:
            valid_end = embargo_end + relativedelta(weeks=valid_weeks)

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

        # Move forward by roll_delta for next split
        current_start = current_start + roll_delta
        split_id += 1

    return splits


def get_locked_predict_dates(end_date: str) -> Tuple[str, str]:
    """
    Get locked prediction set dates.

    For minute data: last N weeks before end_date
    For daily data: last N months before end_date

    Args:
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (start_date, end_date) for locked prediction set
    """
    end = pd.to_datetime(end_date)

    # 优先使用周数配置（适用于分钟数据）
    locked_weeks = getattr(_data_config, 'locked_predict_weeks', None)
    locked_months = getattr(_data_config, 'locked_predict_months', 6)

    if locked_weeks is not None and locked_weeks > 0:
        start = end - relativedelta(weeks=locked_weeks)
    else:
        start = end - relativedelta(months=locked_months)

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
        timeframe: Timeframe ('15min', '30min', '1h', '2h', '4h') for minute data
                   or ('daily', 'weekly', 'monthly') for daily data

    Returns:
        DataFrame or None if not found
    """
    # Determine base timeframe
    if _data_config.data_type == "minute":
        base_tf = "15min"
    else:
        base_tf = "daily"

    if timeframe == base_tf:
        # Load raw data
        df = load_raw_data(symbol)
    else:
        # Load synthesized data
        df = load_synthetic_data(symbol, timeframe)
        # If not cached, try to synthesize from raw data
        if df is None:
            raw_df = load_raw_data(symbol)
            if raw_df is not None and not raw_df.empty:
                try:
                    df = synthesize_timeframe(raw_df, timeframe)
                    save_synthetic_data(symbol, timeframe, df)
                except Exception as e:
                    print(f"Warning: Could not synthesize {timeframe} for {symbol}: {e}")

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