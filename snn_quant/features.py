"""Features use only completed bars; histories never cross session/gap boundaries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DESCRIPTIONS = {
    "move_1": "近1分钟价格变化 / 此前20分钟平均真实振幅",
    "move_3": "近3分钟价格变化 / 此前20分钟平均真实振幅",
    "path_efficiency_5": "近5分钟净价格变化 / 路径绝对价格变化之和，带方向",
    "distance_resistance": "(此前20分钟最高价 - 当前收盘价) / 历史振幅；负值表示越过历史高点",
    "distance_support": "(当前收盘价 - 此前20分钟最低价) / 历史振幅；负值表示越过历史低点",
    "close_location": "(2*收盘价-最高价-最低价)/(最高价-最低价)",
    "volatility_compression": "log(最近5分钟价格差标准差 / 此前20分钟价格差标准差)",
    "relative_range": "log(本分钟高低价差 / 此前20分钟平均真实振幅)",
    "relative_volume": "log1p(本分钟成交量)-log1p(此前20分钟平均成交量)",
    "open_interest_change": "(本分钟持仓量-前一分钟持仓量)/前一分钟持仓量",
}
FEATURE_NAMES = list(FEATURE_DESCRIPTIONS)


def load_bars(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    needed = ["date", "open", "high", "low", "close", "volume", "open_interest"]
    missing = set(needed) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    df = raw[needed].copy()
    df["datetime"] = pd.to_datetime(df.pop("date"), errors="raise")
    if df["datetime"].duplicated().any() or not df["datetime"].is_monotonic_increasing:
        raise ValueError("Input timestamps must be unique and increasing; no silent sort.")
    numeric = ["open", "high", "low", "close", "volume", "open_interest"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Missing or nonfinite raw observations require explicit review.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Price must be positive for this IC experiment.")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
            (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC observations.")
    if (df[["volume", "open_interest"]] < 0).any().any():
        raise ValueError("Negative volume or open interest.")
    # Optional future data: reset feature history also at known contract changes.
    contract_column = next((c for c in ("dominant_id", "contract_id", "order_book_id")
                            if c in raw.columns), None)
    if contract_column:
        if raw[contract_column].isna().any():
            raise ValueError("Partly missing contract identity is not silently accepted.")
        df["contract_id"] = raw[contract_column].astype(str)
    else:
        df["contract_id"] = "UNKNOWN_CONTINUOUS"
    return attach_segments(df)


def attach_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trading_date"] = df.datetime.dt.strftime("%Y-%m-%d")
    minutes = df.datetime.dt.hour * 60 + df.datetime.dt.minute
    df["session"] = np.where(minutes < 12 * 60, "AM", "PM")
    if "contract_id" not in df:
        df["contract_id"] = "UNKNOWN_CONTINUOUS"
    breaks = (df.datetime.diff().ne(pd.Timedelta(minutes=1)) |
              df.trading_date.ne(df.trading_date.shift()) |
              df.session.ne(df.session.shift()) |
              df.contract_id.ne(df.contract_id.shift()))
    df["segment_id"] = breaks.cumsum().astype(int) - 1
    return df.reset_index(drop=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series,
                 zero_value: float | None = None) -> pd.Series:
    result = numerator / denominator.where(denominator.abs() > 1e-12)
    if zero_value is not None:
        result = result.mask(denominator.abs() <= 1e-12, zero_value)
    return result


def build_features(bars: pd.DataFrame, lookback: int = 20,
                   horizon: int = 5) -> pd.DataFrame:
    if lookback < 5 or horizon < 1:
        raise ValueError("lookback must be >= 5 and horizon >= 1")
    chunks = []
    for _, segment in bars.groupby("segment_id", sort=False):
        s = segment.copy()
        close, high, low = s.close, s.high, s.low
        previous = close.shift(1)
        change = close.diff()
        true_range = pd.concat([high - low, (high - previous).abs(),
                                (low - previous).abs()], axis=1).max(axis=1)
        true_range = true_range.mask(previous.isna())
        past_range = true_range.shift(1).rolling(lookback, min_periods=lookback).mean()
        past_sigma = change.shift(1).rolling(lookback, min_periods=lookback).std(ddof=0)
        short_sigma = change.rolling(5, min_periods=5).std(ddof=0)
        s["reference_high"] = high.shift(1).rolling(lookback, min_periods=lookback).max()
        s["reference_low"] = low.shift(1).rolling(lookback, min_periods=lookback).min()
        s["past_range"] = past_range
        s["move_1"] = _safe_divide(change, past_range)
        s["move_3"] = _safe_divide(close - close.shift(3), past_range)
        path = change.abs().rolling(5, min_periods=5).sum()
        s["path_efficiency_5"] = _safe_divide(close - close.shift(5), path, 0.0)
        s["distance_resistance"] = _safe_divide(s.reference_high - close, past_range)
        s["distance_support"] = _safe_divide(close - s.reference_low, past_range)
        s["close_location"] = _safe_divide(2 * close - high - low, high - low, 0.0)
        # Ratio zero is a legitimate extreme; tiny clipping prevents log(0).
        # Missing/zero long-term volatility still invalidates the observation.
        s["volatility_compression"] = np.log(_safe_divide(short_sigma, past_sigma).clip(lower=1e-6))
        s["relative_range"] = np.log(_safe_divide(high - low, past_range).clip(lower=1e-6))
        mean_volume = s.volume.shift(1).rolling(lookback, min_periods=lookback).mean()
        s["relative_volume"] = np.log1p(s.volume) - np.log1p(mean_volume)
        s["open_interest_change"] = _safe_divide(s.open_interest.diff(), s.open_interest.shift(1))
        # This is a forecast target, not an assumed executable entry/exit return.
        s["future_return_bps"] = 1e4 * np.log(close.shift(-horizon) / close)
        s["target_datetime"] = s.datetime.shift(-horizon)
        s["feature_valid"] = np.isfinite(s[FEATURE_NAMES].to_numpy()).all(axis=1)
        chunks.append(s)
    return pd.concat(chunks, axis=0).sort_index().reset_index(drop=True)


def split_labels(dates: pd.Series, train_end: str, validation_end: str) -> np.ndarray:
    if train_end >= validation_end:
        raise ValueError("Train end must precede validation end")
    return np.where(dates <= train_end, "train",
                    np.where(dates <= validation_end, "validation", "test"))


@dataclass
class WindowData:
    x: np.ndarray
    metadata: pd.DataFrame


def build_windows(features: pd.DataFrame, sequence_minutes: int,
                  train_end: str, validation_end: str) -> WindowData:
    if sequence_minutes < 2:
        raise ValueError("A temporal experiment requires at least two minutes")
    windows, indices = [], []
    for _, s in features.groupby("segment_id", sort=False):
        values = s[FEATURE_NAMES].to_numpy(dtype=np.float32)
        valid = s.feature_valid.to_numpy()
        for local_end in range(sequence_minutes - 1, len(s)):
            begin = local_end + 1 - sequence_minutes
            row = s.iloc[local_end]
            if not valid[begin:local_end + 1].all() or not np.isfinite(row.future_return_bps):
                continue
            # Exclude labels that extend into another chronological split.
            decision_split = split_labels(pd.Series([row.trading_date]), train_end, validation_end)[0]
            target_split = split_labels(pd.Series([row.target_datetime.strftime("%Y-%m-%d")]),
                                        train_end, validation_end)[0]
            if decision_split != target_split:
                continue
            windows.append(values[begin:local_end + 1])
            indices.append(s.index[local_end])
    if not windows:
        raise ValueError("No complete within-session windows")
    columns = ["datetime", "trading_date", "session", "segment_id", "contract_id",
               "target_datetime", "future_return_bps"]
    meta = features.loc[indices, columns].copy().reset_index(drop=True)
    meta["split"] = split_labels(meta.trading_date, train_end, validation_end)
    return WindowData(np.stack(windows), meta)


def fit_scales(features: pd.DataFrame, train_end: str, quantile: float = 0.99) -> np.ndarray:
    if not 0 < quantile <= 1:
        raise ValueError("Invalid scale quantile")
    train = features.loc[(features.trading_date <= train_end) & features.feature_valid, FEATURE_NAMES]
    if train.empty:
        raise ValueError("No training features for normalization")
    scale = np.quantile(np.abs(train.to_numpy()), quantile, axis=0)
    return np.maximum(scale, 1e-8).astype(np.float32)


def normalize(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # No mean subtraction: zero retains the financial meaning of each feature.
    return np.clip(x / scale, -1.0, 1.0).astype(np.float32)
