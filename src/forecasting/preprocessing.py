from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLS = ["net_kwh", "price_eur_kwh"]
KNOWN_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
STANDARDIZE_COLS = TARGET_COLS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = out.index.hour.to_numpy(dtype=float)
    dow = out.index.dayofweek.to_numpy(dtype=float)
    m = out.index.month.to_numpy(dtype=float) - 1.0
    out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    return out


def make_splits(index: pd.Index) -> dict:
    train_end = pd.Timestamp("2022-09-30 23:59:59")
    val_end = pd.Timestamp("2022-10-31 23:59:59")
    return {
        "train": index <= train_end,
        "val": (index > train_end) & (index <= val_end),
        "test": index > val_end,
    }


class Standardizer:
    def __init__(self, cols: list[str]):
        self.cols = cols
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        self.mean_ = df[self.cols].to_numpy(dtype=np.float64).mean(axis=0)
        self.std_ = df[self.cols].to_numpy(dtype=np.float64).std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.cols] = (df[self.cols].to_numpy(dtype=np.float64) - self.mean_) / self.std_
        return out

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std_ + self.mean_

    def inverse_target(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std_ + self.mean_


def build_supervised(
    df: pd.DataFrame, lookback: int = 24, horizon: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = TARGET_COLS + KNOWN_COLS
    x = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)
    xs, ys, ys_raw_idx = [], [], []
    for t in range(lookback, len(df) - horizon + 1):
        xs.append(x[t - lookback : t])
        ys.append(y[t : t + horizon])
        ys_raw_idx.append(t)
    return np.stack(xs), np.stack(ys), np.array(ys_raw_idx), np.array(ys_raw_idx)


def split_arrays(
    xs: np.ndarray, ys: np.ndarray, idx: np.ndarray, masks: dict
) -> dict:
    m_train = masks["train"][idx]
    m_val = masks["val"][idx]
    m_test = masks["test"][idx]
    return {
        "X_train": xs[m_train], "y_train": ys[m_train],
        "X_val": xs[m_val], "y_val": ys[m_val],
        "X_test": xs[m_test], "y_test": ys[m_test],
        "idx_train": idx[m_train], "idx_val": idx[m_val], "idx_test": idx[m_test],
    }
