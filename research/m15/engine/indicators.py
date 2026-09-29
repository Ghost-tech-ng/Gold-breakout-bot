from __future__ import annotations

import numpy as np
import pandas as pd

FRACTAL = 3


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    prev = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def fractal_highs(high: np.ndarray, k: int = FRACTAL) -> np.ndarray:
    """Indices i where high[i] is the strict maximum of high[i-k..i+k]. Known only at bar i+k."""
    n = len(high)
    if n < 2 * k + 1:
        return np.empty(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(high, 2 * k + 1)
    centre = win[:, k]
    others = np.delete(win, k, axis=1)
    idx = np.nonzero(centre > others.max(axis=1))[0] + k
    return idx.astype(np.int64)


def fractal_lows(low: np.ndarray, k: int = FRACTAL) -> np.ndarray:
    return fractal_highs(-low, k)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = a + b*x. Returns (a, b, r2)."""
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return float(ym), 0.0, 0.0
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    ss_tot = ((y - ym) ** 2).sum()
    ss_res = ((y - (a + b * x)) ** 2).sum()
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)
