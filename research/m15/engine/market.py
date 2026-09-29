"""Precomputed, causal view of M15 + H1 data that every setup reads from.

Short setups are detected by running the long logic on a mirrored market
(prices negated, highs and lows swapped), so each rule is written once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.indicators import FRACTAL, ema, fractal_highs, fractal_lows, wilder_atr

M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)


@dataclass
class Market:
    time: np.ndarray  # bar open time, UTC, datetime64[ns]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray
    atr: np.ndarray
    atr5: np.ndarray
    atr50: np.ndarray
    atr100: np.ndarray
    ema50: np.ndarray
    h1_close: np.ndarray
    h1_ema50: np.ndarray
    h1_ema200: np.ndarray
    h1_ema50_prev5: np.ndarray
    h1_atr: np.ndarray
    h1_idx: np.ndarray  # index of the latest completed H1 bar at each M15 close, -1 if none
    swing_hi: np.ndarray
    swing_lo: np.ndarray
    h1_high: np.ndarray
    h1_low: np.ndarray
    h1_swing_hi: np.ndarray
    h1_swing_lo: np.ndarray
    sign: int = 1
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.c)

    def swings_hi_known(self, t: int, lookback: int) -> np.ndarray:
        """Swing-high indices confirmed by the close of bar t, within the last `lookback` bars."""
        hi = np.searchsorted(self.swing_hi, t - FRACTAL, side="right")
        lo = np.searchsorted(self.swing_hi, t - lookback, side="left")
        return self.swing_hi[lo:hi]

    def swings_lo_known(self, t: int, lookback: int) -> np.ndarray:
        hi = np.searchsorted(self.swing_lo, t - FRACTAL, side="right")
        lo = np.searchsorted(self.swing_lo, t - lookback, side="left")
        return self.swing_lo[lo:hi]

    def h1_swings_known(self, t: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        j = int(self.h1_idx[t])
        if j < 0:
            return np.empty(0), np.empty(0)
        def pick(arr: np.ndarray, px: np.ndarray) -> np.ndarray:
            a = np.searchsorted(arr, j - lookback, side="left")
            b = np.searchsorted(arr, j - FRACTAL, side="right")
            return px[arr[a:b]]
        return pick(self.h1_swing_hi, self.h1_high), pick(self.h1_swing_lo, self.h1_low)


def to_h1(m15: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return m15.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["open"])


def build_market(m15: pd.DataFrame, mirror: bool = False) -> Market:
    """m15: UTC-indexed bars (index = open time) with open/high/low/close/spread (bid prices)."""
    df = m15
    if mirror:
        df = m15.copy()
        df["open"], df["close"] = -m15["open"], -m15["close"]
        df["high"], df["low"] = -m15["low"], -m15["high"]
    h1 = to_h1(df)

    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    h1h, h1l, h1c = (h1[k].to_numpy(dtype=float) for k in ("high", "low", "close"))

    h1_close_time = (h1.index + H1).to_numpy()
    m15_close_time = (df.index + M15).to_numpy()
    h1_idx = np.searchsorted(h1_close_time, m15_close_time, side="right") - 1

    h1_ema50 = ema(h1c, 50)
    h1_ema200 = ema(h1c, 200)
    prev5 = np.concatenate((np.full(5, np.nan), h1_ema50[:-5]))
    h1_atr = wilder_atr(h1h, h1l, h1c, 14)

    def at(arr: np.ndarray) -> np.ndarray:
        out = np.full(len(df), np.nan)
        ok = h1_idx >= 0
        out[ok] = arr[h1_idx[ok]]
        return out

    spread = df["spread"].to_numpy(dtype=float) if "spread" in df else np.zeros(len(df))
    return Market(
        time=df.index.to_numpy(),
        o=o, h=h, l=l, c=c,
        spread=spread,
        atr=wilder_atr(h, l, c, 14),
        atr5=wilder_atr(h, l, c, 5),
        atr50=wilder_atr(h, l, c, 50),
        atr100=wilder_atr(h, l, c, 100),
        ema50=ema(c, 50),
        h1_close=at(h1c),
        h1_ema50=at(h1_ema50),
        h1_ema200=at(h1_ema200),
        h1_ema50_prev5=at(prev5),
        h1_atr=at(h1_atr),
        h1_idx=h1_idx,
        swing_hi=fractal_highs(h),
        swing_lo=fractal_lows(l),
        h1_high=h1h,
        h1_low=h1l,
        h1_swing_hi=fractal_highs(h1h),
        h1_swing_lo=fractal_lows(h1l),
        sign=-1 if mirror else 1,
    )
