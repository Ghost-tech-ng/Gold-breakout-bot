"""Long-only gold trend follower, run as three independent books (H1, H2, H4).

Each book: when yesterday's daily close > EMA50 > EMA200, buy a close above the
highest high of the previous `lookback` bars. Initial stop k_stop * ATR below entry,
then a chandelier trail (highest high since entry - k_trail * ATR) that only moves up.
No take-profit: winners are left to run, which is where the edge comes from.

The backtest and the live bot share `Position.stop_hit` / `Position.trail`, so they run identical rules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TF_HOURS = {"1h": 1, "2h": 2, "4h": 4}


@dataclass(frozen=True)
class Params:
    symbol: str = "XAU/USD"
    timeframes: tuple[str, ...] = ("1h", "2h", "4h")
    lookback: int = 20
    k_stop: float = 2.5
    k_trail: float = 5.0
    atr_n: int = 14
    ema_fast: int = 50
    ema_slow: int = 200
    risk_pct_per_signal: float = 0.5
    contract_oz: float = 100.0
    stop_alert_min_r: float = 0.25

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        raw = json.loads(Path(path).read_text())
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if "timeframes" in known:
            known["timeframes"] = tuple(known["timeframes"])
        bad = [tf for tf in known.get("timeframes", ()) if tf not in TF_HOURS]
        if bad:
            raise ValueError(f"unsupported timeframes {bad}; use {list(TF_HOURS)}")
        return cls(**known)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """UTC-midnight-aligned, left-labelled bars. Keeps a `spread` column if present."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in df.columns:
        agg["spread"] = "median"
    return df.resample(tf, label="left", closed="left").agg(agg).dropna()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev).abs(), (df["low"] - prev).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def daily_uptrend(daily_close: pd.Series, p: Params) -> pd.Series:
    """Trend state as of each day's close (index = day label)."""
    fast = daily_close.ewm(span=p.ema_fast).mean()
    slow = daily_close.ewm(span=p.ema_slow).mean()
    return (daily_close > fast) & (fast > slow)


def uptrend_on(index: pd.DatetimeIndex, daily_up: pd.Series) -> np.ndarray:
    """Previous completed day's state for each intraday bar, so nothing peeks at today."""
    prev = daily_up.astype(float).shift(1)
    prev.index = prev.index.normalize()
    return prev.reindex(index.normalize()).eq(1.0).to_numpy()


def breakout(df: pd.DataFrame, p: Params) -> np.ndarray:
    ceiling = df["high"].rolling(p.lookback).max().shift(1)
    return (df["close"] > ceiling).to_numpy()


@dataclass
class Position:
    tf: str
    entry_time: str
    entry: float
    stop: float
    risk: float
    best: float
    signal_close: float = 0.0
    announced_stop: float = 0.0

    @classmethod
    def open(cls, tf: str, when: pd.Timestamp, price: float, atr_now: float, p: Params) -> "Position":
        return cls(tf=tf, entry_time=when.isoformat(), entry=price, stop=price - p.k_stop * atr_now,
                   risk=p.k_stop * atr_now, best=price, signal_close=price,
                   announced_stop=price - p.k_stop * atr_now)

    def stop_hit(self, o: float, low: float, slip: float = 0.0) -> float | None:
        """Exit price if this bar trades through the resting stop (gap-aware)."""
        if low <= self.stop:
            return min(o, self.stop) - slip
        return None

    def trail(self, high: float, atr_now: float, p: Params) -> bool:
        """Ratchet the stop at a bar close. Returns True if it moved."""
        self.best = max(self.best, high)
        new = max(self.stop, self.best - p.k_trail * atr_now)
        moved = new > self.stop + 1e-9
        self.stop = new
        return moved

    def r(self, exit_price: float) -> float:
        return (exit_price - self.entry) / self.risk

    def to_dict(self) -> dict:
        return asdict(self)


def backtest(bars: pd.DataFrame, up: np.ndarray, tf: str, p: Params,
             slip_atr: float = 0.05, warmup: int = 250) -> pd.DataFrame:
    """One book on one timeframe. Signal on bar i close, fill at bar i+1 open (ask + slippage);
    each later bar: stop check first (bid), then trail, then a fresh entry if flat."""
    o, h, l = (bars[c].to_numpy() for c in ("open", "high", "low"))
    spread = bars["spread"].to_numpy() if "spread" in bars.columns else np.zeros(len(bars))
    a = atr(bars, p.atr_n).to_numpy()
    sig = breakout(bars, p) & up
    idx = bars.index
    rows: list[tuple] = []
    pos: Position | None = None
    for i in range(warmup, len(bars) - 1):
        if pos is not None:
            px = pos.stop_hit(o[i], l[i], slip_atr * a[i])
            if px is not None:
                rows.append((tf, pd.Timestamp(pos.entry_time), idx[i], pos.entry, px, pos.r(px)))
                pos = None
            else:
                pos.trail(h[i], a[i], p)
        if pos is None and sig[i] and not np.isnan(a[i]):
            fill = o[i + 1] + spread[i + 1] + slip_atr * a[i]
            pos = Position.open(tf, idx[i + 1], fill, a[i], p)
    return pd.DataFrame(rows, columns=["tf", "entry_time", "exit_time", "entry", "exit", "r"])
