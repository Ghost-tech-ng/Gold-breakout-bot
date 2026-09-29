"""Asian-range / London-open breakout: the simplest well-known gold breakout, used
as the baseline the strategy has to beat under identical costs and management."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.run import Data, Scanner
from engine.market import M15
from engine.params import Params
from engine.signals import Signal


def asian_range_scanner(d: Data, buffer_atr: float = 0.1, max_range_atr: float = 4.0) -> Scanner:
    idx = d.m15.index
    day = idx.normalize()
    asian = np.asarray(idx.hour < 7)
    frame = pd.DataFrame({"h": d.m15["high"].where(asian), "l": d.m15["low"].where(asian), "day": day})
    rng = frame.groupby("day")[["h", "l"]].transform("max"), frame.groupby("day")[["h", "l"]].transform("min")
    hi = rng[0]["h"].to_numpy()
    lo = rng[1]["l"].to_numpy()
    window = np.asarray((idx.hour >= 7) & (idx.hour < 10))
    traded: set[pd.Timestamp] = set()

    def scanner(dd: Data, t: int, p: Params) -> list[Signal]:
        if not window[t] or day[t] in traded or np.isnan(hi[t]) or np.isnan(dd.long.atr[t]):
            return []
        atr = dd.long.atr[t]
        if hi[t] - lo[t] > max_range_atr * atr:
            return []
        c = dd.m15["close"].iat[t]
        when = idx[t] + M15
        for direction, level, stop in (("long", hi[t], lo[t]), ("short", lo[t], hi[t])):
            s = 1 if direction == "long" else -1
            if s * (c - level) > buffer_atr * atr:
                risk = min(abs(c - stop), 2.0 * atr)
                traded.add(day[t])
                return [Signal(setup="ASIA", direction=direction, t=t, time=when, entry_type="market",
                               entry_ref=c, limit_price=np.nan, stop=c - s * risk,
                               tp2=c + s * 2.0 * risk, level=level, atr=atr, score=100.0, risk_mult=1.0)]
        return []

    return scanner
