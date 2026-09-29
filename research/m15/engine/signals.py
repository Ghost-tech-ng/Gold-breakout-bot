"""Turns raw setup candidates into filtered, scored, tradeable signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.market import M15, Market
from engine.params import Params
from engine.setups import Candidate, Level, candle_ok, levels, setup_a, setup_b, setup_c

# Where bars drop out of the pipeline; read by the backtester to tune filters.
FUNNEL: Counter[str] = Counter()


def _drop(reason: str) -> None:
    FUNNEL[reason] += 1


@dataclass
class TimeFlags:
    in_session: np.ndarray
    prime: np.ndarray
    news_block: np.ndarray
    friday_late: np.ndarray


@dataclass
class Signal:
    setup: str
    direction: str  # "long" | "short"
    t: int
    time: pd.Timestamp  # close time of the signal bar = earliest entry time
    entry_type: str
    entry_ref: float
    limit_price: float
    stop: float
    tp2: float
    level: float
    atr: float
    score: float
    risk_mult: float
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1


def time_flags(bar_open: pd.DatetimeIndex, news: list[tuple[pd.Timestamp, int, int]]) -> TimeFlags:
    """Flags evaluated at each bar's close, which is when a signal would be sent."""
    close = (bar_open + M15).tz_convert("UTC") if bar_open.tz else (bar_open + M15).tz_localize("UTC")
    ldn = close.tz_convert("Europe/London")
    ny = close.tz_convert("America/New_York")
    lm = ldn.hour * 60 + ldn.minute
    nm = ny.hour * 60 + ny.minute
    in_session = ((lm >= 8 * 60) & (lm < 12 * 60)) | ((nm >= 8 * 60) & (nm < 11 * 60 + 30))
    prime = ((lm >= 8 * 60) & (lm < 10 * 60)) | ((nm >= 8 * 60 + 30) & (nm < 10 * 60 + 30))
    friday_late = (close.weekday == 4) & (close.hour >= 18)

    block = np.zeros(len(close), dtype=bool)
    ts = close.asi8
    for when, before, after in news:
        w = pd.Timestamp(when).tz_convert("UTC").value if pd.Timestamp(when).tz else pd.Timestamp(when).tz_localize("UTC").value
        lo = np.searchsorted(ts, w - before * 60_000_000_000, side="left")
        hi = np.searchsorted(ts, w + after * 60_000_000_000, side="right")
        block[lo:hi] = True
    return TimeFlags(np.asarray(in_session), np.asarray(prime), block, np.asarray(friday_late))


def h1_bullish(m: Market, t: int) -> bool:
    return bool(
        m.h1_close[t] > m.h1_ema50[t] > m.h1_ema200[t] and m.h1_ema50[t] > m.h1_ema50_prev5[t]
    )


def _score(m: Market, t: int, cand: Candidate, room_r: float, prime: bool,
           body_frac: float, close_pos: float, confluence: bool) -> dict[str, float]:
    candle = 10 * min(1.0, body_frac / 0.8) + 10 * np.clip((close_pos - 0.7) / 0.3, 0, 1)
    comp_ratio = m.atr5[t - 1] / m.atr50[t - 1] if m.atr50[t - 1] > 0 else 1.0
    compression = 15 * float(np.clip((1.0 - comp_ratio) / 0.2, 0, 1))
    if cand.compression:
        compression = max(compression, 10.0)
    room = 15 * float(np.clip((room_r - 1.5) / 1.0, 0, 1))
    strength = (m.h1_close[t] - m.h1_ema50[t]) / m.h1_atr[t] if m.h1_atr[t] > 0 else 0.0
    h1 = 15 * float(np.clip(strength, 0, 1))
    return {
        "candle": float(candle),
        "level": 20 * cand.level_quality,
        "compression": compression,
        "room": room,
        "h1": h1,
        "prime": 10.0 if prime else 0.0,
        "confluence": 5.0 if confluence else 0.0,
    }


def scan_side(m: Market, t: int, p: Params, flags: TimeFlags) -> Signal | None:
    """Evaluate bar t (just closed) for a long on market m (mirrored m gives shorts)."""
    if t < 300 or np.isnan(m.atr100[t]) or np.isnan(m.h1_ema200[t]):
        return None
    if p.use_session and not flags.in_session[t]:
        return _drop("session")
    if flags.friday_late[t]:
        return _drop("friday")
    if p.use_news and flags.news_block[t]:
        return _drop("news")
    bull = h1_bullish(m, t)
    if p.use_h1_bias and not bull:
        return _drop("h1_bias")
    atr = m.atr[t]
    vol = atr / m.atr100[t]
    if vol < p.vol_min:
        return _drop("low_vol")
    ok, body_frac, close_pos = candle_ok(m, t, p)
    if not ok:
        return _drop("candle")

    lv: list[Level] = levels(m, t, p)
    cands: list[Candidate] = []
    if "A" in p.setups and (c := setup_a(m, t, p)):
        cands.append(c)
    if "B" in p.setups and (c := setup_b(m, t, p, lv)):
        cands.append(c)
    if "C" in p.setups and (c := setup_c(m, t, p, bull)):
        cands.append(c)
    if not cands:
        return _drop("no_setup")

    best: Signal | None = None
    for cand in cands:
        entry = cand.limit_price if cand.entry_type == "limit" else m.c[t]
        stop = cand.stop
        risk = entry - stop
        if risk > p.max_stop_atr * atr:
            _drop(f"{cand.setup}:stop_wide")
            continue
        if risk < p.min_stop_atr * atr:
            stop = entry - p.min_stop_atr * atr
            risk = entry - stop
        if m.spread[t] > p.max_spread_frac * risk:
            _drop(f"{cand.setup}:spread")
            continue
        above = [L.price for L in lv if L.price > entry + 0.10 * atr and L.price > cand.level + 0.3 * atr]
        nearest = min(above) if above else np.inf
        room_r = (nearest - entry) / risk if np.isfinite(nearest) else p.level_range_atr * atr / risk
        if room_r < p.min_room_r:
            _drop(f"{cand.setup}:room")
            continue
        if cand.setup == "C" and np.isfinite(cand.target):
            tp2 = cand.target
        else:
            tp2 = min(nearest - 0.10 * atr, entry + p.tp2_max_r * risk)
        parts = _score(m, t, cand, room_r, bool(flags.prime[t]), body_frac, close_pos, len(cands) > 1)
        score = sum(parts.values())
        if score < p.min_score:
            _drop(f"{cand.setup}:score")
            continue
        s = m.sign
        sig = Signal(
            setup="+".join(sorted(c.setup for c in cands)) if len(cands) > 1 else cand.setup,
            direction="long" if s == 1 else "short",
            t=t,
            time=pd.Timestamp(m.time[t]).tz_localize("UTC") + M15 if pd.Timestamp(m.time[t]).tz is None else pd.Timestamp(m.time[t]) + M15,
            entry_type=cand.entry_type,
            entry_ref=s * float(entry),
            limit_price=s * float(cand.limit_price) if cand.entry_type == "limit" else np.nan,
            stop=s * float(stop),
            tp2=s * float(tp2),
            level=s * float(cand.level),
            atr=float(atr),
            score=float(score),
            risk_mult=0.5 if vol > p.vol_high else 1.0,
            parts=parts,
        )
        _drop(f"{cand.setup}:PASS")
        if best is None or sig.score > best.score:
            best = sig
    return best


def scan(long_m: Market, short_m: Market, t: int, p: Params, flags: TimeFlags) -> list[Signal]:
    out: list[Signal] = []
    if "long" in p.directions and (s := scan_side(long_m, t, p, flags)):
        out.append(s)
    if "short" in p.directions and (s := scan_side(short_m, t, p, flags)):
        out.append(s)
    return out
