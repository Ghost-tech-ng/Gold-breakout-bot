"""Long-side breakout detectors. Shorts run the same code on a mirrored Market."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.indicators import fit_line
from engine.market import Market
from engine.params import Params


@dataclass
class Level:
    price: float
    touches: int
    last_touch: int  # M15 bar index of the latest M15 touch, -1 if H1-only
    h1: bool


@dataclass
class Candidate:
    setup: str
    t: int
    level: float  # the line/level that was broken, for the failed-break exit
    stop: float
    entry_type: str  # "market" | "limit"
    limit_price: float = np.nan
    target: float = np.nan  # measured-move target (setup C)
    level_quality: float = 0.0  # 0..1
    compression: bool = False


def candle_ok(m: Market, t: int, p: Params) -> tuple[bool, float, float]:
    rng = m.h[t] - m.l[t]
    if rng <= 0:
        return False, 0.0, 0.0
    body = m.c[t] - m.o[t]
    body_frac = body / rng
    close_pos = (m.c[t] - m.l[t]) / rng
    ok = body > 0 and body_frac >= p.min_body_frac and close_pos >= p.min_close_pos
    ok = ok and rng <= p.max_range_atr * m.atr[t]
    return ok, body_frac, close_pos


def levels(m: Market, t: int, p: Params) -> list[Level]:
    atr = m.atr[t]
    hi = m.swings_hi_known(t, p.b_m15_lookback)
    lo = m.swings_lo_known(t, p.b_m15_lookback)
    idx = np.concatenate((hi, lo))
    px = np.concatenate((m.h[hi], m.l[lo]))
    h1_hi, h1_lo = m.h1_swings_known(t, p.b_h1_lookback)
    h1_px = np.concatenate((h1_hi, h1_lo))

    all_px = np.concatenate((px, h1_px))
    all_idx = np.concatenate((idx, np.full(len(h1_px), -1)))
    is_h1 = np.concatenate((np.zeros(len(px), bool), np.ones(len(h1_px), bool)))
    near = np.abs(all_px - m.c[t]) <= p.level_range_atr * atr + p.b_cluster_atr * atr
    all_px, all_idx, is_h1 = all_px[near], all_idx[near], is_h1[near]
    if len(all_px) == 0:
        return []

    order = np.argsort(all_px)
    all_px, all_idx, is_h1 = all_px[order], all_idx[order], is_h1[order]
    out: list[Level] = []
    start = 0
    tol = p.b_cluster_atr * atr
    for i in range(1, len(all_px) + 1):
        if i == len(all_px) or all_px[i] - all_px[start] > tol:
            grp = slice(start, i)
            m15_touch = all_idx[grp][~is_h1[grp]]
            out.append(Level(
                price=float(all_px[grp].mean()),
                touches=int(i - start),
                last_touch=int(m15_touch.max()) if len(m15_touch) else -1,
                h1=bool(is_h1[grp].any()),
            ))
            start = i
    return out


def setup_a(m: Market, t: int, p: Params) -> Candidate | None:
    atr = m.atr[t]
    sw = m.swings_hi_known(t, p.a_lookback)
    if len(sw) < p.a_min_swings:
        return None
    run = [int(sw[-1])]
    for j in sw[-2::-1]:
        if m.h[j] > m.h[run[0]]:
            run.insert(0, int(j))
        else:
            break
    if len(run) < p.a_min_swings:
        return None
    x = np.asarray(run, dtype=float)
    y = m.h[run]
    a, b, r2 = fit_line(x, y)
    if r2 < p.a_min_r2 or b > p.a_max_slope_atr * atr:
        return None
    touches = int((np.abs(y - (a + b * x)) <= p.a_touch_tol_atr * atr).sum())
    if touches < p.a_min_swings:
        return None

    first = run[0]
    span = np.arange(first, t)
    if (m.c[first:t] > a + b * span + 0.10 * atr).any():
        return None
    line_prev = a + b * (t - 1)
    line_now = a + b * t
    if not (m.c[t - 1] <= line_prev and m.c[t] >= line_now + p.a_break_atr * atr):
        return None
    if m.c[t] - line_now > p.max_extension_atr * atr:
        return None
    if m.c[t] < m.ema50[t] - 0.3 * atr:
        return None

    stop = float(m.l[first : t + 1].min() - p.a_stop_buffer_atr * atr)
    return Candidate("A", t, float(line_now), stop, "market",
                     level_quality=1.0 if touches >= 4 else 0.5)


def setup_b(m: Market, t: int, p: Params, lv: list[Level]) -> Candidate | None:
    atr = m.atr[t]
    body = m.c[t] - m.o[t]
    if body < p.b_min_body_atr * atr:
        return None
    best: Candidate | None = None
    for L in lv:
        if L.touches < p.b_min_touches:
            continue
        if not L.h1 and t - L.last_touch > p.b_max_age:
            continue
        lvl = L.price
        if not (m.c[t - 1] < lvl and m.c[t] >= lvl + p.b_break_atr * atr):
            continue
        if m.c[t] - lvl > p.max_extension_atr * atr:
            continue
        lo = max(0, t - 20)
        if (m.c[lo : t - 1] > lvl).any():
            continue
        stop = float(min(m.l[t], lvl - 0.5 * atr) - 0.10 * atr)
        quality = 1.0 if (L.touches >= 3 or L.h1) else 0.5
        compression = bool(m.atr5[t - 1] / m.atr50[t - 1] < 0.8) if m.atr50[t - 1] > 0 else False
        cand = Candidate(
            "B", t, lvl, stop,
            "limit" if p.b_entry == "limit" else "market",
            limit_price=lvl + p.b_limit_offset_atr * atr,
            level_quality=quality,
            compression=compression,
        )
        if best is None or lvl > best.level:
            best = cand
    return best


def setup_c(m: Market, t: int, p: Params, h1_bullish: bool) -> Candidate | None:
    atr = m.atr[t]
    hi = m.swings_hi_known(t, p.c_lookback)
    lo = m.swings_lo_known(t, p.c_lookback)
    if len(hi) < 2 or len(lo) < 2 or len(hi) + len(lo) < p.c_min_touches:
        return None
    start = int(min(hi[0], lo[0]))
    if t - start < p.c_min_span:
        return None
    ua, ub, ur2 = fit_line(hi.astype(float), m.h[hi])
    la, lb, lr2 = fit_line(lo.astype(float), m.l[lo])
    if ur2 < p.c_min_r2 or lr2 < p.c_min_r2:
        return None
    w0 = (ua + ub * start) - (la + lb * start)
    w1 = (ua + ub * (t - 1)) - (la + lb * (t - 1))
    if w0 < p.c_min_height_atr * atr or w1 <= 0:
        return None
    ratio = w1 / w0
    triangle = ratio <= 0.70
    flag = abs(ratio - 1) < 0.20 and ub < 0 and lb < 0 and h1_bullish
    if not (triangle or flag):
        return None

    span = np.arange(start, t)
    up = ua + ub * span
    dn = la + lb * span
    if (m.c[start:t] > up + 0.10 * atr).any() or (m.c[start:t] < dn - 0.10 * atr).any():
        return None
    upper_now = ua + ub * t
    if not (m.c[t - 1] <= ua + ub * (t - 1) and m.c[t] >= upper_now + p.c_break_atr * atr):
        return None
    if m.c[t] - upper_now > p.max_extension_atr * atr:
        return None

    stop = float(m.l[lo[-1]] - 0.10 * atr)
    target = float(m.c[t] + min(w0, p.c_target_cap_atr * atr))
    touches = len(hi) + len(lo)
    return Candidate("C", t, float(upper_now), stop, "market", target=target,
                     level_quality=1.0 if touches >= 6 else 0.5, compression=True)
