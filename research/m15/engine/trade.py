"""Trade lifecycle shared by the backtester and the live signal tracker.

All internal arithmetic is done in "signed" prices (price * sign), so a short
behaves exactly like a long: stop below, targets above, favourable = up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.params import Params
from engine.signals import Signal


@dataclass
class Bar:
    """One bar on the side we would trade against: ask for buys/short exits, bid for sells/long exits."""
    o: float
    h: float
    l: float
    c: float


def signed(bar: Bar, s: int) -> Bar:
    if s == 1:
        return bar
    return Bar(-bar.o, -bar.l, -bar.h, -bar.c)


@dataclass
class Trade:
    sig: Signal
    p: Params
    s: int = 0
    state: str = "pending"  # pending | open | closed | cancelled
    entry: float = np.nan  # signed
    stop: float = np.nan
    risk: float = np.nan
    tp1: float = np.nan
    tp2: float = np.nan
    level: float = np.nan
    remaining: float = 1.0
    r: float = 0.0
    tp1_hit: bool = False
    bars_pending: int = 0
    bars_open: int = 0
    extreme: float = -np.inf
    mfe_r: float = 0.0
    mae_r: float = 0.0
    entry_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None
    exit_reason: str = ""
    events: list[tuple[pd.Timestamp, str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.s = self.sig.sign
        self.level = self.s * self.sig.level

    # ---- helpers -------------------------------------------------------
    def price(self, signed_px: float) -> float:
        return self.s * signed_px

    def _close_part(self, frac: float, px: float, when: pd.Timestamp, reason: str) -> None:
        frac = min(frac, self.remaining)
        self.r += frac * (px - self.entry) / self.risk
        self.remaining -= frac
        self.events.append((when, reason, self.price(px)))
        if self.remaining <= 1e-9:
            self.state = "closed"
            self.exit_time = when
            self.exit_reason = reason

    def _fill(self, px: float, when: pd.Timestamp) -> bool:
        stop = self.s * self.sig.stop
        risk = px - stop
        atr = self.sig.atr
        if risk <= 0.2 * atr:
            self.state = "cancelled"
            self.exit_reason = "gap_through_stop"
            return False
        if risk < self.p.min_stop_atr * atr:
            stop = px - self.p.min_stop_atr * atr
            risk = px - stop
        self.entry, self.stop, self.risk = px, stop, risk
        self.tp1 = px + self.p.tp1_r * risk
        self.tp2 = max(self.s * self.sig.tp2, self.tp1 + 0.5 * risk)
        self.extreme = px
        self.state = "open"
        self.entry_time = when
        self.events.append((when, "entry", self.price(px)))
        return True

    # ---- entry ---------------------------------------------------------
    def enter_market(self, entry_side_open: float, when: pd.Timestamp, slippage: float) -> bool:
        px = self.s * entry_side_open + slippage
        return self._fill(px, when)

    def try_limit(self, entry_side: Bar, when: pd.Timestamp) -> bool:
        b = signed(entry_side, self.s)
        lim = self.s * self.sig.limit_price
        if b.l <= lim:
            return self._fill(min(b.o, lim), when)
        return False

    # ---- intrabar ------------------------------------------------------
    def on_bar(self, exit_side: Bar, when: pd.Timestamp, slippage: float) -> None:
        """Process one (M1) bar of exit-side prices. Stop is assumed to hit before targets."""
        if self.state != "open":
            return
        b = signed(exit_side, self.s)
        if b.o <= self.stop:
            self._close_part(1.0, b.o - slippage, when, "stop_gap")
            return
        if b.o >= self.tp2:
            self._close_part(1.0, b.o, when, "tp2")
            return
        if b.l <= self.stop:
            self._close_part(1.0, self.stop - slippage, when, "trail" if self.tp1_hit else "stop")
            self._track(b)
            return
        self._track(b)
        if not self.tp1_hit and b.h >= self.tp1:
            self.tp1_hit = True
            self._close_part(self.p.tp1_frac, self.tp1, when, "tp1")
            self.stop = max(self.stop, self.entry + self.p.be_buffer_atr * self.sig.atr)
            if self.state == "closed":
                return
            if b.c <= self.stop:
                self._close_part(1.0, self.stop - slippage, when, "breakeven")
                return
        if b.h >= self.tp2:
            self._close_part(1.0, self.tp2, when, "tp2")

    def _track(self, b: Bar) -> None:
        self.extreme = max(self.extreme, b.h)
        self.mfe_r = max(self.mfe_r, (b.h - self.entry) / self.risk)
        self.mae_r = min(self.mae_r, (b.l - self.entry) / self.risk)

    # ---- bar-close management -------------------------------------------
    def on_close(self, exit_side_close: float, atr: float, when: pd.Timestamp,
                 slippage: float, news_soon: bool = False) -> None:
        """Called at each M15 close while the trade is live."""
        if self.state == "pending":
            self.bars_pending += 1
            if self.s * exit_side_close < self.level or self.bars_pending >= self.p.b_limit_bars:
                self.state = "cancelled"
                self.exit_reason = "limit_expired"
            return
        if self.state != "open":
            return
        self.bars_open += 1
        c = self.s * exit_side_close

        if when.weekday() == 4 and when.hour >= 20:
            self._close_part(1.0, c - slippage, when, "friday_close")
            return
        if self.bars_open <= self.p.failed_break_bars and c < self.level - self.p.failed_break_atr * atr:
            self._close_part(1.0, c - slippage, when, "failed_break")
            return
        if (not self.tp1_hit and self.bars_open >= self.p.time_stop_bars
                and self.mfe_r < self.p.time_stop_r):
            self._close_part(1.0, c - slippage, when, "time_stop")
            return
        if self.bars_open >= self.p.max_hold_bars:
            self._close_part(1.0, c - slippage, when, "max_hold")
            return
        if news_soon and c > self.entry:
            self.stop = max(self.stop, self.entry + self.p.be_buffer_atr * atr)
        if self.tp1_hit:
            self.stop = max(self.stop, self.extreme - self.p.trail_atr * atr)

    def force_close(self, exit_side_close: float, when: pd.Timestamp, reason: str) -> None:
        if self.state == "open":
            self._close_part(1.0, self.s * exit_side_close, when, reason)


def _side(o: float, h: float, l: float, c: float, spread: float, use_ask: bool) -> Bar:
    add = spread if use_ask else 0.0
    return Bar(o + add, h + add, l + add, c + add)


def advance(tr: Trade, times: pd.DatetimeIndex, o: np.ndarray, h: np.ndarray, l: np.ndarray,
            c: np.ndarray, spread: np.ndarray, bar_close: pd.Timestamp, atr: float,
            slippage: float, news_soon: bool) -> None:
    """Run one M15 bar's worth of M1 bid bars through a live trade, then the bar-close rules.

    Longs buy at ask and sell at bid, shorts the reverse; ask = bid + spread.
    Shared by the backtester and the live tracker so both fill and exit identically.
    """
    long = tr.sig.direction == "long"
    for i in range(len(times)):
        when = times[i]
        if tr.state == "pending":
            entry = _side(o[i], h[i], l[i], c[i], spread[i], use_ask=long)
            if tr.sig.entry_type == "market":
                if not tr.enter_market(entry.o, when, slippage):
                    return
            else:
                if not tr.try_limit(entry, when):
                    continue
                # Filled inside this minute: only the stop can be judged on the same bar.
                xb = signed(_side(o[i], h[i], l[i], c[i], spread[i], use_ask=not long), tr.s)
                if xb.l <= tr.stop:
                    tr._close_part(1.0, tr.stop - slippage, when, "stop")
                    return
                continue
        tr.on_bar(_side(o[i], h[i], l[i], c[i], spread[i], use_ask=not long), when, slippage)
        if tr.state == "closed":
            return
    if len(times) and tr.state in ("open", "pending"):
        last = len(times) - 1
        xc = c[last] + (0.0 if long else spread[last])
        tr.on_close(xc, atr, bar_close, slippage, news_soon)
