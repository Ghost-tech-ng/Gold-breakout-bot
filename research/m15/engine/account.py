"""Portfolio-level gates: open-trade caps, cooldowns, zone locks and loss limits."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from engine.params import Params
from engine.signals import Signal


@dataclass
class ClosedResult:
    time: pd.Timestamp
    r: float  # already multiplied by the trade's size multiplier


@dataclass
class Account:
    p: Params
    open_dirs: list[str] = field(default_factory=list)
    last_signal: dict[str, tuple[int, float]] = field(default_factory=dict)  # dir -> (t, entry)
    closed: list[ClosedResult] = field(default_factory=list)
    locks: list[tuple[str, int, float, float]] = field(default_factory=list)  # dir, until_t, price, width

    def allows(self, sig: Signal, t: int, now: pd.Timestamp) -> tuple[bool, str]:
        p = self.p
        if len(self.open_dirs) >= p.max_open:
            return False, "max_open"
        if sig.direction in self.open_dirs:
            return False, "one_per_direction"
        last = self.last_signal.get(sig.direction)
        if last and t - last[0] <= p.cooldown_bars:
            return False, "cooldown"
        for d, until, px, width in self.locks:
            if d == sig.direction and t <= until and abs(sig.entry_ref - px) <= width:
                return False, "zone_lock"

        day = now.normalize()
        today = [c.r for c in self.closed if c.time >= day]
        if sum(today) <= p.daily_loss_r:
            return False, "daily_loss"
        streak = 0
        for r in reversed(today):
            if r < 0:
                streak += 1
            else:
                break
        if streak >= p.max_consec_losses_day:
            return False, "loss_streak"
        week = day - pd.Timedelta(days=day.weekday())
        if sum(c.r for c in self.closed if c.time >= week) <= p.weekly_loss_r:
            return False, "weekly_loss"
        return True, ""

    def size_multiplier(self, sig: Signal) -> float:
        mult = sig.risk_mult
        last_two = [c.r for c in self.closed[-2:]]
        if len(last_two) == 2 and all(r < 0 for r in last_two):
            mult *= 0.5
        return mult

    def on_signal(self, sig: Signal, t: int) -> None:
        self.open_dirs.append(sig.direction)
        self.last_signal[sig.direction] = (t, sig.entry_ref)
        self.locks.append((sig.direction, t + self.p.zone_lock_bars, sig.entry_ref,
                           self.p.zone_lock_atr * sig.atr))
        self.locks = [lk for lk in self.locks if lk[1] >= t]

    def on_close(self, direction: str, when: pd.Timestamp, r_weighted: float | None,
                 t: int, level: float | None = None, atr: float = 0.0) -> None:
        if direction in self.open_dirs:
            self.open_dirs.remove(direction)
        if r_weighted is not None:
            self.closed.append(ClosedResult(when, r_weighted))
            if len(self.closed) > 500:
                self.closed = self.closed[-500:]
        if level is not None:
            self.locks.append((direction, t + 20, level, 0.5 * atr))
