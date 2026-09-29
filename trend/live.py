"""Live signal engine. `process` is pure: completed H1 bars + daily closes in, state + events out.

Stops are checked on every completed H1 bar (the stop only changes at a book's bar close,
so this triggers exactly when the backtest's per-bar check would). At each 1h/2h/4h bar
close the book trails its stop, then looks for a new breakout. A signal is announced at
the signal bar's close and re-anchored to the next bar's open, which is the fill the
backtest assumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from trend.core import TF_HOURS, Params, Position, atr, breakout, daily_uptrend, resample

H1 = pd.Timedelta(hours=1)
# A bucket whose last H1 bar never arrives is closed on the clock only after this long,
# so a vendor that is a few minutes late can't make us close a half-built bar.
MARKET_CLOSED_AFTER = pd.Timedelta(minutes=90)
TD_URL = "https://api.twelvedata.com/time_series"


class DataError(RuntimeError):
    pass


@dataclass
class Event:
    kind: str  # entry | fill | trail | exit
    tf: str
    time: pd.Timestamp
    data: dict[str, Any] = field(default_factory=dict)


def bucket(t: pd.Timestamp, tf: str) -> pd.Timestamp:
    return t.floor(f"{TF_HOURS[tf]}h")


def trend_up_for(day: pd.Timestamp, daily_close: pd.Series, p: Params) -> bool:
    """Trend state as of the last completed day before `day`."""
    past = daily_close[daily_close.index.normalize() < day.normalize()]
    if len(past) < p.ema_slow:
        return False
    return bool(daily_uptrend(past, p).iloc[-1])


def new_state() -> dict[str, Any]:
    return {"last_h1": None, "closed": {}, "positions": {}}


def process(state: dict[str, Any], h1: pd.DataFrame, daily_close: pd.Series,
            now: pd.Timestamp, p: Params) -> tuple[dict[str, Any], list[Event]]:
    """`h1` must hold completed bars only; `daily_close` completed UTC days only."""
    events: list[Event] = []
    if h1.empty:
        return state, events

    if state.get("last_h1") is None:
        last = h1.index[-1]
        state["last_h1"] = last.isoformat()
        for tf in p.timeframes:
            b = bucket(last, tf)
            done = b + pd.Timedelta(hours=TF_HOURS[tf]) <= last + H1
            state["closed"][tf] = (b if done else b - pd.Timedelta(hours=TF_HOURS[tf])).isoformat()
        return state, events

    positions: dict[str, Position] = {tf: Position(**d) for tf, d in state["positions"].items()}
    pending: set[str] = set(state.get("pending", []))
    closed = {tf: pd.Timestamp(v) for tf, v in state["closed"].items()}
    last_seen = pd.Timestamp(state["last_h1"])

    def close_bucket(tf: str, b: pd.Timestamp) -> None:
        if b <= closed.get(tf, pd.Timestamp.min.tz_localize("UTC")):
            return
        closed[tf] = b
        bars = resample(h1[h1.index < b + pd.Timedelta(hours=TF_HOURS[tf])], tf)
        if len(bars) < max(p.lookback, p.atr_n) + 1 or bars.index[-1] != b:
            return
        a_now = float(atr(bars, p.atr_n).iloc[-1])
        bar = bars.iloc[-1]
        pos = positions.get(tf)
        if pos is not None and tf not in pending:
            pos.trail(float(bar["high"]), a_now, p)
            if pos.stop - pos.announced_stop >= p.stop_alert_min_r * pos.risk:
                events.append(Event("trail", tf, b, {"from": pos.announced_stop, "to": pos.stop,
                                                     "open_r": pos.r(float(bar["close"]))}))
                pos.announced_stop = pos.stop
        if pos is None and breakout(bars, p)[-1] and trend_up_for(b, daily_close, p):
            pos = Position.open(tf, b + pd.Timedelta(hours=TF_HOURS[tf]), float(bar["close"]), a_now, p)
            positions[tf] = pos
            pending.add(tf)
            events.append(Event("entry", tf, b, {"price": pos.entry, "stop": pos.stop, "risk": pos.risk, "atr": a_now}))

    for t, bar in h1[h1.index > last_seen].iterrows():
        for tf in p.timeframes:
            prev_b = bucket(last_seen, tf)
            if prev_b < bucket(t, tf):
                close_bucket(tf, prev_b)
        o, lo = float(bar["open"]), float(bar["low"])
        for tf in list(pending):
            pos = positions[tf]
            gap = o - pos.signal_close
            pos.entry, pos.best, pos.stop = o, o, o - pos.risk
            pos.announced_stop = pos.stop
            pos.entry_time = t.isoformat()
            pending.discard(tf)
            if abs(gap) > 0.1 * pos.risk:
                events.append(Event("fill", tf, t, {"price": o, "stop": pos.stop, "gap": gap}))
        for tf, pos in list(positions.items()):
            px = pos.stop_hit(o, lo)
            if px is not None:
                events.append(Event("exit", tf, t, {"price": px, "entry": pos.entry, "entry_time": pos.entry_time,
                                                    "r": pos.r(px)}))
                del positions[tf]
        for tf in p.timeframes:
            if (t + H1).hour % TF_HOURS[tf] == 0:
                close_bucket(tf, bucket(t, tf))
        last_seen = t

    for tf in p.timeframes:
        b = bucket(last_seen, tf)
        if b + pd.Timedelta(hours=TF_HOURS[tf]) + MARKET_CLOSED_AFTER <= now:
            close_bucket(tf, b)

    state["last_h1"] = last_seen.isoformat()
    state["closed"] = {tf: b.isoformat() for tf, b in closed.items()}
    state["positions"] = {tf: pos.to_dict() for tf, pos in positions.items()}
    state["pending"] = sorted(pending)
    return state, events


def fetch_series(symbol: str, interval: str, outputsize: int, api_key: str) -> pd.DataFrame:
    try:
        resp = requests.get(TD_URL, timeout=30, params={
            "symbol": symbol, "interval": interval, "outputsize": outputsize,
            "timezone": "UTC", "order": "ASC", "apikey": api_key})
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataError(f"TwelveData {interval} request failed: {exc}") from exc
    if body.get("status") != "ok" or not body.get("values"):
        raise DataError(f"TwelveData {interval}: {body.get('message', 'no data')}")
    df = pd.DataFrame(body["values"])
    df.index = pd.to_datetime(df.pop("datetime"), utc=True)
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def fetch_market(p: Params, api_key: str, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """Completed H1 bars and completed-day closes (UTC days; H1-derived where H1 covers them,
    since the backtest's days are UTC days and the vendor's daily bars may not be)."""
    h1 = fetch_series(p.symbol, "1h", 5000, api_key)
    h1 = h1[h1.index + H1 <= now]
    daily = fetch_series(p.symbol, "1day", 1000, api_key)["close"]
    daily.index = daily.index.normalize()
    from_h1 = h1["close"].resample("1D").last().dropna()
    closes = pd.concat([daily[daily.index < from_h1.index[0]], from_h1])
    return h1, closes[closes.index < now.normalize()]


def lots_for(risk_dist: float, p: Params) -> float | None:
    raw = os.getenv("ACCOUNT_BALANCE", "").strip()
    try:
        balance = float(raw)
    except ValueError:
        return None
    if balance <= 0 or risk_dist <= 0:
        return None
    return round(balance * p.risk_pct_per_signal / 100 / (risk_dist * p.contract_oz), 2)


def format_event(e: Event, p: Params) -> str:
    d = e.data
    book = f"{p.symbol} · {e.tf} book"
    if e.kind == "entry":
        lots = lots_for(d["risk"], p)
        size = f"\nSize: <b>{lots} lots</b> ({p.risk_pct_per_signal}% risk)" if lots else \
            f"\nSize: risk {p.risk_pct_per_signal}% of equity over ${d['risk']:.2f}/oz"
        return (f"🟢 <b>BUY {book}</b>\n"
                f"Entry: market (signal close {d['price']:.2f})\n"
                f"Stop: <b>{d['stop']:.2f}</b> ({d['risk']:.2f} below)\n"
                f"Target: none, trailing stop manages the exit{size}")
    if e.kind == "fill":
        return (f"↕️ <b>{book}</b>: market opened at {d['price']:.2f} ({d['gap']:+.2f} vs signal).\n"
                f"Stop re-anchored to <b>{d['stop']:.2f}</b>")
    if e.kind == "trail":
        return (f"🔼 <b>Move stop</b> {book}: {d['from']:.2f} → <b>{d['to']:.2f}</b>\n"
                f"Open P/L: {d['open_r']:+.2f}R")
    if e.kind == "exit":
        icon = "✅" if d["r"] > 0 else "🔴"
        return (f"{icon} <b>Stopped out</b> {book}\n"
                f"Entry {d['entry']:.2f} → exit {d['price']:.2f}: <b>{d['r']:+.2f}R</b>\n"
                f"If your broker stop hasn't filled, close this position at market.")
    return f"{e.kind} {book}"


def regime_line(daily_close: pd.Series, p: Params) -> str:
    if len(daily_close) < p.ema_slow:
        return "Regime: not enough daily history"
    up = bool(daily_uptrend(daily_close, p).iloc[-1])
    fast = daily_close.ewm(span=p.ema_fast).mean().iloc[-1]
    slow = daily_close.ewm(span=p.ema_slow).mean().iloc[-1]
    word = "UPTREND, longs allowed" if up else "no uptrend, standing aside"
    return f"Regime: <b>{word}</b> (close {daily_close.iloc[-1]:.2f}, EMA50 {fast:.2f}, EMA200 {slow:.2f})"


def positions_summary(state: dict[str, Any]) -> list[str]:
    out = []
    for tf, d in sorted(state.get("positions", {}).items(), key=lambda kv: TF_HOURS[kv[0]]):
        out.append(f"{tf}: long from {d['entry']:.2f}, stop {d['stop']:.2f}")
    return out

