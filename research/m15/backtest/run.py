"""Event-driven backtest: signals on closed M15 bars, fills and exits on M1 bid/ask."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.account import Account  # noqa: E402
from engine.market import Market, build_market  # noqa: E402
from engine.news import historical_events  # noqa: E402
from engine.params import Params  # noqa: E402
from engine.signals import FUNNEL, Signal, TimeFlags, scan, time_flags  # noqa: E402
from engine.trade import Bar, Trade, advance  # noqa: E402


@dataclass
class Data:
    m15: pd.DataFrame
    long: Market
    short: Market
    flags: TimeFlags
    m1_o: np.ndarray
    m1_h: np.ndarray
    m1_l: np.ndarray
    m1_c: np.ndarray
    m1_spread: np.ndarray
    m1_time: pd.DatetimeIndex
    bar_start: np.ndarray  # M1 row where each M15 bar starts; len = len(m15) + 1


def load(data_dir: Path, use_news: bool = True) -> Data:
    m15 = pd.read_parquet(data_dir / "xauusd_m15.parquet")
    m1 = pd.read_parquet(data_dir / "xauusd_m1.parquet")
    if m15.index.tz is None:
        m15.index = m15.index.tz_localize("UTC")
    if m1.index.tz is None:
        m1.index = m1.index.tz_localize("UTC")
    news = historical_events(m15.index[0].date(), m15.index[-1].date()) if use_news else []
    starts = np.searchsorted(m1.index.asi8, m15.index.asi8, side="left")
    bar_start = np.append(starts, len(m1))
    return Data(
        m15=m15,
        long=build_market(m15),
        short=build_market(m15, mirror=True),
        flags=time_flags(m15.index, news),
        m1_o=m1["open"].to_numpy(), m1_h=m1["high"].to_numpy(),
        m1_l=m1["low"].to_numpy(), m1_c=m1["close"].to_numpy(),
        m1_spread=m1["spread"].to_numpy(),
        m1_time=m1.index,
        bar_start=bar_start,
    )


def _bars(d: Data, i: int, side: str, direction: str) -> Bar:
    """side 'entry' or 'exit'. Longs buy at ask and sell at bid; shorts the reverse."""
    use_ask = (side == "entry") == (direction == "long")
    add = d.m1_spread[i] if use_ask else 0.0
    return Bar(d.m1_o[i] + add, d.m1_h[i] + add, d.m1_l[i] + add, d.m1_c[i] + add)


Scanner = Callable[["Data", int, Params], list[Signal]]


def default_scanner(d: "Data", t: int, p: Params) -> list[Signal]:
    return scan(d.long, d.short, t, p, d.flags)


def simulate(d: Data, p: Params, start: str | None = None, end: str | None = None,
             scanner: Scanner = default_scanner) -> pd.DataFrame:
    idx = d.m15.index
    t0 = max(300, int(idx.searchsorted(pd.Timestamp(start, tz="UTC")))) if start else 300
    t1 = int(idx.searchsorted(pd.Timestamp(end, tz="UTC"))) if end else len(idx) - 1
    t1 = min(t1, len(idx) - 2)

    acct = Account(p)
    live: list[tuple[Trade, float, int]] = []  # trade, size multiplier, signal bar
    rows: list[dict] = []
    atr = d.long.atr
    news_soon = d.flags.news_block

    for t in range(t0, t1 + 1):
        a, b = d.bar_start[t], d.bar_start[t + 1]
        now_close = idx[t] + pd.Timedelta(minutes=15)
        slip = p.slippage_atr * atr[t]

        still: list[tuple[Trade, float, int]] = []
        for tr, mult, ts in live:
            direction = tr.sig.direction
            advance(tr, d.m1_time[a:b], d.m1_o[a:b], d.m1_h[a:b], d.m1_l[a:b], d.m1_c[a:b],
                    d.m1_spread[a:b], now_close, atr[t], slip, bool(news_soon[t]))

            if tr.state in ("closed", "cancelled"):
                filled = tr.entry_time is not None
                r_w = tr.r * mult if filled else None
                failed = tr.exit_reason == "failed_break"
                acct.on_close(direction, now_close, r_w, t,
                              level=tr.sig.level if failed else None, atr=atr[t])
                if filled:
                    rows.append({
                        "signal_time": tr.sig.time, "entry_time": tr.entry_time, "exit_time": tr.exit_time,
                        "setup": tr.sig.setup, "direction": direction, "score": tr.sig.score,
                        "entry_type": tr.sig.entry_type, "r": tr.r, "mult": mult, "r_w": tr.r * mult,
                        "reason": tr.exit_reason, "tp1": tr.tp1_hit, "mfe": tr.mfe_r, "mae": tr.mae_r,
                        "risk_atr": tr.risk / tr.sig.atr, "bars": tr.bars_open,
                        "entry": tr.price(tr.entry), "atr": tr.sig.atr,
                        **{f"s_{k}": v for k, v in tr.sig.parts.items()},
                    })
            else:
                still.append((tr, mult, ts))
        live = still

        for sig in scanner(d, t, p):
            ok, _ = acct.allows(sig, t, now_close)
            if not ok:
                continue
            mult = acct.size_multiplier(sig)
            acct.on_signal(sig, t)
            live.append((Trade(sig, p), mult, t))

    for tr, mult, _ in live:
        if tr.state == "open":
            last = d.bar_start[t1 + 1] - 1
            tr.force_close(_bars(d, last, "exit", tr.sig.direction).c, d.m1_time[last], "end")
            rows.append({"signal_time": tr.sig.time, "entry_time": tr.entry_time, "exit_time": tr.exit_time,
                         "setup": tr.sig.setup, "direction": tr.sig.direction, "score": tr.sig.score,
                         "entry_type": tr.sig.entry_type, "r": tr.r, "mult": mult, "r_w": tr.r * mult,
                         "reason": tr.exit_reason, "tp1": tr.tp1_hit, "mfe": tr.mfe_r, "mae": tr.mae_r,
                         "risk_atr": tr.risk / tr.sig.atr, "bars": tr.bars_open,
                         "entry": tr.price(tr.entry), "atr": tr.sig.atr})
    return pd.DataFrame(rows)


def stats(tr: pd.DataFrame, risk_pct: float = 0.75) -> dict[str, float]:
    if tr.empty:
        return {"trades": 0}
    r = tr["r_w"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    dd = cum - np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    eq = np.cumprod(1 + risk_pct / 100 * r)
    eq_dd = eq / np.maximum.accumulate(np.concatenate(([1.0], eq)))[1:] - 1
    q = tr.set_index("exit_time")["r_w"].resample("QE").sum()
    neg = (q < 0).astype(int).to_numpy()
    worst_run = max((len(s) for s in "".join(map(str, neg)).split("0")), default=0)
    years = max((tr["exit_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 365.25, 1e-9)
    return {
        "trades": int(len(r)),
        "per_month": round(len(r) / (years * 12), 1),
        "win_rate": round(float((r > 0).mean()), 3),
        "pf": round(float(wins / losses), 2) if losses > 0 else float("inf"),
        "exp_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 1),
        "max_dd_r": round(float(dd.min()), 1),
        "return_pct": round(float(eq[-1] - 1) * 100, 1),
        "cagr_pct": round(float(eq[-1] ** (1 / years) - 1) * 100, 1),
        "max_dd_pct": round(float(eq_dd.min()) * 100, 1),
        "losing_q_run": int(worst_run),
        "losing_q": int(neg.sum()),
        "quarters": int(len(neg)),
    }


def breakdown(tr: pd.DataFrame, by: str | pd.Series) -> pd.DataFrame:
    g = tr.groupby(by)
    return pd.DataFrame({k: stats(v) for k, v in g}).T[["trades", "win_rate", "pf", "exp_r", "total_r", "max_dd_r"]]


def report(tr: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(stats(tr)))
    if tr.empty:
        return
    print(breakdown(tr, "setup").to_string())
    print(breakdown(tr, "direction").to_string())
    print(breakdown(tr, tr["exit_time"].dt.year).to_string())
    buckets = pd.cut(tr["score"], [0, 65, 70, 75, 80, 101])
    print(breakdown(tr, buckets.astype(str)).to_string())
    print(tr["reason"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../gold-data")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--params", default=None, help="JSON file of Params overrides")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--out", default=None, help="write trades CSV here")
    a = ap.parse_args()

    p = Params()
    if a.params:
        p = Params.from_dict({**p.__dict__, **json.loads(Path(a.params).read_text())})
    for kv in a.set:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        val = tuple(v.split(",")) if isinstance(cur, tuple) else type(cur)(v if not isinstance(cur, bool) else v.lower() == "true")
        p = p.with_(**{k: val})
    d = load(Path(a.data), use_news=p.use_news)
    trades = simulate(d, p, a.start, a.end)
    report(trades, f"{a.start} to {a.end}")
    print("funnel:", dict(sorted(FUNNEL.items())))
    if a.out:
        trades.to_csv(a.out, index=False)
