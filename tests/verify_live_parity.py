"""Replay historical H1 bars through the live engine and compare with the backtest.

    python tests/verify_live_parity.py --data ../gold-data --start 2023-08-01 --end 2025-01-01

Both run without spread or slippage. Trades should match exactly, except when price gaps
through a stop inside a multi-hour bar: the backtest fills that at the stop, the live engine
(checking hourly) at the gapped open, which is the more realistic of the two.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trend.core import Params, backtest, daily_uptrend, resample, uptrend_on  # noqa: E402
from trend.live import H1, new_state, process  # noqa: E402

WINDOW = 3000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../gold-data")
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2025-01-01")
    a = ap.parse_args()
    p = Params()

    m15 = pd.read_parquet(Path(a.data) / "xauusd_m15.parquet").drop(columns="spread")
    m15 = m15[m15.index < a.end]
    h1 = resample(m15, "1h")
    daily = m15["close"].resample("1D").last().dropna()

    daily_up = daily_uptrend(daily, p)
    bt = []
    for tf in p.timeframes:
        bars = resample(m15, tf)
        bt.append(backtest(bars, uptrend_on(bars.index, daily_up), tf, p, slip_atr=0.0))
    bt = pd.concat(bt)

    state = new_state()
    live = []
    start = h1.index.searchsorted(pd.Timestamp(a.start, tz="UTC"))
    for i in range(start, len(h1)):
        t = h1.index[i]
        now = t + H1
        state, events = process(state, h1.iloc[max(0, i - WINDOW):i + 1],
                                daily[daily.index < now.normalize()], now, p)
        live += [(e.tf, pd.Timestamp(e.data["entry_time"]), e.time, e.data["entry"], e.data["price"], e.data["r"])
                 for e in events if e.kind == "exit"]
    live = pd.DataFrame(live, columns=["tf", "entry_time", "exit_time", "entry", "exit", "r"])

    first_exit = live["exit_time"].min() if not live.empty else pd.Timestamp(a.end, tz="UTC")
    settled = pd.Timestamp(a.start, tz="UTC") + pd.Timedelta(days=30)
    bt = bt[(bt["entry_time"] >= settled)].reset_index(drop=True)
    live = live[(live["entry_time"] >= settled)].reset_index(drop=True)

    # Match on fill price: across a market break the backtest labels the fill with the
    # resampled bar's start, the live engine with the first H1 bar that actually traded.
    key = ["tf", "fill"]
    for df in (bt, live):
        df["fill"] = df["entry"].round(2)
    m = bt.merge(live, on=key, how="outer", suffixes=("_bt", "_live"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["r_bt"] - both["r_live"]).abs()
    print(f"backtest trades {len(bt)}, live trades {len(live)}, matched {len(both)}, first live exit {first_exit}")
    print(f"max |R diff| on matched: {diff.max():.6f}")
    print(f"total R backtest {bt['r'].sum():.2f}, live {live['r'].sum():.2f}")
    off = both[diff > 1e-6]
    gap_only = ((both["r_live"] <= both["r_bt"] + 1e-9) & (diff < 0.1)).all()
    if not off.empty:
        print("R mismatches:\n", off[key + ["exit_time_bt", "exit_time_live", "exit_bt", "exit_live",
                                            "r_bt", "r_live"]].to_string())
    extra = m[m["_merge"] != "both"]
    if not extra.empty:
        print("unmatched:\n", extra[key + ["_merge", "r_bt", "r_live"]].to_string())
    return 0 if extra.empty and gap_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
