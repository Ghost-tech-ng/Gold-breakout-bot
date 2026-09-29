"""Backtest the three-book trend strategy on the HistData M15 dataset.

    python -m trend.backtest --data ../gold-data --risk 0.5
    python -m trend.backtest --start 2025-01-01 --end 2026-09-26

The dataset (bid OHLC + modelled spread) is built by research/data/fetch_histdata.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trend.core import Params, backtest, daily_uptrend, resample, uptrend_on

WINDOWS = {
    "in-sample": ("2019-01-01", "2023-01-01"),
    "2023 H1 (patchy data)": ("2023-01-01", "2023-08-01"),
    "validation": ("2023-08-01", "2025-01-01"),
    "holdout": ("2025-01-01", "2026-09-26"),
}


def run_all(m15: pd.DataFrame, p: Params, slip_atr: float) -> pd.DataFrame:
    daily_up = daily_uptrend(m15["close"].resample("1D").last().dropna(), p)
    books = []
    for tf in p.timeframes:
        bars = resample(m15, tf)
        books.append(backtest(bars, uptrend_on(bars.index, daily_up), tf, p, slip_atr))
    return pd.concat(books, ignore_index=True).sort_values("exit_time")


def trade_stats(tr: pd.DataFrame) -> dict[str, float]:
    if tr.empty:
        return {"trades": 0}
    r = tr["r"].to_numpy()
    cum = np.cumsum(r)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": len(r),
        "win_rate": round(float((r > 0).mean()), 2),
        "pf": round(float(wins / losses), 2) if losses else float("inf"),
        "exp_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 1),
        "max_dd_r": round(float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min()), 1),
    }


def equity(tr: pd.DataFrame, start: str, end: str, risk_pct: float) -> pd.Series:
    days = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    growth = (1 + risk_pct / 100 * tr["r"]).groupby(tr["exit_time"].dt.floor("D")).prod()
    return growth.reindex(days, fill_value=1.0).cumprod()


def curve_stats(eq: pd.Series) -> dict[str, float]:
    years = len(eq) / 365.25
    daily = eq.pct_change().fillna(0)
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    return {
        "return_pct": round((float(eq.iloc[-1]) - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(dd * 100, 1),
        "sharpe": round(float(daily.mean() / daily.std() * np.sqrt(365)), 2) if daily.std() > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../gold-data")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--risk", type=float, default=None, help="%% of equity risked per signal")
    ap.add_argument("--slip", type=float, default=0.05, help="slippage per fill, in ATR")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="write the trade list to this CSV")
    a = ap.parse_args()

    p = Params.load(a.config) if Path(a.config).exists() else Params()
    risk = a.risk if a.risk is not None else p.risk_pct_per_signal
    m15 = pd.read_parquet(Path(a.data) / "xauusd_m15.parquet")
    trades = run_all(m15, p, a.slip)
    if a.out:
        trades.to_csv(a.out, index=False)

    windows = {"custom": (a.start, a.end)} if a.start and a.end else WINDOWS
    for name, (lo, hi) in windows.items():
        tr = trades[(trades["entry_time"] >= lo) & (trades["entry_time"] < hi)]
        closes = m15["close"][(m15.index >= lo) & (m15.index < hi)].resample("1D").last().dropna()
        print(f"\n=== {name}: {lo} to {hi} ===")
        print("  all books      ", trade_stats(tr))
        for tf in p.timeframes:
            print(f"  {tf:>3} book       ", trade_stats(tr[tr["tf"] == tf]))
        print(f"  equity @{risk}%/signal", curve_stats(equity(tr, lo, hi, risk)))
        print("  gold buy & hold", curve_stats(closes / closes.iloc[0]))
        print("  R by year      ", tr.groupby(tr["exit_time"].dt.year)["r"].sum().round(1).to_dict())


if __name__ == "__main__":
    main()
