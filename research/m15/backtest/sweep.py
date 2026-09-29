"""Load the data once and run a grid of parameter variants over one date range.

    python backtest/sweep.py --start 2019-01-01 --end 2023-01-01 \
        --grid setups=A,B,C min_score=55,60,70 --base use_h1_bias=false
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.benchmark import asian_range_scanner  # noqa: E402
from backtest.run import load, simulate, stats  # noqa: E402
from engine.params import Params  # noqa: E402


def parse_value(p: Params, key: str, raw: str) -> object:
    cur = getattr(p, key)
    if isinstance(cur, tuple):
        return tuple(raw.split("+"))
    if isinstance(cur, bool):
        return raw.lower() == "true"
    return type(cur)(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../gold-data")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--base", nargs="*", default=[], help="key=value applied to every run")
    ap.add_argument("--grid", nargs="*", default=[], help="key=v1,v2 (tuples use + inside a value)")
    ap.add_argument("--bench", action="store_true", help="also run the Asian-range benchmark")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    base = Params()
    for kv in a.base:
        k, v = kv.split("=", 1)
        base = base.with_(**{k: parse_value(base, k, v)})
    keys, values = [], []
    for kv in a.grid:
        k, v = kv.split("=", 1)
        keys.append(k)
        values.append([parse_value(base, k, x) for x in v.split(",")])

    d = load(Path(a.data), use_news=True)
    rows = []
    if a.bench:
        tr = simulate(d, base.with_(use_news=True), a.start, a.end, scanner=asian_range_scanner(d))
        rows.append({"variant": "BENCH asian-range", **stats(tr)})
    for combo in itertools.product(*values) if keys else [()]:
        p = base.with_(**dict(zip(keys, combo)))
        tr = simulate(d, p, a.start, a.end)
        label = " ".join(f"{k}={v if not isinstance(v, tuple) else '+'.join(v)}" for k, v in zip(keys, combo)) or "base"
        rows.append({"variant": label, **stats(tr)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows).set_index("variant")
    cols = ["trades", "per_month", "win_rate", "pf", "exp_r", "total_r", "max_dd_r", "cagr_pct", "max_dd_pct", "losing_q_run"]
    print(table.reindex(columns=cols).to_string())
    if a.csv:
        table.to_csv(a.csv)


if __name__ == "__main__":
    main()
