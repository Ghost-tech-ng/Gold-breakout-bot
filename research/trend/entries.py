import itertools, warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trend import bars, st, atr, daily_trend
W = {"IS": ("2019-01-01", "2023-01-01"), "VAL": ("2023-08-01", "2025-01-01")}
which = sys.argv[1:] or ["IS"]
def cut(tr, w): lo, hi = W[w]; return tr[(tr.entry >= lo) & (tr.entry < hi)]
def rsi(c, n):
    d = c.diff(); u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn)
def run(b, sig, ks, kt, slip=0.05, flip_exit=False, up=None):
    o, h, l, c, sp = (b[x].to_numpy() for x in ("open", "high", "low", "close", "spread")); a = atr(b).to_numpy(); idx = b.index
    out = []; pos = None
    for i in range(250, len(b) - 1):
        if pos:
            if l[i] <= pos["stop"]:
                px = min(o[i], pos["stop"]) - slip * a[i]; out.append((pos["t"], idx[i], "long", (px - pos["e"]) / pos["risk"])); pos = None
            elif flip_exit and not up[i]:
                px = o[i + 1] - slip * a[i]; out.append((pos["t"], idx[i + 1], "long", (px - pos["e"]) / pos["risk"])); pos = None; continue
            else:
                pos["best"] = max(pos["best"], h[i]); pos["stop"] = max(pos["stop"], pos["best"] - kt * a[i])
        if pos is None and sig[i]:
            e = o[i + 1] + sp[i + 1] + slip * a[i]; pos = {"t": idx[i + 1], "e": e, "stop": e - ks * a[i], "risk": ks * a[i], "best": e}
    return pd.DataFrame(out, columns=["entry", "exit", "dir", "r"])
for tf in ("1h", "2h", "4h"):
    b = bars(tf); up, _ = daily_trend(b); c = b.close
    ema20 = c.ewm(span=20).mean()
    S = {
        "breakout20": (c > b.high.rolling(20).max().shift(1)).to_numpy() & up,
        "dip_low5": (c < b.low.rolling(5).min().shift(1)).to_numpy() & up,
        "dip_low10": (c < b.low.rolling(10).min().shift(1)).to_numpy() & up,
        "rsi2<10": (rsi(c, 2) < 10).to_numpy() & up,
        "rsi3<20": (rsi(c, 3) < 20).to_numpy() & up,
        "below_ema20": ((c < ema20) & (c.shift() >= ema20.shift())).to_numpy() & up,
        "any_bar": up.copy(),
    }
    for name, sig in S.items():
        for ks, kt, fx in [(2.5, 5.0, False), (2.0, 4.0, False), (3.0, 6.0, False), (2.5, 5.0, True)]:
            tr = run(b, sig, ks, kt, flip_exit=fx, up=up)
            line = []
            for w in which:
                s = st(cut(tr, w)); line.append(f"{w} n {s['n']:4d} pf {s['pf']:.2f} exp {s['exp']:+.3f} dd {s['dd']:6.1f}")
            print(f"{tf} {name:12s} ks{ks} kt{kt} flip{int(fx)} | " + " | ".join(line), flush=True)
