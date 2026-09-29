import itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trend import bars, sim, st, atr, daily_trend
W = {"IS": ("2019-01-01", "2023-01-01"), "VAL": ("2023-08-01", "2025-01-01"), "ALL18-25": ("2018-06-01", "2025-01-01")}
def cut(tr, w): lo, hi = W[w]; return tr[(tr.entry >= lo) & (tr.entry < hi)]
for tf in ("4h", "2h"):
    b = bars(tf); rows = []
    for n, ks, kt in itertools.product([10, 15, 20, 30, 40], [1.5, 2.0, 2.5, 3.0], [3.0, 4.0, 5.0, 6.0, 8.0]):
        tr = sim(b, n, ks, kt, True, dirs=("long",))
        for w in W:
            s = st(cut(tr, w)); s.pop("yrs", None); rows.append(dict(w=w, look=n, ks=ks, kt=kt, **s))
    g = pd.DataFrame(rows)
    for w in W:
        x = g[g.w == w]
        print(f"{tf} {w}: PF>1 {(x.pf>1).mean():.0%} PF>1.2 {(x.pf>1.2).mean():.0%} med pf {x.pf.median():.2f} exp {x.exp.median():.3f} n {x.n.median():.0f} min pf {x.pf.min():.2f}")
    g.to_csv(f"grid_{tf}_isval.csv", index=False)
    # random control on this tf
    up, _ = daily_trend(b); hh = b.high.rolling(20).max().shift(1).to_numpy(); c = b.close.to_numpy()
    rate = ((c > hh) & up).sum() / up.sum()
    o, h, l, sp = (b[x].to_numpy() for x in ("open", "high", "low", "spread")); a = atr(b).to_numpy(); idx = b.index
    for w in ("IS", "VAL"):
        res = []
        for seed in range(20):
            coin = np.random.default_rng(seed).random(len(b)) < rate; out = []; pos = None
            for i in range(250, len(b) - 1):
                if pos:
                    if l[i] <= pos["stop"]:
                        px = min(o[i], pos["stop"]) - .05 * a[i]; out.append((pos["t"], idx[i], "long", (px - pos["e"]) / pos["risk"])); pos = None
                    else:
                        pos["best"] = max(pos["best"], h[i]); pos["stop"] = max(pos["stop"], pos["best"] - 5 * a[i])
                if pos is None and coin[i] and up[i]:
                    e = o[i + 1] + sp[i + 1] + .05 * a[i]; pos = {"t": idx[i + 1], "e": e, "stop": e - 2.5 * a[i], "risk": 2.5 * a[i], "best": e}
            s = st(cut(pd.DataFrame(out, columns=["entry", "exit", "dir", "r"]), w)); res.append((s["n"], s["pf"], s["exp"]))
        r = pd.DataFrame(res, columns=["n", "pf", "exp"])
        real = st(cut(sim(b, 20, 2.5, 5.0, True, dirs=("long",)), w))
        print(f"  {tf} {w} breakout N20: n {real['n']} pf {real['pf']} exp {real['exp']} | random mean pf {r.pf.mean():.2f} exp {r.exp.mean():.3f} max pf {r.pf.max():.2f}")
