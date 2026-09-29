import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import trend
from trend import bars, sim, st, atr, daily_trend
lo, hi = sys.argv[1], sys.argv[2]
b = bars("1h")
real = sim(b, 20, 2.5, 5.0, True, dirs=("long",))
real = real[(real.entry >= lo) & (real.entry < hi)]
print("breakout:", {k: v for k, v in st(real).items()})
# random control: replace the Donchian condition by a coin flip at the same hit rate, same filter/exits
up, _ = daily_trend(b)
hh = b.high.rolling(20).max().shift(1).to_numpy(); c = b.close.to_numpy()
rate = ((c > hh) & up).sum() / up.sum()
res = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    coin = rng.random(len(b)) < rate
    fake = b.copy()
    # force "breakout" exactly where coin is true by making the lookback high tiny/huge
    orig = trend.sim
    hi_arr = np.where(coin, -np.inf, np.inf)
    b2 = b.assign(high=b.high)  # unchanged prices
    def sim_rand(bb, n, ks, kt, filt, dirs=("long",), slip=0.05):
        o, h, l, cc, sp = (bb[x].to_numpy() for x in ("open", "high", "low", "close", "spread"))
        a = atr(bb).to_numpy(); out = []; pos = None; idx = bb.index
        for i in range(250, len(bb) - 1):
            if pos is not None:
                if l[i] <= pos["stop"]:
                    px = min(o[i], pos["stop"]) - slip * a[i]
                    out.append((pos["t"], idx[i], "long", (px - pos["e"]) / pos["risk"])); pos = None
                else:
                    pos["best"] = max(pos["best"], h[i]); pos["stop"] = max(pos["stop"], pos["best"] - kt * a[i])
            if pos is None and coin[i] and up[i]:
                e = o[i + 1] + sp[i + 1] + slip * a[i]
                pos = {"t": idx[i + 1], "e": e, "stop": e - ks * a[i], "risk": ks * a[i], "best": e}
        return pd.DataFrame(out, columns=["entry", "exit", "dir", "r"])
    tr = sim_rand(b, 20, 2.5, 5.0, True)
    tr = tr[(tr.entry >= lo) & (tr.entry < hi)]
    x = st(tr); res.append((x["n"], x["pf"], x["exp"], x["tot"]))
r = pd.DataFrame(res, columns=["n", "pf", "exp", "tot"])
print("random-entry control (20 seeds):"); print(r.describe().loc[["mean", "50%", "max"]].round(2))
