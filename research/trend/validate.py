import sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trend import bars, sim, st, m15

def eq(tr, risk=0.0075):
    if tr.empty: return 0.0, 0.0
    e = np.cumprod(1 + risk * tr.sort_values("exit").r.to_numpy())
    dd = (e / np.maximum.accumulate(np.r_[1.0, e])[1:] - 1).min()
    return round((e[-1] - 1) * 100, 1), round(dd * 100, 1)

def bh(lo, hi):
    d = m15.close[(m15.index >= lo) & (m15.index < hi)].resample("1D").last().dropna()
    return round((d.iloc[-1] / d.iloc[0] - 1) * 100, 1), round((d / d.cummax() - 1).min() * 100, 1)

wins = [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
h1, h4 = bars("1h"), bars("4h")
cache = {}
def get(b, tf, n, ks, kt, dirs):
    k = (tf, n, ks, kt, dirs)
    if k not in cache: cache[k] = sim(b, n, ks, kt, True, dirs=dirs)
    return cache[k]
def cut(tr, lo, hi): return tr[(tr.entry >= lo) & (tr.entry < hi)]

for lo, hi in wins:
    print(f"\n######## {lo} .. {hi}   buy&hold ret%/maxDD% = {bh(lo, hi)}")
    x = cut(get(h1, "1h", 20, 2.5, 5.0, ("long",)), lo, hi)
    print("CHOSEN H1 N20 2.5/5 long:", st(x), " equity@0.75% ret/dd:", eq(x))
    rows = []
    for n, ks, kt in itertools.product([10, 15, 20, 30, 40, 55], [1.5, 2.0, 2.5, 3.0], [3.0, 4.0, 5.0, 6.0, 8.0]):
        s = st(cut(get(h1, "1h", n, ks, kt, ("long",)), lo, hi)); s.pop("yrs", None)
        rows.append(dict(look=n, ks=ks, kt=kt, **s))
    g = pd.DataFrame(rows)
    print(f"H1 long grid ({len(g)} variants): PF>1 {(g.pf>1).mean():.0%}  PF>1.2 {(g.pf>1.2).mean():.0%}  median pf {g.pf.median():.2f} exp {g.exp.median():.3f} n {g.n.median():.0f}  worst pf {g.pf.min():.2f}")
    nb = g[g.look.isin([15, 20, 30]) & g.ks.isin([2.0, 2.5, 3.0]) & g.kt.isin([4.0, 5.0, 6.0])]
    print(f"  neighbourhood (27): PF>1 {(nb.pf>1).mean():.0%}  median pf {nb.pf.median():.2f}  min pf {nb.pf.min():.2f}")
    s = cut(get(h1, "1h", 20, 2.5, 5.0, ("short",)), lo, hi); print("info H1 short:", {k: v for k, v in st(s).items() if k != 'yrs'})
    for d in ("long", "short"):
        s = cut(get(h4, "4h", 20, 2.5, 5.0, (d,)), lo, hi); print(f"info H4 {d}:", {k: v for k, v in st(s).items() if k != 'yrs'})
    for slip in (0.10, 0.15):
        s = cut(sim(h1, 20, 2.5, 5.0, True, dirs=("long",), slip=slip), lo, hi); print(f"slip {slip}:", {k: v for k, v in st(s).items() if k != 'yrs'})
