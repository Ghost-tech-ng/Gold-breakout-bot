import itertools, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from trend import bars, sim, st
lo, hi, tf = sys.argv[1], sys.argv[2], sys.argv[3]
b = bars(tf)
rows = []
for n, ks, kt in itertools.product([10, 15, 20, 30, 40, 55], [1.5, 2.0, 2.5, 3.0], [3.0, 4.0, 5.0, 6.0, 8.0]):
    tr = sim(b, n, ks, kt, True)
    tr = tr[(tr.entry >= lo) & (tr.entry < hi)]
    for d in ("long", "short"):
        x = st(tr[tr.dir == d]); y = x.pop("yrs", {})
        rows.append(dict(look=n, ks=ks, kt=kt, dir=d, **x, worst_yr=min(y.values()) if y else None))
df = pd.DataFrame(rows)
df.to_csv(f"grid_{tf}_{lo[:4]}.csv", index=False)
for d in ("long", "short"):
    x = df[df.dir == d]
    print(d, tf, "PF pivot (rows n, cols ks, mean over kt)")
    print(x.pivot_table(index="look", columns="ks", values="pf").round(2))
    print(x.pivot_table(index="look", columns="kt", values="pf").round(2))
    print("share PF>1.2:", round((x.pf > 1.2).mean(), 2), "median exp:", x.exp.median(), "median n:", x.n.median())
