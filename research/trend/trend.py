"""Fast bar-level research sim: Donchian breakout + ATR chandelier trail on H1/H4.
Bid OHLC + per-bar spread; long buys ask / sells bid. Stop checked before anything else."""
import itertools, os, sys
import numpy as np, pandas as pd

D = os.environ.get("GOLD_DATA", "../gold-data").rstrip("/") + "/"
m15 = pd.read_parquet(D + "xauusd_m15.parquet")

def bars(tf):
    return m15.resample(tf, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "median"}).dropna()

def atr(b, n=14):
    pc = b.close.shift()
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def daily_trend(b):
    d = m15.close.resample("1D").last().dropna()
    e50, e200 = d.ewm(span=50).mean(), d.ewm(span=200).mean()
    up = (d > e50) & (e50 > e200)
    dn = (d < e50) & (e50 < e200)
    # yesterday's daily state, forward-filled onto intraday bars (causal)
    up = up.shift(1).reindex(b.index, method="ffill").fillna(False).to_numpy()
    dn = dn.shift(1).reindex(b.index, method="ffill").fillna(False).to_numpy()
    return up, dn

def sim(b, n, k_stop, k_trail, filt, dirs=("long", "short"), slip=0.05, exit_n=None):
    o, h, l, c, sp = (b[x].to_numpy() for x in ("open", "high", "low", "close", "spread"))
    a = atr(b).to_numpy()
    hh = b.high.rolling(n).max().shift(1).to_numpy()
    ll = b.low.rolling(n).min().shift(1).to_numpy()
    up, dn = daily_trend(b) if filt else (np.ones(len(b), bool), np.ones(len(b), bool))
    t_idx = b.index
    out = []
    for direction in dirs:
        s = 1 if direction == "long" else -1
        pos = None
        for i in range(250, len(b) - 1):
            if pos is not None:
                # exit side: long exits at bid, short exits at ask (bid+spread)
                xo, xh, xl = (o[i], h[i], l[i]) if s == 1 else (o[i] + sp[i], h[i] + sp[i], l[i] + sp[i])
                stop = pos["stop"]
                hit = xl <= stop if s == 1 else xh >= stop
                if hit:
                    px = (min(xo, stop) if s == 1 else max(xo, stop)) - s * slip * a[i]
                    out.append((pos["t"], t_idx[i], direction, s * (px - pos["e"]) / pos["risk"]))
                    pos = None
                else:
                    xc = c[i] if s == 1 else c[i] + sp[i]
                    ext = (h[i] if s == 1 else -(l[i] + sp[i]))
                    pos["best"] = max(pos["best"], ext)
                    trail = s * (pos["best"] - k_trail * a[i]) if s == 1 else -(pos["best"] - k_trail * a[i])
                    pos["stop"] = max(stop, trail) if s == 1 else min(stop, trail)
                    if exit_n and ((s == 1 and c[i] < ll_x[i]) or (s == -1 and c[i] > hh_x[i])):
                        px = o[i + 1] + (0 if s == 1 else sp[i + 1])
                        out.append((pos["t"], t_idx[i + 1], direction, s * (px - pos["e"]) / pos["risk"]))
                        pos = None
            if pos is None and not np.isnan(hh[i]) and not np.isnan(a[i]):
                sig = (c[i] > hh[i] and up[i]) if s == 1 else (c[i] < ll[i] and dn[i])
                if sig:
                    e = o[i + 1] + (sp[i + 1] if s == 1 else 0) + s * slip * a[i]
                    stop = e - s * k_stop * a[i]
                    pos = {"t": t_idx[i + 1], "e": e, "stop": stop, "risk": k_stop * a[i],
                           "best": e if s == 1 else -e}
        if pos is not None:
            out.append((pos["t"], t_idx[-1], direction, s * ((c[-1] if s == 1 else c[-1] + sp[-1]) - pos["e"]) / pos["risk"]))
    return pd.DataFrame(out, columns=["entry", "exit", "dir", "r"]).sort_values("exit")

def st(tr):
    if tr.empty:
        return dict(n=0)
    r = tr.r.to_numpy()
    cum = np.cumsum(r); dd = (cum - np.maximum.accumulate(np.r_[0, cum])[1:]).min()
    w, lo = r[r > 0].sum(), -r[r < 0].sum()
    yrs = tr.groupby(tr.exit.dt.year).r.sum().round(1).to_dict()
    return dict(n=len(r), win=round((r > 0).mean(), 2), pf=round(w / lo, 2) if lo else np.inf,
                exp=round(r.mean(), 3), tot=round(r.sum(), 1), dd=round(dd, 1), yrs=yrs)

if __name__ == "__main__":
    lo, hi = sys.argv[1], sys.argv[2]
    rows = []
    for tf, n, ks, kt, f in itertools.product(["1h", "4h"], [20, 55], [2.0, 3.0], [3.0, 5.0], [False, True]):
        b = bars(tf)
        ll_x = hh_x = None
        tr = sim(b, n, ks, kt, f)
        tr = tr[(tr.entry >= lo) & (tr.entry < hi)]
        for d in ("long", "short"):
            x = st(tr[tr.dir == d])
            rows.append((tf, n, ks, kt, f, d, x))
            print(tf, n, ks, kt, f, d, {k: v for k, v in x.items() if k != "yrs"}, x.get("yrs"), flush=True)
