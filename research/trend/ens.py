import warnings, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trend import bars, sim, st, m15
W = {"IS": ("2019-01-01", "2023-01-01"), "H1-23": ("2023-01-01", "2023-08-01"), "VAL": ("2023-08-01", "2025-01-01")}
if "--holdout" in sys.argv: W = {"HOLDOUT": ("2025-01-01", "2026-09-26")}
def cut(tr, lo, hi): return tr[(tr.entry >= lo) & (tr.entry < hi)]
def curve(tr, lo, hi, risk):
    """daily equity from closed trades, compounding, risk per trade = risk (fraction)"""
    tr = tr.sort_values("exit")
    days = pd.date_range(lo, hi, freq="D", tz="UTC")
    pnl = tr.groupby(tr.exit.dt.floor("D")).apply(lambda x: np.prod(1 + risk * x.r.to_numpy() * x.w.to_numpy()) - 1)
    return (1 + pnl.reindex(days, fill_value=0)).cumprod()
def metr(eq):
    r = eq.pct_change().fillna(0); yrs = len(eq) / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    return f"ret {100*(eq.iloc[-1]-1):6.1f}%  cagr {100*cagr:5.1f}%  maxDD {100*dd:6.1f}%  sharpe {sh:4.2f}  calmar {cagr/abs(dd) if dd else 0:4.2f}"
sets = {tf: sim(bars(tf), 20, 2.5, 5.0, True, dirs=("long",)).assign(w=1.0, tf=tf) for tf in ("1h", "2h", "4h")}
for w, (lo, hi) in W.items():
    d = m15.close[(m15.index >= lo) & (m15.index < hi)].resample("1D").last().ffill()
    print(f"\n#### {w} {lo}..{hi}\n  buy&hold 1x          {metr(d / d.iloc[0])}")
    for tf, tr in sets.items():
        x = cut(tr, lo, hi); s = st(x)
        print(f"  {tf} breakout   n {s['n']:3d} pf {s.get('pf',0):.2f} exp {s.get('exp',0):+.3f} dd {s.get('dd',0):5.1f}R  yrs {s.get('yrs')}\n     @0.75%/trade    {metr(curve(x, lo, hi, .0075))}")
    ens = pd.concat([cut(t, lo, hi).assign(w=1 / 3) for t in sets.values()])
    s = st(ens.assign(r=ens.r / 3))
    print(f"  ENSEMBLE 1h+2h+4h (1/3 each) n {s['n']} pf {s['pf']} exp/trade {s['exp']:+.3f} totR {s['tot']} ddR {s['dd']} yrs {s['yrs']}")
    for risk in (0.0075, 0.015):
        print(f"     @{risk*100:.2f}%/signal   {metr(curve(ens, lo, hi, risk))}")
