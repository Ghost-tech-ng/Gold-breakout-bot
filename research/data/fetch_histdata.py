"""Build the backtest dataset from HistData.com free XAUUSD M1 bid bars.

HistData timestamps are EST without daylight saving (fixed UTC-5). It has no ask
side, so the spread comes from a model fitted on the Dukascopy H1 bid/ask months
already cached by fetch_dukascopy.py: median spread as a fraction of price for
each hour of the week, applied to the current price. That keeps spreads realistic
as gold moved from ~1300 to ~4000.
"""

from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd/{path}"
GET = "https://www.histdata.com/get.php"
MIN_SPREAD = 0.10

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"


def _fetch_zip(path: str, cache: Path) -> pd.DataFrame | None:
    out = cache / f"histdata_{path.replace('/', '-')}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    page_url = PAGE.format(path=path)
    for attempt in range(5):
        try:
            page = _session.get(page_url, timeout=60)
            page.raise_for_status()
            form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', page.text))
            if "tk" not in form:
                print(f"  {path}: no download form (not published yet?)", flush=True)
                return None
            r = _session.post(GET, data=form, headers={"Referer": page_url}, timeout=300)
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
        except (requests.RequestException, zipfile.BadZipFile) as e:
            print(f"  {path}: {e}; retry {attempt + 1}", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(zf.open(name), sep=";", header=None,
                          names=["ts", "open", "high", "low", "close", "volume"])
        idx = pd.to_datetime(raw["ts"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
        df = raw.drop(columns="ts").set_index(idx.dt.tz_localize("UTC")).astype(float)
        df.index.name = None
        df.to_parquet(out)
        print(f"  {path}: {len(df)} rows {df.index[0]} .. {df.index[-1]}", flush=True)
        time.sleep(2)
        return df
    return None


def spread_model(duka_cache: Path) -> pd.Series:
    """Median (ask-bid)/price per hour of week (0..167), from Dukascopy H1 files."""
    bids, asks = sorted(duka_cache.glob("BID_H1_*.parquet")), sorted(duka_cache.glob("ASK_H1_*.parquet"))
    if not bids or not asks:
        raise SystemExit("no Dukascopy H1 bid/ask files to fit the spread model on")
    hb = pd.concat(pd.read_parquet(f) for f in bids).sort_index()
    ha = pd.concat(pd.read_parquet(f) for f in asks).sort_index()
    both = hb[["open", "close"]].join(ha[["open", "close"]], rsuffix="_ask", how="inner")
    spread = ((both["open_ask"] - both["open"]) + (both["close_ask"] - both["close"])) / 2
    frac = (spread.clip(lower=MIN_SPREAD) / both["close"]).rename("frac")
    how = frac.index.weekday * 24 + frac.index.hour
    model = frac.groupby(how).median().reindex(range(168))
    print(f"spread model from {len(frac)} H1 bars {frac.index[0].date()}..{frac.index[-1].date()}; "
          f"median {frac.median() * 1e4:.2f} bp", flush=True)
    return model.fillna(frac.median())


def build(start_year: int, end: date, out: Path, duka_cache: Path) -> None:
    cache = out / "histdata"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [str(y) for y in range(start_year, end.year)]
    paths += [f"{end.year}/{m}" for m in range(1, end.month + 1)]
    frames = [df for p in paths if (df := _fetch_zip(p, cache)) is not None]
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated()]

    model = spread_model(duka_cache)
    how = m1.index.weekday * 24 + m1.index.hour
    m1["spread"] = np.maximum(model.to_numpy()[how] * m1["close"].to_numpy(), MIN_SPREAD)
    m1.to_parquet(out / "xauusd_m1.parquet")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "spread": "median"}
    m15 = m1.resample("15min", label="left", closed="left").agg(agg).dropna(subset=["open"])
    m15.to_parquet(out / "xauusd_m15.parquet")
    print("M1 rows:", len(m1), "M15 rows:", len(m15), m15.index[0], "..", m15.index[-1], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default="data")
    ap.add_argument("--duka-cache", default="data/cache")
    a = ap.parse_args()
    build(a.start_year, date.fromisoformat(a.end), Path(a.out), Path(a.duka_cache))
