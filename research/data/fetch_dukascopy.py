"""Download XAUUSD history from Dukascopy's public datafeed.

M1 bid candles per day drive the simulation; monthly H1 ask candles give the
bid/ask spread per hour (one file per month instead of one per day, which keeps
us under Dukascopy's rate limit). Everything is cached, so the script resumes.
"""

from __future__ import annotations

import argparse
import lzma
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
POINT = 1000.0
MIN_SPREAD = 0.10

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"


def _download(url: str, pause: float) -> bytes | None:
    """File bytes, b"" when the file does not exist, None if the server kept refusing."""
    wait = 30.0
    for _ in range(10):
        try:
            r = _session.get(url, timeout=45)
        except requests.RequestException:
            time.sleep(wait)
            wait = min(wait * 2, 600)
            continue
        if r.status_code == 200:
            time.sleep(pause)
            return r.content
        if r.status_code == 404:
            return b""
        print(f"  {r.status_code} on {url.rsplit('/XAUUSD/', 1)[1]}, sleeping {wait:.0f}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 600)
    return None


def _decode(content: bytes, start: pd.Timestamp) -> pd.DataFrame:
    raw = lzma.decompress(content)
    n = len(raw) // 24
    u = np.frombuffer(raw[: n * 24], dtype=">u4").reshape(n, 6)
    vol = np.frombuffer(raw[: n * 24], dtype=">f4").reshape(n, 6)[:, 5]
    df = pd.DataFrame(
        {"open": u[:, 1] / POINT, "close": u[:, 2] / POINT, "low": u[:, 3] / POINT,
         "high": u[:, 4] / POINT, "volume": vol.astype(float)},
        index=start + pd.to_timedelta(u[:, 0].astype(np.int64), unit="s"),
    )
    return df[df["volume"] > 0]


def _cached(path: Path, url: str, start: pd.Timestamp, pause: float) -> pd.DataFrame | None:
    empty = path.with_suffix(".empty")
    if path.exists():
        return pd.read_parquet(path)
    if empty.exists():
        return None
    content = _download(url, pause)
    if content is None:
        return None
    df = _decode(content, start) if content else pd.DataFrame()
    if df.empty:
        empty.touch()
        return None
    df.to_parquet(path)
    return df


def fetch_m1(day: date, cache: Path, pause: float) -> pd.DataFrame | None:
    url = f"{ROOT}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    return _cached(cache / f"BID_{day.isoformat()}.parquet", url, pd.Timestamp(day, tz="UTC"), pause)


def fetch_h1(year: int, month: int, side: str, cache: Path, pause: float) -> pd.DataFrame | None:
    url = f"{ROOT}/{year}/{month - 1:02d}/{side}_candles_hour_1.bi5"
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    return _cached(cache / f"{side}_H1_{year}-{month:02d}.parquet", url, start, pause)


def build(start: date, end: date, out: Path, pause: float) -> None:
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    today = date.today()

    months = pd.period_range(start, end, freq="M")
    h1_bid, h1_ask = [], []
    for per in months:
        if per.start_time.date() > today or (per.year == today.year and per.month == today.month):
            continue
        for side, bucket in (("BID", h1_bid), ("ASK", h1_ask)):
            df = fetch_h1(per.year, per.month, side, cache, pause)
            if df is not None:
                bucket.append(df)
    print("H1 months:", len(h1_bid), len(h1_ask), flush=True)

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() != 5 and d < today]
    frames, missing = [], []
    for i, d in enumerate(days, 1):
        df = fetch_m1(d, cache, pause)
        if df is not None:
            frames.append(df)
        elif not (cache / f"BID_{d.isoformat()}.empty").exists():
            missing.append(d.isoformat())
        if i % 100 == 0:
            print(f"{i}/{len(days)} days, {len(missing)} unreachable", flush=True)
    if missing:
        (out / "missing_days.txt").write_text(chr(10).join(missing))

    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated()]

    spread = pd.Series(dtype=float)
    if h1_bid and h1_ask:
        hb = pd.concat(h1_bid).sort_index()
        ha = pd.concat(h1_ask).sort_index()
        both = hb[["open", "close"]].join(ha[["open", "close"]], rsuffix="_ask", how="inner")
        spread = ((both["open_ask"] - both["open"]) + (both["close_ask"] - both["close"])) / 2
        spread = spread.clip(lower=MIN_SPREAD)
    hour = m1.index.floor("1h")
    m1["spread"] = spread.reindex(hour).to_numpy() if len(spread) else np.nan
    fallback = m1["spread"].rolling(60 * 24 * 5, min_periods=1).median()
    m1["spread"] = m1["spread"].fillna(fallback).fillna(0.35).clip(lower=MIN_SPREAD)
    m1.to_parquet(out / "xauusd_m1.parquet")

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "spread": "median"}
    m15 = m1.resample("15min", label="left", closed="left").agg(agg).dropna(subset=["open"])
    m15.to_parquet(out / "xauusd_m15.parquet")
    print("M1 rows:", len(m1), "M15 rows:", len(m15), m15.index[0], m15.index[-1],
          "missing days:", len(missing), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--out", default="data")
    p.add_argument("--pause", type=float, default=0.4)
    a = p.parse_args()
    build(date.fromisoformat(a.start), date.fromisoformat(a.end), Path(a.out), a.pause)
