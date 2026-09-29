"""High-impact USD event calendars: a historical approximation for backtests and
the live ForexFactory weekly feed."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BEFORE = 30
NEWS_AFTER = 30
FOMC_AFTER = 120

FOMC_DATES = """
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
""".split()

NewsEvent = tuple[pd.Timestamp, int, int]


def _ny(d: date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(d.year, d.month, d.day, hh, mm, tz="America/New_York").tz_convert("UTC")


def historical_events(start: date, end: date) -> list[NewsEvent]:
    """FOMC statements (14:00 NY) and NFP (first Friday, 08:30 NY).

    CPI and other 08:30 prints are not included because their exact historical
    dates are not in this dataset; the backtest therefore slightly overstates
    what the live news filter will allow.
    """
    ev: list[NewsEvent] = [(_ny(date.fromisoformat(d), 14, 0), NEWS_BEFORE, FOMC_AFTER) for d in FOMC_DATES]
    d = date(start.year, start.month, 1)
    while d <= end:
        first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
        ev.append((_ny(first_friday, 8, 30), NEWS_BEFORE, NEWS_AFTER))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return sorted(ev)


def live_events(timeout: float = 15.0) -> list[NewsEvent] | None:
    """High-impact USD events for the current week. None if the feed is unreachable."""
    try:
        r = requests.get(FF_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        rows = r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("news feed unavailable: %s", e)
        return None
    out: list[NewsEvent] = []
    for row in rows:
        if row.get("country") != "USD" or row.get("impact") != "High":
            continue
        try:
            when = pd.Timestamp(row["date"]).tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        after = FOMC_AFTER if "FOMC" in row.get("title", "") or "Federal Funds" in row.get("title", "") else NEWS_AFTER
        out.append((when, NEWS_BEFORE, after))
    return sorted(out)
