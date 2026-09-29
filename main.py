"""Gold trend signal bot: hourly scan of XAU/USD, Telegram alerts for entries, stop moves and exits.

Strategy and backtest: see README.md and trend/core.py.
"""

from __future__ import annotations

import html
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import schedule
from dotenv import load_dotenv

from db import init_db
from keep_alive import bot_status, keep_alive, update_bot_status
from trend import store, telegram
from trend.core import Params
from trend.live import DataError, fetch_market, format_event, new_state, positions_summary, process, regime_line

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

ALERT_AFTER_FAILURES = 3


def load_params() -> Params:
    path = Path("config.json")
    if not path.exists():
        log.warning("config.json not found, using defaults")
        return Params()
    return Params.load(path)


def run_cycle(p: Params, api_key: str) -> str | None:
    """One scan. Returns the regime line on success, None on failure."""
    now = pd.Timestamp.now(tz="UTC")
    try:
        h1, daily = fetch_market(p, api_key, now)
        state = store.load_state() or new_state()
        state, events = process(state, h1, daily, now, p)
        for e in events:
            if e.kind == "exit":
                store.record_trade(e.tf, e.data["entry_time"], e.time.isoformat(),
                                   e.data["entry"], e.data["price"], e.data["r"])
        store.save_state(state)
    except DataError as exc:
        return _failed(str(exc), alert=False)
    except Exception:
        return _failed(traceback.format_exc(), alert=True)

    for e in events:
        log.info("event %s %s %s", e.kind, e.tf, e.data)
        telegram.send(format_event(e, p))
    regime = regime_line(daily, p)
    if bot_status["consecutive_failures"] >= ALERT_AFTER_FAILURES:
        telegram.send("✅ Data feed recovered, scanning again.")
    update_bot_status(runs=bot_status["runs"] + 1, last_success=now.isoformat(), consecutive_failures=0,
                      last_error=None, regime=regime)
    log.info("scan ok: %d H1 bars, last %s, %d events", len(h1), h1.index[-1], len(events))
    return regime


def _failed(message: str, alert: bool) -> None:
    failures = bot_status["consecutive_failures"] + 1
    update_bot_status(consecutive_failures=failures, last_error=message[-500:])
    log.error("scan failed (%d in a row): %s", failures, message)
    if alert or failures == ALERT_AFTER_FAILURES:
        telegram.send(f"⚠️ Scan failed ({failures} in a row):\n<code>{html.escape(message[-600:])}</code>")
    return None


def main() -> None:
    load_dotenv()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        sys.exit("TWELVE_DATA_API_KEY is not set")
    if not (os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")):
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; alerts will only be logged")
    try:
        p = load_params()
    except (OSError, ValueError, TypeError) as exc:
        sys.exit(f"bad config.json: {exc}")

    init_db()
    keep_alive()

    fresh = store.load_state() is None
    regime = run_cycle(p, api_key)
    state = store.load_state() or {}
    lines = [f"🤖 <b>Gold trend bot online</b> ({', '.join(p.timeframes)} books, "
             f"{p.risk_pct_per_signal}% risk per signal)", regime or "Regime: unknown (first scan failed)"]
    if fresh:
        lines.append("First start: tracking from now, no past signals replayed.")
    lines += positions_summary(state) or ["No open positions."]
    telegram.send("\n".join(lines))

    schedule.every().hour.at(":02").do(run_cycle, p, api_key)
    while True:
        schedule.run_pending()
        time.sleep(20)


if __name__ == "__main__":
    main()
