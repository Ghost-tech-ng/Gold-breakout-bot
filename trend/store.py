"""Persistence for engine state and closed trades (Postgres on Render, SQLite locally)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from db import get_db, q

STATE_KEY = "engine"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict[str, Any] | None:
    with get_db() as (_, cur):
        cur.execute(q("SELECT value FROM bot_state WHERE key = ?"), (STATE_KEY,))
        row = cur.fetchone()
    return json.loads(row[0]) if row else None


def save_state(state: dict[str, Any]) -> None:
    upsert = ("INSERT INTO bot_state (key, value, updated_at) VALUES (?, ?, ?) "
              "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
    with get_db() as (_, cur):
        cur.execute(q(upsert), (STATE_KEY, json.dumps(state), _now()))


def record_trade(book: str, entry_time: str, exit_time: str, entry: float, exit_price: float, r: float) -> None:
    sql = ("INSERT INTO trend_trades (book, entry_time, exit_time, entry_price, exit_price, r) "
           "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (book, entry_time) DO NOTHING")
    with get_db() as (_, cur):
        cur.execute(q(sql), (book, entry_time, exit_time, entry, exit_price, r))


def recent_trades(limit: int = 20) -> list[dict[str, Any]]:
    with get_db() as (_, cur):
        cur.execute(q("SELECT book, entry_time, exit_time, entry_price, exit_price, r FROM trend_trades "
                      "ORDER BY exit_time DESC LIMIT ?"), (limit,))
        rows = cur.fetchall()
    cols = ("book", "entry_time", "exit_time", "entry", "exit", "r")
    return [dict(zip(cols, row)) for row in rows]


def totals() -> dict[str, float]:
    with get_db() as (_, cur):
        cur.execute("SELECT COUNT(*), COALESCE(SUM(r), 0), "
                    "COALESCE(SUM(CASE WHEN r > 0 THEN r ELSE 0 END), 0), "
                    "COALESCE(SUM(CASE WHEN r < 0 THEN -r ELSE 0 END), 0) FROM trend_trades")
        n, total, won, lost = cur.fetchone()
    return {"trades": int(n), "total_r": round(float(total), 2),
            "profit_factor": round(float(won) / float(lost), 2) if lost else None}


def ping() -> bool:
    with get_db() as (_, cur):
        cur.execute("SELECT 1")
        return cur.fetchone() is not None

