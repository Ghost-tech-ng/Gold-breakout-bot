"""
Database abstraction layer.

Uses PostgreSQL when DATABASE_URL env var is set (Render hosting),
falls back to SQLite for local development.

Set DATABASE_URL in Render environment variables:
  - Go to Render dashboard → your service → Environment
  - Add DATABASE_URL from your Render PostgreSQL instance
"""

import os
from contextlib import contextmanager

# Render provides postgres:// but psycopg2 requires postgresql://
DATABASE_URL = os.getenv('DATABASE_URL', '')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

IS_POSTGRES = DATABASE_URL.startswith('postgresql://')
SQLITE_PATH = 'trade_history.db'

# SQL dialect constants
PH = '%s' if IS_POSTGRES else '?'                           # placeholder
AUTOINC = 'SERIAL PRIMARY KEY' if IS_POSTGRES else 'INTEGER PRIMARY KEY AUTOINCREMENT'


@contextmanager
def get_db():
    """
    Context manager that yields (conn, cursor).
    Auto-commits on success, rolls back on error, always closes connection.
    """
    if IS_POSTGRES:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
    else:
        import sqlite3
        conn = sqlite3.connect(SQLITE_PATH)

    try:
        cur = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def q(sql: str) -> str:
    """Adapt SQLite ? placeholders to PostgreSQL %s placeholders."""
    if IS_POSTGRES:
        return sql.replace('?', '%s')
    return sql


def init_db():
    """Create all tables if they do not already exist."""
    with get_db() as (conn, cur):
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS positions (
                id {AUTOINC},
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_reward REAL NOT NULL,
                entry_time TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                pnl REAL,
                exit_reason TEXT,
                ml_confidence REAL,
                market_sentiment TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS signal_history (
                id {AUTOINC},
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                price_level REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id {AUTOINC},
                date TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                avg_rr REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                consecutive_losses INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(date)
            )
        """)

    db_type = 'PostgreSQL' if IS_POSTGRES else 'SQLite'
    print(f"✅ Database initialized ({db_type})")
