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
    """Create the tables the signal engine needs if they do not already exist."""
    with get_db() as (conn, cur):
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS trend_trades (
                id {AUTOINC},
                book TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                r REAL NOT NULL,
                UNIQUE(book, entry_time)
            )
        """)

    db_type = 'PostgreSQL' if IS_POSTGRES else 'SQLite'
    print(f"Database initialized ({db_type})")
