"""Small Flask server so Render keeps the worker up, plus read-only status endpoints."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import Any

from flask import Flask, Response, jsonify

log = logging.getLogger(__name__)
app = Flask(__name__)

STALE_AFTER = timedelta(hours=2, minutes=15)

bot_status: dict[str, Any] = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "runs": 0,
    "last_success": None,
    "last_error": None,
    "consecutive_failures": 0,
    "regime": None,
}


def update_bot_status(**fields: Any) -> None:
    bot_status.update(fields)


@app.route("/")
def home() -> Response:
    return jsonify({"status": "alive", "service": "Gold trend signal bot", "started_at": bot_status["started_at"]})


@app.route("/health")
def health() -> tuple[Response, int]:
    from trend import store

    checks: dict[str, str] = {}
    try:
        checks["database"] = "ok" if store.ping() else "error"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    last = bot_status["last_success"]
    started = datetime.fromisoformat(bot_status["started_at"])
    reference = datetime.fromisoformat(last) if last else started
    checks["engine"] = "ok" if datetime.now(timezone.utc) - reference < STALE_AFTER else "stale"

    healthy = all(v == "ok" for v in checks.values())
    return jsonify({"status": "healthy" if healthy else "degraded", "checks": checks,
                    "bot_status": bot_status}), 200 if healthy else 503


@app.route("/stats")
def stats() -> tuple[Response, int]:
    from trend import store

    try:
        state = store.load_state() or {}
        return jsonify({"open_positions": state.get("positions", {}), "totals": store.totals(),
                        "recent_trades": store.recent_trades(20), "bot_status": bot_status}), 200
    except Exception as exc:
        log.exception("stats failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/config")
def get_config() -> tuple[Response, int]:
    try:
        with open("config.json", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500


def keep_alive() -> None:
    port = int(os.getenv("PORT", "5000"))
    Thread(target=lambda: app.run(host="0.0.0.0", port=port, debug=False), daemon=True).start()
    log.info("Web server listening on port %s", port)
