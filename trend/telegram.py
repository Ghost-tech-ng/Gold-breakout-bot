from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


def send(text: str) -> bool:
    """Post an HTML message to the configured chat. Never raises: a Telegram outage must not stop the engine."""
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("Telegram not configured; message dropped: %s", text[:80])
        return False
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=15, json={
            "chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False
