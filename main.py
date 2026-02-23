import os
from keep_alive import keep_alive
import time
import schedule
import asyncio
import json
from datetime import datetime, timedelta
import requests
from breakouts import detect_breakouts
from logger import log_trade, log_position_update, log_performance_summary, log_system_event
from gpt_brain import (get_market_sentiment, should_skip_news,
                       get_trading_session_multiplier, apply_volatility_adaptation,
                       determine_h1_bias)
from position_manager import position_manager
from risk_manager import initialize_risk_manager
from performance_analytics import analytics
from twelvedata import TDClient
from collections import deque
import pandas as pd

# Load configuration
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("❌ Error: config.json not found. Please ensure it exists.")
    exit(1)
except json.JSONDecodeError:
    print("❌ Error: config.json is malformed. Please check its syntax.")
    exit(1)

# API and Telegram setup with environment variables
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not TWELVE_DATA_API_KEY:
    print("❌ Error: TWELVE_DATA_API_KEY not set in Secrets. Please add it.")
    exit(1)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        "❌ Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in Secrets. Please add them."
    )
    exit(1)

try:
    TELEGRAM_CHAT_ID = int(TELEGRAM_CHAT_ID)
except ValueError:
    print("❌ Error: TELEGRAM_CHAT_ID must be a valid integer.")
    exit(1)

# M15 only — tight scalp signals with trendline breakouts
timeframe_queue = deque(["M15"])

# Confidence threshold for M15 scalping (lower than before)
CONFIDENCE_THRESHOLD = 0.5

# H1 bias cache — refreshed every h1_bias_cache_minutes
_h1_bias_cache: dict = {"bias": "ranging", "fetched_at": None}


def send_telegram_message(message):
    """Send message to Telegram chat with detailed error handling."""
    if not message or not message.strip():
        print(f"❌ Error: Attempted to send empty message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        print(f"📤 Sending Telegram message: {message[:50]}...")
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print(f"✅ Telegram message sent successfully!")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error sending Telegram message: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Telegram message: {e}")
    except Exception as e:
        print(f"❌ Unexpected error sending Telegram message: {e}")


def get_direction_emoji(direction):
    return "🟢📈" if direction.lower() == "long" else "🔴📉"


def get_strategy_emoji(strategy):
    emoji_map = {
        "resistance_breakout":           "🚀",
        "support_breakout":              "⬇️",
        "triangle_breakout":             "🔺",
        "channel_breakout":              "📊",
        "ascending_triangle_breakout":   "📈🔺",
        "descending_triangle_breakout":  "📉🔻",
        "symmetrical_triangle_breakout": "⚖️🔺",
        "downtrend_line_break":          "📉➡️📈",   # trendline break long
        "uptrend_line_break":            "📈➡️📉",   # trendline break short
    }
    return emoji_map.get(strategy, "📊")


def get_rr_emoji(rr_ratio):
    if rr_ratio >= 3.0:
        return "💎🔥"
    elif rr_ratio >= 2.0:
        return "⭐✨"
    elif rr_ratio >= 1.5:
        return "🎯"
    else:
        return "📊"


def send_telegram_signal(signal):
    """Send enhanced trade signal to Telegram chat."""
    if not signal:
        print("❌ Error: Attempted to send signal with empty data. Aborting.")
        return

    direction_emoji = get_direction_emoji(signal['direction'])
    strategy_emoji  = get_strategy_emoji(signal['strategy'])
    rr_emoji        = get_rr_emoji(signal['rr'])

    pip_value        = abs(signal['entry'] - signal['sl'])
    profit_potential = abs(signal['tp'] - signal['entry'])

    # Extra info for trendline signals
    tl_info = ""
    if signal['strategy'] in ("downtrend_line_break", "uptrend_line_break"):
        touches = signal.get("trendline_touches", "?")
        tl_info = f"\n🔗 <b>Trendline Touches:</b> {touches}"

    message = f"""
🏆 <b>GOLD SCALP SIGNAL</b> 🏆
{strategy_emoji} <b>Strategy:</b> {signal['strategy'].replace('_', ' ').title()}
{direction_emoji} <b>Direction:</b> {signal['direction'].upper()}

💰 <b>TRADE DETAILS</b>
🎯 <b>Entry:</b> ${signal['entry']:.3f}
🛡️ <b>Stop Loss:</b> ${signal['sl']:.3f}
🎁 <b>Take Profit:</b> ${signal['tp']:.3f}
{rr_emoji} <b>Risk:Reward:</b> 1:{signal['rr']:.2f}

📊 <b>ANALYSIS</b>
📈 <b>Symbol:</b> {signal['symbol']}
⏰ <b>Timeframe:</b> {signal['timeframe']} (Scalp)
💎 <b>Risk:</b> ${pip_value:.3f}
🎯 <b>Profit Potential:</b> ${profit_potential:.3f}{tl_info}

🕐 <b>Signal Time:</b> {signal['timestamp']}

⚠️ <b>Trade Responsibly!</b>
🔒 <b>Risk Management is Key!</b>
"""
    send_telegram_message(message)


def send_startup_message(current_price, opening_price, closing_price,
                         market_sentiment=""):
    """Send enhanced startup message."""
    price_change = closing_price - opening_price if opening_price and closing_price else 0
    change_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"

    if current_price is not None:
        message = f"""
🤖 <b>GOLD BOT ACTIVATED</b> 🤖
⚡ <b>System Status:</b> ONLINE ✅

💰 <b>CURRENT GOLD PRICE</b>
🏅 <b>XAU/USD:</b> ${current_price:.3f}
📊 <b>Latest Candle:</b>
   • Open: ${opening_price:.3f}
   • Close: ${closing_price:.3f}
   {change_emoji} Change: ${price_change:.3f}

🎯 <b>MONITORING:</b>
• Timeframe: M15 (Scalping Mode)
• Trendline Breakouts (NEW)
• Support/Resistance Breakouts
• Chart Pattern Breakouts
• Fakeout Detection

{market_sentiment}

🚀 <b>Ready to Hunt Gold Scalp Breakouts!</b>
"""
    else:
        message = f"""
🤖 <b>GOLD BOT ACTIVATED</b> 🤖
⚡ <b>System Status:</b> ONLINE ✅

❌ <b>Unable to fetch current prices</b>
🔄 <b>Retrying price fetch...</b>

🎯 <b>MONITORING:</b>
• Timeframe: M15 (Scalping Mode)
• Trendline Breakouts (NEW)
• Support/Resistance Breakouts

🚀 <b>Ready to Hunt Gold Scalp Breakouts!</b>
"""
    send_telegram_message(message)


async def fetch_data(symbol, timeframe):
    """Fetch OHLCV data from Twelve Data with improved error handling."""
    interval_map = {
        "M5":  "5min",
        "M15": "15min",
        "M30": "30min",
        "H1":  "1h",
        "H4":  "4h",
        "D1":  "1day"
    }

    interval = interval_map.get(timeframe, "15min")
    td = TDClient(apikey=TWELVE_DATA_API_KEY)

    for attempt in range(3):
        try:
            print(f"🔍 Fetching {symbol} data for {timeframe} (attempt {attempt + 1})")

            from keep_alive import update_bot_status
            update_bot_status("api_call_start", datetime.now().isoformat())

            ts = td.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=150,
                timezone="UTC",
                order="ASC").as_pandas()

            if ts.empty:
                print(f"⚠️ Warning: No data returned for {symbol} ({timeframe})")
                update_bot_status("api_call_failed", "Empty data returned")
                return None

            required_columns = ["open", "high", "low", "close"]
            if not all(col in ts.columns for col in required_columns):
                print(f"❌ Missing required columns for {symbol} ({timeframe})")
                update_bot_status("api_call_failed", "Missing columns in data")
                return None

            if "volume" in ts.columns:
                ts = ts[required_columns + ["volume"]].copy()
            else:
                try:
                    volume_data = td.time_series(
                        symbol=symbol,
                        interval=interval,
                        outputsize=150,
                        timezone="UTC",
                        order="ASC"
                    ).with_volume().as_pandas()

                    if not volume_data.empty and "volume" in volume_data.columns:
                        ts = ts[required_columns].copy()
                        ts["volume"] = volume_data["volume"].reindex(ts.index, fill_value=0)
                    else:
                        ts = ts[required_columns].copy()
                        ts["volume"] = abs(ts["close"].diff()).fillna(0) * 1000
                        print("⚠️ Using price change as volume proxy")
                except Exception as vol_error:
                    print(f"⚠️ Could not fetch volume: {vol_error}, using proxy")
                    ts = ts[required_columns].copy()
                    ts["volume"] = abs(ts["close"].diff()).fillna(0) * 1000

            ts = ts.sort_index()
            ts = ts.dropna()

            for col in required_columns + ["volume"]:
                if col in ts.columns:
                    ts[col] = pd.to_numeric(ts[col], errors='coerce')

            print(f"✅ Successfully fetched {len(ts)} candles for {symbol} ({timeframe})")
            update_bot_status("api_call_success", datetime.now().isoformat())

            return ts.tail(100)

        except Exception as e:
            print(f"❌ Error fetching data (attempt {attempt + 1}): {e}")
            from keep_alive import update_bot_status
            update_bot_status("api_call_failed", str(e))

            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return None

    return None


async def get_latest_prices():
    """Fetch the latest prices for XAU/USD."""
    data = await fetch_data("XAU/USD", "M15")
    if data is not None and not data.empty:
        latest_candle = data.iloc[-1]
        current_price = float(latest_candle["close"])
        opening_price = float(latest_candle["open"])
        closing_price = float(latest_candle["close"])
        if 1000 <= current_price <= 5000:
            return current_price, opening_price, closing_price
    return None, None, None


async def get_h1_bias() -> str:
    """Return cached H1 bias, refreshing when the cache has expired."""
    cache_minutes = config.get("h1_bias_cache_minutes", 60)
    now = datetime.utcnow()

    fetched_at = _h1_bias_cache.get("fetched_at")
    if fetched_at is None or (now - fetched_at).total_seconds() > cache_minutes * 60:
        print("🏗️ Fetching H1 data for top-down bias...")
        h1_data = await fetch_data("XAU/USD", "H1")
        if h1_data is not None and len(h1_data) >= 30:
            bias = determine_h1_bias(h1_data)
            _h1_bias_cache["bias"] = bias
            _h1_bias_cache["fetched_at"] = now
            print(f"🏗️ H1 bias updated: {bias.upper()}")
        else:
            print("⚠️ Could not fetch H1 data — keeping previous bias")

    return _h1_bias_cache["bias"]


async def scan_markets():
    """Enhanced market scanning: trendline + S/R breakouts on M15."""
    if not timeframe_queue:
        return

    timeframe = timeframe_queue[0]   # Always M15, no rotation needed
    symbol = "XAU/USD"

    print(f"🔍 Scanning {symbol} on {timeframe}")

    from keep_alive import update_bot_status
    update_bot_status("last_scan", datetime.now().isoformat())
    update_bot_status("total_scans", None)

    if should_skip_news(config):
        print("📰 Skipping scan due to high-impact news event")
        return

    data = await fetch_data(symbol, timeframe)
    if data is None:
        print(f"❌ Skipping {timeframe} due to data fetch failure.")
        return

    market_sentiment = get_market_sentiment(data)
    session_mult, current_session = get_trading_session_multiplier(config)

    # Fetch (or return cached) H1 bias for top-down analysis
    h1_bias = await get_h1_bias()
    bias_emoji = {"bullish": "🟢", "bearish": "🔴", "ranging": "🟡"}.get(h1_bias, "⚪")

    print(f"📊 Session: {current_session} (multiplier: {session_mult}x) | "
          f"Confidence: {market_sentiment.get('confidence', 0):.2f} | "
          f"H1 Bias: {h1_bias.upper()}")

    # First scan notification
    from keep_alive import bot_status
    if bot_status.get("total_scans", 0) <= 1:
        latest_candle = data.iloc[-1]
        candle_time = data.index[-1] if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        message = f"""
📊 <b>FIRST SCAN COMPLETE</b> ✅

💰 <b>Latest {symbol} Candle ({timeframe})</b>
🕐 <b>Time:</b> {candle_time}

📈 <b>OHLC Data:</b>
   • Open: ${latest_candle['open']:.3f}
   • High: ${latest_candle['high']:.3f}
   • Low: ${latest_candle['low']:.3f}
   • Close: ${latest_candle['close']:.3f}
   • Volume: {latest_candle['volume']:.0f}

📊 <b>Market Analysis:</b>
   • Session: {current_session.title()} ({session_mult}x)
   • Sentiment: {market_sentiment.get('analysis', 'Neutral')}
   • Confidence: {market_sentiment.get('confidence', 0):.1%}
   • {bias_emoji} H1 Bias: {h1_bias.upper()}

✅ <b>Data fetching working properly!</b>
🔍 <b>Monitoring trendline + S/R breakouts on M15...</b>
"""
        send_telegram_message(message)

    signals = detect_breakouts(data, symbol, timeframe, config, h1_bias=h1_bias)

    print(f"📊 Scan complete: {len(signals)} signal(s) detected on {timeframe}")
    if not signals:
        print(f"ℹ️ No breakouts on {timeframe} — market consolidating or trendline not broken yet")

    for signal in signals:
        print(f"🎯 Signal: {signal['strategy']} {signal['direction']} "
              f"| Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']}")

        signal = apply_volatility_adaptation(signal, data, config)

        can_send, reason = position_manager.can_send_signal(signal, config)
        if not can_send:
            print(f"🚫 Blocked by position manager: {reason}")
            log_trade(signal, result=f"blocked_{reason}")
            continue

        adjusted_confidence = market_sentiment.get("confidence", 0) * session_mult

        if (signal["rr"] >= config["min_rr"]
                and not signal["fakeout_detected"]
                and adjusted_confidence > CONFIDENCE_THRESHOLD):

            signal_id = position_manager.add_position(
                signal,
                ml_confidence=0.0,
                market_sentiment=market_sentiment.get("analysis", "")
            )

            if signal_id:
                send_telegram_signal(signal)
                log_trade(signal, result="sent",
                          session=current_session,
                          volatility_regime=signal.get("volatility_regime", "unknown"))
                print(f"✅ Signal sent and tracked: {signal_id}")
            else:
                print("❌ Failed to track position in database")

        elif signal["fakeout_detected"]:
            print(f"🚫 Fakeout detected for {signal['strategy']}")
            log_trade(signal, result="fakeout")
        else:
            reasons = []
            if signal["rr"] < config["min_rr"]:
                reasons.append(f"RR {signal['rr']:.2f} < {config['min_rr']}")
            if adjusted_confidence <= CONFIDENCE_THRESHOLD:
                reasons.append(f"confidence {adjusted_confidence:.2f} ≤ {CONFIDENCE_THRESHOLD}")
            filter_reason = ", ".join(reasons) or "Unknown"
            print(f"⚠️ Signal filtered ({filter_reason}): {signal['strategy']}")
            log_trade(signal, result=f"filtered_{filter_reason}")


async def monitor_positions():
    """Monitor open positions and process risk management actions."""
    try:
        open_positions = position_manager.get_open_positions()
        if not open_positions:
            return

        print(f"📊 Monitoring {len(open_positions)} open position(s)")

        data = await fetch_data("XAU/USD", "M15")
        if data is None:
            print("⚠️ Could not fetch data for position monitoring")
            return

        from risk_manager import risk_manager
        if risk_manager is None:
            print("⚠️ Risk manager not initialized")
            return

        actions = risk_manager.process_position_updates(open_positions, data)

        for action in actions:
            action_type = action['type']
            signal_id   = action['signal_id']

            if action_type == 'close_position':
                position_manager.update_position(
                    signal_id,
                    exit_price=action['exit_price'],
                    exit_reason=action['reason']
                )
                log_position_update(action)
                reason_emoji = "🎯" if action['reason'] == 'tp_hit' else "🛑"
                send_telegram_message(f"""
{reason_emoji} <b>POSITION CLOSED</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
💰 <b>Exit Price:</b> ${action['exit_price']:.3f}
📊 <b>P&L:</b> ${action['pnl']:.3f}
🔔 <b>Reason:</b> {action['reason'].replace('_', ' ').title()}
""")
                risk_manager.reset_position_tracking(signal_id)

            elif action_type == 'move_breakeven':
                log_position_update(action)
                send_telegram_message(f"""
🔒 <b>BREAKEVEN MOVE</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
📈 <b>Stop Loss:</b> ${action['old_sl']:.3f} → ${action['new_sl']:.3f}
📊 <b>Current R:R:</b> {action['current_rr']:.2f}

✅ Risk eliminated - Position now risk-free!
""")

            elif action_type == 'update_trailing_stop':
                log_position_update(action)
                send_telegram_message(f"""
📈 <b>TRAILING STOP UPDATE</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
🔄 <b>Stop Loss:</b> ${action['old_sl']:.3f} → ${action['new_sl']:.3f}
📊 <b>Current R:R:</b> {action['current_rr']:.2f}

🎯 Locking in profits!
""")

            elif action_type == 'partial_close':
                log_position_update(action)
                send_telegram_message(f"""
💰 <b>PARTIAL PROFIT TAKEN</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
📊 <b>Closed:</b> {action['percentage']}% @ ${action['exit_price']:.3f}
💵 <b>Profit:</b> ${action['pnl']:.3f}

🎯 Remaining position still active!
""")

    except Exception as e:
        print(f"❌ Error monitoring positions: {e}")
        from logger import log_error
        log_error("position_monitoring", str(e))


async def send_daily_summary():
    """Send daily performance summary to Telegram."""
    try:
        metrics = analytics.get_comprehensive_metrics(days=1)
        if metrics.get('total_trades', 0) == 0:
            return
        report  = analytics.generate_performance_report(days=1)
        send_telegram_message(f"""
📊 <b>DAILY PERFORMANCE SUMMARY</b>

{report}

🕐 <b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        log_performance_summary(metrics, period="daily")
    except Exception as e:
        print(f"❌ Error sending daily summary: {e}")


def run_schedule():
    """Run the scheduler with improved timing and position monitoring."""
    global risk_manager
    risk_manager = initialize_risk_manager(config)
    print("✅ Risk manager initialized")

    log_system_event("startup", "Gold Breakout Bot started", {
        "timeframe": "M15",
        "strategy": "trendline_breakout + sr_breakout"
    })

    # Scan every 5 minutes (M15 candles, catches breaks quickly)
    schedule.every(5).minutes.do(lambda: asyncio.run(scan_markets()))

    # Monitor positions every 2 minutes
    schedule.every(2).minutes.do(lambda: asyncio.run(monitor_positions()))

    # Daily summary at 23:00 UTC
    schedule.every().day.at("23:00").do(lambda: asyncio.run(send_daily_summary()))

    current_price, opening_price, closing_price = asyncio.run(get_latest_prices())
    market_sentiment = ""

    if current_price:
        try:
            data = asyncio.run(fetch_data("XAU/USD", "M15"))
            if data is not None:
                sentiment = get_market_sentiment(data)
                market_sentiment = f"🧠 <b>Market Sentiment:</b> {sentiment.get('analysis', 'Neutral')}"
        except Exception as e:
            print(f"⚠️ Could not get initial sentiment: {e}")

    send_startup_message(current_price, opening_price, closing_price, market_sentiment)

    print("🤖 Gold Breakout Bot is running (M15 Scalp Mode)")
    print("⏰ Scanning every 5 minutes")
    print("📊 Monitoring positions every 2 minutes")
    print("📈 Daily summary at 23:00 UTC")
    print("🎯 Strategies: Trendline Break + S/R Break + Chart Patterns")

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"❌ Error in scheduler: {e}")
            send_telegram_message(
                f"🚨 <b>BOT ERROR</b> 🚨\n❌ Scheduler error: {str(e)}\n🔄 Attempting to recover...")
            time.sleep(60)


if __name__ == "__main__":
    print("🚀 Starting Gold Breakout Bot (M15 Scalp Mode)...")
    keep_alive()
    try:
        asyncio.run(scan_markets())
        run_schedule()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        send_telegram_message("🤖 <b>GOLD BOT STOPPED</b> 🤖\n👋 Bot manually stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        send_telegram_message(
            f"🚨 <b>CRITICAL ERROR</b> 🚨\n❌ {str(e)}\n🔄 Please restart the bot")
