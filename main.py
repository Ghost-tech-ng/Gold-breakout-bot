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
                       get_trading_session_multiplier, apply_volatility_adaptation)
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

# Timeframe queue for XAUUSD - optimized order
timeframe_queue = deque(["M15", "M30", "H1"])  # Removed M5 for better accuracy


def send_telegram_message(message):
    """Send message to Telegram chat with detailed error handling."""
    if not message or not message.strip():
        print(
            f"❌ Error: Attempted to send empty message. Message was: '{message}'"
        )
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
    """Get appropriate emoji for trade direction."""
    return "🟢📈" if direction.lower() == "long" else "🔴📉"


def get_strategy_emoji(strategy):
    """Get appropriate emoji for strategy type."""
    emoji_map = {
        "resistance_breakout": "🚀",
        "support_breakout": "⬇️",
        "triangle_breakout": "🔺",
        "channel_breakout": "📊",
        "ascending_triangle_breakout": "📈🔺",
        "descending_triangle_breakout": "📉🔻",
        "symmetrical_triangle_breakout": "⚖️🔺"
    }
    return emoji_map.get(strategy, "📊")


def get_rr_emoji(rr_ratio):
    """Get emoji based on risk-reward ratio."""
    if rr_ratio >= 5.0:
        return "💎🔥"
    elif rr_ratio >= 4.0:
        return "⭐✨"
    elif rr_ratio >= 3.0:
        return "🎯"
    else:
        return "📊"


def send_telegram_signal(signal):
    """Send enhanced trade signal to Telegram chat."""
    if not signal:
        print("❌ Error: Attempted to send signal with empty data. Aborting.")
        return

    direction_emoji = get_direction_emoji(signal['direction'])
    strategy_emoji = get_strategy_emoji(signal['strategy'])
    rr_emoji = get_rr_emoji(signal['rr'])

    # Calculate pip value and percentages
    pip_value = abs(signal['entry'] - signal['sl'])
    profit_potential = abs(signal['tp'] - signal['entry'])

    message = f"""
🏆 <b>GOLD BREAKOUT SIGNAL</b> 🏆
{strategy_emoji} <b>Strategy:</b> {signal['strategy'].replace('_', ' ').title()}
{direction_emoji} <b>Direction:</b> {signal['direction'].upper()}

💰 <b>TRADE DETAILS</b>
🎯 <b>Entry:</b> ${signal['entry']:.3f}
🛡️ <b>Stop Loss:</b> ${signal['sl']:.3f}
🎁 <b>Take Profit:</b> ${signal['tp']:.3f}
{rr_emoji} <b>Risk:Reward:</b> 1:{signal['rr']:.2f}

📊 <b>ANALYSIS</b>
📈 <b>Symbol:</b> {signal['symbol']}
⏰ <b>Timeframe:</b> {signal['timeframe']}
💎 <b>Risk:</b> ${pip_value:.3f}
🎯 <b>Profit Potential:</b> ${profit_potential:.3f}

🕐 <b>Signal Time:</b> {signal['timestamp']}

⚠️ <b>Trade Responsibly!</b> 
🔒 <b>Risk Management is Key!</b>
"""
    send_telegram_message(message)


def send_startup_message(current_price,
                         opening_price,
                         closing_price,
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
• Timeframes: M15, M30, H1
• Breakout Patterns
• Support/Resistance Levels
• Fakeout Detection

{market_sentiment}

🚀 <b>Ready to Hunt Gold Breakouts!</b>
"""
    else:
        message = f"""
🤖 <b>GOLD BOT ACTIVATED</b> 🤖
⚡ <b>System Status:</b> ONLINE ✅

❌ <b>Unable to fetch current prices</b>
🔄 <b>Retrying price fetch...</b>

🎯 <b>MONITORING:</b>
• Timeframes: M15, M30, H1
• Breakout Patterns
• Support/Resistance Levels

🚀 <b>Ready to Hunt Gold Breakouts!</b>
"""
    send_telegram_message(message)


async def fetch_data(symbol, timeframe):
    """Fetch OHLCV data from Twelve Data with improved error handling."""
    interval_map = {
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1day"
    }

    interval = interval_map.get(timeframe, "15min")
    td = TDClient(apikey=TWELVE_DATA_API_KEY)

    for attempt in range(3):
        try:
            print(
                f"🔍 Fetching {symbol} data for {timeframe} (attempt {attempt + 1})"
            )

            # Use more specific parameters for better accuracy
            ts = td.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=150,  # Increased for better analysis
                timezone="UTC",
                order="ASC").as_pandas()

            if ts.empty:
                print(
                    f"⚠️ Warning: No data returned for {symbol} ({timeframe})")
                return None

            # Ensure we have the required columns
            required_columns = ["open", "high", "low", "close"]
            if not all(col in ts.columns for col in required_columns):
                print(f"❌ Missing required columns for {symbol} ({timeframe})")
                return None

            # Clean and prepare data
            # Check if volume column exists, if not fetch it separately
            if "volume" in ts.columns:
                ts = ts[required_columns + ["volume"]].copy()
            else:
                # Fetch volume data separately for forex pairs
                try:
                    print(f"📊 Fetching volume data for {symbol}...")
                    volume_data = td.time_series(
                        symbol=symbol,
                        interval=interval,
                        outputsize=150,
                        timezone="UTC",
                        order="ASC"
                    ).with_volume().as_pandas()
                    
                    if not volume_data.empty and "volume" in volume_data.columns:
                        ts = ts[required_columns].copy()
                        # Align volume data with price data
                        ts["volume"] = volume_data["volume"].reindex(ts.index, fill_value=0)
                        print(f"✅ Volume data fetched successfully")
                    else:
                        # Fallback: Use close price changes as volume proxy
                        ts = ts[required_columns].copy()
                        ts["volume"] = abs(ts["close"].diff()).fillna(0) * 1000
                        print(f"⚠️ Using price change as volume proxy")
                except Exception as vol_error:
                    print(f"⚠️ Could not fetch volume: {vol_error}, using price change proxy")
                    ts = ts[required_columns].copy()
                    ts["volume"] = abs(ts["close"].diff()).fillna(0) * 1000
            
            ts = ts.sort_index()
            ts = ts.dropna()  # Remove any NaN values

            # Convert to float for precision
            for col in required_columns + ["volume"]:
                if col in ts.columns:
                    ts[col] = pd.to_numeric(ts[col], errors='coerce')

            print(
                f"✅ Successfully fetched {len(ts)} candles for {symbol} ({timeframe})"
            )
            return ts.tail(100)  # Return last 100 candles

        except Exception as e:
            print(
                f"❌ Attempt {attempt + 1} failed for {symbol} ({timeframe}): {e}"
            )
            if attempt < 2:
                await asyncio.sleep(5)  # Use asyncio.sleep for async function
            else:
                return None

    return None


async def get_latest_prices():
    """Fetch the latest prices for XAU/USD with improved accuracy."""
    # Try multiple timeframes for better accuracy
    for timeframe in ["M5", "M15"]:
        data = await fetch_data("XAU/USD", timeframe)
        if data is not None and not data.empty:
            latest_candle = data.iloc[-1]
            current_price = float(latest_candle["close"])
            opening_price = float(latest_candle["open"])
            closing_price = float(latest_candle["close"])

            # Validate prices (Gold typically trades between 1000-4000)
            if 1000 <= current_price <= 4000:
                return current_price, opening_price, closing_price

    return None, None, None


async def scan_markets():
    """Enhanced market scanning with GPT brain, session detection, and volatility adaptation."""
    if not timeframe_queue:
        return

    timeframe = timeframe_queue.popleft()
    timeframe_queue.append(timeframe)
    symbol = "XAU/USD"

    print(f"🔍 Scanning {symbol} on {timeframe}")

    # Check if we should skip due to news
    if should_skip_news(config):
        print("📰 Skipping scan due to high-impact news event")
        return

    data = await fetch_data(symbol, timeframe)
    if data is None:
        print(f"❌ Skipping {timeframe} due to data fetch failure.")
        return

    # Get market sentiment from GPT brain
    market_sentiment = get_market_sentiment(data)
    
    # Get current trading session and multiplier
    session_mult, current_session = get_trading_session_multiplier(config)
    print(f"📊 Session: {current_session} (multiplier: {session_mult}x)")

    signals = detect_breakouts(data, symbol, timeframe, config)

    for signal in signals:
        print(
            f"🎯 Detected signal: {signal['strategy']} - {signal['direction']}")
        
        # Apply volatility adaptation to signal
        signal = apply_volatility_adaptation(signal, data, config)

        # Check position manager before processing signal
        can_send, reason = position_manager.can_send_signal(signal, config)
        
        if not can_send:
            print(f"🚫 Signal blocked by position manager: {reason}")
            log_trade(signal, result=f"blocked_{reason}")
            continue
        
        # Apply session multiplier to confidence
        adjusted_confidence = market_sentiment.get("confidence", 0) * session_mult

        # Enhanced filtering with GPT brain insights, session, and volatility
        if (signal["rr"] >= config["min_rr"] 
                and not signal["fakeout_detected"]
                and adjusted_confidence > 0.6):

            # Add position to database
            signal_id = position_manager.add_position(
                signal, 
                ml_confidence=0.0,  # Will be updated when ML is enabled
                market_sentiment=market_sentiment.get("analysis", "")
            )
            
            if signal_id:
                # Send signal to Telegram
                send_telegram_signal(signal)
                log_trade(signal, result="sent", 
                         session=current_session,
                         volatility_regime=signal.get("volatility_regime", "unknown"))
                print(f"✅ Signal sent and tracked: {signal_id}")
            else:
                print(f"❌ Failed to track position")

        elif signal["fakeout_detected"]:
            print(f"🚫 Fakeout detected for {signal['strategy']}")
            log_trade(signal, result="fakeout")
        else:
            print(
                f"⚠️ Signal filtered out: RR={signal['rr']:.2f}, Confidence={adjusted_confidence:.2f} (session: {session_mult}x)"
            )


async def monitor_positions():
    """Monitor open positions and process risk management actions."""
    try:
        # Get open positions
        open_positions = position_manager.get_open_positions()
        
        if not open_positions:
            return
        
        print(f"📊 Monitoring {len(open_positions)} open position(s)")
        
        # Get current data for XAU/USD
        data = await fetch_data("XAU/USD", "M15")
        if data is None:
            print("⚠️ Could not fetch data for position monitoring")
            return
        
        # Process each position with risk manager
        from risk_manager import risk_manager
        if risk_manager is None:
            print("⚠️ Risk manager not initialized")
            return
        
        actions = risk_manager.process_position_updates(open_positions, data)
        
        # Execute actions
        for action in actions:
            action_type = action['type']
            signal_id = action['signal_id']
            
            if action_type == 'close_position':
                # Close position in database
                position_manager.update_position(
                    signal_id,
                    exit_price=action['exit_price'],
                    exit_reason=action['reason']
                )
                
                # Log the action
                log_position_update(action)
                
                # Send Telegram notification
                reason_emoji = "🎯" if action['reason'] == 'tp_hit' else "🛑"
                message = f"""
{reason_emoji} <b>POSITION CLOSED</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
💰 <b>Exit Price:</b> ${action['exit_price']:.3f}
📊 <b>P&L:</b> ${action['pnl']:.3f}
🔔 <b>Reason:</b> {action['reason'].replace('_', ' ').title()}
"""
                send_telegram_message(message)
                
                # Reset tracking
                risk_manager.reset_position_tracking(signal_id)
            
            elif action_type == 'move_breakeven':
                # Log the action
                log_position_update(action)
                
                # Send Telegram notification
                message = f"""
🔒 <b>BREAKEVEN MOVE</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
📈 <b>Stop Loss:</b> ${action['old_sl']:.3f} → ${action['new_sl']:.3f}
📊 <b>Current R:R:</b> {action['current_rr']:.2f}

✅ Risk eliminated - Position now risk-free!
"""
                send_telegram_message(message)
            
            elif action_type == 'update_trailing_stop':
                # Log the action
                log_position_update(action)
                
                # Send Telegram notification
                message = f"""
📈 <b>TRAILING STOP UPDATE</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
🔄 <b>Stop Loss:</b> ${action['old_sl']:.3f} → ${action['new_sl']:.3f}
📊 <b>Current R:R:</b> {action['current_rr']:.2f}

🎯 Locking in profits!
"""
                send_telegram_message(message)
            
            elif action_type == 'partial_close':
                # Log the action
                log_position_update(action)
                
                # Send Telegram notification
                message = f"""
💰 <b>PARTIAL PROFIT TAKEN</b>

📍 <b>Signal ID:</b> {signal_id[:20]}...
📊 <b>Closed:</b> {action['percentage']}% @ ${action['exit_price']:.3f}
💵 <b>Profit:</b> ${action['pnl']:.3f}

🎯 Remaining position still active!
"""
                send_telegram_message(message)
    
    except Exception as e:
        print(f"❌ Error monitoring positions: {e}")
        from logger import log_error
        log_error("position_monitoring", str(e))


async def send_daily_summary():
    """Send daily performance summary to Telegram."""
    try:
        # Get performance metrics
        metrics = analytics.get_comprehensive_metrics(days=1)
        
        if metrics.get('total_trades', 0) == 0:
            return
        
        # Generate report
        report = analytics.generate_performance_report(days=1)
        
        # Send to Telegram
        message = f"""
📊 <b>DAILY PERFORMANCE SUMMARY</b>

{report}

🕐 <b>Report Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        send_telegram_message(message)
        
        # Log performance
        log_performance_summary(metrics, period="daily")
        
    except Exception as e:
        print(f"❌ Error sending daily summary: {e}")


def run_schedule():
    """Run the scheduler with improved timing and position monitoring."""
    # Initialize risk manager
    global risk_manager
    risk_manager = initialize_risk_manager(config)
    print("✅ Risk manager initialized")
    
    # Log system startup
    log_system_event("startup", "Gold Breakout Bot started", {
        "timeframes": list(timeframe_queue)
    })
    
    # Schedule market scanning every 5 minutes
    schedule.every(5).minutes.do(lambda: asyncio.run(scan_markets()))
    
    # Schedule position monitoring every 2 minutes
    schedule.every(2).minutes.do(lambda: asyncio.run(monitor_positions()))
    
    # Schedule daily summary at 23:00
    schedule.every().day.at("23:00").do(lambda: asyncio.run(send_daily_summary()))

    # Get startup prices and market sentiment
    current_price, opening_price, closing_price = asyncio.run(
        get_latest_prices())
    market_sentiment = ""

    if current_price:
        # Get initial market sentiment
        try:
            data = asyncio.run(fetch_data("XAU/USD", "M15"))
            if data is not None:
                sentiment = get_market_sentiment(data)
                market_sentiment = f"🧠 <b>Market Sentiment:</b> {sentiment.get('analysis', 'Neutral')}"
        except Exception as e:
            print(f"⚠️ Could not get initial sentiment: {e}")

    send_startup_message(current_price, opening_price, closing_price,
                         market_sentiment)

    print("🤖 Gold Breakout Bot is now running...")
    print("⏰ Scanning every 5 minutes")
    print("📊 Monitoring positions every 2 minutes")
    print("📈 Daily summary at 23:00")
    print("🎯 Monitoring M15, M30, H1 timeframes")

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            print(f"❌ Error in scheduler: {e}")
            error_message = f"🚨 <b>BOT ERROR</b> 🚨\n❌ Scheduler error: {str(e)}\n🔄 Attempting to recover..."
            send_telegram_message(error_message)
            time.sleep(60)


if __name__ == "__main__":
    print("🚀 Starting Gold Breakout Bot...")
    keep_alive()  # Start the keep-alive web server
    try:
        asyncio.run(scan_markets())
        run_schedule()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        send_telegram_message(
            "🤖 <b>GOLD BOT STOPPED</b> 🤖\n👋 Bot manually stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        send_telegram_message(
            f"🚨 <b>CRITICAL ERROR</b> 🚨\n❌ {str(e)}\n🔄 Please restart the bot")
