"""
Enhanced Logger Module
Comprehensive logging for trades, positions, and performance metrics.
"""

import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, Optional


def log_trade(signal: Dict, result: str = "pending", **kwargs):
    """
    Log trade signal to CSV with comprehensive details.
    
    Args:
        signal: Signal dictionary with trade details
        result: Trade result status
        **kwargs: Additional fields (ml_confidence, market_sentiment, etc.)
    """
    log_file = "trade_log.csv"
    
    # Base log entry
    log_entry = {
        "timestamp": signal.get("timestamp", datetime.now().isoformat()),
        "symbol": signal["symbol"],
        "timeframe": signal["timeframe"],
        "strategy": signal["strategy"],
        "direction": signal["direction"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "rr": signal["rr"],
        "result": result,
        "fakeout_detected": signal.get("fakeout_detected", False),
        "retest_confirmed": signal.get("retest_confirmed", False),
        "retest_quality": signal.get("retest_quality", 0.0),
        "pattern": signal.get("pattern", ""),
        "trend_strength": signal.get("trend_strength", 0.0),
        "confidence": signal.get("confidence", 0.0),
    }
    
    # Add optional fields from kwargs
    log_entry.update({
        "ml_confidence": kwargs.get("ml_confidence", 0.0),
        "market_sentiment": kwargs.get("market_sentiment", ""),
        "session": kwargs.get("session", ""),
        "volatility_regime": kwargs.get("volatility_regime", ""),
        "exit_price": kwargs.get("exit_price", ""),
        "exit_time": kwargs.get("exit_time", ""),
        "pnl": kwargs.get("pnl", ""),
        "exit_reason": kwargs.get("exit_reason", ""),
        "actual_rr": kwargs.get("actual_rr", ""),
    })
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)
    
    print(f"📝 Logged: {signal['strategy']} {signal['direction']} @ {signal['entry']} | Result: {result}")


def log_position_update(action: Dict):
    """
    Log position management actions (breakeven, trailing stop, partial close).
    
    Args:
        action: Action dictionary from risk_manager
    """
    log_file = "position_updates.csv"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "signal_id": action.get("signal_id", ""),
        "action_type": action.get("type", ""),
        "old_sl": action.get("old_sl", ""),
        "new_sl": action.get("new_sl", ""),
        "exit_price": action.get("exit_price", ""),
        "pnl": action.get("pnl", ""),
        "percentage": action.get("percentage", ""),
        "current_rr": action.get("current_rr", ""),
        "reason": action.get("reason", ""),
    }
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)
    
    action_type = action.get("type", "unknown")
    signal_id = action.get("signal_id", "")
    
    if action_type == "move_breakeven":
        print(f"🔒 Breakeven: {signal_id} | SL moved to {action.get('new_sl')}")
    elif action_type == "update_trailing_stop":
        print(f"📈 Trailing: {signal_id} | SL {action.get('old_sl')} → {action.get('new_sl')}")
    elif action_type == "partial_close":
        print(f"💰 Partial: {signal_id} | {action.get('percentage')}% @ {action.get('exit_price')}")
    elif action_type == "close_position":
        print(f"🏁 Closed: {signal_id} | {action.get('reason')} | PnL: {action.get('pnl')}")


def log_performance_summary(metrics: Dict, period: str = "daily"):
    """
    Log performance summary to JSON file.
    
    Args:
        metrics: Performance metrics dictionary
        period: Time period (daily, weekly, monthly)
    """
    log_file = f"performance_{period}.json"
    
    # Add timestamp
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["period"] = period
    
    # Load existing data
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Append new metrics
    data.append(metrics)
    
    # Keep only last 90 days for daily, 52 weeks for weekly
    max_entries = 90 if period == "daily" else 52 if period == "weekly" else 24
    data = data[-max_entries:]
    
    # Save
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"📊 Performance logged: {period} | Win Rate: {metrics.get('win_rate', 0):.1f}% | PnL: {metrics.get('total_pnl', 0):.2f}")


def log_error(error_type: str, error_message: str, context: Optional[Dict] = None):
    """
    Log errors to dedicated error log file.
    
    Args:
        error_type: Type of error (api_error, database_error, etc.)
        error_message: Error message
        context: Additional context dictionary
    """
    log_file = "error_log.csv"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "error_message": error_message,
        "context": json.dumps(context) if context else ""
    }
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)
    
    print(f"❌ Error logged: {error_type} | {error_message}")


def log_system_event(event_type: str, message: str, data: Optional[Dict] = None):
    """
    Log system events (startup, shutdown, config changes, etc.).
    
    Args:
        event_type: Type of event
        message: Event message
        data: Additional event data
    """
    log_file = "system_events.csv"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "message": message,
        "data": json.dumps(data) if data else ""
    }
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)
    
    print(f"ℹ️  System: {event_type} | {message}")


def get_trade_statistics(days: int = 7) -> Dict:
    """
    Calculate trading statistics from trade log.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Dictionary with statistics
    """
    log_file = "trade_log.csv"
    
    if not os.path.exists(log_file):
        return {
            "total_signals": 0,
            "sent_signals": 0,
            "fakeout_rate": 0.0,
            "retest_confirmed_rate": 0.0,
            "avg_rr": 0.0,
            "avg_confidence": 0.0
        }
    
    try:
        df = pd.read_csv(log_file)
        
        # Filter by date
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = datetime.now() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
        
        total = len(df)
        sent = len(df[df["result"] == "sent"])
        fakeouts = len(df[df["fakeout_detected"] == True])
        retest_confirmed = len(df[df["retest_confirmed"] == True])
        
        return {
            "total_signals": total,
            "sent_signals": sent,
            "blocked_signals": total - sent,
            "fakeout_rate": (fakeouts / total * 100) if total > 0 else 0.0,
            "retest_confirmed_rate": (retest_confirmed / total * 100) if total > 0 else 0.0,
            "avg_rr": df["rr"].mean() if "rr" in df.columns else 0.0,
            "avg_confidence": df["confidence"].mean() if "confidence" in df.columns else 0.0,
            "avg_retest_quality": df["retest_quality"].mean() if "retest_quality" in df.columns else 0.0
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            "total_signals": 0,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test enhanced logger
    print("🧪 Testing Enhanced Logger...")
    
    # Test trade log
    test_signal = {
        "symbol": "XAU/USD",
        "timeframe": "M15",
        "strategy": "resistance_breakout",
        "direction": "long",
        "entry": 2050.5,
        "sl": 2045.0,
        "tp": 2065.0,
        "rr": 2.64,
        "timestamp": datetime.now().isoformat(),
        "fakeout_detected": False,
        "retest_confirmed": True,
        "retest_quality": 0.75,
        "pattern": "ascending_triangle",
        "trend_strength": 5.2,
        "confidence": 0.85
    }
    
    log_trade(test_signal, result="sent", 
              ml_confidence=0.72, 
              market_sentiment="Bullish",
              session="london")
    
    # Test position update log
    test_action = {
        "type": "move_breakeven",
        "signal_id": "TEST_001",
        "old_sl": 2045.0,
        "new_sl": 2050.5,
        "current_rr": 1.2
    }
    
    log_position_update(test_action)
    
    # Test performance summary
    test_metrics = {
        "total_trades": 10,
        "winning_trades": 7,
        "losing_trades": 3,
        "win_rate": 70.0,
        "total_pnl": 45.5,
        "avg_rr": 2.3
    }
    
    log_performance_summary(test_metrics, period="daily")
    
    # Test statistics
    stats = get_trade_statistics(days=7)
    print(f"Statistics: {stats}")
    
    print("✅ Enhanced Logger test complete")