# Unchanged
import pandas as pd
import os

def log_trade(signal, result="pending"):
    """Log trade signal to CSV with fakeout and news skip fields."""
    log_file = "trade_log.csv"
    log_entry = {
        "symbol": signal["symbol"],
        "timeframe": signal["timeframe"],
        "strategy": signal["strategy"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "rr": signal["rr"],
        "result": result,
        "fakeout_detected": signal["fakeout_detected"],
        "timestamp": signal["timestamp"]
    }

    df = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, mode="w", header=True, index=False)
    print(f"Logged trade: {signal['strategy']} on {signal['symbol']} (Fakeout: {signal['fakeout_detected']}, Result: {result})")