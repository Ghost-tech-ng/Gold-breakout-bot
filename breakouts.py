import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# Indicator calculations
# ────────────────────────────────────────────────────────────────────────────────
def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) using an EMA for smoother results."""
    if len(data) < period:
        return pd.Series([0.0] * len(data), index=data.index)

    high_low  = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close  = (data["low"]  - data["close"].shift()).abs()

    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def calculate_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper / middle / lower Bollinger Bands."""
    sma = data["close"].rolling(window=period).mean()
    std = data["close"].rolling(window=period).std()

    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


# ────────────────────────────────────────────────────────────────────────────────
# Support & Resistance
# ────────────────────────────────────────────────────────────────────────────────
def detect_support_resistance_enhanced(
    data: pd.DataFrame,
    window: int = 20,
    min_touches: int = 2
) -> Dict[str, List[float]]:
    """Return up to three key resistance & support levels."""
    if len(data) < window:
        return {"resistance": [], "support": []}

    highs = data["high"].rolling(window=5, center=True).max()
    lows  = data["low"].rolling(window=5, center=True).min()

    pivot_highs, pivot_lows = [], []

    for i in range(2, len(data) - 2):
        # pivot high
        if (
            data["high"].iloc[i] == highs.iloc[i]
            and data["high"].iloc[i] > data["high"].iloc[i - 1]
            and data["high"].iloc[i] > data["high"].iloc[i + 1]
        ):
            pivot_highs.append((i, data["high"].iloc[i]))

        # pivot low
        if (
            data["low"].iloc[i] == lows.iloc[i]
            and data["low"].iloc[i] < data["low"].iloc[i - 1]
            and data["low"].iloc[i] < data["low"].iloc[i + 1]
        ):
            pivot_lows.append((i, data["low"].iloc[i]))

    atr_series = calculate_atr(data)
    res_levels, sup_levels = [], []

    # resistance levels
    for _, level in pivot_highs:
        touches = sum(
            abs(data["high"].iloc[j] - level) <= atr_series.iloc[j] * 0.5
            for j in range(len(data))
        )
        if touches >= min_touches:
            res_levels.append(level)

    # support levels
    for _, level in pivot_lows:
        touches = sum(
            abs(data["low"].iloc[j] - level) <= atr_series.iloc[j] * 0.5
            for j in range(len(data))
        )
        if touches >= min_touches:
            sup_levels.append(level)

    res_levels = sorted({round(r, 3) for r in res_levels}, reverse=True)[:3]
    sup_levels = sorted({round(s, 3) for s in sup_levels})[:3]

    return {"resistance": res_levels, "support": sup_levels}


# ────────────────────────────────────────────────────────────────────────────────
# Trendline detection
# ────────────────────────────────────────────────────────────────────────────────
def detect_trendline_enhanced(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """Return slope, strength, direction, and correlation of trend."""
    if len(data) < window:
        return {"slope": 0.0, "strength": 0.0, "direction": "neutral", "correlation": 0.0}

    close = data["close"].tail(window)
    high  = data["high"].tail(window)
    low   = data["low"].tail(window)
    x = np.arange(len(close))

    close_slope, _, close_r, _, _ = linregress(x, close.values)
    high_slope,  _, high_r,  _, _ = linregress(x, high.values)
    low_slope,   _, low_r,   _, _ = linregress(x, low.values)

    slopes     = np.array([close_slope, high_slope, low_slope])
    weights    = np.abs([close_r, high_r, low_r])
    w_slope    = np.average(slopes, weights=weights)
    avg_corr   = weights.mean()

    if w_slope > 0.1:
        direction, raw_strength = "bullish", w_slope * 100
    elif w_slope < -0.1:
        direction, raw_strength = "bearish", abs(w_slope) * 100
    else:
        direction, raw_strength = "neutral", 0.0

    return {
        "slope": w_slope,
        "strength": min(raw_strength, 10) * avg_corr,
        "direction": direction,
        "correlation": avg_corr,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Chart-pattern detection
# ────────────────────────────────────────────────────────────────────────────────
def detect_chart_patterns_enhanced(data: pd.DataFrame, window: int = 20) -> Dict[str, object]:
    """Identify triangles, channels, and wedges with confidence/target."""
    if len(data) < window:
        return {"pattern": None, "confidence": 0.0, "breakout_target": None}

    recent = data.tail(window)
    highs, lows = recent["high"], recent["low"]
    x = np.arange(window)

    high_slope, _, high_r, _, _ = linregress(x, highs.values)
    low_slope,  _, low_r,  _, _ = linregress(x, lows.values)

    patterns: List[Tuple[str, float]] = []

    # triangle patterns
    if abs(high_slope) > 0.01 and abs(low_slope) > 0.01:
        if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
            patterns.append(("ascending_triangle", abs(high_r) + abs(low_r)))
        elif high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
            patterns.append(("descending_triangle", abs(high_r) + abs(low_r)))
        elif high_slope < 0 < low_slope:
            patterns.append(("symmetrical_triangle", abs(high_r) + abs(low_r)))

    # channel patterns
    if abs(high_slope - low_slope) < 0.005:
        if high_slope > 0:
            patterns.append(("ascending_channel", abs(high_r) + abs(low_r)))
        elif high_slope < 0:
            patterns.append(("descending_channel", abs(high_r) + abs(low_r)))
        else:
            patterns.append(("horizontal_channel", abs(high_r) + abs(low_r)))

    # wedge patterns
    if high_slope < 0 < low_slope and abs(high_slope) > abs(low_slope):
        patterns.append(("rising_wedge", abs(high_r) + abs(low_r)))
    elif high_slope > 0 > low_slope and high_slope < abs(low_slope):
        patterns.append(("falling_wedge", abs(high_r) + abs(low_r)))

    if not patterns:
        return {"pattern": None, "confidence": 0.0, "breakout_target": None}

    pattern, conf = max(patterns, key=lambda p: p[1])
    pr_range = highs.max() - lows.min()
    price_now = data["close"].iloc[-1]

    if any(key in pattern for key in ("ascending", "rising")):
        target = price_now + pr_range
    elif any(key in pattern for key in ("descending", "falling")):
        target = price_now - pr_range
    else:
        target = None

    return {"pattern": pattern, "confidence": min(conf, 1.0), "breakout_target": target}


# ────────────────────────────────────────────────────────────────────────────────
# Fake-out detection
# ────────────────────────────────────────────────────────────────────────────────
def enhanced_fakeout_detection(data: pd.DataFrame, signal: Dict, cfg: Dict) -> bool:
    """Return True if ≥2 fake-out criteria are met."""
    if len(data) < 10:
        return False

    recent5 = data.tail(5)
    brk  = data.iloc[-1]      # breakout candle
    prev = data.iloc[-2]

    atr_series = calculate_atr(data)
    atr_curr = atr_series.iloc[-1]

    fake_signals = 0

    # 1. Volume
    if "volume" in data.columns and data["volume"].sum() > 0:
        avg_vol = data["volume"].rolling(20).mean().iloc[-1]
        if brk["volume"] < avg_vol * cfg.get("volume_threshold", 1.5):
            fake_signals += 1

    # 2. Candle body
    body = abs(brk["close"] - brk["open"])
    avg_body = abs(data["close"] - data["open"]).rolling(20).mean().iloc[-1]
    if body < avg_body * 0.5:
        fake_signals += 1

    # 3. Wick
    if signal["direction"] == "long":
        upper_wick = brk["high"] - max(brk["open"], brk["close"])
        if upper_wick > body * 2:
            fake_signals += 1
    else:
        lower_wick = min(brk["open"], brk["close"]) - brk["low"]
        if lower_wick > body * 2:
            fake_signals += 1

    # 4. Immediate reversal
    lvl = signal["entry"]
    for _, candle in recent5.iterrows():
        if signal["direction"] == "long" and candle["low"] < lvl - atr_curr * 0.3:
            fake_signals += 1
            break
        if signal["direction"] == "short" and candle["high"] > lvl + atr_curr * 0.3:
            fake_signals += 1
            break

    # 5. Momentum divergence
    if len(data) >= 14:
        recent_hi = data["high"].tail(14)
        recent_lo = data["low"].tail(14)
        if (
            signal["direction"] == "long"
            and recent_hi.iloc[-1] > recent_hi.iloc[-5]
            and brk["close"] < prev["close"]
        ):
            fake_signals += 1
        elif (
            signal["direction"] == "short"
            and recent_lo.iloc[-1] < recent_lo.iloc[-5]
            and brk["close"] > prev["close"]
        ):
            fake_signals += 1

    return fake_signals >= 2


# ────────────────────────────────────────────────────────────────────────────────
# Dynamic risk targets
# ────────────────────────────────────────────────────────────────────────────────
def calculate_dynamic_targets(data: pd.DataFrame, signal: Dict, cfg: Dict) -> Dict[str, float]:
    """Return SL/TP and their distances."""
    atr_now = calculate_atr(data).iloc[-1]
    levels = detect_support_resistance_enhanced(data)
    price  = signal["entry"]

    if signal["direction"] == "long":
        supports = [s for s in levels["support"] if s < price]
        if supports:
            nearest_sup = max(supports)
            dyn_sl = nearest_sup - atr_now * 0.5
        else:
            dyn_sl = price - atr_now * cfg.get("sl_buffer", 1.5)

        resistances = [r for r in levels["resistance"] if r > price]
        if resistances:
            nearest_res = min(resistances)
            dyn_tp = nearest_res - atr_now * 0.3
        else:
            dyn_tp = price + (price - dyn_sl) * cfg.get("min_rr", 2.0)

    else:  # short
        resistances = [r for r in levels["resistance"] if r > price]
        if resistances:
            nearest_res = min(resistances)
            dyn_sl = nearest_res + atr_now * 0.5
        else:
            dyn_sl = price + atr_now * cfg.get("sl_buffer", 1.5)

        supports = [s for s in levels["support"] if s < price]
        if supports:
            nearest_sup = max(supports)
            dyn_tp = nearest_sup + atr_now * 0.3
        else:
            dyn_tp = price - (dyn_sl - price) * cfg.get("min_rr", 2.0)

    return {
        "sl": round(dyn_sl, 3),
        "tp": round(dyn_tp, 3),
        "sl_distance": abs(price - dyn_sl),
        "tp_distance": abs(dyn_tp - price),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Breakout scanner
# ────────────────────────────────────────────────────────────────────────────────
def detect_breakouts(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cfg: Dict
) -> List[Dict]:
    """Return breakout trade signals list."""
    if len(data) < 50:
        return []

    atr_series = calculate_atr(data)
    curr_atr  = atr_series.iloc[-1]

    curr, prev = data.iloc[-1], data.iloc[-2]
    levels = detect_support_resistance_enhanced(data, window=cfg.get("support_resistance_window", 20))
    pattern_info = detect_chart_patterns_enhanced(data, window=cfg.get("pattern_window", 20))
    trend_info   = detect_trendline_enhanced(data, window=cfg.get("pattern_window", 20))
    bb_u, bb_m, bb_l = calculate_bollinger_bands(data)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    signals: List[Dict] = []

    # 1️⃣ Support/Resistance breakouts
    for res in levels["resistance"]:
        if curr["close"] > res >= prev["close"] and curr["close"] > curr["open"]:
            entry = curr["close"]
            tgt   = calculate_dynamic_targets(data, {"entry": entry, "direction": "long"}, cfg)
            rr    = tgt["tp_distance"] / tgt["sl_distance"] if tgt["sl_distance"] else 0
            if rr >= cfg["min_rr"]:
                sig = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": "resistance_breakout",
                    "direction": "long",
                    "entry": round(entry, 3),
                    "sl": tgt["sl"],
                    "tp": tgt["tp"],
                    "rr": round(rr, 2),
                    "timestamp": timestamp,
                    "fakeout_detected": False,
                    "pattern": pattern_info["pattern"],
                    "trend_strength": trend_info["strength"],
                    "confidence": pattern_info["confidence"],
                }
                sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                signals.append(sig)

    for sup in levels["support"]:
        if curr["close"] < sup <= prev["close"] and curr["close"] < curr["open"]:
            entry = curr["close"]
            tgt   = calculate_dynamic_targets(data, {"entry": entry, "direction": "short"}, cfg)
            rr    = tgt["tp_distance"] / tgt["sl_distance"] if tgt["sl_distance"] else 0
            if rr >= cfg["min_rr"]:
                sig = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": "support_breakout",
                    "direction": "short",
                    "entry": round(entry, 3),
                    "sl": tgt["sl"],
                    "tp": tgt["tp"],
                    "rr": round(rr, 2),
                    "timestamp": timestamp,
                    "fakeout_detected": False,
                    "pattern": pattern_info["pattern"],
                    "trend_strength": trend_info["strength"],
                    "confidence": pattern_info["confidence"],
                }
                sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                signals.append(sig)

    # 2️⃣ Pattern breakouts
    if pattern_info["pattern"] and pattern_info["confidence"] > 0.6:
        pat = pattern_info["pattern"]
        if "ascending" in pat and levels["resistance"]:
            res = min(levels["resistance"])
            if curr["close"] > res:
                entry = curr["close"]
                tgt   = calculate_dynamic_targets(data, {"entry": entry, "direction": "long"}, cfg)
                rr    = tgt["tp_distance"] / tgt["sl_distance"] if tgt["sl_distance"] else 0
                if rr >= cfg["min_rr"]:
                    sig = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": f"{pat}_breakout",
                        "direction": "long",
                        "entry": round(entry, 3),
                        "sl": tgt["sl"],
                        "tp": tgt["tp"],
                        "rr": round(rr, 2),
                        "timestamp": timestamp,
                        "fakeout_detected": False,
                        "pattern": pat,
                        "trend_strength": trend_info["strength"],
                        "confidence": pattern_info["confidence"],
                    }
                    sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                    signals.append(sig)

        elif "descending" in pat and levels["support"]:
            sup = max(levels["support"])
            if curr["close"] < sup:
                entry = curr["close"]
                tgt   = calculate_dynamic_targets(data, {"entry": entry, "direction": "short"}, cfg)
                rr    = tgt["tp_distance"] / tgt["sl_distance"] if tgt["sl_distance"] else 0
                if rr >= cfg["min_rr"]:
                    sig = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "strategy": f"{pat}_breakout",
                        "direction": "short",
                        "entry": round(entry, 3),
                        "sl": tgt["sl"],
                        "tp": tgt["tp"],
                        "rr": round(rr, 2),
                        "timestamp": timestamp,
                        "fakeout_detected": False,
                        "pattern": pat,
                        "trend_strength": trend_info["strength"],
                        "confidence": pattern_info["confidence"],
                    }
                    sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                    signals.append(sig)

    # 3️⃣ Trend & Bollinger filtering
    filtered = []
    for sig in signals:
        align = (
            (trend_info["direction"] == "bullish" and sig["direction"] == "long")
            or (trend_info["direction"] == "bearish" and sig["direction"] == "short")
            or trend_info["direction"] == "neutral"
            or trend_info["strength"] <= 3
        )

        bb_pos = (
            "above_upper" if curr["close"] > bb_u.iloc[-1]
            else "below_lower" if curr["close"] < bb_l.iloc[-1]
            else "inside"
        )

        if (bb_pos == "above_upper" and sig["direction"] == "long") or (
            bb_pos == "below_lower" and sig["direction"] == "short"
        ):
            sig["confidence"] *= 0.8

        if align and sig["confidence"] > 0.4:
            filtered.append(sig)

    return filtered


# ────────────────────────────────────────────────────────────────────────────────
# Retest confirmation
# ────────────────────────────────────────────────────────────────────────────────
def confirm_retest(data: pd.DataFrame, signal: Dict) -> bool:
    """Return True if breakout level has been retested once."""
    if len(data) < 10:
        return False

    candles = data.tail(10)
    level   = signal["entry"]
    atr_ten = calculate_atr(data).iloc[-1]
    tol     = atr_ten * 0.3

    for _, c in candles.iterrows():
        if signal["direction"] == "long":
            if c["low"] <= level + tol and c["close"] > level:
                return True
        else:
            if c["high"] >= level - tol and c["close"] < level:
                return True
    return False


# ────────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing Enhanced Breakout Detection...")

    # synthetic 15-minute XAU/USD data
    rng = pd.date_range("2024-01-01", periods=100, freq="15min")
    np.random.seed(42)
    base = 2000
    trend = np.linspace(0, 50, 100)     # gentle uptrend
    noise = np.random.normal(0, 2, 100)

    price = base + trend + noise

    df = pd.DataFrame(
        {
            "open":  price + np.random.normal(0, 0.5, 100),
            "high":  price + np.random.uniform(0.5, 2.0, 100),
            "low":   price - np.random.uniform(0.5, 2.0, 100),
            "close": price,
            "volume": np.random.randint(1000, 5000, 100),
        },
        index=rng,
    )

    cfg = {
        "min_rr": 2.0,
        "sl_buffer": 1.5,
        "support_resistance_window": 20,
        "pattern_window": 20,
        "volume_threshold": 1.5,
    }

    sigs = detect_breakouts(df, "XAU/USD", "M15", cfg)

    print("✅ Test completed!")
    print(f"📊 Found {len(sigs)} potential breakout signal(s)")
    for s in sigs:
        print(f"🎯 {s['strategy']} | {s['direction']} | RR {s['rr']:.2f}")