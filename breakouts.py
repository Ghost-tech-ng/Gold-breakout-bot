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
        if brk["volume"] < avg_vol * cfg.get("volume_threshold", 1.2):
            fake_signals += 1

    # 2. Candle body
    body = abs(brk["close"] - brk["open"])
    avg_body = abs(data["close"] - data["open"]).rolling(20).mean().iloc[-1]
    if body < avg_body * 0.4:
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
# Scalp targets (ATR-based, tight — primary for M15)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_scalp_targets(data: pd.DataFrame, signal: Dict, cfg: Dict) -> Dict[str, float]:
    """
    Tight ATR-based SL/TP for M15 scalping.
    SL = entry ± ATR * sl_atr_mult (default 0.8)
    TP = entry ± SL_distance * min_rr
    Also considers nearest S/R level to place SL just beyond it.
    """
    atr = calculate_atr(data).iloc[-1]
    entry = signal["entry"]
    direction = signal["direction"]
    sl_mult = cfg.get("sl_atr_mult", 0.8)
    min_rr  = cfg.get("min_rr", 1.5)

    levels = detect_support_resistance_enhanced(data)

    if direction == "long":
        # Try to anchor SL at nearest support
        supports = [s for s in levels["support"] if s < entry]
        if supports and (entry - max(supports)) <= atr * sl_mult * 1.5:
            sl = max(supports) - atr * 0.15
        else:
            sl = entry - atr * sl_mult

        sl_dist = abs(entry - sl)
        tp = entry + sl_dist * min_rr

    else:  # short
        resistances = [r for r in levels["resistance"] if r > entry]
        if resistances and (min(resistances) - entry) <= atr * sl_mult * 1.5:
            sl = min(resistances) + atr * 0.15
        else:
            sl = entry + atr * sl_mult

        sl_dist = abs(sl - entry)
        tp = entry - sl_dist * min_rr

    tp_dist = abs(tp - entry)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    return {
        "sl": round(sl, 3),
        "tp": round(tp, 3),
        "sl_distance": sl_dist,
        "tp_distance": tp_dist,
        "rr": round(rr, 2),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Dynamic risk targets (S/R anchored — used for pattern breakouts)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_dynamic_targets(data: pd.DataFrame, signal: Dict, cfg: Dict) -> Dict[str, float]:
    """Return SL/TP anchored to nearest S/R levels, capped at 2× ATR for scalping."""
    atr_now = calculate_atr(data).iloc[-1]
    levels = detect_support_resistance_enhanced(data)
    price  = signal["entry"]
    max_sl = atr_now * cfg.get("sl_atr_mult", 0.8) * 2  # never wider than 2× scalp SL

    if signal["direction"] == "long":
        supports = [s for s in levels["support"] if s < price]
        if supports:
            nearest_sup = max(supports)
            dyn_sl = nearest_sup - atr_now * 0.3
        else:
            dyn_sl = price - atr_now * cfg.get("sl_buffer", 0.8)

        # Cap SL distance for scalping
        if (price - dyn_sl) > max_sl:
            dyn_sl = price - max_sl

        resistances = [r for r in levels["resistance"] if r > price]
        if resistances:
            nearest_res = min(resistances)
            dyn_tp = nearest_res - atr_now * 0.2
        else:
            dyn_tp = price + (price - dyn_sl) * cfg.get("min_rr", 1.5)

    else:  # short
        resistances = [r for r in levels["resistance"] if r > price]
        if resistances:
            nearest_res = min(resistances)
            dyn_sl = nearest_res + atr_now * 0.3
        else:
            dyn_sl = price + atr_now * cfg.get("sl_buffer", 0.8)

        if (dyn_sl - price) > max_sl:
            dyn_sl = price + max_sl

        supports = [s for s in levels["support"] if s < price]
        if supports:
            nearest_sup = max(supports)
            dyn_tp = nearest_sup + atr_now * 0.2
        else:
            dyn_tp = price - (dyn_sl - price) * cfg.get("min_rr", 1.5)

    return {
        "sl": round(dyn_sl, 3),
        "tp": round(dyn_tp, 3),
        "sl_distance": abs(price - dyn_sl),
        "tp_distance": abs(dyn_tp - price),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Trendline breakout detection (NEW — primary strategy)
# ────────────────────────────────────────────────────────────────────────────────
def _find_swing_points(data: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Return (swing_highs, swing_lows) as lists of (bar_index, price).
    Uses ±2 bar confirmation window.
    """
    swing_highs: List[Tuple[int, float]] = []
    swing_lows:  List[Tuple[int, float]] = []

    for i in range(2, len(data) - 2):
        h = data["high"].iloc[i]
        if (h >= data["high"].iloc[i - 1] and h >= data["high"].iloc[i + 1]
                and h >= data["high"].iloc[i - 2] and h >= data["high"].iloc[i + 2]):
            swing_highs.append((i, h))

        l = data["low"].iloc[i]
        if (l <= data["low"].iloc[i - 1] and l <= data["low"].iloc[i + 1]
                and l <= data["low"].iloc[i - 2] and l <= data["low"].iloc[i + 2]):
            swing_lows.append((i, l))

    return swing_highs, swing_lows


def detect_trendline_breakouts(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cfg: Dict
) -> List[Dict]:
    """
    Detect price breaking through a trendline:
      • Downtrend trendline break (price was below descending highs, now closes above) → LONG
      • Uptrend trendline break   (price was above ascending lows,  now closes below) → SHORT

    Requirements for a valid trendline:
      - At least 2 swing points that form the line
      - R² ≥ trendline_r2_min (default 0.70) — points must align well
      - At least trendline_min_touches (default 2) — price respected the line
      - Breaking candle must have a real body (≥ 0.25 × ATR)
      - SL never wider than 1.5 × ATR (keeps it a scalp)
    """
    signals: List[Dict] = []

    lookback = cfg.get("trendline_lookback", 60)
    min_r2   = cfg.get("trendline_r2_min", 0.70)
    min_tch  = cfg.get("trendline_min_touches", 2)

    if len(data) < 30:
        return signals

    atr_series = calculate_atr(data)
    atr = atr_series.iloc[-1]
    if atr == 0:
        return signals

    curr = data.iloc[-1]
    prev = data.iloc[-2]
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Work on a recent slice (reset index to 0-based for regression)
    window_size = min(len(data), lookback)
    recent = data.tail(window_size).reset_index(drop=True)
    n = len(recent)

    swing_highs, swing_lows = _find_swing_points(recent)

    tolerance = atr * 0.4   # within this distance = "touching" the trendline
    max_sl_dist = atr * 1.5  # scalp guard: SL must be tighter than 1.5 × ATR

    # ── DOWNTREND LINE BREAK → LONG ─────────────────────────────────────────────
    if len(swing_highs) >= 2:
        pts = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        x_arr = np.array([p[0] for p in pts], dtype=float)
        y_arr = np.array([p[1] for p in pts], dtype=float)

        slope, intercept, r_value, _, _ = linregress(x_arr, y_arr)

        # Must be a genuine downtrend with good fit
        if slope < -atr * 0.005 and abs(r_value) >= min_r2:
            tl_curr = slope * (n - 1) + intercept
            tl_prev = slope * (n - 2) + intercept

            # Count how many bars respected the trendline from below
            touches = sum(
                1 for i in range(n)
                if abs(recent["high"].iloc[i] - (slope * i + intercept)) <= tolerance
                and recent["close"].iloc[i] < (slope * i + intercept)
            )

            body = abs(curr["close"] - curr["open"])

            if (
                touches >= min_tch
                and prev["close"] <= tl_prev + tolerance      # was at/below line
                and curr["close"] > tl_curr                    # now broke above
                and curr["close"] > curr["open"]               # bullish candle
                and body >= atr * 0.25                         # real body
            ):
                entry = curr["close"]
                # SL just below the broken trendline or candle low
                sl_from_line = tl_curr - atr * 0.4
                sl_from_low  = curr["low"] - atr * 0.15
                sl = min(sl_from_line, sl_from_low)
                sl_dist = abs(entry - sl)

                if sl_dist <= max_sl_dist and sl_dist > 0:
                    tp_dist = sl_dist * cfg.get("min_rr", 1.5)
                    tp = entry + tp_dist
                    rr = tp_dist / sl_dist

                    if rr >= cfg.get("min_rr", 1.5):
                        sig = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": "downtrend_line_break",
                            "direction": "long",
                            "entry": round(entry, 3),
                            "sl": round(sl, 3),
                            "tp": round(tp, 3),
                            "rr": round(rr, 2),
                            "timestamp": timestamp,
                            "fakeout_detected": False,
                            "retest_confirmed": True,
                            "retest_quality": min(touches / max(min_tch, 4), 1.0),
                            "pattern": "downtrend_line",
                            "trend_strength": abs(slope) * 100,
                            "confidence": round(abs(r_value), 3),
                            "trendline_touches": touches,
                        }
                        sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                        signals.append(sig)
                        print(f"📉➡️📈 Downtrend line break detected: {touches} touches, "
                              f"R={r_value:.2f}, SL dist=${sl_dist:.2f}")

    # ── UPTREND LINE BREAK → SHORT ───────────────────────────────────────────────
    if len(swing_lows) >= 2:
        pts = swing_lows[-4:] if len(swing_lows) >= 4 else swing_lows
        x_arr = np.array([p[0] for p in pts], dtype=float)
        y_arr = np.array([p[1] for p in pts], dtype=float)

        slope, intercept, r_value, _, _ = linregress(x_arr, y_arr)

        # Must be a genuine uptrend with good fit
        if slope > atr * 0.005 and abs(r_value) >= min_r2:
            tl_curr = slope * (n - 1) + intercept
            tl_prev = slope * (n - 2) + intercept

            # Count how many bars respected the trendline from above
            touches = sum(
                1 for i in range(n)
                if abs(recent["low"].iloc[i] - (slope * i + intercept)) <= tolerance
                and recent["close"].iloc[i] > (slope * i + intercept)
            )

            body = abs(curr["close"] - curr["open"])

            if (
                touches >= min_tch
                and prev["close"] >= tl_prev - tolerance      # was at/above line
                and curr["close"] < tl_curr                    # now broke below
                and curr["close"] < curr["open"]               # bearish candle
                and body >= atr * 0.25                         # real body
            ):
                entry = curr["close"]
                # SL just above the broken trendline or candle high
                sl_from_line = tl_curr + atr * 0.4
                sl_from_high = curr["high"] + atr * 0.15
                sl = max(sl_from_line, sl_from_high)
                sl_dist = abs(sl - entry)

                if sl_dist <= max_sl_dist and sl_dist > 0:
                    tp_dist = sl_dist * cfg.get("min_rr", 1.5)
                    tp = entry - tp_dist
                    rr = tp_dist / sl_dist

                    if rr >= cfg.get("min_rr", 1.5):
                        sig = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": "uptrend_line_break",
                            "direction": "short",
                            "entry": round(entry, 3),
                            "sl": round(sl, 3),
                            "tp": round(tp, 3),
                            "rr": round(rr, 2),
                            "timestamp": timestamp,
                            "fakeout_detected": False,
                            "retest_confirmed": True,
                            "retest_quality": min(touches / max(min_tch, 4), 1.0),
                            "pattern": "uptrend_line",
                            "trend_strength": slope * 100,
                            "confidence": round(abs(r_value), 3),
                            "trendline_touches": touches,
                        }
                        sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                        signals.append(sig)
                        print(f"📈➡️📉 Uptrend line break detected: {touches} touches, "
                              f"R={r_value:.2f}, SL dist=${sl_dist:.2f}")

    return signals


# ────────────────────────────────────────────────────────────────────────────────
# Retest confirmation
# ────────────────────────────────────────────────────────────────────────────────
def enhanced_retest_confirmation(data: pd.DataFrame, signal: Dict, cfg: Dict) -> Dict:
    """
    Enhanced retest confirmation with detailed analysis.
    Returns dict with retest status, quality score, and details.
    """
    if not cfg.get("retest_enabled", False):
        return {"confirmed": True, "quality": 1.0, "reason": "retest_disabled"}

    if len(data) < cfg.get("retest_patience_candles", 10):
        return {"confirmed": False, "quality": 0.0, "reason": "insufficient_data"}

    patience_candles = cfg.get("retest_patience_candles", 10)
    max_distance_atr = cfg.get("retest_max_distance_atr", 0.5)
    min_quality = cfg.get("retest_min_quality_score", 0.6)

    candles = data.tail(patience_candles)
    level = signal["entry"]
    atr = calculate_atr(data).iloc[-1]
    tolerance = atr * max_distance_atr

    best_retest_quality = 0.0
    retest_candle_idx = -1

    for idx, (_, c) in enumerate(candles.iterrows()):
        quality = 0.0
        is_retest = False

        if signal["direction"] == "long":
            if c["low"] <= level + tolerance and c["close"] > level:
                is_retest = True
                body = abs(c["close"] - c["open"])
                lower_wick = min(c["open"], c["close"]) - c["low"]
                total_range = c["high"] - c["low"]
                if total_range > 0:
                    quality += min((lower_wick / total_range) * 0.6, 0.3)
                if c["close"] > c["open"]:
                    quality += 0.2
                if "volume" in candles.columns and candles["volume"].sum() > 0:
                    avg_vol = candles["volume"].mean()
                    vol_ratio = c["volume"] / avg_vol if avg_vol > 0 else 0
                    quality += min(vol_ratio * 0.25, 0.25)
                quality += (idx / patience_candles) * 0.15
                distance = abs(c["low"] - level)
                quality += max(0, 1 - (distance / tolerance)) * 0.1
        else:
            if c["high"] >= level - tolerance and c["close"] < level:
                is_retest = True
                body = abs(c["close"] - c["open"])
                upper_wick = c["high"] - max(c["open"], c["close"])
                total_range = c["high"] - c["low"]
                if total_range > 0:
                    quality += min((upper_wick / total_range) * 0.6, 0.3)
                if c["close"] < c["open"]:
                    quality += 0.2
                if "volume" in candles.columns and candles["volume"].sum() > 0:
                    avg_vol = candles["volume"].mean()
                    vol_ratio = c["volume"] / avg_vol if avg_vol > 0 else 0
                    quality += min(vol_ratio * 0.25, 0.25)
                quality += (idx / patience_candles) * 0.15
                distance = abs(c["high"] - level)
                quality += max(0, 1 - (distance / tolerance)) * 0.1

        if is_retest and quality > best_retest_quality:
            best_retest_quality = quality
            retest_candle_idx = idx

    if best_retest_quality >= min_quality:
        return {
            "confirmed": True,
            "quality": best_retest_quality,
            "candle_index": retest_candle_idx,
            "reason": f"quality_retest_{best_retest_quality:.2f}"
        }
    elif best_retest_quality > 0:
        return {
            "confirmed": False,
            "quality": best_retest_quality,
            "candle_index": retest_candle_idx,
            "reason": f"low_quality_{best_retest_quality:.2f}"
        }
    else:
        return {
            "confirmed": False,
            "quality": 0.0,
            "candle_index": -1,
            "reason": "no_retest_found"
        }


# ────────────────────────────────────────────────────────────────────────────────
# Main breakout scanner
# ────────────────────────────────────────────────────────────────────────────────
def detect_breakouts(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cfg: Dict
) -> List[Dict]:
    """
    Return all breakout trade signals.

    Sources (in priority order):
      1. Trendline breakouts  (uptrend/downtrend line breaks)
      2. S/R breakouts        (resistance/support level breaks)
      3. Chart pattern breakouts (triangles, etc.)

    All signals use tight ATR-based scalp targets suitable for M15.
    """
    if len(data) < 50:
        return []

    atr_series = calculate_atr(data)
    curr_atr  = atr_series.iloc[-1]

    curr, prev = data.iloc[-1], data.iloc[-2]
    levels       = detect_support_resistance_enhanced(data, window=cfg.get("support_resistance_window", 20))
    pattern_info = detect_chart_patterns_enhanced(data, window=cfg.get("pattern_window", 20))
    trend_info   = detect_trendline_enhanced(data, window=cfg.get("pattern_window", 20))
    bb_u, bb_m, bb_l = calculate_bollinger_bands(data)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    signals: List[Dict] = []

    # ── 1. TRENDLINE BREAKOUTS (new primary strategy) ───────────────────────────
    trendline_signals = detect_trendline_breakouts(data, symbol, timeframe, cfg)
    signals.extend(trendline_signals)

    # ── 2. S/R BREAKOUTS ────────────────────────────────────────────────────────
    for res in levels["resistance"]:
        if curr["close"] > res >= prev["close"] and curr["close"] > curr["open"]:
            entry = curr["close"]
            tgt   = calculate_scalp_targets(data, {"entry": entry, "direction": "long"}, cfg)
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
                    "retest_confirmed": False,
                    "retest_quality": 0.0,
                    "pattern": pattern_info["pattern"],
                    "trend_strength": trend_info["strength"],
                    "confidence": pattern_info["confidence"],
                }
                sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                retest_result = enhanced_retest_confirmation(data, sig, cfg)
                sig["retest_confirmed"] = retest_result["confirmed"]
                sig["retest_quality"] = retest_result["quality"]
                if retest_result["confirmed"]:
                    signals.append(sig)

    for sup in levels["support"]:
        if curr["close"] < sup <= prev["close"] and curr["close"] < curr["open"]:
            entry = curr["close"]
            tgt   = calculate_scalp_targets(data, {"entry": entry, "direction": "short"}, cfg)
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
                    "retest_confirmed": False,
                    "retest_quality": 0.0,
                    "pattern": pattern_info["pattern"],
                    "trend_strength": trend_info["strength"],
                    "confidence": pattern_info["confidence"],
                }
                sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                retest_result = enhanced_retest_confirmation(data, sig, cfg)
                sig["retest_confirmed"] = retest_result["confirmed"]
                sig["retest_quality"] = retest_result["quality"]
                if retest_result["confirmed"]:
                    signals.append(sig)

    # ── 3. CHART PATTERN BREAKOUTS ───────────────────────────────────────────────
    if pattern_info["pattern"] and pattern_info["confidence"] > 0.6:
        pat = pattern_info["pattern"]
        if "ascending" in pat and levels["resistance"]:
            res = min(levels["resistance"])
            if curr["close"] > res:
                entry = curr["close"]
                tgt   = calculate_scalp_targets(data, {"entry": entry, "direction": "long"}, cfg)
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
                        "retest_confirmed": True,
                        "retest_quality": pattern_info["confidence"],
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
                tgt   = calculate_scalp_targets(data, {"entry": entry, "direction": "short"}, cfg)
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
                        "retest_confirmed": True,
                        "retest_quality": pattern_info["confidence"],
                        "pattern": pat,
                        "trend_strength": trend_info["strength"],
                        "confidence": pattern_info["confidence"],
                    }
                    sig["fakeout_detected"] = enhanced_fakeout_detection(data, sig, cfg)
                    signals.append(sig)

    # ── 4. TREND + BOLLINGER FILTER ─────────────────────────────────────────────
    filtered = []
    for sig in signals:
        align = (
            (trend_info["direction"] == "bullish" and sig["direction"] == "long")
            or (trend_info["direction"] == "bearish" and sig["direction"] == "short")
            or trend_info["direction"] == "neutral"
            or trend_info["strength"] <= 3
        )

        # Trendline breaks are counter-trend by definition — always allow them
        if sig["strategy"] in ("downtrend_line_break", "uptrend_line_break"):
            align = True

        bb_pos = (
            "above_upper" if curr["close"] > bb_u.iloc[-1]
            else "below_lower" if curr["close"] < bb_l.iloc[-1]
            else "inside"
        )

        if (bb_pos == "above_upper" and sig["direction"] == "long") or (
            bb_pos == "below_lower" and sig["direction"] == "short"
        ):
            sig["confidence"] *= 0.8

        # Use a lower confidence threshold for M15 scalping
        min_conf = cfg.get("min_signal_confidence", 0.3)
        if align and sig["confidence"] > min_conf:
            filtered.append(sig)
        else:
            reason = "trend misalign" if not align else f"conf {sig['confidence']:.2f} < {min_conf}"
            print(f"⚠️  Signal filtered ({reason}): {sig['strategy']} {sig['direction']}")

    # Deduplicate: if trendline and S/R signals overlap in direction, keep trendline
    seen_directions = set()
    deduped = []
    # Put trendline signals first (higher priority)
    for sig in sorted(filtered, key=lambda s: 0 if "line_break" in s["strategy"] else 1):
        key = sig["direction"]
        if key not in seen_directions:
            seen_directions.add(key)
            deduped.append(sig)

    return deduped


# ────────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing Enhanced Breakout Detection with Trendline Breakouts...")

    rng = pd.date_range("2024-01-01", periods=100, freq="15min")
    np.random.seed(42)
    base = 2000
    # Downtrend then breakout
    trend = np.concatenate([np.linspace(50, 0, 70), np.linspace(0, 30, 30)])
    noise = np.random.normal(0, 1, 100)

    price = base + trend + noise

    df = pd.DataFrame(
        {
            "open":   price + np.random.normal(0, 0.5, 100),
            "high":   price + np.random.uniform(0.5, 2.0, 100),
            "low":    price - np.random.uniform(0.5, 2.0, 100),
            "close":  price,
            "volume": np.random.randint(1000, 5000, 100),
        },
        index=rng,
    )

    cfg = {
        "min_rr": 1.5,
        "sl_buffer": 0.8,
        "sl_atr_mult": 0.8,
        "tp_atr_mult": 1.5,
        "support_resistance_window": 20,
        "pattern_window": 20,
        "volume_threshold": 1.2,
        "trendline_lookback": 60,
        "trendline_r2_min": 0.70,
        "trendline_min_touches": 2,
        "min_signal_confidence": 0.3,
    }

    sigs = detect_breakouts(df, "XAU/USD", "M15", cfg)

    print("✅ Test completed!")
    print(f"📊 Found {len(sigs)} potential breakout signal(s)")
    for s in sigs:
        print(f"🎯 {s['strategy']} | {s['direction']} | Entry: {s['entry']} | "
              f"SL: {s['sl']} | TP: {s['tp']} | RR {s['rr']:.2f}")
