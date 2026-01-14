import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional


class MarketAnalyzer:
    """Advanced market analysis using multiple indicators and patterns."""

    def __init__(self):
        self.news_events = []
        self.market_sessions = {
            "asian": {
                "start": 0,
                "end": 8
            },
            "london": {
                "start": 8,
                "end": 16
            },
            "new_york": {
                "start": 13,
                "end": 21
            }
        }

    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure for trend and key levels."""
        if len(data) < 50:
            return {"trend": "unknown", "strength": 0, "key_levels": []}

        # Calculate trend using multiple methods
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        current_price = data['close'].iloc[-1]

        # Trend determination
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "bullish"
            strength = min(
                ((current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]) * 100,
                10)
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = "bearish"
            strength = min(
                ((sma_50.iloc[-1] - current_price) / sma_50.iloc[-1]) * 100,
                10)
        else:
            trend = "sideways"
            strength = 0

        # Key levels identification
        recent_data = data.tail(50)
        resistance_levels = self._find_resistance_levels(recent_data)
        support_levels = self._find_support_levels(recent_data)

        return {
            "trend": trend,
            "strength": strength,
            "key_levels": {
                "resistance": resistance_levels,
                "support": support_levels
            },
            "current_price": current_price
        }

    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find key resistance levels."""
        highs = data['high'].rolling(window=5).max()
        peaks = []

        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] > highs.iloc[i - 1]
                    and highs.iloc[i] > highs.iloc[i + 1]
                    and highs.iloc[i] > highs.iloc[i - 2]
                    and highs.iloc[i] > highs.iloc[i + 2]):
                peaks.append(highs.iloc[i])

        # Return top 3 resistance levels
        return sorted(list(set(peaks)), reverse=True)[:3]

    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find key support levels."""
        lows = data['low'].rolling(window=5).min()
        troughs = []

        for i in range(2, len(lows) - 2):
            if (lows.iloc[i] < lows.iloc[i - 1]
                    and lows.iloc[i] < lows.iloc[i + 1]
                    and lows.iloc[i] < lows.iloc[i - 2]
                    and lows.iloc[i] < lows.iloc[i + 2]):
                troughs.append(lows.iloc[i])

        # Return top 3 support levels
        return sorted(list(set(troughs)), reverse=True)[:3]

    def calculate_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """Calculate current volatility regime."""
        if len(data) < 20:
            return {"regime": "unknown", "percentile": 0}

        # Calculate ATR-based volatility
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close],
                               axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()

        current_atr = atr.iloc[-1]
        atr_percentile = (atr.iloc[-1] / atr.quantile(0.8)) * 100

        if atr_percentile > 120:
            regime = "high"
        elif atr_percentile < 80:
            regime = "low"
        else:
            regime = "medium"

        return {
            "regime": regime,
            "percentile": atr_percentile,
            "current_atr": current_atr
        }

    def analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators."""
        if len(data) < 20:
            return {"rsi": 50, "momentum": "neutral", "divergence": False}

        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # Momentum classification
        if current_rsi > 70:
            momentum = "overbought"
        elif current_rsi < 30:
            momentum = "oversold"
        elif current_rsi > 55:
            momentum = "bullish"
        elif current_rsi < 45:
            momentum = "bearish"
        else:
            momentum = "neutral"

        # Simple divergence detection
        price_trend = data['close'].iloc[-5:].is_monotonic_increasing
        rsi_trend = rsi.iloc[-5:].is_monotonic_increasing
        divergence = price_trend != rsi_trend

        return {
            "rsi": current_rsi,
            "momentum": momentum,
            "divergence": divergence
        }

    def get_market_session(self) -> str:
        """Get current market session with overlap detection."""
        current_hour = datetime.utcnow().hour
        
        # Check for overlap periods first (highest priority)
        # London-NY overlap: 13:00-16:00 UTC
        if 13 <= current_hour < 16:
            return "overlap"
        
        # Individual sessions
        if self.market_sessions["asian"]["start"] <= current_hour < self.market_sessions["asian"]["end"]:
            return "asian"
        elif self.market_sessions["london"]["start"] <= current_hour < self.market_sessions["london"]["end"]:
            return "london"
        elif self.market_sessions["new_york"]["start"] <= current_hour < self.market_sessions["new_york"]["end"]:
            return "new_york"
        else:
            return "off_hours"

    def check_news_events(self, blackout_minutes: int = 30) -> Dict:
        """Check for upcoming high-impact news events with time-based detection."""
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Skip weekends
        if current_weekday >= 5:  # Saturday or Sunday
            return {"has_news": False, "next_event": None, "impact": "none", "reason": "weekend"}
        
        # High-impact news times (UTC) - typical economic calendar
        high_impact_times = [
            # US Data (12:30 UTC) - NFP, CPI, Retail Sales, etc.
            {"hour": 12, "minute": 30, "event": "US Economic Data", "days": [4]},  # Usually Friday for NFP
            {"hour": 12, "minute": 30, "event": "US CPI/PPI", "days": [1, 2, 3, 4]},  # Mid-week
            
            # FOMC (14:00 UTC) - Usually Wednesday
            {"hour": 14, "minute": 0, "event": "FOMC Decision", "days": [2]},
            
            # UK Data (08:30 UTC)
            {"hour": 8, "minute": 30, "event": "UK Economic Data", "days": [0, 1, 2, 3, 4]},
            
            # ECB (12:45 UTC) - Usually Thursday
            {"hour": 12, "minute": 45, "event": "ECB Decision", "days": [3]},
            
            # US Fed Speeches (18:00 UTC)
            {"hour": 18, "minute": 0, "event": "Fed Speech", "days": [0, 1, 2, 3, 4]},
        ]
        
        # Check if current time is within blackout window of any event
        for event in high_impact_times:
            # Check if event is scheduled for today
            if current_weekday not in event["days"]:
                continue
            
            event_hour = event["hour"]
            event_minute = event["minute"]
            
            # Calculate time difference in minutes
            event_time_minutes = event_hour * 60 + event_minute
            current_time_minutes = current_hour * 60 + current_minute
            time_diff = abs(event_time_minutes - current_time_minutes)
            
            # If within blackout window
            if time_diff <= blackout_minutes:
                return {
                    "has_news": True,
                    "next_event": event["event"],
                    "impact": "high",
                    "time_to_event": event_time_minutes - current_time_minutes,
                    "reason": f"Within {blackout_minutes}min of {event['event']}"
                }
        
        return {"has_news": False, "next_event": None, "impact": "none", "reason": "clear"}


# Global analyzer instance
analyzer = MarketAnalyzer()


def get_market_sentiment(data: pd.DataFrame) -> Dict:
    """Main function to get comprehensive market sentiment."""
    try:
        # Analyze market structure
        structure = analyzer.analyze_market_structure(data)

        # Analyze volatility
        volatility = analyzer.calculate_volatility_regime(data)

        # Analyze momentum
        momentum = analyzer.analyze_momentum(data)

        # Get current session
        session = analyzer.get_market_session()

        # Calculate overall confidence
        confidence_factors = []

        # Trend strength factor
        if structure["strength"] > 5:
            confidence_factors.append(0.3)
        elif structure["strength"] > 2:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)

        # Volatility factor
        if volatility["regime"] == "medium":
            confidence_factors.append(0.2)
        elif volatility["regime"] == "high":
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(0.15)

        # Momentum factor
        if momentum["momentum"] in ["bullish", "bearish"]:
            confidence_factors.append(0.25)
        else:
            confidence_factors.append(0.15)

        # Session factor
        if session in ["london", "new_york"]:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)

        overall_confidence = sum(confidence_factors)

        # Generate analysis text
        analysis_parts = []

        if structure["trend"] == "bullish":
            analysis_parts.append("🟢 Bullish Trend")
        elif structure["trend"] == "bearish":
            analysis_parts.append("🔴 Bearish Trend")
        else:
            analysis_parts.append("🟡 Sideways Market")

        if volatility["regime"] == "high":
            analysis_parts.append("⚡ High Volatility")
        elif volatility["regime"] == "low":
            analysis_parts.append("😴 Low Volatility")
        else:
            analysis_parts.append("📊 Normal Volatility")

        if momentum["momentum"] == "overbought":
            analysis_parts.append("🔥 Overbought")
        elif momentum["momentum"] == "oversold":
            analysis_parts.append("❄️ Oversold")

        analysis_text = " | ".join(analysis_parts)

        return {
            "confidence": overall_confidence,
            "analysis": analysis_text,
            "structure": structure,
            "volatility": volatility,
            "momentum": momentum,
            "session": session,
            "recommendation":
            "proceed" if overall_confidence > 0.6 else "caution"
        }

    except Exception as e:
        print(f"❌ Error in market sentiment analysis: {e}")
        return {
            "confidence": 0.5,
            "analysis": "Analysis Error",
            "recommendation": "caution"
        }


def should_skip_news(config: Dict = None) -> bool:
    """Check if current time is near high-impact news."""
    try:
        blackout_minutes = config.get("news_blackout_minutes", 30) if config else 30
        news_info = analyzer.check_news_events(blackout_minutes)
        
        if news_info.get("has_news", False) and news_info.get("impact") == "high":
            print(f"📰 News blackout active: {news_info.get('reason', 'Unknown')}")
            return True
        
        return False
    except Exception as e:
        print(f"⚠️ Error checking news events: {e}")
        return False


def get_trading_session_multiplier(config: Dict = None) -> tuple[float, str]:
    """Get multiplier based on current trading session from config."""
    session = analyzer.get_market_session()
    
    # Get multipliers from config or use defaults
    if config and "session_multipliers" in config:
        multipliers = config["session_multipliers"]
    else:
        multipliers = {
            "asian": 0.8,
            "london": 1.0,
            "new_york": 1.1,
            "overlap": 1.15,
            "off_hours": 0.7
        }
    
    multiplier = multipliers.get(session, 1.0)
    return multiplier, session


def enhance_signal_with_sentiment(signal: Dict, sentiment: Dict) -> Dict:
    """Enhance signal with sentiment analysis."""
    enhanced_signal = signal.copy()

    # Adjust risk-reward based on confidence
    confidence = sentiment.get("confidence", 0.5)
    if confidence > 0.8:
        enhanced_signal["rr"] = signal["rr"] * 1.1  # Increase target
    elif confidence < 0.4:
        enhanced_signal["rr"] = signal["rr"] * 0.9  # Decrease target

    # Add sentiment info
    enhanced_signal["sentiment"] = sentiment.get("analysis", "")
    enhanced_signal["confidence"] = confidence

    return enhanced_signal


def apply_volatility_adaptation(signal: Dict, data: pd.DataFrame, config: Dict) -> Dict:
    """
    Apply volatility-based adjustments to signal parameters.
    Adjusts SL, TP, and position size based on current volatility regime.
    """
    adapted_signal = signal.copy()
    
    # Get volatility regime
    volatility = analyzer.calculate_volatility_regime(data)
    regime = volatility.get("regime", "medium")
    
    # Get adaptation parameters from config
    if "volatility_adaptation" in config and regime in config["volatility_adaptation"]:
        adaptation = config["volatility_adaptation"][regime]
        
        sl_mult = adaptation.get("sl_mult", 1.0)
        tp_mult = adaptation.get("tp_mult", 1.0)
        size_mult = adaptation.get("size_mult", 1.0)
        
        # Adjust stop loss
        entry = signal["entry"]
        sl = signal["sl"]
        sl_distance = abs(entry - sl)
        new_sl_distance = sl_distance * sl_mult
        
        if signal["direction"].lower() == "long":
            adapted_signal["sl"] = round(entry - new_sl_distance, 3)
        else:
            adapted_signal["sl"] = round(entry + new_sl_distance, 3)
        
        # Adjust take profit
        tp = signal["tp"]
        tp_distance = abs(tp - entry)
        new_tp_distance = tp_distance * tp_mult
        
        if signal["direction"].lower() == "long":
            adapted_signal["tp"] = round(entry + new_tp_distance, 3)
        else:
            adapted_signal["tp"] = round(entry - new_tp_distance, 3)
        
        # Recalculate RR
        new_sl_dist = abs(adapted_signal["entry"] - adapted_signal["sl"])
        new_tp_dist = abs(adapted_signal["tp"] - adapted_signal["entry"])
        adapted_signal["rr"] = round(new_tp_dist / new_sl_dist, 2) if new_sl_dist > 0 else signal["rr"]
        
        # Add volatility info
        adapted_signal["volatility_regime"] = regime
        adapted_signal["volatility_adapted"] = True
        adapted_signal["position_size_mult"] = size_mult
        
        print(f"📊 Volatility adaptation applied: {regime} regime (SL: {sl_mult}x, TP: {tp_mult}x, Size: {size_mult}x)")
    else:
        adapted_signal["volatility_regime"] = regime
        adapted_signal["volatility_adapted"] = False
        adapted_signal["position_size_mult"] = 1.0
    
    return adapted_signal


def log_sentiment_analysis(data: pd.DataFrame, symbol: str = "XAU/USD"):
    """Log sentiment analysis for future training."""
    try:
        sentiment = get_market_sentiment(data)

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "sentiment": sentiment,
            "price": data['close'].iloc[-1] if not data.empty else None
        }

        # In production, this would be saved to a database
        print(f"📊 Sentiment logged: {sentiment['analysis']}")

    except Exception as e:
        print(f"❌ Error logging sentiment: {e}")


if __name__ == "__main__":
    # Test the sentiment analysis
    print("🧠 Testing GPT Brain Market Analysis...")

    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    np.random.seed(42)

    # Generate realistic gold price data
    base_price = 2000
    price_changes = np.random.normal(0, 5, 100)
    prices = base_price + np.cumsum(price_changes)

    test_data = pd.DataFrame(
        {
            'open': prices + np.random.normal(0, 1, 100),
            'high': prices + np.random.uniform(1, 5, 100),
            'low': prices - np.random.uniform(1, 5, 100),
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        },
        index=dates)

    sentiment = get_market_sentiment(test_data)
    print(f"✅ Analysis Complete!")
    print(f"📊 Confidence: {sentiment['confidence']:.2f}")
    print(f"🧠 Analysis: {sentiment['analysis']}")
    print(f"📈 Recommendation: {sentiment['recommendation']}")
