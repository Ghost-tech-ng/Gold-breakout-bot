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
        """Get current market session."""
        current_hour = datetime.utcnow().hour

        if self.market_sessions["asian"][
                "start"] <= current_hour < self.market_sessions["asian"]["end"]:
            return "asian"
        elif self.market_sessions["london"][
                "start"] <= current_hour < self.market_sessions["london"][
                    "end"]:
            return "london"
        elif self.market_sessions["new_york"][
                "start"] <= current_hour < self.market_sessions["new_york"][
                    "end"]:
            return "new_york"
        else:
            return "off_hours"

    def check_news_events(self) -> Dict:
        """Check for upcoming high-impact news events."""
        # This would integrate with a news API in production
        # For now, return a simple structure
        current_time = datetime.utcnow()

        # Simulate news check - in production, integrate with ForexFactory API
        high_impact_times = [{
            "time": "08:30",
            "event": "US NFP",
            "impact": "high"
        }, {
            "time": "14:00",
            "event": "FOMC",
            "impact": "high"
        }, {
            "time": "12:30",
            "event": "US CPI",
            "impact": "high"
        }]

        return {"has_news": False, "next_event": None, "impact": "none"}


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


def should_skip_news() -> bool:
    """Check if current time is near high-impact news."""
    try:
        news_info = analyzer.check_news_events()
        return news_info.get("has_news",
                             False) and news_info.get("impact") == "high"
    except:
        return False


def get_trading_session_multiplier() -> float:
    """Get multiplier based on current trading session."""
    session = analyzer.get_market_session()

    multipliers = {
        "london": 1.2,  # High activity
        "new_york": 1.1,  # High activity
        "asian": 0.9,  # Medium activity
        "off_hours": 0.7  # Low activity
    }

    return multipliers.get(session, 1.0)


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
