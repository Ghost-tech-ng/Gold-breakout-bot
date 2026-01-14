"""
Risk Manager Module
Handles trailing stops, breakeven moves, partial profit taking, and position monitoring.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from position_manager import position_manager


class RiskManager:
    """Manages dynamic risk adjustments for open positions."""
    
    def __init__(self, config: Dict):
        """Initialize risk manager with configuration."""
        self.config = config
        self.trailing_stops = {}  # {signal_id: current_stop_level}
        self.breakeven_moved = set()  # signal_ids that moved to breakeven
        self.partial_closed = set()  # signal_ids with partial TP taken
    
    def calculate_chandelier_stop(self, data: pd.DataFrame, direction: str, 
                                  atr_multiplier: float = 2.5) -> float:
        """
        Calculate Chandelier trailing stop.
        Long: Highest High - (ATR * multiplier)
        Short: Lowest Low + (ATR * multiplier)
        """
        if len(data) < 14:
            return 0.0
        
        # Calculate ATR
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
        
        # Get recent high/low (last 10 candles)
        recent_data = data.tail(10)
        
        if direction.lower() == "long":
            highest_high = recent_data["high"].max()
            stop = highest_high - (atr * atr_multiplier)
        else:  # short
            lowest_low = recent_data["low"].min()
            stop = lowest_low + (atr * atr_multiplier)
        
        return round(stop, 3)
    
    def should_move_to_breakeven(self, position: Dict, current_price: float) -> bool:
        """
        Check if position should move to breakeven.
        Triggers when profit >= 1:1 RR (or configured threshold).
        """
        if position['signal_id'] in self.breakeven_moved:
            return False
        
        entry = position['entry']
        sl = position['sl']
        direction = position['direction']
        
        risk = abs(entry - sl)
        breakeven_threshold = self.config.get("breakeven_rr_threshold", 1.0)
        required_profit = risk * breakeven_threshold
        
        if direction.lower() == "long":
            current_profit = current_price - entry
            return current_profit >= required_profit
        else:  # short
            current_profit = entry - current_price
            return current_profit >= required_profit
    
    def should_take_partial_profit(self, position: Dict, current_price: float) -> bool:
        """
        Check if partial profit should be taken.
        Triggers when profit >= configured RR threshold (default 1.5:1).
        """
        if not self.config.get("partial_tp_enabled", False):
            return False
        
        if position['signal_id'] in self.partial_closed:
            return False
        
        entry = position['entry']
        sl = position['sl']
        direction = position['direction']
        
        risk = abs(entry - sl)
        partial_threshold = self.config.get("partial_tp_rr", 1.5)
        required_profit = risk * partial_threshold
        
        if direction.lower() == "long":
            current_profit = current_price - entry
            return current_profit >= required_profit
        else:  # short
            current_profit = entry - current_price
            return current_profit >= required_profit
    
    def update_trailing_stop(self, signal_id: str, position: Dict, 
                            data: pd.DataFrame) -> Optional[float]:
        """
        Update trailing stop for a position.
        Returns new stop level if updated, None otherwise.
        """
        if not self.config.get("trailing_stop_enabled", False):
            return None
        
        direction = position['direction']
        current_sl = position['sl']
        atr_mult = self.config.get("trailing_stop_atr_multiplier", 2.5)
        
        # Calculate new chandelier stop
        new_stop = self.calculate_chandelier_stop(data, direction, atr_mult)
        
        if direction.lower() == "long":
            # For long, only move stop up
            if new_stop > current_sl:
                # Store in tracking dict
                self.trailing_stops[signal_id] = new_stop
                return new_stop
        else:  # short
            # For short, only move stop down
            if new_stop < current_sl:
                self.trailing_stops[signal_id] = new_stop
                return new_stop
        
        return None
    
    def move_to_breakeven(self, signal_id: str, position: Dict) -> float:
        """
        Move stop loss to breakeven (entry price).
        Returns new stop level.
        """
        entry = position['entry']
        self.breakeven_moved.add(signal_id)
        return entry
    
    def check_stop_hit(self, position: Dict, current_price: float) -> Optional[str]:
        """
        Check if stop loss or take profit has been hit.
        Returns 'sl_hit', 'tp_hit', or None.
        """
        direction = position['direction']
        sl = position['sl']
        tp = position['tp']
        
        # Check if we have a trailing stop for this position
        signal_id = position['signal_id']
        if signal_id in self.trailing_stops:
            sl = self.trailing_stops[signal_id]
        
        if direction.lower() == "long":
            if current_price <= sl:
                return "sl_hit"
            elif current_price >= tp:
                return "tp_hit"
        else:  # short
            if current_price >= sl:
                return "sl_hit"
            elif current_price <= tp:
                return "tp_hit"
        
        return None
    
    def calculate_current_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate current unrealized PnL for a position."""
        entry = position['entry']
        direction = position['direction']
        
        if direction.lower() == "long":
            pnl = current_price - entry
        else:  # short
            pnl = entry - current_price
        
        return round(pnl, 3)
    
    def calculate_current_rr(self, position: Dict, current_price: float) -> float:
        """Calculate current risk-reward ratio."""
        entry = position['entry']
        sl = position['sl']
        direction = position['direction']
        
        # Check if we have a trailing stop
        signal_id = position['signal_id']
        if signal_id in self.trailing_stops:
            sl = self.trailing_stops[signal_id]
        
        risk = abs(entry - sl)
        
        if direction.lower() == "long":
            reward = current_price - entry
        else:  # short
            reward = entry - current_price
        
        if risk == 0:
            return 0.0
        
        return round(reward / risk, 2)
    
    def get_position_status(self, position: Dict, current_price: float) -> Dict:
        """
        Get comprehensive status for a position.
        Returns dict with PnL, RR, stop levels, and recommendations.
        """
        signal_id = position['signal_id']
        
        # Get current metrics
        pnl = self.calculate_current_pnl(position, current_price)
        current_rr = self.calculate_current_rr(position, current_price)
        
        # Check for actions
        stop_hit = self.check_stop_hit(position, current_price)
        should_breakeven = self.should_move_to_breakeven(position, current_price)
        should_partial = self.should_take_partial_profit(position, current_price)
        
        # Get current stop level
        current_sl = self.trailing_stops.get(signal_id, position['sl'])
        
        return {
            'signal_id': signal_id,
            'current_price': current_price,
            'unrealized_pnl': pnl,
            'current_rr': current_rr,
            'current_sl': current_sl,
            'original_sl': position['sl'],
            'tp': position['tp'],
            'stop_hit': stop_hit,
            'should_move_breakeven': should_breakeven,
            'should_take_partial': should_partial,
            'breakeven_moved': signal_id in self.breakeven_moved,
            'partial_taken': signal_id in self.partial_closed,
            'trailing_active': signal_id in self.trailing_stops
        }
    
    def process_position_updates(self, positions: List[Dict], 
                                 current_data: pd.DataFrame) -> List[Dict]:
        """
        Process all open positions and return list of actions to take.
        Returns list of dicts with action details.
        """
        if current_data.empty or len(current_data) < 1:
            return []
        
        current_price = float(current_data.iloc[-1]['close'])
        actions = []
        
        for position in positions:
            signal_id = position['signal_id']
            status = self.get_position_status(position, current_price)
            
            # Check for stop/TP hit
            if status['stop_hit']:
                actions.append({
                    'type': 'close_position',
                    'signal_id': signal_id,
                    'reason': status['stop_hit'],
                    'exit_price': current_price,
                    'pnl': status['unrealized_pnl']
                })
                continue
            
            # Check for breakeven move
            if status['should_move_breakeven'] and not status['breakeven_moved']:
                new_sl = self.move_to_breakeven(signal_id, position)
                actions.append({
                    'type': 'move_breakeven',
                    'signal_id': signal_id,
                    'old_sl': position['sl'],
                    'new_sl': new_sl,
                    'current_rr': status['current_rr']
                })
            
            # Check for partial profit
            if status['should_take_partial'] and not status['partial_taken']:
                self.partial_closed.add(signal_id)
                partial_pct = self.config.get("partial_tp_percentage", 50)
                actions.append({
                    'type': 'partial_close',
                    'signal_id': signal_id,
                    'percentage': partial_pct,
                    'exit_price': current_price,
                    'pnl': status['unrealized_pnl']
                })
            
            # Check for trailing stop update
            if self.config.get("trailing_stop_enabled", False):
                new_stop = self.update_trailing_stop(signal_id, position, current_data)
                if new_stop:
                    actions.append({
                        'type': 'update_trailing_stop',
                        'signal_id': signal_id,
                        'old_sl': status['current_sl'],
                        'new_sl': new_stop,
                        'current_rr': status['current_rr']
                    })
        
        return actions
    
    def reset_position_tracking(self, signal_id: str):
        """Remove tracking for a closed position."""
        self.trailing_stops.pop(signal_id, None)
        self.breakeven_moved.discard(signal_id)
        self.partial_closed.discard(signal_id)


# Global instance
risk_manager = None

def initialize_risk_manager(config: Dict):
    """Initialize global risk manager instance."""
    global risk_manager
    risk_manager = RiskManager(config)
    return risk_manager


if __name__ == "__main__":
    # Test risk manager
    print("🧪 Testing Risk Manager...")
    
    test_config = {
        "trailing_stop_enabled": True,
        "trailing_stop_atr_multiplier": 2.5,
        "breakeven_rr_threshold": 1.0,
        "partial_tp_enabled": True,
        "partial_tp_rr": 1.5,
        "partial_tp_percentage": 50
    }
    
    rm = RiskManager(test_config)
    
    # Create test data
    import numpy as np
    dates = pd.date_range(start='2024-01-01', periods=50, freq='15min')
    np.random.seed(42)
    
    base_price = 2050
    prices = base_price + np.cumsum(np.random.randn(50) * 0.5)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(50) * 0.2,
        'high': prices + np.random.uniform(0.5, 1.5, 50),
        'low': prices - np.random.uniform(0.5, 1.5, 50),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 50)
    }, index=dates)
    
    # Test position
    test_position = {
        'signal_id': 'TEST_001',
        'entry': 2050.0,
        'sl': 2045.0,
        'tp': 2062.5,
        'direction': 'long',
        'rr': 2.5
    }
    
    current_price = 2055.0  # 5 points profit (1:1 RR)
    
    # Test breakeven check
    should_be = rm.should_move_to_breakeven(test_position, current_price)
    print(f"Should move to breakeven at {current_price}: {should_be}")
    
    # Test chandelier stop
    chandelier = rm.calculate_chandelier_stop(test_data, 'long', 2.5)
    print(f"Chandelier stop: {chandelier}")
    
    # Test position status
    status = rm.get_position_status(test_position, current_price)
    print(f"Position status: PnL={status['unrealized_pnl']}, RR={status['current_rr']}")
    
    print("✅ Risk Manager test complete")
