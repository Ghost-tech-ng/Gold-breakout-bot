"""
Position Manager Module
Handles position tracking, duplicate signal prevention, and risk management.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class PositionManager:
    """Manages trading positions and prevents duplicate signals."""
    
    def __init__(self, db_path: str = "trade_history.db"):
        """Initialize position manager with database."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_reward REAL NOT NULL,
                entry_time TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                pnl REAL,
                exit_reason TEXT,
                ml_confidence REAL,
                market_sentiment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Signal history table (for cooldown tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                price_level REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                avg_rr REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                consecutive_losses INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(date)
            )
        """)
        
        conn.commit()
        conn.close()
        print("✅ Position database initialized")
    
    def generate_signal_id(self, signal: Dict) -> str:
        """Generate unique signal ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{signal['symbol']}_{signal['timeframe']}_{signal['strategy']}_{timestamp}"
    
    def can_send_signal(self, signal: Dict, config: Dict) -> tuple[bool, str]:
        """
        Check if signal can be sent based on position limits and cooldown.
        Returns (can_send, reason)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check 1: Maximum concurrent positions
        cursor.execute("""
            SELECT COUNT(*) FROM positions 
            WHERE status = 'open'
        """)
        open_positions = cursor.fetchone()[0]
        max_positions = config.get("max_concurrent_positions", 3)
        
        if open_positions >= max_positions:
            conn.close()
            return False, f"Maximum concurrent positions reached ({open_positions}/{max_positions})"
        
        # Check 2: Cooldown period
        cooldown_minutes = config.get("cooldown_minutes", 30)
        cooldown_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        cursor.execute("""
            SELECT COUNT(*) FROM signal_history 
            WHERE symbol = ? AND timeframe = ? 
            AND datetime(timestamp) > datetime(?)
        """, (signal['symbol'], signal['timeframe'], cooldown_time.isoformat()))
        
        recent_signals = cursor.fetchone()[0]
        if recent_signals > 0:
            conn.close()
            return False, f"Cooldown active (last signal < {cooldown_minutes} min ago)"
        
        # Check 3: Price proximity to existing positions
        min_distance_atr = config.get("min_distance_atr", 0.5)
        cursor.execute("""
            SELECT entry_price, stop_loss FROM positions 
            WHERE symbol = ? AND status = 'open'
        """, (signal['symbol'],))
        
        existing_positions = cursor.fetchall()
        for entry_price, stop_loss in existing_positions:
            atr_estimate = abs(entry_price - stop_loss) / config.get("sl_buffer", 1.2)
            min_distance = atr_estimate * min_distance_atr
            
            if abs(signal['entry'] - entry_price) < min_distance:
                conn.close()
                return False, f"Too close to existing position (< {min_distance_atr} ATR)"
        
        # Check 4: Daily loss limit
        max_daily_loss = config.get("max_daily_loss_percent", 3.0)
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT total_pnl FROM performance_metrics 
            WHERE date = ?
        """, (today,))
        
        result = cursor.fetchone()
        if result and result[0] is not None:
            daily_pnl = result[0]
            # Assuming hypothetical account size of 10000 for percentage calculation
            if daily_pnl < -(max_daily_loss * 100):  # -3% of 10000 = -300
                conn.close()
                return False, f"Daily loss limit reached ({daily_pnl:.2f})"
        
        # Check 5: Consecutive losses
        max_consecutive_losses = config.get("max_consecutive_losses", 5)
        cursor.execute("""
            SELECT consecutive_losses FROM performance_metrics 
            WHERE date = ?
        """, (today,))
        
        result = cursor.fetchone()
        if result and result[0] is not None:
            consecutive_losses = result[0]
            if consecutive_losses >= max_consecutive_losses:
                conn.close()
                return False, f"Max consecutive losses reached ({consecutive_losses})"
        
        conn.close()
        return True, "OK"
    
    def add_position(self, signal: Dict, ml_confidence: float = 0.0, 
                     market_sentiment: str = "") -> str:
        """Add new position to database."""
        signal_id = self.generate_signal_id(signal)
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO positions (
                    signal_id, symbol, timeframe, strategy, direction,
                    entry_price, stop_loss, take_profit, risk_reward,
                    entry_time, status, ml_confidence, market_sentiment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, signal['symbol'], signal['timeframe'], 
                signal['strategy'], signal['direction'],
                signal['entry'], signal['sl'], signal['tp'], signal['rr'],
                signal['timestamp'], 'open', ml_confidence, market_sentiment, now
            ))
            
            # Add to signal history for cooldown tracking
            cursor.execute("""
                INSERT INTO signal_history (
                    symbol, timeframe, strategy, direction, price_level, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal['symbol'], signal['timeframe'], signal['strategy'],
                signal['direction'], signal['entry'], signal['timestamp'], now
            ))
            
            conn.commit()
            print(f"✅ Position added: {signal_id}")
            return signal_id
            
        except sqlite3.IntegrityError as e:
            print(f"❌ Position already exists: {e}")
            return ""
        finally:
            conn.close()
    
    def update_position(self, signal_id: str, exit_price: float, 
                       exit_reason: str = "manual"):
        """Update position with exit information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get position details
        cursor.execute("""
            SELECT entry_price, direction, stop_loss, take_profit 
            FROM positions WHERE signal_id = ?
        """, (signal_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            print(f"❌ Position not found: {signal_id}")
            return
        
        entry_price, direction, sl, tp = result
        
        # Calculate PnL
        if direction.lower() == "long":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        # Update position
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE positions 
            SET status = 'closed', exit_price = ?, exit_time = ?, 
                pnl = ?, exit_reason = ?
            WHERE signal_id = ?
        """, (exit_price, now, pnl, exit_reason, signal_id))
        
        conn.commit()
        conn.close()
        
        # Update performance metrics
        self._update_performance_metrics(pnl)
        
        print(f"✅ Position updated: {signal_id}, PnL: {pnl:.3f}")
    
    def _update_performance_metrics(self, pnl: float):
        """Update daily performance metrics."""
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get or create today's metrics
        cursor.execute("""
            SELECT id, total_trades, winning_trades, losing_trades, 
                   total_pnl, consecutive_losses
            FROM performance_metrics WHERE date = ?
        """, (today,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing
            metric_id, total_trades, winning_trades, losing_trades, total_pnl, consecutive_losses = result
            
            total_trades += 1
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
                consecutive_losses = 0
            else:
                losing_trades += 1
                consecutive_losses += 1
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            cursor.execute("""
                UPDATE performance_metrics 
                SET total_trades = ?, winning_trades = ?, losing_trades = ?,
                    total_pnl = ?, win_rate = ?, consecutive_losses = ?
                WHERE id = ?
            """, (total_trades, winning_trades, losing_trades, total_pnl, 
                  win_rate, consecutive_losses, metric_id))
        else:
            # Create new
            winning_trades = 1 if pnl > 0 else 0
            losing_trades = 0 if pnl > 0 else 1
            consecutive_losses = 0 if pnl > 0 else 1
            win_rate = (winning_trades / 1 * 100)
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    date, total_trades, winning_trades, losing_trades,
                    total_pnl, win_rate, consecutive_losses, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, 1, winning_trades, losing_trades, pnl, 
                  win_rate, consecutive_losses, now))
        
        conn.commit()
        conn.close()
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT signal_id, symbol, timeframe, strategy, direction,
                   entry_price, stop_loss, take_profit, risk_reward, entry_time
            FROM positions WHERE status = 'open'
            ORDER BY entry_time DESC
        """)
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'signal_id': row[0],
                'symbol': row[1],
                'timeframe': row[2],
                'strategy': row[3],
                'direction': row[4],
                'entry': row[5],
                'sl': row[6],
                'tp': row[7],
                'rr': row[8],
                'entry_time': row[9]
            })
        
        conn.close()
        return positions
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT SUM(total_trades), SUM(winning_trades), SUM(losing_trades),
                   SUM(total_pnl), AVG(win_rate)
            FROM performance_metrics 
            WHERE date >= ?
        """, (start_date,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            total_trades, winning, losing, total_pnl, avg_win_rate = result
            return {
                'total_trades': total_trades or 0,
                'winning_trades': winning or 0,
                'losing_trades': losing or 0,
                'total_pnl': total_pnl or 0.0,
                'win_rate': avg_win_rate or 0.0,
                'profit_factor': abs(total_pnl / losing) if losing and total_pnl else 0.0
            }
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    def cleanup_old_signals(self, days: int = 30):
        """Remove old signal history entries."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM signal_history 
            WHERE datetime(created_at) < datetime(?)
        """, (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"🧹 Cleaned up {deleted} old signal history entries")


# Global instance
position_manager = PositionManager()


if __name__ == "__main__":
    # Test the position manager
    print("🧪 Testing Position Manager...")
    
    pm = PositionManager("test_trade_history.db")
    
    # Test signal
    test_signal = {
        'symbol': 'XAU/USD',
        'timeframe': 'M15',
        'strategy': 'resistance_breakout',
        'direction': 'long',
        'entry': 2050.5,
        'sl': 2045.0,
        'tp': 2065.0,
        'rr': 2.64,
        'timestamp': datetime.now().isoformat()
    }
    
    test_config = {
        'max_concurrent_positions': 3,
        'cooldown_minutes': 30,
        'min_distance_atr': 0.5,
        'sl_buffer': 1.2,
        'max_daily_loss_percent': 3.0,
        'max_consecutive_losses': 5
    }
    
    # Test can_send_signal
    can_send, reason = pm.can_send_signal(test_signal, test_config)
    print(f"Can send signal: {can_send}, Reason: {reason}")
    
    if can_send:
        # Add position
        signal_id = pm.add_position(test_signal, ml_confidence=0.75, 
                                    market_sentiment="Bullish")
        print(f"Added position: {signal_id}")
        
        # Get open positions
        open_pos = pm.get_open_positions()
        print(f"Open positions: {len(open_pos)}")
        
        # Simulate exit
        pm.update_position(signal_id, exit_price=2065.0, exit_reason="tp_hit")
        
        # Get performance
        perf = pm.get_performance_summary(days=1)
        print(f"Performance: {perf}")
    
    print("✅ Position Manager test complete")
