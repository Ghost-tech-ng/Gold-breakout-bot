"""
Position Manager Module
Handles position tracking, duplicate signal prevention, and risk management.
Uses PostgreSQL on Render (DATABASE_URL env var) or SQLite locally.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from db import get_db, q, init_db


class PositionManager:
    """Manages trading positions and prevents duplicate signals."""

    def __init__(self):
        """Initialize position manager and ensure tables exist."""
        init_db()

    def generate_signal_id(self, signal: Dict) -> str:
        """Generate unique signal ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{signal['symbol']}_{signal['timeframe']}_{signal['strategy']}_{timestamp}"

    def can_send_signal(self, signal: Dict, config: Dict) -> tuple:
        """
        Check if signal can be sent based on position limits and cooldown.
        Returns (can_send: bool, reason: str)
        """
        with get_db() as (conn, cur):
            # Check 1: Maximum concurrent positions
            cur.execute(q("SELECT COUNT(*) FROM positions WHERE status = 'open'"))
            open_positions = cur.fetchone()[0]
            max_positions = config.get("max_concurrent_positions", 3)

            if open_positions >= max_positions:
                return False, f"Maximum concurrent positions reached ({open_positions}/{max_positions})"

            # Check 2: Cooldown period
            cooldown_minutes = config.get("cooldown_minutes", 30)
            cooldown_time = (datetime.now() - timedelta(minutes=cooldown_minutes)).isoformat()

            cur.execute(q("""
                SELECT COUNT(*) FROM signal_history
                WHERE symbol = ? AND timeframe = ? AND timestamp > ?
            """), (signal['symbol'], signal['timeframe'], cooldown_time))

            recent_signals = cur.fetchone()[0]
            if recent_signals > 0:
                return False, f"Cooldown active (last signal < {cooldown_minutes} min ago)"

            # Check 3: Price proximity to existing positions
            min_distance_atr = config.get("min_distance_atr", 0.5)
            cur.execute(q("""
                SELECT entry_price, stop_loss FROM positions
                WHERE symbol = ? AND status = 'open'
            """), (signal['symbol'],))

            existing_positions = cur.fetchall()
            for entry_price, stop_loss in existing_positions:
                atr_estimate = abs(entry_price - stop_loss) / config.get("sl_buffer", 0.8)
                min_distance = atr_estimate * min_distance_atr

                if abs(signal['entry'] - entry_price) < min_distance:
                    return False, f"Too close to existing position (< {min_distance_atr} ATR)"

            # Check 4: Daily loss limit
            max_daily_loss = config.get("max_daily_loss_percent", 3.0)
            today = datetime.now().strftime("%Y-%m-%d")

            cur.execute(q("""
                SELECT total_pnl FROM performance_metrics WHERE date = ?
            """), (today,))

            result = cur.fetchone()
            if result and result[0] is not None:
                daily_pnl = result[0]
                if daily_pnl < -(max_daily_loss * 100):
                    return False, f"Daily loss limit reached ({daily_pnl:.2f})"

            # Check 5: Consecutive losses
            max_consecutive_losses = config.get("max_consecutive_losses", 5)
            cur.execute(q("""
                SELECT consecutive_losses FROM performance_metrics WHERE date = ?
            """), (today,))

            result = cur.fetchone()
            if result and result[0] is not None:
                consecutive_losses = result[0]
                if consecutive_losses >= max_consecutive_losses:
                    return False, f"Max consecutive losses reached ({consecutive_losses})"

        return True, "OK"

    def add_position(self, signal: Dict, ml_confidence: float = 0.0,
                     market_sentiment: str = "") -> str:
        """Add new position to database. Returns signal_id or empty string on failure."""
        signal_id = self.generate_signal_id(signal)
        now = datetime.now().isoformat()

        try:
            with get_db() as (conn, cur):
                cur.execute(q("""
                    INSERT INTO positions (
                        signal_id, symbol, timeframe, strategy, direction,
                        entry_price, stop_loss, take_profit, risk_reward,
                        entry_time, status, ml_confidence, market_sentiment, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """), (
                    signal_id, signal['symbol'], signal['timeframe'],
                    signal['strategy'], signal['direction'],
                    signal['entry'], signal['sl'], signal['tp'], signal['rr'],
                    signal['timestamp'], 'open', ml_confidence, market_sentiment, now
                ))

                cur.execute(q("""
                    INSERT INTO signal_history (
                        symbol, timeframe, strategy, direction, price_level, timestamp, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """), (
                    signal['symbol'], signal['timeframe'], signal['strategy'],
                    signal['direction'], signal['entry'], signal['timestamp'], now
                ))

            print(f"✅ Position added: {signal_id}")
            return signal_id

        except Exception as e:
            print(f"❌ Failed to add position: {e}")
            return ""

    def update_position(self, signal_id: str, exit_price: float,
                        exit_reason: str = "manual"):
        """Update position with exit information and calculate PnL."""
        with get_db() as (conn, cur):
            cur.execute(q("""
                SELECT entry_price, direction, stop_loss, take_profit
                FROM positions WHERE signal_id = ?
            """), (signal_id,))

            result = cur.fetchone()
            if not result:
                print(f"❌ Position not found: {signal_id}")
                return

            entry_price, direction, sl, tp = result

            pnl = (exit_price - entry_price) if direction.lower() == "long" \
                  else (entry_price - exit_price)

            now = datetime.now().isoformat()
            cur.execute(q("""
                UPDATE positions
                SET status = 'closed', exit_price = ?, exit_time = ?,
                    pnl = ?, exit_reason = ?
                WHERE signal_id = ?
            """), (exit_price, now, pnl, exit_reason, signal_id))

        self._update_performance_metrics(pnl)
        print(f"✅ Position updated: {signal_id}, PnL: {pnl:.3f}")

    def _update_performance_metrics(self, pnl: float):
        """Update daily performance metrics."""
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().isoformat()

        with get_db() as (conn, cur):
            cur.execute(q("""
                SELECT id, total_trades, winning_trades, losing_trades,
                       total_pnl, consecutive_losses
                FROM performance_metrics WHERE date = ?
            """), (today,))

            result = cur.fetchone()

            if result:
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

                cur.execute(q("""
                    UPDATE performance_metrics
                    SET total_trades = ?, winning_trades = ?, losing_trades = ?,
                        total_pnl = ?, win_rate = ?, consecutive_losses = ?
                    WHERE id = ?
                """), (total_trades, winning_trades, losing_trades,
                       total_pnl, win_rate, consecutive_losses, metric_id))
            else:
                winning_trades = 1 if pnl > 0 else 0
                losing_trades  = 0 if pnl > 0 else 1
                consecutive_losses = 0 if pnl > 0 else 1
                win_rate = 100.0 if pnl > 0 else 0.0

                cur.execute(q("""
                    INSERT INTO performance_metrics (
                        date, total_trades, winning_trades, losing_trades,
                        total_pnl, win_rate, consecutive_losses, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """), (today, 1, winning_trades, losing_trades, pnl,
                       win_rate, consecutive_losses, now))

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        with get_db() as (conn, cur):
            cur.execute(q("""
                SELECT signal_id, symbol, timeframe, strategy, direction,
                       entry_price, stop_loss, take_profit, risk_reward, entry_time
                FROM positions WHERE status = 'open'
                ORDER BY entry_time DESC
            """))

            positions = []
            for row in cur.fetchall():
                positions.append({
                    'signal_id': row[0],
                    'symbol':    row[1],
                    'timeframe': row[2],
                    'strategy':  row[3],
                    'direction': row[4],
                    'entry':     row[5],
                    'sl':        row[6],
                    'tp':        row[7],
                    'rr':        row[8],
                    'entry_time': row[9],
                })

        return positions

    def get_performance_summary(self, days: int = 7) -> Dict:
        """Get performance summary for last N days."""
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with get_db() as (conn, cur):
            cur.execute(q("""
                SELECT SUM(total_trades), SUM(winning_trades), SUM(losing_trades),
                       SUM(total_pnl), AVG(win_rate)
                FROM performance_metrics
                WHERE date >= ?
            """), (start_date,))

            result = cur.fetchone()

        if result and result[0]:
            total_trades, winning, losing, total_pnl, avg_win_rate = result
            return {
                'total_trades':    total_trades or 0,
                'winning_trades':  winning or 0,
                'losing_trades':   losing or 0,
                'total_pnl':       total_pnl or 0.0,
                'win_rate':        avg_win_rate or 0.0,
                'profit_factor':   abs(total_pnl / losing) if losing and total_pnl else 0.0,
            }

        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'total_pnl': 0.0,  'win_rate': 0.0,    'profit_factor': 0.0,
        }

    def cleanup_old_signals(self, days: int = 30):
        """Remove old signal history entries."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with get_db() as (conn, cur):
            cur.execute(q("""
                DELETE FROM signal_history WHERE created_at < ?
            """), (cutoff_date,))
            deleted = cur.rowcount

        print(f"🧹 Cleaned up {deleted} old signal history entries")


# Global instance
position_manager = PositionManager()


if __name__ == "__main__":
    print("🧪 Testing Position Manager...")

    pm = PositionManager()

    test_signal = {
        'symbol': 'XAU/USD',
        'timeframe': 'M15',
        'strategy': 'downtrend_line_break',
        'direction': 'long',
        'entry': 3350.5,
        'sl': 3347.0,
        'tp': 3357.0,
        'rr': 1.86,
        'timestamp': datetime.now().isoformat()
    }

    test_config = {
        'max_concurrent_positions': 3,
        'cooldown_minutes': 30,
        'min_distance_atr': 0.5,
        'sl_buffer': 0.8,
        'max_daily_loss_percent': 3.0,
        'max_consecutive_losses': 5
    }

    can_send, reason = pm.can_send_signal(test_signal, test_config)
    print(f"Can send signal: {can_send}, Reason: {reason}")

    if can_send:
        signal_id = pm.add_position(test_signal, ml_confidence=0.0,
                                    market_sentiment="Bullish")
        print(f"Added position: {signal_id}")

        open_pos = pm.get_open_positions()
        print(f"Open positions: {len(open_pos)}")

        pm.update_position(signal_id, exit_price=3357.0, exit_reason="tp_hit")

        perf = pm.get_performance_summary(days=1)
        print(f"Performance: {perf}")

    print("✅ Position Manager test complete")
