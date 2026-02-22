"""
Performance Analytics Module
Calculates comprehensive trading performance metrics.
Uses PostgreSQL on Render (DATABASE_URL env var) or SQLite locally.
"""

import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from db import get_db, q


class PerformanceAnalytics:
    """Calculate and track trading performance metrics."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss   = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def calculate_average_rr(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        actual_rrs = []
        for trade in trades:
            entry      = trade.get('entry_price', 0)
            sl         = trade.get('stop_loss', 0)
            exit_price = trade.get('exit_price', 0)
            direction  = trade.get('direction', 'long')
            if entry and sl and exit_price:
                risk = abs(entry - sl)
                reward = (exit_price - entry) if direction.lower() == 'long' \
                         else (entry - exit_price)
                if risk > 0:
                    actual_rrs.append(reward / risk)
        return float(np.mean(actual_rrs)) if actual_rrs else 0.0

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        if not trades or len(trades) < 2:
            return 0.0
        returns  = [t.get('pnl', 0) / self.initial_balance for t in trades]
        avg_ret  = np.mean(returns)
        std_ret  = np.std(returns)
        if std_ret == 0:
            return 0.0
        daily_rf = risk_free_rate / 250
        sharpe   = (avg_ret - daily_rf) / std_ret
        return float(sharpe * np.sqrt(250))

    def calculate_max_drawdown(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {
                'max_drawdown': 0.0, 'max_drawdown_pct': 0.0,
                'peak_balance': self.initial_balance,
                'trough_balance': self.initial_balance,
                'recovery_trades': 0
            }
        balance  = self.initial_balance
        balances = [balance]
        for trade in trades:
            balance += trade.get('pnl', 0)
            balances.append(balance)

        peak = balances[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        peak_balance = peak
        trough_balance = peak

        for b in balances:
            if b > peak:
                peak = b
            dd     = peak - b
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd      = dd
                max_dd_pct  = dd_pct
                peak_balance   = peak
                trough_balance = b

        return {
            'max_drawdown':     round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'peak_balance':     round(peak_balance, 2),
            'trough_balance':   round(trough_balance, 2),
            'current_balance':  round(balances[-1], 2),
        }

    def calculate_expectancy(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        win_rate  = self.calculate_win_rate(trades) / 100
        win_pnls  = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        loss_pnls = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        avg_win  = float(np.mean(win_pnls))  if win_pnls  else 0
        avg_loss = abs(float(np.mean(loss_pnls))) if loss_pnls else 0
        return round((win_rate * avg_win) - ((1 - win_rate) * avg_loss), 3)

    def calculate_consecutive_stats(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {
                'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
                'current_streak': 0, 'current_streak_type': 'none'
            }
        max_wins = max_losses = cur_wins = cur_losses = 0

        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                cur_wins += 1; cur_losses = 0
                max_wins = max(max_wins, cur_wins)
            elif pnl < 0:
                cur_losses += 1; cur_wins = 0
                max_losses = max(max_losses, cur_losses)

        if cur_wins > 0:
            current_streak, streak_type = cur_wins, 'winning'
        elif cur_losses > 0:
            current_streak, streak_type = cur_losses, 'losing'
        else:
            current_streak, streak_type = 0, 'none'

        return {
            'max_consecutive_wins':   max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak':         current_streak,
            'current_streak_type':    streak_type,
        }

    def get_comprehensive_metrics(self, days: int = 30) -> Dict:
        """Get comprehensive performance metrics from database."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with get_db() as (conn, cur):
            cur.execute(q("""
                SELECT entry_price, stop_loss, take_profit, exit_price,
                       pnl, direction, entry_time, exit_time
                FROM positions
                WHERE status = 'closed' AND exit_time >= ?
                ORDER BY exit_time ASC
            """), (cutoff_date,))

            trades = [
                {
                    'entry_price': row[0], 'stop_loss': row[1],
                    'take_profit': row[2], 'exit_price': row[3],
                    'pnl': row[4], 'direction': row[5],
                    'entry_time': row[6], 'exit_time': row[7],
                }
                for row in cur.fetchall()
            ]

        if not trades:
            return {'period_days': days, 'total_trades': 0,
                    'message': 'No closed trades in this period'}

        win_rate    = self.calculate_win_rate(trades)
        pf          = self.calculate_profit_factor(trades)
        avg_rr      = self.calculate_average_rr(trades)
        sharpe      = self.calculate_sharpe_ratio(trades)
        drawdown    = self.calculate_max_drawdown(trades)
        expectancy  = self.calculate_expectancy(trades)
        consecutive = self.calculate_consecutive_stats(trades)

        total_pnl     = sum(t.get('pnl', 0) for t in trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing_trades  = sum(1 for t in trades if t.get('pnl', 0) < 0)

        avg_win  = float(np.mean([t['pnl'] for t in trades if t.get('pnl', 0) > 0])) if winning_trades else 0
        avg_loss = float(np.mean([t['pnl'] for t in trades if t.get('pnl', 0) < 0])) if losing_trades  else 0

        return {
            'period_days': days,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 3),
            'avg_loss': round(avg_loss, 3),
            'profit_factor': round(pf, 2),
            'avg_rr_achieved': round(avg_rr, 2),
            'sharpe_ratio': round(sharpe, 2),
            'expectancy': expectancy,
            'max_drawdown': drawdown['max_drawdown'],
            'max_drawdown_pct': drawdown['max_drawdown_pct'],
            'current_balance': drawdown['current_balance'],
            'max_consecutive_wins': consecutive['max_consecutive_wins'],
            'max_consecutive_losses': consecutive['max_consecutive_losses'],
            'current_streak': consecutive['current_streak'],
            'current_streak_type': consecutive['current_streak_type'],
        }

    def generate_performance_report(self, days: int = 30) -> str:
        metrics = self.get_comprehensive_metrics(days)

        if metrics.get('total_trades', 0) == 0:
            return f"📊 No trades in the last {days} days"

        report = f"""📊 <b>PERFORMANCE REPORT</b> ({days} days)

📈 <b>TRADING ACTIVITY</b>
• Total Trades: {metrics['total_trades']}
• Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}
• Win Rate: {metrics['win_rate']}%

💰 <b>PROFITABILITY</b>
• Total P&L: ${metrics['total_pnl']:.2f}
• Avg Win: ${metrics['avg_win']:.2f} | Avg Loss: ${metrics['avg_loss']:.2f}
• Profit Factor: {metrics['profit_factor']:.2f}
• Expectancy: ${metrics['expectancy']:.3f}

📊 <b>RISK METRICS</b>
• Avg RR Achieved: {metrics['avg_rr_achieved']:.2f}
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.1f}%)
• Current Balance: ${metrics['current_balance']:.2f}

🔥 <b>STREAKS</b>
• Max Consecutive Wins: {metrics['max_consecutive_wins']}
• Max Consecutive Losses: {metrics['max_consecutive_losses']}
• Current Streak: {metrics['current_streak']} {metrics['current_streak_type']}

{'✅ <b>EXCELLENT PERFORMANCE</b>' if metrics['win_rate'] >= 60 and metrics['profit_factor'] >= 1.5 else ''}
{'⚠️ <b>NEEDS IMPROVEMENT</b>' if metrics['win_rate'] < 50 or metrics['profit_factor'] < 1.0 else ''}"""
        return report.strip()


# Global instance
analytics = PerformanceAnalytics()


if __name__ == "__main__":
    print("🧪 Testing Performance Analytics...")
    pa = PerformanceAnalytics(initial_balance=10000)
    test_trades = [
        {'pnl': 12, 'entry_price': 3350, 'stop_loss': 3347, 'exit_price': 3354, 'direction': 'long'},
        {'pnl': -7, 'entry_price': 3355, 'stop_loss': 3358, 'exit_price': 3352, 'direction': 'short'},
        {'pnl': 15, 'entry_price': 3348, 'stop_loss': 3345, 'exit_price': 3353, 'direction': 'long'},
        {'pnl': 10, 'entry_price': 3352, 'stop_loss': 3349, 'exit_price': 3356, 'direction': 'long'},
        {'pnl': -8, 'entry_price': 3358, 'stop_loss': 3361, 'exit_price': 3355, 'direction': 'short'},
    ]
    print(f"Win Rate: {pa.calculate_win_rate(test_trades):.1f}%")
    print(f"Profit Factor: {pa.calculate_profit_factor(test_trades):.2f}")
    print(f"Expectancy: ${pa.calculate_expectancy(test_trades):.2f}")
    print("✅ Performance Analytics test complete")
