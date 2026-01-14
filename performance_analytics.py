"""
Performance Analytics Module
Calculates comprehensive trading performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from position_manager import position_manager


class PerformanceAnalytics:
    """Calculate and track trading performance metrics."""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize performance analytics.
        
        Args:
            initial_balance: Starting account balance for calculations
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate percentage."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        Values > 1.0 indicate profitable strategy.
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_average_rr(self, trades: List[Dict]) -> float:
        """Calculate average risk-reward ratio achieved."""
        if not trades:
            return 0.0
        
        # Calculate actual RR for each trade
        actual_rrs = []
        for trade in trades:
            entry = trade.get('entry_price', 0)
            sl = trade.get('stop_loss', 0)
            exit_price = trade.get('exit_price', 0)
            direction = trade.get('direction', 'long')
            
            if entry and sl and exit_price:
                risk = abs(entry - sl)
                if direction.lower() == 'long':
                    reward = exit_price - entry
                else:
                    reward = entry - exit_price
                
                if risk > 0:
                    actual_rrs.append(reward / risk)
        
        return np.mean(actual_rrs) if actual_rrs else 0.0
    
    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        Higher values indicate better risk-adjusted performance.
        
        Args:
            trades: List of trade dictionaries
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        if not trades or len(trades) < 2:
            return 0.0
        
        # Calculate returns for each trade
        returns = [t.get('pnl', 0) / self.initial_balance for t in trades]
        
        # Calculate excess returns
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming ~250 trading days)
        daily_rf_rate = risk_free_rate / 250
        sharpe = (avg_return - daily_rf_rate) / std_return
        
        # Annualize Sharpe ratio
        return sharpe * np.sqrt(250)
    
    def calculate_max_drawdown(self, trades: List[Dict]) -> Dict:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        Returns dict with max_dd, max_dd_pct, and recovery info.
        """
        if not trades:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'peak_balance': self.initial_balance,
                'trough_balance': self.initial_balance,
                'recovery_trades': 0
            }
        
        # Calculate cumulative balance
        balance = self.initial_balance
        balances = [balance]
        
        for trade in trades:
            balance += trade.get('pnl', 0)
            balances.append(balance)
        
        # Find maximum drawdown
        peak = balances[0]
        max_dd = 0
        max_dd_pct = 0
        peak_balance = peak
        trough_balance = peak
        
        for i, balance in enumerate(balances):
            if balance > peak:
                peak = balance
            
            dd = peak - balance
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
                peak_balance = peak
                trough_balance = balance
        
        return {
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'peak_balance': round(peak_balance, 2),
            'trough_balance': round(trough_balance, 2),
            'current_balance': round(balances[-1], 2)
        }
    
    def calculate_expectancy(self, trades: List[Dict]) -> float:
        """
        Calculate expectancy (average amount expected to win/lose per trade).
        Positive values indicate profitable strategy.
        """
        if not trades:
            return 0.0
        
        win_rate = self.calculate_win_rate(trades) / 100
        
        winning_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return round(expectancy, 3)
    
    def calculate_consecutive_stats(self, trades: List[Dict]) -> Dict:
        """Calculate consecutive wins and losses statistics."""
        if not trades:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Determine current streak
        if current_wins > 0:
            current_streak = current_wins
            streak_type = 'winning'
        elif current_losses > 0:
            current_streak = current_losses
            streak_type = 'losing'
        else:
            current_streak = 0
            streak_type = 'none'
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current_streak,
            'current_streak_type': streak_type
        }
    
    def get_comprehensive_metrics(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance metrics from database.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with all performance metrics
        """
        # Get closed positions from database
        import sqlite3
        conn = sqlite3.connect(position_manager.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT entry_price, stop_loss, take_profit, exit_price, 
                   pnl, direction, entry_time, exit_time
            FROM positions 
            WHERE status = 'closed' AND exit_time >= ?
            ORDER BY exit_time ASC
        """, (cutoff_date,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'entry_price': row[0],
                'stop_loss': row[1],
                'take_profit': row[2],
                'exit_price': row[3],
                'pnl': row[4],
                'direction': row[5],
                'entry_time': row[6],
                'exit_time': row[7]
            })
        
        conn.close()
        
        if not trades:
            return {
                'period_days': days,
                'total_trades': 0,
                'message': 'No closed trades in this period'
            }
        
        # Calculate all metrics
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        avg_rr = self.calculate_average_rr(trades)
        sharpe = self.calculate_sharpe_ratio(trades)
        drawdown = self.calculate_max_drawdown(trades)
        expectancy = self.calculate_expectancy(trades)
        consecutive = self.calculate_consecutive_stats(trades)
        
        # Calculate basic stats
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
        
        avg_win = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]) if losing_trades > 0 else 0
        
        return {
            'period_days': days,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 3),
            'avg_loss': round(avg_loss, 3),
            'profit_factor': round(profit_factor, 2),
            'avg_rr_achieved': round(avg_rr, 2),
            'sharpe_ratio': round(sharpe, 2),
            'expectancy': expectancy,
            'max_drawdown': drawdown['max_drawdown'],
            'max_drawdown_pct': drawdown['max_drawdown_pct'],
            'current_balance': drawdown['current_balance'],
            'max_consecutive_wins': consecutive['max_consecutive_wins'],
            'max_consecutive_losses': consecutive['max_consecutive_losses'],
            'current_streak': consecutive['current_streak'],
            'current_streak_type': consecutive['current_streak_type']
        }
    
    def generate_performance_report(self, days: int = 30) -> str:
        """
        Generate formatted performance report.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Formatted report string
        """
        metrics = self.get_comprehensive_metrics(days)
        
        if metrics.get('total_trades', 0) == 0:
            return f"📊 No trades in the last {days} days"
        
        report = f"""
📊 **PERFORMANCE REPORT** ({days} days)

📈 **TRADING ACTIVITY**
• Total Trades: {metrics['total_trades']}
• Winning Trades: {metrics['winning_trades']}
• Losing Trades: {metrics['losing_trades']}
• Win Rate: {metrics['win_rate']}%

💰 **PROFITABILITY**
• Total P&L: ${metrics['total_pnl']:.2f}
• Average Win: ${metrics['avg_win']:.2f}
• Average Loss: ${metrics['avg_loss']:.2f}
• Profit Factor: {metrics['profit_factor']:.2f}
• Expectancy: ${metrics['expectancy']:.3f}

📊 **RISK METRICS**
• Average RR Achieved: {metrics['avg_rr_achieved']:.2f}
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.1f}%)
• Current Balance: ${metrics['current_balance']:.2f}

🔥 **STREAKS**
• Max Consecutive Wins: {metrics['max_consecutive_wins']}
• Max Consecutive Losses: {metrics['max_consecutive_losses']}
• Current Streak: {metrics['current_streak']} {metrics['current_streak_type']}

{'✅ **EXCELLENT PERFORMANCE**' if metrics['win_rate'] >= 60 and metrics['profit_factor'] >= 1.5 else ''}
{'⚠️ **NEEDS IMPROVEMENT**' if metrics['win_rate'] < 50 or metrics['profit_factor'] < 1.0 else ''}
"""
        return report.strip()


# Global instance
analytics = PerformanceAnalytics()


if __name__ == "__main__":
    # Test performance analytics
    print("🧪 Testing Performance Analytics...")
    
    pa = PerformanceAnalytics(initial_balance=10000)
    
    # Create test trades
    test_trades = [
        {'pnl': 50, 'entry_price': 2050, 'stop_loss': 2045, 'exit_price': 2060, 'direction': 'long'},
        {'pnl': -25, 'entry_price': 2055, 'stop_loss': 2060, 'exit_price': 2050, 'direction': 'short'},
        {'pnl': 75, 'entry_price': 2048, 'stop_loss': 2043, 'exit_price': 2063, 'direction': 'long'},
        {'pnl': 40, 'entry_price': 2052, 'stop_loss': 2047, 'exit_price': 2060, 'direction': 'long'},
        {'pnl': -30, 'entry_price': 2058, 'stop_loss': 2063, 'exit_price': 2053, 'direction': 'short'},
    ]
    
    print(f"Win Rate: {pa.calculate_win_rate(test_trades):.1f}%")
    print(f"Profit Factor: {pa.calculate_profit_factor(test_trades):.2f}")
    print(f"Average RR: {pa.calculate_average_rr(test_trades):.2f}")
    print(f"Sharpe Ratio: {pa.calculate_sharpe_ratio(test_trades):.2f}")
    print(f"Expectancy: ${pa.calculate_expectancy(test_trades):.2f}")
    
    drawdown = pa.calculate_max_drawdown(test_trades)
    print(f"Max Drawdown: ${drawdown['max_drawdown']:.2f} ({drawdown['max_drawdown_pct']:.1f}%)")
    
    consecutive = pa.calculate_consecutive_stats(test_trades)
    print(f"Max Consecutive Wins: {consecutive['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {consecutive['max_consecutive_losses']}")
    
    print("✅ Performance Analytics test complete")
