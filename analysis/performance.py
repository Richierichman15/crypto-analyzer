import numpy as np
import pandas as pd
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def analyze_trades(self, trades_history):
        """Analyze trading performance"""
        if not trades_history:
            return self.get_empty_metrics()
        
        df = pd.DataFrame(trades_history)
        
        self.metrics = {
            'win_rate': self.calculate_win_rate(df),
            'profit_factor': self.calculate_profit_factor(df),
            'average_win': self.calculate_average_win(df),
            'average_loss': self.calculate_average_loss(df),
            'largest_drawdown': self.calculate_max_drawdown(df),
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'sortino_ratio': self.calculate_sortino_ratio(df),
            'total_trades': len(df),
            'profitable_trades': len(df[df['profit'] > 0]),
            'losing_trades': len(df[df['profit'] < 0])
        }
        
        return self.metrics
    
    def calculate_win_rate(self, df):
        """Calculate percentage of winning trades"""
        if len(df) == 0:
            return 0.0
        winning_trades = len(df[df['profit'] > 0])
        return (winning_trades / len(df)) * 100
    
    def calculate_profit_factor(self, df):
        """Calculate ratio of gross profits to gross losses"""
        gross_profits = df[df['profit'] > 0]['profit'].sum()
        gross_losses = abs(df[df['profit'] < 0]['profit'].sum())
        return gross_profits / gross_losses if gross_losses != 0 else 0
    
    def calculate_average_win(self, df):
        """Calculate average winning trade"""
        winning_trades = df[df['profit'] > 0]
        return winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
    
    def calculate_average_loss(self, df):
        """Calculate average losing trade"""
        losing_trades = df[df['profit'] < 0]
        return losing_trades['profit'].mean() if len(losing_trades) > 0 else 0
    
    def calculate_max_drawdown(self, df):
        """Calculate maximum drawdown"""
        if 'portfolio_value' not in df.columns:
            return 0.0
        
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        return abs(drawdown.min()) * 100
    
    def calculate_sharpe_ratio(self, df):
        """Calculate Sharpe ratio"""
        if 'portfolio_value' not in df.columns:
            return 0.0
        
        daily_returns = df['portfolio_value'].pct_change()
        if len(daily_returns) < 2:
            return 0.0
            
        return np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    
    def calculate_sortino_ratio(self, df):
        """Calculate Sortino ratio"""
        if 'portfolio_value' not in df.columns:
            return 0.0
        
        daily_returns = df['portfolio_value'].pct_change()
        negative_returns = daily_returns[daily_returns < 0]
        
        if len(negative_returns) < 1:
            return 0.0
            
        return np.sqrt(252) * (daily_returns.mean() / negative_returns.std())
    
    def get_empty_metrics(self):
        """Return empty metrics when no trades"""
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0
        }
    
    def generate_report(self):
        """Generate a formatted performance report"""
        if not self.metrics:
            return "No trading data available for analysis."
        
        report = """
📊 Trading Performance Report
============================

🎯 Trade Statistics
------------------
Total Trades: {}
Win Rate: {:.2f}%
Profitable Trades: {}
Losing Trades: {}

💰 Profitability Metrics
----------------------
Profit Factor: {:.2f}
Average Win: ${:.2f}
Average Loss: ${:.2f}
Maximum Drawdown: {:.2f}%

📈 Risk Metrics
-------------
Sharpe Ratio: {:.2f}
Sortino Ratio: {:.2f}
""".format(
            self.metrics['total_trades'],
            self.metrics['win_rate'],
            self.metrics['profitable_trades'],
            self.metrics['losing_trades'],
            self.metrics['profit_factor'],
            self.metrics['average_win'],
            self.metrics['average_loss'],
            self.metrics['largest_drawdown'],
            self.metrics['sharpe_ratio'],
            self.metrics['sortino_ratio']
        )
        
        return report

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_returns = []
        
    def add_trade(self, trade):
        self.trades.append(trade)
        
    def calculate_metrics(self):
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'average_trade': self.calculate_average_trade()
        }
