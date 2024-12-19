import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradingSimulator:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = {}
        self.trades_history = []
        self.daily_portfolio_value = []
        
    def calculate_position_size(self, price, confidence_score):
        """Calculate how much to invest based on confidence score"""
        max_position_size = self.balance * 0.1  # Max 10% of balance per trade
        position_size = max_position_size * confidence_score
        return min(position_size, self.balance)
    
    def execute_trade(self, date, symbol, price, action, amount, confidence_score):
        """Execute a trade and record it"""
        if action == 'BUY':
            cost = amount * price
            if cost <= self.balance:
                self.balance -= cost
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + amount
                trade_type = 'BUY'
            else:
                return False
        
        elif action == 'SELL':
            if symbol in self.portfolio and self.portfolio[symbol] >= amount:
                self.balance += amount * price
                self.portfolio[symbol] -= amount
                if self.portfolio[symbol] == 0:
                    del self.portfolio[symbol]
                trade_type = 'SELL'
            else:
                return False
        
        # Record the trade
        self.trades_history.append({
            'date': date,
            'symbol': symbol,
            'action': trade_type,
            'price': price,
            'amount': amount,
            'confidence_score': confidence_score,
            'balance_after': self.balance,
            'portfolio_value': self.get_portfolio_value(date, {symbol: price})
        })
        
        return True
    
    def get_portfolio_value(self, date, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.balance
        for symbol, amount in self.portfolio.items():
            if symbol in current_prices:
                portfolio_value += amount * current_prices[symbol]
        
        self.daily_portfolio_value.append({
            'date': date,
            'value': portfolio_value
        })
        
        return portfolio_value
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.daily_portfolio_value:
            return {}
        
        df = pd.DataFrame(self.daily_portfolio_value)
        daily_returns = df['value'].pct_change()
        
        metrics = {
            'total_return': (df['value'].iloc[-1] - self.initial_balance) / self.initial_balance * 100,
            'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
            'max_drawdown': (df['value'].max() - df['value'].min()) / df['value'].max() * 100,
            'total_trades': len(self.trades_history),
            'final_balance': df['value'].iloc[-1],
            'profit_loss': df['value'].iloc[-1] - self.initial_balance
        }
        
        return metrics
