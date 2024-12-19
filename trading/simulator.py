import pandas as pd
import numpy as np
from datetime import datetime

class TradingSimulator:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = {}
        self.trades_history = []
        self.min_trade_value = 50
        self.max_position_size = 0.20  # 20% of portfolio
        
    def calculate_position_size(self, price, confidence_score, volatility):
        """Calculate position size based on confidence and volatility"""
        base_size = self.balance * self.max_position_size
        volatility_factor = 1 - (volatility * 2)  # Reduce size for higher volatility
        confidence_factor = confidence_score * 2
        
        position_size = base_size * volatility_factor * confidence_factor
        return min(position_size, self.balance * 0.20)  # Cap at 20% of balance
    
    def execute_trade(self, date, symbol, price, trade_type, quantity, prediction, fees=0):
        """Execute trade with detailed debugging"""
        try:
            print(f"\n🔄 Attempting trade execution:")
            print(f"Date: {date}")
            print(f"Symbol: {symbol}")
            print(f"Type: {trade_type}")
            print(f"Price: ${price:.4f}")
            print(f"Quantity: {quantity:.4f}")
            print(f"Prediction: {prediction:.4f}")
            print(f"Fees: ${fees:.4f}")
            
            # Check balance
            print(f"Current Balance: ${self.balance:.2f}")
            
            if trade_type == 'BUY':
                total_cost = (price * quantity) + fees
                print(f"Total Cost: ${total_cost:.2f}")
                
                if total_cost > self.balance:
                    print("❌ Insufficient balance for trade!")
                    return None
                    
                self.balance -= total_cost
                
                # Add to portfolio
                self.portfolio[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_date': date,
                    'current_price': price
                }
                
                print(f"✅ Buy executed. New balance: ${self.balance:.2f}")
                
            elif trade_type == 'SELL':
                if symbol not in self.portfolio:
                    print("❌ Symbol not in portfolio!")
                    return None
                    
                position = self.portfolio[symbol]
                total_return = (price * quantity) - fees
                
                self.balance += total_return
                del self.portfolio[symbol]
                
                print(f"✅ Sell executed. New balance: ${self.balance:.2f}")
                
            # Record trade
            trade = {
                'date': date,
                'symbol': symbol,
                'type': trade_type,
                'price': price,
                'quantity': quantity,
                'prediction': prediction,
                'fees': fees
            }
            self.trades_history.append(trade)
            
            return trade
            
        except Exception as e:
            print(f"❌ Trade execution error: {str(e)}")
            return None
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.balance
        
        for symbol, position in self.portfolio.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                portfolio_value += position_value
                
        return portfolio_value

class RiskManager:
    def __init__(self):
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_position_loss = 0.01  # 1% max loss per position
        self.max_correlation = 0.7  # Maximum correlation between positions
        
    def check_risk_limits(self, portfolio, current_prices):
        """Check if any risk limits are breached"""
        daily_pnl = self.calculate_daily_pnl(portfolio, current_prices)
        position_risks = self.calculate_position_risks(portfolio)
        correlation = self.calculate_portfolio_correlation(portfolio)
        
        return {
            'daily_loss_exceeded': abs(daily_pnl) > self.max_daily_loss,
            'position_risk_exceeded': any(risk > self.max_position_loss for risk in position_risks),
            'correlation_exceeded': correlation > self.max_correlation
        }
