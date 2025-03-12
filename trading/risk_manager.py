import numpy as np
from datetime import datetime
import pandas as pd

class RiskManager:
    def __init__(self):
        # Adjusted for smaller account size
        self.max_portfolio_risk = 0.05      # 5% maximum portfolio risk
        self.max_position_risk = 0.02       # 2% maximum position risk
        self.max_correlation = 0.75         # Maximum correlation between assets
        self.max_leverage = 1.0             # No leverage (1.0 = 100% of capital)
        
        # Position-level risk parameters
        self.stop_loss_pct = 0.02          # 2% stop loss
        self.take_profit_pct = 0.04        # 4% take profit
        self.trailing_stop_pct = 0.015     # 1.5% trailing stop
        
        # Volatility-based sizing
        self.volatility_lookback = 20       # Days for volatility calculation
        self.volatility_threshold = 0.03    # 3% daily volatility threshold
        
        # Portfolio diversification
        self.max_sector_exposure = 0.30     # 30% maximum sector exposure
        self.max_single_position = 0.40     # 40% maximum single position
        
        # Trading restrictions
        self.min_trade_value = 5            # Minimum trade size $5
        self.min_volume_threshold = 100000  # Minimum daily volume in USD
        
        # Fee structure (typical crypto exchange fees)
        self.maker_fee = 0.001    # 0.1% maker fee
        self.taker_fee = 0.002    # 0.2% taker fee
        self.slippage = 0.001     # 0.1% estimated slippage
        
        # Adjusted profit targets to account for fees
        self.quick_profit_target = 0.025    # 2.5% profit target (increased to cover fees)
        self.min_profit_after_fees = 0.005  # 0.5% minimum profit after fees
        
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_position_loss = 0.01  # 1% max loss per position
        self.max_concentration = 0.2  # Maximum allocation to one asset
        self.max_positions = 10  # Maximum number of open positions
    
    def calculate_position_size(self, balance, price, volatility, risk_score):
        """Calculate position size based on risk parameters"""
        try:
            # Base position size from portfolio risk
            max_position = balance * self.max_position_risk
            
            # Adjust for volatility
            volatility_factor = self.calculate_volatility_factor(volatility)
            
            # Adjust for risk score (0 to 1)
            risk_adjusted_size = max_position * volatility_factor * risk_score
            
            # Ensure minimum trade size and maximum position size
            position_size = max(
                min(risk_adjusted_size, balance * self.max_single_position),
                self.min_trade_value
            )
            
            return position_size
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0
    
    def calculate_volatility_factor(self, volatility):
        """Calculate position sizing factor based on volatility"""
        if volatility > self.volatility_threshold:
            # Reduce position size for high volatility
            return self.volatility_threshold / volatility
        return 1.0
    
    def check_stop_loss(self, entry_price, current_price, position_type='long'):
        """Check if stop loss has been hit"""
        if position_type == 'long':
            stop_price = entry_price * (1 - self.stop_loss_pct)
            return current_price <= stop_price
        else:
            stop_price = entry_price * (1 + self.stop_loss_pct)
            return current_price >= stop_price
    
    def check_take_profit(self, entry_price, current_price, position_type='long'):
        """Check if take profit has been hit"""
        if position_type == 'long':
            target_price = entry_price * (1 + self.take_profit_pct)
            return current_price >= target_price
        else:
            target_price = entry_price * (1 - self.take_profit_pct)
            return current_price <= target_price
    
    def update_trailing_stop(self, position):
        """Update trailing stop level"""
        try:
            if position['type'] == 'long':
                new_stop = position['current_price'] * (1 - self.trailing_stop_pct)
                position['stop_loss'] = max(position['stop_loss'], new_stop)
            else:
                new_stop = position['current_price'] * (1 + self.trailing_stop_pct)
                position['stop_loss'] = min(position['stop_loss'], new_stop)
                
            return position
            
        except Exception as e:
            print(f"Error updating trailing stop: {str(e)}")
            return position
    
    def check_portfolio_risk(self, portfolio):
        """Check overall portfolio risk metrics"""
        try:
            total_risk = 0
            positions_at_risk = []
            
            for symbol, position in portfolio.items():
                # Calculate position risk
                risk = self.calculate_position_risk(position)
                total_risk += risk
                
                if risk > self.max_position_risk:
                    positions_at_risk.append(symbol)
            
            return {
                'total_risk': total_risk,
                'positions_at_risk': positions_at_risk,
                'risk_level': 'High' if total_risk > self.max_portfolio_risk else 'Normal'
            }
            
        except Exception as e:
            print(f"Error checking portfolio risk: {str(e)}")
            return {'total_risk': 0, 'positions_at_risk': [], 'risk_level': 'Error'}
    
    def calculate_position_risk(self, position):
        """Calculate risk for a single position"""
        try:
            # Risk based on position size and volatility
            position_value = position['quantity'] * position['current_price']
            position_risk = (position_value * position['volatility']) / position['portfolio_value']
            
            return position_risk
            
        except Exception as e:
            print(f"Error calculating position risk: {str(e)}")
            return 0
    
    def should_trade(self, symbol, price, volume, volatility):
        """Determine if trading is allowed based on risk parameters"""
        try:
            # Check minimum volume
            if volume < self.min_volume_threshold:
                return False, "Insufficient volume"
            
            # Check volatility
            if volatility > self.volatility_threshold * 2:
                return False, "Excessive volatility"
            
            # Check minimum trade value
            if price * 1 < self.min_trade_value:  # Minimum 1 unit
                return False, "Below minimum trade value"
            
            return True, "Trade allowed"
            
        except Exception as e:
            print(f"Error in trade validation: {str(e)}")
            return False, "Error in validation"
    
    def calculate_fees(self, price, quantity, trade_type='market'):
        """Calculate total fees for a trade"""
        trade_value = price * quantity
        
        # Use taker fee for market orders, maker fee for limit orders
        fee_rate = self.taker_fee if trade_type == 'market' else self.maker_fee
        
        # Calculate fees
        exchange_fee = trade_value * fee_rate
        slippage_cost = trade_value * self.slippage
        
        return exchange_fee + slippage_cost
    
    def calculate_profit_after_fees(self, entry_price, current_price, quantity):
        """Calculate actual profit after all fees"""
        # Calculate gross profit
        gross_profit = (current_price - entry_price) * quantity
        
        # Calculate total fees (entry + exit)
        entry_fees = self.calculate_fees(entry_price, quantity)
        exit_fees = self.calculate_fees(current_price, quantity)
        total_fees = entry_fees + exit_fees
        
        # Net profit
        net_profit = gross_profit - total_fees
        
        return net_profit
    
    def check_risk_limits(self, portfolio, current_prices):
        """Check if any risk limits are breached"""
        daily_pnl = self.calculate_daily_pnl(portfolio, current_prices)
        position_risks = self.calculate_position_risks(portfolio)
        correlation = self.calculate_portfolio_correlation(portfolio)
        
        return {
            'daily_loss_exceeded': daily_pnl < -self.max_daily_loss,
            'position_risk_exceeded': any(risk > self.max_position_loss for risk in position_risks),
            'correlation_exceeded': correlation > self.max_correlation
        }
        
    def check_trade(self, portfolio, symbol, price, quantity, trade_type):
        """Validate if a trade meets risk management criteria"""
        try:
            # Check number of positions
            if trade_type == 'BUY' and len(portfolio) >= self.max_positions:
                print(f"⚠️ Risk limit: Maximum positions ({self.max_positions}) reached")
                return False
                
            # Check concentration
            if trade_type == 'BUY':
                trade_value = price * quantity
                portfolio_value = sum(pos['quantity'] * pos['current_price'] for pos in portfolio.values())
                
                # If first position, portfolio value might be zero
                if portfolio_value == 0:
                    return True
                    
                concentration = trade_value / (portfolio_value + trade_value)
                if concentration > self.max_concentration:
                    print(f"⚠️ Risk limit: Position concentration too high ({concentration:.2%})")
                    return False
            
            # For now, all sell trades are approved (may add additional checks)
            return True
            
        except Exception as e:
            print(f"❌ Error in risk check: {str(e)}")
            # Default to conservative stance - block trade on error
            return False
    
    def calculate_daily_pnl(self, portfolio, current_prices):
        """Calculate daily P&L as percentage"""
        total_pnl = 0
        total_value = 0
        
        for symbol, position in portfolio.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                position_value = quantity * current_price
                position_pnl = (current_price - entry_price) * quantity
                
                total_pnl += position_pnl
                total_value += position_value
                
        return total_pnl / total_value if total_value > 0 else 0
    
    def calculate_position_risks(self, portfolio):
        """Calculate risk for each position"""
        risks = []
        
        for symbol, position in portfolio.items():
            # Simple risk measure based on position size
            # In reality would include volatility, etc.
            position_size = position['quantity'] * position['entry_price']
            risks.append(position_size / 1000)  # Simplified
            
        return risks
    
    def calculate_portfolio_correlation(self, portfolio):
        """Calculate average correlation between positions"""
        # This is a simplified version - in reality would use price history
        return 0.5  # Placeholder
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from price series"""
        # For future implementation
        return 0.1  # Placeholder
