import numpy as np
import math
from datetime import datetime
import pandas as pd

class RiskManager:
    """Advanced risk management with short selling support"""
    def __init__(self):
        # General risk parameters
        self.max_portfolio_risk = 0.05      # 5% maximum portfolio risk
        self.max_position_risk = 0.02       # 2% maximum position risk
        self.max_correlation = 0.75         # Maximum correlation between assets
        self.max_leverage = 1.0             # No leverage (1.0 = 100% of capital)
        
        # Position-level risk parameters
        self.long_stop_loss_pct = 0.02      # 2% stop loss for long positions
        self.long_take_profit_pct = 0.04    # 4% take profit for long positions
        self.trailing_stop_pct = 0.015      # 1.5% trailing stop
        
        # Short position specific parameters
        self.short_stop_loss_pct = 0.02     # 2% stop loss for short positions (price increases)
        self.short_take_profit_pct = 0.04   # 4% take profit for short positions (price decreases)
        self.max_short_exposure = 0.3       # Maximum 30% of portfolio in short positions
        
        # Market trend parameters
        self.bearish_threshold = -0.02      # -2% threshold to consider market bearish
        self.bullish_threshold = 0.02       # +2% threshold to consider market bullish
        
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
    
    def detect_market_trend(self, prices):
        """
        Detect market trend based on recent prices
        
        Parameters:
        - prices: List of recent prices (newest last)
        
        Returns:
        - String indicating trend: 'bullish', 'bearish', or 'neutral'
        """
        if len(prices) < 3:
            return "neutral"  # Not enough data
            
        # Convert to numpy array for easier calculations
        price_array = np.array(prices)
        
        # Calculate returns
        returns = np.diff(price_array) / price_array[:-1]
        
        # Calculate metrics
        avg_return = np.mean(returns)
        recent_returns = returns[-3:]  # Last 3 returns
        recent_avg = np.mean(recent_returns)
        
        # Check if prices are consistently rising or falling
        consistent_direction = all(r > 0 for r in recent_returns) or all(r < 0 for r in recent_returns)
        
        # Calculate moving averages if we have enough data
        if len(price_array) >= 5:
            ma5 = np.mean(price_array[-5:])
            ma10 = np.mean(price_array[-10:]) if len(price_array) >= 10 else ma5
            
            # Current price vs moving averages
            current_price = price_array[-1]
            price_above_ma5 = current_price > ma5
            price_above_ma10 = current_price > ma10
            
            # Trend detection based on multiple factors
            if recent_avg > 0.005 and consistent_direction and price_above_ma5 and price_above_ma10:
                return "bullish"
            elif recent_avg < -0.005 and consistent_direction and not price_above_ma5 and not price_above_ma10:
                return "bearish"
            
        # Simpler detection for fewer data points
        if recent_avg > 0.01 and consistent_direction:
            return "bullish"
        elif recent_avg < -0.01 and consistent_direction:
            return "bearish"
            
        return "neutral"
    
    def calculate_position_size(self, balance, price, volatility, risk_score, position_type='long'):
        """Calculate position size based on risk parameters"""
        try:
            # Base position size from portfolio risk
            max_position = balance * self.max_position_risk
            
            # Adjust for volatility
            volatility_factor = self.calculate_volatility_factor(volatility)
            
            # Adjust for risk score (0 to 1)
            risk_adjusted_size = max_position * volatility_factor * risk_score
            
            # Different size adjustments based on position type
            if position_type == 'short':
                # Be more conservative with short positions
                risk_adjusted_size *= 0.8  # 20% smaller positions for shorts
            
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
            stop_price = entry_price * (1 - self.long_stop_loss_pct)
            return current_price <= stop_price
        else:  # short position
            stop_price = entry_price * (1 + self.short_stop_loss_pct)
            return current_price >= stop_price
    
    def check_take_profit(self, entry_price, current_price, position_type='long'):
        """Check if take profit has been hit"""
        if position_type == 'long':
            target_price = entry_price * (1 + self.long_take_profit_pct)
            return current_price >= target_price
        else:  # short position
            target_price = entry_price * (1 - self.short_take_profit_pct)
            return current_price <= target_price
    
    def update_trailing_stop(self, position):
        """Update trailing stop level"""
        try:
            position_type = position.get('position_type', 'long')
            
            if position_type == 'long':
                new_stop = position['current_price'] * (1 - self.trailing_stop_pct)
                position['stop_loss'] = max(position['stop_loss'], new_stop)
            else:  # short position
                new_stop = position['current_price'] * (1 + self.trailing_stop_pct)
                position['stop_loss'] = min(position['stop_loss'], new_stop)
                
            return position
            
        except Exception as e:
            print(f"Error updating trailing stop: {str(e)}")
            return position
    
    def check_portfolio_risk(self, portfolio, short_portfolio):
        """Check overall portfolio risk metrics"""
        try:
            total_risk = 0
            positions_at_risk = []
            
            # Check long positions
            for symbol, position in portfolio.items():
                # Calculate position risk
                risk = self.calculate_position_risk(position, 'long')
                total_risk += risk
                
                if risk > self.max_position_risk:
                    positions_at_risk.append(f"{symbol} (long)")
            
            # Check short positions
            for symbol, position in short_portfolio.items():
                # Calculate position risk
                risk = self.calculate_position_risk(position, 'short')
                total_risk += risk
                
                if risk > self.max_position_risk:
                    positions_at_risk.append(f"{symbol} (short)")
            
            # Check short exposure
            short_exposure = self.calculate_short_exposure(short_portfolio)
            if short_exposure > self.max_short_exposure:
                positions_at_risk.append("Excessive short exposure")
            
            return {
                'total_risk': total_risk,
                'positions_at_risk': positions_at_risk,
                'risk_level': 'High' if total_risk > self.max_portfolio_risk else 'Normal',
                'short_exposure': short_exposure
            }
            
        except Exception as e:
            print(f"Error checking portfolio risk: {str(e)}")
            return {'total_risk': 0, 'positions_at_risk': [], 'risk_level': 'Error'}
    
    def calculate_short_exposure(self, short_portfolio):
        """Calculate the total short exposure as percentage of total capital"""
        try:
            total_short_value = sum(
                position['quantity'] * position['entry_price']
                for position in short_portfolio.values()
            )
            
            # This is a simplification - would need total capital for exact ratio
            # Using 1000 as a placeholder for total capital
            return total_short_value / 1000
            
        except Exception as e:
            print(f"Error calculating short exposure: {str(e)}")
            return 0
    
    def calculate_position_risk(self, position, position_type='long'):
        """Calculate risk for a single position"""
        try:
            # Risk based on position size and volatility
            position_value = position['quantity'] * position['entry_price']
            
            # For shorts, consider the potential for unlimited losses
            if position_type == 'short':
                # Apply a higher risk factor for short positions
                position_risk = (position_value * 1.5) / 1000  # Using 1000 as placeholder
            else:
                position_risk = position_value / 1000  # Using 1000 as placeholder
            
            return position_risk
            
        except Exception as e:
            print(f"Error calculating position risk: {str(e)}")
            return 0
    
    def should_trade(self, symbol, price, volume, volatility, market_trend, position_type='long'):
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
            
            # Check market trend for short positions
            if position_type == 'short' and market_trend != 'bearish':
                return False, f"Market not bearish for shorting ({market_trend})"
            
            # Check market trend for long positions
            if position_type == 'long' and market_trend == 'bearish':
                return False, "Bearish market for long position"
            
            return True, "Trade allowed"
            
        except Exception as e:
            print(f"Error in trade validation: {str(e)}")
            return False, "Error in validation"
    
    def check_risk_limits(self, portfolio, short_portfolio, current_prices):
        """Check if any risk limits are breached"""
        daily_pnl = self.calculate_daily_pnl(portfolio, short_portfolio, current_prices)
        position_risks = self.calculate_position_risks(portfolio, short_portfolio)
        correlation = self.calculate_portfolio_correlation(portfolio, short_portfolio)
        short_exposure = self.calculate_short_exposure(short_portfolio)
        
        return {
            'daily_loss_exceeded': daily_pnl < -self.max_daily_loss,
            'position_risk_exceeded': any(risk > self.max_position_loss for risk in position_risks),
            'correlation_exceeded': correlation > self.max_correlation,
            'short_exposure_exceeded': short_exposure > self.max_short_exposure
        }
        
    def check_trade(self, portfolio, short_portfolio, symbol, price, quantity, trade_type):
        """Validate if a trade meets risk management criteria"""
        try:
            # Check number of positions
            total_positions = len(portfolio) + len(short_portfolio)
            if trade_type in ['BUY', 'SHORT'] and total_positions >= self.max_positions:
                print(f"⚠️ Risk limit: Maximum positions ({self.max_positions}) reached")
                return False
                
            # Check concentration for long positions
            if trade_type == 'BUY':
                trade_value = price * quantity
                portfolio_value = sum(pos['quantity'] * pos['entry_price'] for pos in portfolio.values())
                
                # If first position, portfolio value might be zero
                if portfolio_value == 0:
                    return True
                    
                concentration = trade_value / (portfolio_value + trade_value)
                if concentration > self.max_concentration:
                    print(f"⚠️ Risk limit: Position concentration too high ({concentration:.2%})")
                    return False
            
            # Check short exposure
            if trade_type == 'SHORT':
                new_short_value = price * quantity
                current_short_value = sum(pos['quantity'] * pos['entry_price'] for pos in short_portfolio.values())
                total_capital = 1000  # Placeholder, should be actual total capital
                
                new_exposure = (current_short_value + new_short_value) / total_capital
                if new_exposure > self.max_short_exposure:
                    print(f"⚠️ Risk limit: Short exposure too high ({new_exposure:.2%})")
                    return False
            
            # For now, all closing trades are approved
            return True
            
        except Exception as e:
            print(f"❌ Error in risk check: {str(e)}")
            # Default to conservative stance - block trade on error
            return False
    
    def calculate_daily_pnl(self, portfolio, short_portfolio, current_prices):
        """Calculate daily P&L as percentage"""
        total_pnl = 0
        total_value = 0
        
        # Calculate P&L for long positions
        for symbol, position in portfolio.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                position_value = quantity * current_price
                position_pnl = (current_price - entry_price) * quantity
                
                total_pnl += position_pnl
                total_value += position_value
        
        # Calculate P&L for short positions
        for symbol, position in short_portfolio.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # For shorts, P&L is positive when price decreases
                position_value = quantity * entry_price  # Original position value
                position_pnl = (entry_price - current_price) * quantity
                
                total_pnl += position_pnl
                total_value += position_value
                
        return total_pnl / total_value if total_value > 0 else 0
    
    def calculate_position_risks(self, portfolio, short_portfolio):
        """Calculate risk for each position"""
        risks = []
        
        # Calculate risks for long positions
        for symbol, position in portfolio.items():
            position_size = position['quantity'] * position['entry_price']
            risks.append(position_size / 1000)  # Simplified
        
        # Calculate risks for short positions (higher risk)
        for symbol, position in short_portfolio.items():
            position_size = position['quantity'] * position['entry_price']
            risks.append((position_size * 1.5) / 1000)  # Higher risk for shorts
            
        return risks
    
    def calculate_portfolio_correlation(self, portfolio, short_portfolio):
        """Calculate average correlation between positions"""
        # This is a simplified version - in reality would use price history
        return 0.5  # Placeholder
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from price series"""
        # For future implementation
        return 0.1  # Placeholder
