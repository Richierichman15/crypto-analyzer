from datetime import datetime
import numpy as np
import pandas as pd

class TradingMonitor:
    def __init__(self):
        self.alerts = []
        self.metrics = {}
        self.max_drawdown_threshold = 0.10  # 10% drawdown alert
        self.max_exposure_threshold = 0.25   # 25% max exposure per asset
        self.correlation_threshold = 0.75    # Correlation warning threshold
        
    def monitor_portfolio(self, portfolio):
        """Monitor portfolio health and risk metrics"""
        try:
            # Check portfolio metrics
            self.check_drawdown(portfolio)
            self.check_exposure(portfolio)
            self.check_correlation(portfolio)
            
            # Store metrics
            self.metrics['timestamp'] = datetime.now()
            self.metrics['portfolio_size'] = len(portfolio)
            
        except Exception as e:
            self.generate_alert('ERROR', f"Monitoring error: {str(e)}")
    
    def check_drawdown(self, portfolio):
        """Monitor drawdown levels"""
        if not portfolio:
            return
            
        try:
            # Calculate drawdown
            equity_curve = pd.Series([pos['value'] for pos in portfolio.values()])
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            
            current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
            
            if abs(current_drawdown) > self.max_drawdown_threshold:
                self.generate_alert(
                    'WARNING',
                    f"High drawdown detected: {current_drawdown:.2%}"
                )
                
        except Exception as e:
            self.generate_alert('ERROR', f"Drawdown calculation error: {str(e)}")
    
    def check_exposure(self, portfolio):
        """Monitor position exposure"""
        if not portfolio:
            return
            
        try:
            total_value = sum(pos['value'] for pos in portfolio.values())
            
            for symbol, position in portfolio.items():
                exposure = position['value'] / total_value if total_value > 0 else 0
                
                if exposure > self.max_exposure_threshold:
                    self.generate_alert(
                        'WARNING',
                        f"High exposure in {symbol}: {exposure:.2%}"
                    )
                    
        except Exception as e:
            self.generate_alert('ERROR', f"Exposure calculation error: {str(e)}")
    
    def check_correlation(self, portfolio):
        """Monitor portfolio correlation"""
        if len(portfolio) < 2:
            return
            
        try:
            # Get returns for all positions
            returns = pd.DataFrame({
                symbol: pos['returns'] 
                for symbol, pos in portfolio.items()
                if 'returns' in pos
            })
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Check for high correlations
            high_corr = np.where(
                (corr_matrix > self.correlation_threshold) & 
                (corr_matrix < 1.0)
            )
            
            if len(high_corr[0]) > 0:
                pairs = list(zip(high_corr[0], high_corr[1]))
                for i, j in pairs:
                    self.generate_alert(
                        'INFO',
                        f"High correlation between {returns.columns[i]} and {returns.columns[j]}"
                    )
                    
        except Exception as e:
            self.generate_alert('ERROR', f"Correlation calculation error: {str(e)}")
    
    def generate_alert(self, alert_type, message):
        """Generate and store alerts"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now()
        }
        self.alerts.append(alert)
        
        # Print alert
        print(f"🚨 {alert_type}: {message}")
