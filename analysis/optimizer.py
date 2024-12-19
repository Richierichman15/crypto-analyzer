import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor

class StrategyOptimizer:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.best_params = None
        self.best_performance = None
        
    def optimize_parameters(self):
        """Find optimal strategy parameters"""
        param_grid = {
            'sma_short': range(5, 15, 2),      # Short-term MA
            'sma_long': range(20, 50, 5),      # Long-term MA
            'rsi_period': range(10, 20, 2),    # RSI period
            'volume_threshold': np.arange(1.0, 3.0, 0.2)  # Volume threshold
        }
        
        print("\n🔄 Optimizing strategy parameters...")
        
        # Generate all parameter combinations
        param_combinations = list(self.generate_param_combinations(param_grid))
        
        # Run parallel optimization
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.backtest_strategy, param_combinations))
        
        # Find best parameters
        best_idx = np.argmax([r['sharpe_ratio'] for r in results])
        self.best_params = param_combinations[best_idx]
        self.best_performance = results[best_idx]
        
        self.print_optimization_results()
        return self.best_params
    
    def generate_param_combinations(self, param_grid):
        """Generate all possible parameter combinations"""
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))
    
    def backtest_strategy(self, params):
        """Run backtest with specific parameters"""
        try:
            returns = []
            positions = []
            
            for symbol, data in self.data.items():
                # Apply strategy with current parameters
                signals = self.apply_strategy(data, params)
                
                # Calculate returns
                daily_returns = data['Close'].pct_change()
                strategy_returns = daily_returns * signals.shift(1)
                
                returns.append(strategy_returns)
                positions.append(signals)
            
            # Combine returns across all symbols
            portfolio_returns = pd.concat(returns, axis=1).mean(axis=1)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(portfolio_returns)
            
            return metrics
            
        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            return {'sharpe_ratio': -999}
    
    def apply_strategy(self, data, params):
        """Apply trading strategy with given parameters"""
        # Calculate indicators
        data['SMA_Short'] = data['Close'].rolling(window=params['sma_short']).mean()
        data['SMA_Long'] = data['Close'].rolling(window=params['sma_long']).mean()
        data['RSI'] = self.calculate_rsi(data['Close'], params['rsi_period'])
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Long signal conditions
        long_condition = (
            (data['SMA_Short'] > data['SMA_Long']) &
            (data['RSI'] < 70) &
            (data['Volume_Ratio'] > params['volume_threshold'])
        )
        
        # Short signal conditions
        short_condition = (
            (data['SMA_Short'] < data['SMA_Long']) &
            (data['RSI'] > 30) &
            (data['Volume_Ratio'] > params['volume_threshold'])
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals
    
    def calculate_performance_metrics(self, returns):
        """Calculate strategy performance metrics"""
        try:
            # Annualized Sharpe Ratio
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            
            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding(min_periods=1).max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Win Rate
            win_rate = len(returns[returns > 0]) / len(returns)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {'sharpe_ratio': -999}
    
    @staticmethod
    def calculate_rsi(prices, period):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def print_optimization_results(self):
        """Print optimization results"""
        print("\n📊 Optimization Results:")
        print("="*50)
        print("\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        print("\nPerformance Metrics:")
        print(f"Sharpe Ratio: {self.best_performance['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.best_performance['max_drawdown']:.2%}")
        print(f"Win Rate: {self.best_performance['win_rate']:.2%}")
