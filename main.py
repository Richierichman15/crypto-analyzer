from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback
import argparse
import copy

from data.data_fetcher import DataFetcher
from data.historical_data import HistoricalDataFetcher
from trading.simulator import TradingSimulator
from trading.risk_manager import RiskManager
from analysis.performance import PerformanceTracker
from monitoring.monitor import TradingMonitor
from analysis.optimizer import StrategyOptimizer
from data.market_data import MarketDataFetcher

class TradingBot:
    """Advanced Trading Bot with Machine Learning and Risk Management"""
    
    def __init__(self, initial_balance=2000.0):
        """Initialize the trading bot with enhanced configuration"""
        self.simulator = TradingSimulator(initial_balance=initial_balance)
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.monitor = TradingMonitor()
        self.learning_metrics = {
            'predictions': [],
            'accuracy': [],
            'trades': 0,
            'successful_trades': 0
        }
        self.ml_model = None
        self.ml_data = {
            'features': [],
            'targets': []
        }
        self.portfolio_history = []
        
        # Initialize the data fetchers
        self.market_data_fetcher = MarketDataFetcher()
        self.historical_data_fetcher = HistoricalDataFetcher()
        
        print("✅ Trading Bot initialized with balance: ${:.2f}".format(initial_balance))
        
    def backtest(self, start_date, end_date, provided_symbols=None):
        """
        Run a backtest on historical data with both long-only and short-selling strategies
        
        Parameters:
        - start_date: Start date for backtest (format: 'YYYYMMDD')
        - end_date: End date for backtest (format: 'YYYYMMDD')
        - provided_symbols: List of symbols to use for backtest (optional)
        """
        # Initialize metrics for normal strategy
        self.learning_metrics = {
            'predictions': 0,
            'accuracy': [],
            'trades': 0,
            'successful_trades': 0
        }
        
        # Get list of symbols to analyze
        if provided_symbols:
            symbols = provided_symbols
            print(f"🔍 Using provided symbols: {len(symbols)} symbols")
        else:
            # Fetch market data if no symbols provided
            print(f"📊 Fetching market data...")
            market_data = self.market_data_fetcher.fetch_top_coins(limit=10)
            
            if not market_data or len(market_data) == 0:
                print("❌ No market data available. Check API connectivity.")
                return
                
            symbols = [item['symbol'] for item in market_data]
            print(f"📊 Analyzing {len(symbols)} symbols from market data")
        
        # Create a new simulator for each strategy to track performance independently
        original_simulator = self.simulator
        
        # Convert date strings to datetime
        try:
            # First try with format='%Y%m%d'
            start = pd.to_datetime(start_date, format='%Y%m%d')
        except ValueError:
            # Then try with format='%Y-%m-%d'
            start = pd.to_datetime(start_date)
            
        try:
            # First try with format='%Y%m%d'
            end = pd.to_datetime(end_date, format='%Y%m%d')
        except ValueError:
            # Then try with format='%Y-%m-%d'
            end = pd.to_datetime(end_date)
            
        print(f"🔍 Running backtest from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        # Fetch historical data for all symbols
        historical_data = {}
        
        # Use mock data for testing
        print("📊 Generating mock data for testing...")
        historical_data = self.historical_data_fetcher.generate_mock_data(symbols, start, end)
            
        if not historical_data:
            print("❌ No historical data available for any symbol.")
            return
            
        print(f"📈 Backtesting with {len(historical_data)} symbols")
        
        # Get unique dates across all symbols
        all_dates = set()
        for symbol, data in historical_data.items():
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
            all_dates.update(date_range)
        
        # Sort dates
        backtest_dates = sorted(list(all_dates))
        
        # Run backtest with normal strategy
        print("\n🚀 Running backtest with long-only strategy...")
        self.simulator = TradingSimulator(initial_balance=2000)
        daily_values_normal = []
        
        for date in backtest_dates:
            if date.date() < start.date() or date.date() > end.date():
                continue
                
            self.execute_trading_day(date, historical_data)
            
            # Get current prices for portfolio valuation
            current_prices = {}
            for symbol, data in historical_data.items():
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                day_data = data[data.index.date == date.date()]
                if not day_data.empty:
                    current_prices[symbol] = day_data['Close'].iloc[-1]
            
            # Calculate portfolio value
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            total_value = portfolio_value
            daily_values_normal.append((date, total_value))
        
        # Save normal strategy metrics
        normal_strategy_metrics = {
            'final_balance': self.simulator.balance,
            'final_portfolio_value': portfolio_value,
            'final_total_value': total_value,
            'trades': self.learning_metrics['trades'],
            'successful_trades': self.learning_metrics.get('successful_trades', 0),
            'success_rate': (self.learning_metrics.get('successful_trades', 0) / 
                             self.learning_metrics['trades'] * 100) if self.learning_metrics['trades'] > 0 else 0
        }
        
        # Save ML data from normal strategy
        normal_ml_data = copy.deepcopy(self.ml_data)
        
        # Reset for short-selling strategy
        self.learning_metrics = {
            'predictions': 0,
            'accuracy': [],
            'trades': 0,
            'successful_trades': 0
        }
        
        # Run backtest with short-selling strategy
        print("\n🚀 Running backtest with short-selling strategy...")
        self.simulator = TradingSimulator(initial_balance=2000)
        daily_values_short = []
        
        for date in backtest_dates:
            if date.date() < start.date() or date.date() > end.date():
                continue
                
            self.execute_trading_day(date, historical_data)
            
            # Get current prices for portfolio valuation
            current_prices = {}
            for symbol, data in historical_data.items():
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                day_data = data[data.index.date == date.date()]
                if not day_data.empty:
                    current_prices[symbol] = day_data['Close'].iloc[-1]
            
            # Calculate portfolio value
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            total_value = portfolio_value
            daily_values_short.append((date, total_value))
        
        # Save short-selling strategy metrics
        short_strategy_metrics = {
            'final_balance': self.simulator.balance,
            'final_portfolio_value': portfolio_value,
            'final_total_value': total_value,
            'trades': self.learning_metrics['trades'],
            'successful_trades': self.learning_metrics.get('successful_trades', 0),
            'success_rate': (self.learning_metrics.get('successful_trades', 0) / 
                             self.learning_metrics['trades'] * 100) if self.learning_metrics['trades'] > 0 else 0
        }
        
        # Combine ML data from both strategies
        combined_ml_data = {
            'features': normal_ml_data.get('features', []) + self.ml_data.get('features', []),
            'targets': normal_ml_data.get('targets', []) + self.ml_data.get('targets', [])
        }
        self.ml_data = combined_ml_data
        
        # Restore original simulator
        self.simulator = original_simulator
        
        # Compare strategies
        normal_return = ((normal_strategy_metrics['final_total_value'] / 2000) - 1) * 100
        short_return = ((short_strategy_metrics['final_total_value'] / 2000) - 1) * 100
        
        print("\n📊 Strategy Comparison:")
        print(f"Long-only strategy: ${normal_strategy_metrics['final_total_value']:.2f} ({normal_return:+.2f}%)")
        print(f"Short-selling strategy: ${short_strategy_metrics['final_total_value']:.2f} ({short_return:+.2f}%)")
        
        if normal_return > short_return:
            print("🔍 Long-only strategy performed better in this period.")
        elif short_return > normal_return:
            print("🔍 Short-selling strategy performed better in this period.")
        else:
            print("🔍 Both strategies performed equally in this period.")
        
        # Print detailed metrics for both strategies
        print("\n📈 Long-only Strategy Metrics:")
        print(f"Final Balance: ${normal_strategy_metrics['final_balance']:.2f}")
        print(f"Final Portfolio Value: ${normal_strategy_metrics['final_portfolio_value']:.2f}")
        print(f"Total Trades: {normal_strategy_metrics['trades']}")
        print(f"Successful Trades: {normal_strategy_metrics['successful_trades']}")
        print(f"Success Rate: {normal_strategy_metrics['success_rate']:.2f}%")
        
        print("\n📉 Short-selling Strategy Metrics:")
        print(f"Final Balance: ${short_strategy_metrics['final_balance']:.2f}")
        print(f"Final Portfolio Value: ${short_strategy_metrics['final_portfolio_value']:.2f}")
        print(f"Total Trades: {short_strategy_metrics['trades']}")
        print(f"Successful Trades: {short_strategy_metrics['successful_trades']}")
        print(f"Success Rate: {short_strategy_metrics['success_rate']:.2f}%")
        
        # Train ML model with combined data
        print("\n🧠 Training ML model with combined data from both strategies...")
        self.train_ml_model()
        
        return {
            'normal_strategy': normal_strategy_metrics,
            'short_strategy': short_strategy_metrics,
            'normal_daily_values': daily_values_normal,
            'short_daily_values': daily_values_short
        }
    
    def get_current_day_data(self, historical_data, date):
        """
        Get data for the current trading day, including historical context for indicators
        
        Parameters:
        - historical_data: Dictionary of historical data frames
        - date: Current date
        
        Returns:
        - DataFrame with data for the current day and all symbols
        """
        current_data = pd.DataFrame()
        
        # For each symbol, get data up to the current date
        for symbol, df in historical_data.items():
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Get all data up to and including the current date
            # This ensures we have enough history for indicators
            symbol_data = df[df.index <= date].copy()
            
            # Only include if we have data for the current date
            if not symbol_data.empty and date in symbol_data.index:
                # Add to the current data
                current_data = pd.concat([current_data, symbol_data])
        
        return current_data
    
    def make_predictions(self, current_data):
        """Make predictions with machine learning model integration"""
        try:
            predictions = []
            print("\n🔮 Making predictions...")
            
            # Prepare features for ML model
            ml_features = []
            symbols = []
            
            for symbol in current_data['symbol'].unique():
                try:
                    # Get symbol data
                    symbol_data = current_data[current_data['symbol'] == symbol].copy()
                    
                    if len(symbol_data) < 2:
                        print(f"⚠️ Insufficient data for {symbol}, skipping")
                        predictions.append(0)
                        continue
                    
                    price = symbol_data['Close'].iloc[-1]
                    
                    # Calculate features
                    price_change = symbol_data['Close'].pct_change().iloc[-1]
                    volume_change = symbol_data['Volume'].pct_change().iloc[-1]
                    
                    # Calculate RSI
                    delta = symbol_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                    
                    # Calculate MACD
                    ema12 = symbol_data['Close'].ewm(span=12, adjust=False).mean()
                    ema26 = symbol_data['Close'].ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9, adjust=False).mean()
                    current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
                    current_macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
                    
                    # Create feature vector
                    features = [
                        price,
                        price_change,
                        volume_change,
                        current_rsi,
                        current_macd,
                        current_macd_signal
                    ]
                    
                    ml_features.append(features)
                    symbols.append(symbol)
                    
                    # For now, use a rule-based prediction
                    # This will be replaced by ML model once we have enough training data
                    prediction = 0.0
                    signal_summary = []
                    
                    # Buy signals
                    if price_change > 0.001:  # Price up 0.1%
                        prediction += 0.05
                        signal_summary.append("Price+")
                        
                    if volume_change > 0.01:  # Volume up 1%
                        prediction += 0.05
                        signal_summary.append("Vol+")
                    
                    if current_rsi < 30:  # Oversold
                        prediction += 0.1
                        signal_summary.append("RSI<30")
                        
                    if current_macd > current_macd_signal:  # MACD crossover
                        prediction += 0.1
                        signal_summary.append("MACD+")
                        
                    # Sell signals
                    if price_change < -0.001:  # Price down 0.1%
                        prediction -= 0.05
                        signal_summary.append("Price-")
                        
                    if volume_change < -0.01:  # Volume down 1%
                        prediction -= 0.05
                        signal_summary.append("Vol-")
                    
                    if current_rsi > 70:  # Overbought
                        prediction -= 0.1
                        signal_summary.append("RSI>70")
                        
                    if current_macd < current_macd_signal:  # MACD crossover down
                        prediction -= 0.1
                        signal_summary.append("MACD-")
                    
                    # Only show prediction if there's a significant signal
                    if abs(prediction) >= 0.1:
                        signal_str = ", ".join(signal_summary)
                        print(f"{symbol}: ${price:.4f} | Signals: {signal_str} | Prediction: {prediction:.4f}")
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {str(e)}")
                    predictions.append(0)
            
            # If we have collected enough historical data (from previous runs)
            # We could train and use an XGBoost model here
            if hasattr(self, 'ml_model') and len(ml_features) > 0:
                try:
                    from xgboost import XGBRegressor
                    
                    # Use ML model for prediction
                    ml_predictions = self.ml_model.predict(np.array(ml_features))
                    
                    # Combine rule-based and ML predictions (gradually increase ML weight)
                    print("\n🧠 Enhancing predictions with ML model...")
                    ml_weight = min(0.5, len(self.learning_metrics['predictions']) / 1000)
                    print(f"ML model weight: {ml_weight:.2f}")
                    
                    for i in range(len(predictions)):
                        old_pred = predictions[i]
                        ml_pred = ml_predictions[i]
                        combined_prediction = (old_pred * (1 - ml_weight)) + (ml_pred * ml_weight)
                        predictions[i] = combined_prediction
                        
                        # Only show significant changes
                        if abs(combined_prediction - old_pred) > 0.05 and abs(combined_prediction) >= 0.1:
                            print(f"{symbols[i]}: Rule: {old_pred:.2f} → ML: {combined_prediction:.2f}")
                except Exception as e:
                    print(f"❌ Error using ML model: {str(e)}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error in make_predictions: {str(e)}")
            traceback.print_exc()
            return [0] * len(current_data['symbol'].unique())
    
    def calculate_technical_indicators(self, symbol_data):
        """Calculate all necessary technical indicators"""
        try:
            # RSI
            delta = symbol_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['RSI'] = 100 - (100 / (1 + rs))

            # Moving Averages
            symbol_data['SMA_5'] = symbol_data['Close'].rolling(window=5).mean()
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()

            # MACD
            symbol_data['EMA_12'] = symbol_data['Close'].ewm(span=12, adjust=False).mean()
            symbol_data['EMA_26'] = symbol_data['Close'].ewm(span=26, adjust=False).mean()
            symbol_data['MACD'] = symbol_data['EMA_12'] - symbol_data['EMA_26']
            symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9, adjust=False).mean()

            # Fill NaN values to avoid issues in calculations
            symbol_data.fillna(method='bfill', inplace=True)
            symbol_data.fillna(method='ffill', inplace=True)

            return symbol_data

        except Exception as e:
            print(f"❌ Error calculating technical indicators for symbol: {e}")
            return symbol_data
    
    def analyze_trends(self, data):
        """Analyze multiple timeframe trends"""
        try:
            trends = []
            
            # Short-term trend (5-day)
            if 'Close' in data and 'SMA_5' in data:
                short_trend = 1 if data['Close'] > data['SMA_5'] else -1
                trends.append(short_trend * 0.4)  # 40% weight
            
            # Medium-term trend (20-day)
            if 'SMA_20' in data:
                medium_trend = 1 if data['Close'] > data['SMA_20'] else -1
                trends.append(medium_trend * 0.3)  # 30% weight
            
            # Long-term trend (50-day)
            if 'SMA_50' in data:
                long_trend = 1 if data['Close'] > data['SMA_50'] else -1
                trends.append(long_trend * 0.3)  # 30% weight
            
            # Combine trend signals
            return sum(trends) if trends else 0
            
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")
            return 0
    
    def analyze_volume_trend(self, data):
        """Analyze volume trends for confirmation"""
        try:
            volume_signals = []
            
            # Volume increase/decrease
            if 'Volume' in data and 'Volume_MA' in data:
                vol_change = (data['Volume'] / data['Volume_MA']) - 1
                volume_signals.append(np.clip(vol_change, -1, 1))
            
            # Price-volume correlation
            if 'Daily_Return' in data:
                price_direction = np.sign(data['Daily_Return'])
                volume_direction = 1 if data['Volume'] > data['Volume_MA'] else -1
                correlation = price_direction * volume_direction
                volume_signals.append(correlation)
            
            return np.mean(volume_signals) if volume_signals else 0
            
        except Exception as e:
            print(f"Error in volume analysis: {str(e)}")
            return 0
    
    def analyze_momentum(self, data):
        """Analyze price momentum"""
        try:
            momentum_signals = []
            
            # RSI
            if 'RSI' in data:
                rsi = data['RSI']
                # Convert RSI to -1 to 1 scale
                rsi_signal = (rsi - 50) / 50
                momentum_signals.append(rsi_signal)
            
            # Price momentum
            if 'Daily_Return' in data:
                momentum = np.clip(data['Daily_Return'] * 10, -1, 1)  # Scale returns
                momentum_signals.append(momentum)
            
            # MACD
            if all(x in data for x in ['MACD', 'MACD_Signal']):
                macd_hist = data['MACD'] - data['MACD_Signal']
                macd_signal = np.clip(macd_hist, -1, 1)
                momentum_signals.append(macd_signal)
            
            return np.mean(momentum_signals) if momentum_signals else 0
            
        except Exception as e:
            print(f"Error in momentum analysis: {str(e)}")
            return 0
    
    def execute_trading_day(self, date, historical_data):
        """Execute trades with enhanced strategy including ML, short selling, and improved risk management"""
        try:
            current_data = self.get_current_day_data(historical_data, date)
            if current_data.empty:
                print("❌ No data available for this date")
                return
                
            # Get symbols available on this day
            symbols_today = []
            for symbol in current_data['symbol'].unique():
                symbol_data = current_data[current_data['symbol'] == symbol]
                if date in symbol_data.index:
                    symbols_today.append(symbol)
                    
            if not symbols_today:
                print("❌ No symbols available for this date")
                return
                
            print(f"📊 Data available for {len(symbols_today)}/{len(historical_data)} symbols")

            # Create current prices dictionary for portfolio valuation
            current_prices = {}
            for symbol in symbols_today:
                symbol_data = current_data[(current_data['symbol'] == symbol) & (current_data.index == date)]
                if not symbol_data.empty:
                    current_prices[symbol] = symbol_data['Close'].iloc[0]
            
            # Check portfolio risk limits
            portfolio_reduced = False
            if self.simulator.portfolio or self.simulator.short_portfolio:
                risk_status = self.risk_manager.check_risk_limits(
                    self.simulator.portfolio, 
                    self.simulator.short_portfolio,
                    current_prices
                )
                
                if any(risk_status.values()):
                    risk_issues = [k.replace('_', ' ').title() for k, v in risk_status.items() if v]
                    print(f"⚠️ Risk limits exceeded: {', '.join(risk_issues)}")
                    
                    # Force liquidate most risky positions if needed
                    if risk_status.get('daily_loss_exceeded', False):
                        # Liquidate long positions
                        long_positions_liquidated = []
                        for symbol in list(self.simulator.portfolio.keys()):  # Use list to avoid modification during iteration
                            if symbol in current_prices:
                                price = current_prices[symbol]
                                position = self.simulator.portfolio[symbol]
                                entry_price = position['entry_price']
                                profit_pct = (price - entry_price) / entry_price * 100
                                
                                # Liquidate losing positions first
                                if profit_pct < 0:
                                    trade = self.simulator.execute_trade(
                                        date, symbol, price, 'SELL',
                                        position['quantity'], profit_pct/100
                                    )
                                    
                                    if trade:
                                        long_positions_liquidated.append(f"{symbol} ({profit_pct:.1f}%)")
                                        portfolio_reduced = True
                        
                        # Cover short positions
                        short_positions_liquidated = []
                        for symbol in list(self.simulator.short_portfolio.keys()):
                            if symbol in current_prices:
                                price = current_prices[symbol]
                                position = self.simulator.short_portfolio[symbol]
                                entry_price = position['entry_price']
                                profit_pct = (entry_price - price) / entry_price * 100
                                
                                # Cover losing short positions first
                                if profit_pct < 0:
                                    trade = self.simulator.execute_trade(
                                        date, symbol, price, 'COVER',
                                        position['quantity'], profit_pct/100
                                    )
                                    
                                    if trade:
                                        short_positions_liquidated.append(f"{symbol} ({profit_pct:.1f}%)")
                                        portfolio_reduced = True
                        
                        if long_positions_liquidated:
                            print(f"🔄 Risk management liquidated long: {', '.join(long_positions_liquidated)}")
                        if short_positions_liquidated:
                            print(f"🔄 Risk management covered shorts: {', '.join(short_positions_liquidated)}")

            # Analyze market trend for shorting opportunities
            market_trend = "neutral"
            
            # Get list of available symbols from current data
            available_symbols = current_data['symbol'].unique().tolist()
            
            # Use our updated market trend detection function with available symbols
            market_trend = self.risk_manager.detect_market_trend(available_symbols, historical_data)
            print(f"📈 Market trend: {market_trend.upper()}")

            # Calculate technical indicators and predictions
            ml_features = []
            symbols_analyzed = []

            # Process all symbols to gather data for prediction
            for symbol in current_data['symbol'].unique():
                try:
                    symbol_data = current_data[current_data['symbol'] == symbol].copy()
                    
                    if len(symbol_data) < 2:
                        print(f"⚠️ Not enough data for {symbol}: {len(symbol_data)} rows")
                        continue

                    print(f"\n📊 Processing {symbol} data:")
                    print(f"  - Data rows: {len(symbol_data)}")
                    
                    # Calculate indicators
                    # Calculate RSI
                    delta = symbol_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    symbol_data['RSI'] = 100 - (100 / (1 + rs))

                    # Calculate moving averages
                    symbol_data['SMA_5'] = symbol_data['Close'].rolling(window=5).mean()
                    symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()

                    # Calculate MACD
                    symbol_data['EMA_12'] = symbol_data['Close'].ewm(span=12, adjust=False).mean()
                    symbol_data['EMA_26'] = symbol_data['Close'].ewm(span=26, adjust=False).mean()
                    symbol_data['MACD'] = symbol_data['EMA_12'] - symbol_data['EMA_26']
                    symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9, adjust=False).mean()

                    # Get current values
                    price = symbol_data['Close'].iloc[-1]
                    rsi = symbol_data['RSI'].iloc[-1] if not pd.isna(symbol_data['RSI'].iloc[-1]) else 50
                    macd = symbol_data['MACD'].iloc[-1] if not pd.isna(symbol_data['MACD'].iloc[-1]) else 0
                    macd_signal = symbol_data['MACD_Signal'].iloc[-1] if not pd.isna(symbol_data['MACD_Signal'].iloc[-1]) else 0
                    price_change = symbol_data['Close'].pct_change().iloc[-1] if not pd.isna(symbol_data['Close'].pct_change().iloc[-1]) else 0
                    volume_change = symbol_data['Volume'].pct_change().iloc[-1] if not pd.isna(symbol_data['Volume'].pct_change().iloc[-1]) else 0

                    print(f"  - Price: ${price:.2f}")
                    print(f"  - RSI: {rsi:.2f}")
                    print(f"  - MACD: {macd:.4f}")
                    print(f"  - Price change: {price_change:.4f}")
                    print(f"  - Volume change: {volume_change:.4f}")
                    
                    # Create feature vector for ML model
                    features = [
                        price,
                        price_change,
                        volume_change,
                        rsi,
                        macd,
                        macd_signal
                    ]
                    
                    ml_features.append(features)
                    symbols_analyzed.append(symbol)
                    print(f"  ✅ Features extracted successfully")

                except Exception as e:
                    print(f"❌ Error processing {symbol}: {str(e)}")
                    continue

            # Make predictions
            predictions = []
            
            # Attempt ML predictions if model exists and we have features
            if hasattr(self, 'long_ml_model') and hasattr(self, 'short_ml_model') and ml_features:
                try:
                    # Use our new prediction function with market trend awareness
                    ml_predictions = []
                    for i, features in enumerate(ml_features):
                        ml_pred = self.predict_with_ml(features, market_trend)
                        if ml_pred is not None:
                            ml_predictions.append(ml_pred)
                        else:
                            # Fall back to rule-based if ML fails
                            rule_prediction = self.calculate_rule_based_prediction(
                                price=ml_features[i][0],
                                price_change=ml_features[i][1],
                                volume_change=ml_features[i][2],
                                rsi=ml_features[i][3],
                                macd=ml_features[i][4],
                                macd_signal=ml_features[i][5],
                                market_trend=market_trend
                            )
                            ml_predictions.append(rule_prediction)
                    
                    # Create predictions with symbols
                    for i, symbol in enumerate(symbols_analyzed):
                        if i < len(ml_predictions):
                            predictions.append((symbol, ml_predictions[i]))
                
                except Exception as e:
                    print(f"⚠️ ML prediction error: {str(e)}")
                    # Continue with rule-based fallback
            
            # Fall back to rule-based if no ML or if ML failed
            if not predictions and ml_features:
                print("⚠️ Using rule-based predictions as fallback")
                print(f"  - ML features available: {len(ml_features)}")
                print(f"  - Symbols analyzed: {len(symbols_analyzed)}")
                
                for i, symbol in enumerate(symbols_analyzed):
                    print(f"\n🔮 Calculating rule-based prediction for {symbol}")
                    rule_prediction = self.calculate_rule_based_prediction(
                        price=ml_features[i][0],
                        price_change=ml_features[i][1],
                        volume_change=ml_features[i][2],
                        rsi=ml_features[i][3],
                        macd=ml_features[i][4],
                        macd_signal=ml_features[i][5],
                        market_trend=market_trend
                    )
                    predictions.append((symbol, rule_prediction))
                    print(f"  ✅ Added prediction for {symbol}: {rule_prediction:.4f}")
                    
            # Log initial status
            print(f"\n📅 Trading day: {date.strftime('%Y-%m-%d')} | Market trend: {market_trend.upper()}")
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            print(f"💰 Starting balance: ${self.simulator.balance:.2f} | Portfolio value: ${portfolio_value:.2f}")
            
            # Trading activity trackers
            buys_executed = []
            sells_executed = []
            shorts_executed = []
            covers_executed = []
            trading_activity = False
            
            # Debug: Print all predictions
            print("\n🔮 Predictions:")
            for symbol, prediction in predictions:
                price = current_prices.get(symbol, 0)
                if price > 0:
                    signal_str = f"{symbol} @ ${price:.2f}: "
                    signal_str += f"{'BUY 📈' if prediction > 0.1 else 'SELL 📉' if prediction < -0.1 else 'HOLD ⏸️'}"
                    signal_str += f" (Signal: {prediction:.4f})"
                    print(signal_str)
            
            for symbol, prediction in predictions:
                # Skip if prediction is too weak or no price data
                price = current_prices.get(symbol, 0)
                if abs(prediction) < 0.1 or price == 0:
                    continue
                
                # Check existing positions
                has_long = symbol in self.simulator.portfolio
                has_short = symbol in self.simulator.short_portfolio
                
                # Debug: Print trade evaluation
                print(f"\n🔍 Evaluating trade for {symbol}:")
                print(f"  - Price: ${price:.2f}")
                print(f"  - Prediction: {prediction:.4f}")
                print(f"  - Market trend: {market_trend}")
                print(f"  - Has long position: {has_long}")
                print(f"  - Has short position: {has_short}")
                
                # Long position logic (buy)
                if prediction > 0.1 and not has_long and not has_short and market_trend != "bearish":
                    print(f"  ✅ Buy conditions met")
                    # More aggressive position sizing for bullish trend + strong signal
                    if market_trend == "bullish" and prediction > 0.15:
                        # Use up to 15% of balance for strong signals in bullish trend
                        position_size = self.simulator.balance * (0.03 + (prediction * 0.5))
                        position_size = min(position_size, self.simulator.balance * 0.15)
                    else:
                        # Standard sizing: 3% base + up to 5% based on confidence
                        position_size = self.simulator.balance * (0.03 + (prediction * 0.25))
                        position_size = min(position_size, self.simulator.balance * 0.08)
                    
                    # Enforce minimum trade size of $10
                    if position_size < 10:
                        position_size = min(10, self.simulator.balance * 0.05)
                    
                    print(f"  - Position size: ${position_size:.2f}")
                    
                    # Calculate quantity and check additional risk factors
                    quantity = position_size / price
                    
                    # Additional risk check
                    ok_to_trade = self.risk_manager.check_trade(
                        self.simulator.portfolio, 
                        self.simulator.short_portfolio,
                        symbol, 
                        price, 
                        quantity, 
                        'BUY'
                    )
                    
                    print(f"  - Risk check passed: {ok_to_trade}")
                    
                    if ok_to_trade:
                        print(f"  🔄 Executing BUY trade")
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'BUY',
                            quantity, prediction
                        )
                        
                        if trade:
                            trading_activity = True
                            self.learning_metrics['trades'] += 1
                            buys_executed.append(f"{symbol} (${price:.2f}, ${position_size:.2f})")
                            
                            # Store features and target for ML model
                            # Target will be updated when position is closed
                            for i, sym in enumerate(symbols_analyzed):
                                if sym == symbol:
                                    self.ml_data['features'].append(ml_features[i])
                                    self.ml_data['targets'].append(0)  # Placeholder until we sell
                
                # Short position logic (short)
                elif prediction < -0.1 and not has_short and not has_long and market_trend == "bearish":
                    print(f"  ✅ Short conditions met")
                    # More aggressive position sizing for bearish trend + strong signal
                    if market_trend == "bearish" and prediction < -0.15:
                        # Use up to 15% of balance for strong signals in bearish trend
                        position_size = self.simulator.balance * (0.03 + (abs(prediction) * 0.5))
                        position_size = min(position_size, self.simulator.balance * 0.15)
                    else:
                        # Standard sizing: 3% base + up to 5% based on confidence
                        position_size = self.simulator.balance * (0.03 + (abs(prediction) * 0.25))
                        position_size = min(position_size, self.simulator.balance * 0.08)
                    
                    # Enforce minimum trade size of $10
                    if position_size < 10:
                        position_size = min(10, self.simulator.balance * 0.05)
                    
                    print(f"  - Position size: ${position_size:.2f}")
                    
                    # Calculate quantity
                    quantity = position_size / price
                    
                    # Additional risk check (relaxed for shorts in bearish markets)
                    ok_to_trade = self.risk_manager.check_trade(
                        self.simulator.portfolio, 
                        self.simulator.short_portfolio,
                        symbol, 
                        price, 
                        quantity, 
                        'SHORT',
                        strict_check=False  # Relax check for shorts in bearish market
                    )
                    
                    print(f"  - Risk check passed: {ok_to_trade}")
                    
                    if ok_to_trade:
                        print(f"  🔄 Executing SHORT trade")
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'SHORT',
                            quantity, abs(prediction)
                        )
                        
                        if trade:
                            trading_activity = True
                            self.learning_metrics['trades'] += 1
                            shorts_executed.append(f"{symbol} (${price:.2f}, ${position_size:.2f})")
                            
                            # Store features and target for ML model (short)
                            for i, sym in enumerate(symbols_analyzed):
                                if sym == symbol:
                                    self.ml_data['features'].append(ml_features[i])
                                    self.ml_data['targets'].append(0)  # Placeholder until we cover
                
                # Manage existing long positions with improved exit strategies
                elif has_long:
                    position = self.simulator.portfolio[symbol]
                    entry_price = position['entry_price']
                    profit_pct = (price - entry_price) / entry_price * 100
                    
                    # Get dynamic exit thresholds from risk manager
                    take_profit_threshold = self.risk_manager.take_profit_pct
                    stop_loss_threshold = -self.risk_manager.stop_loss_pct
                    
                    # Should we sell? Collect all exit signals
                    sell_signals = []
                    
                    # ML signal turned strongly negative
                    if prediction < -0.15:
                        sell_signals.append(f"Strong sell signal ({prediction:.2f})")
                    
                    # Market trend turned bearish
                    if market_trend == "bearish":
                        sell_signals.append("Bearish market")
                    
                    # Take profit triggered
                    if profit_pct >= take_profit_threshold:
                        sell_signals.append(f"Take profit {profit_pct:.1f}% > {take_profit_threshold:.1f}%")
                    
                    # Stop loss triggered - stricter in bearish markets
                    adjusted_stop_loss = stop_loss_threshold * (1.2 if market_trend == "bearish" else 1.0)
                    if profit_pct <= adjusted_stop_loss:
                        sell_signals.append(f"Stop loss {profit_pct:.1f}% < {adjusted_stop_loss:.1f}%")
                    
                    # Calculate holding period
                    entry_date = position['entry_date']
                    if hasattr(entry_date, 'date'):
                        days_held = (date.date() - entry_date.date()).days
                    else:
                        days_held = 0
                    
                    # Exit based on time - shorter holding in bearish markets
                    max_hold_days = 5 if market_trend == "bearish" else 7
                    if days_held >= max_hold_days and profit_pct < 1.0:
                        sell_signals.append(f"Time limit ({days_held}d)")
                    
                    # Exit if we hit any exit condition
                    if sell_signals:
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'SELL',
                            position['quantity'], profit_pct/100
                        )
                        
                        if trade:
                            trading_activity = True
                            # Update ML training data with actual profit/loss
                            for i, features in enumerate(self.ml_data['features']):
                                # Simplified matching - in real system would need position ID
                                if self.ml_data['targets'][i] == 0:  # Placeholder target
                                    self.ml_data['targets'][i] = profit_pct / 100  # Update target with actual profit
                                    break
                            
                            # Record the trade
                            sell_msg = f"{symbol} ({profit_pct:+.1f}%, reason: {', '.join(sell_signals)})"
                            sells_executed.append(sell_msg)
                            
                            # Update metrics for learning
                            if profit_pct > 0:
                                self.learning_metrics['successful_trades'] += 1
                                self.learning_metrics['accuracy'].append(1)
                            else:
                                self.learning_metrics['accuracy'].append(0)
                
                # Manage existing short positions with improved exit strategies
                elif has_short:
                    position = self.simulator.short_portfolio[symbol]
                    entry_price = position['entry_price']
                    # For shorts, profit is when price goes down
                    profit_pct = (entry_price - price) / entry_price * 100
                    
                    # Get dynamic exit thresholds from risk manager - more aggressive for shorts
                    take_profit_threshold = self.risk_manager.short_take_profit_pct
                    stop_loss_threshold = -self.risk_manager.short_stop_loss_pct
                    
                    # Should we cover the short? Collect all exit signals
                    cover_signals = []
                    
                    # ML signal turned strongly positive
                    if prediction > 0.15:
                        cover_signals.append(f"Strong buy signal ({prediction:.2f})")
                    
                    # Market trend is no longer bearish
                    if market_trend != "bearish":
                        cover_signals.append(f"Market trend ({market_trend})")
                    
                    # Take profit triggered - be more aggressive on covering profitable shorts
                    if profit_pct >= take_profit_threshold:
                        cover_signals.append(f"Take profit {profit_pct:.1f}% > {take_profit_threshold:.1f}%")
                    
                    # Stop loss triggered - be more tolerant for shorts in bearish markets
                    adjusted_stop_loss = stop_loss_threshold * (0.8 if market_trend == "bearish" else 1.0)
                    if profit_pct <= adjusted_stop_loss:
                        cover_signals.append(f"Stop loss {profit_pct:.1f}% < {adjusted_stop_loss:.1f}%")
                    
                    # Calculate holding period
                    entry_date = position['entry_date']
                    if hasattr(entry_date, 'date'):
                        days_held = (date.date() - entry_date.date()).days
                    else:
                        days_held = 0
                    
                    # Cover based on time - longer holding allowed in bearish markets
                    max_hold_days = 6 if market_trend == "bearish" else 4
                    if days_held >= max_hold_days and profit_pct < 1.0:
                        cover_signals.append(f"Time limit ({days_held}d)")
                    
                    # Cover if we hit any exit condition
                    if cover_signals:
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'COVER',
                            position['quantity'], profit_pct/100
                        )
                        
                        if trade:
                            trading_activity = True
                            # Update ML training data with actual profit/loss
                            for i, features in enumerate(self.ml_data['features']):
                                if self.ml_data['targets'][i] == 0:  # Placeholder target
                                    self.ml_data['targets'][i] = profit_pct / 100  # Update target with actual profit
                                    break
                            
                            # Record the trade
                            cover_msg = f"{symbol} ({profit_pct:+.1f}%, reason: {', '.join(cover_signals)})"
                            covers_executed.append(cover_msg)
                            
                            # Update metrics for learning
                            if profit_pct > 0:
                                self.learning_metrics['successful_trades'] += 1
                                self.learning_metrics['accuracy'].append(1)
                            else:
                                self.learning_metrics['accuracy'].append(0)

            # Show trading activity summary
            if buys_executed:
                print(f"🛒 Buys: {', '.join(buys_executed)}")
            if sells_executed:
                print(f"💰 Sells: {', '.join(sells_executed)}")
            if shorts_executed:
                print(f"📉 Shorts: {', '.join(shorts_executed)}")
            if covers_executed:
                print(f"📈 Covers: {', '.join(covers_executed)}")
            
            # Only show portfolio summary if we had trading activity or risk management
            if trading_activity:
                portfolio_value = self.simulator.get_portfolio_value(current_prices)
                total_value = portfolio_value
                profit_pct = ((total_value / self.simulator.initial_balance) - 1) * 100
                
                long_positions = len(self.simulator.portfolio)
                short_positions = len(self.simulator.short_portfolio)
                
                print(f"📊 Portfolio: ${portfolio_value:.2f} | Cash: ${self.simulator.balance:.2f}")
                print(f"Total Value: ${total_value:.2f} ({profit_pct:+.2f}%)")
                print(f"Positions: {long_positions} long, {short_positions} short")

        except Exception as e:
            print(f"❌ Error in execute_trading_day: {str(e)}")
            traceback.print_exc()
            
    def calculate_rule_based_prediction(self, price, price_change, volume_change, rsi, macd, macd_signal, market_trend="neutral"):
        """Calculate rule-based prediction score (positive for long, negative for short)"""
        prediction = 0.0
        
        print(f"\n🧮 Rule-based prediction inputs:")
        print(f"  - Price: ${price:.2f}")
        print(f"  - Price change: {price_change:.4f}")
        print(f"  - Volume change: {volume_change:.4f}")
        print(f"  - RSI: {rsi:.2f}")
        print(f"  - MACD: {macd:.4f}")
        print(f"  - MACD Signal: {macd_signal:.4f}")
        print(f"  - Market trend: {market_trend}")
        
        # RSI signals (oversold/overbought)
        if rsi < 30:  # Oversold - bullish signal
            prediction += 0.1
            print(f"  ✅ RSI oversold signal: +0.1")
        elif rsi > 70:  # Overbought - bearish signal
            prediction -= 0.1
            print(f"  ✅ RSI overbought signal: -0.1")
        
        # MACD signals
        if macd > 0 and macd > macd_signal:  # Bullish
            prediction += 0.15
            print(f"  ✅ MACD bullish signal: +0.15")
        elif macd < 0 and macd < macd_signal:  # Bearish
            prediction -= 0.15
            print(f"  ✅ MACD bearish signal: -0.15")
        
        # Price movement signals
        if price_change > 0.02:  # Strong up move
            # In a bullish trend, this is continuation; in bearish, could be a reversal
            if market_trend == "bullish":
                prediction += 0.1  # Higher weight in bullish market
                print(f"  ✅ Strong up move in bullish market: +0.1")
            else:
                prediction += 0.05  # Lower weight in bearish market
                print(f"  ✅ Strong up move in non-bullish market: +0.05")
        elif price_change < -0.02:  # Strong down move
            # In a bearish trend, this is continuation; in bullish, could be a reversal
            if market_trend == "bearish":
                prediction -= 0.1  # Higher weight in bearish market
                print(f"  ✅ Strong down move in bearish market: -0.1")
            else:
                prediction -= 0.05  # Lower weight in bullish market
                print(f"  ✅ Strong down move in non-bearish market: -0.05")
        
        # Volume confirmation
        if abs(price_change) > 0.01 and volume_change > 0.5:
            # High volume confirms the move direction
            if price_change > 0:
                prediction += 0.05
                print(f"  ✅ High volume confirming up move: +0.05")
            else:
                prediction -= 0.05
                print(f"  ✅ High volume confirming down move: -0.05")
        
        # Market trend influence
        if market_trend == "bullish":
            # In bullish market, amplify positive signals, reduce negative ones
            old_prediction = prediction
            prediction = prediction * 1.2 if prediction > 0 else prediction * 0.8
            print(f"  ✅ Bullish market adjustment: {old_prediction:.4f} → {prediction:.4f}")
        elif market_trend == "bearish":
            # In bearish market, amplify negative signals, reduce positive ones
            old_prediction = prediction
            prediction = prediction * 0.8 if prediction > 0 else prediction * 1.2
            print(f"  ✅ Bearish market adjustment: {old_prediction:.4f} → {prediction:.4f}")
        
        # Cap prediction in reasonable range
        final_prediction = max(-0.25, min(0.25, prediction))
        if final_prediction != prediction:
            print(f"  ✅ Capped prediction: {prediction:.4f} → {final_prediction:.4f}")
        
        print(f"  📊 Final prediction: {final_prediction:.4f}")
        return final_prediction

    def get_success_rate(self):
        """Calculate success rate of trades"""
        if self.learning_metrics['trades'] == 0:
            return 0.0
        return (self.learning_metrics['successful_trades'] / 
                self.learning_metrics['trades'] * 100)

    def get_accuracy_trend(self):
        """Calculate accuracy trend of predictions"""
        if not self.learning_metrics['accuracy']:
            return "No accuracy data available"
        accuracy_percentage = (sum(self.learning_metrics['accuracy']) / 
                               len(self.learning_metrics['accuracy'])) * 100
        return f"{accuracy_percentage:.2f}%"

    def display_results(self):
        """Display final backtest results"""
        print("\n Backtest Results")
        print("="*50)
        
        # Get portfolio metrics
        initial_balance = self.simulator.initial_balance
        final_balance = self.simulator.balance
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        # Display summary
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.simulator.trades_history)}")
        
        # Display performance metrics if available
        if hasattr(self, 'analyzer'):
            metrics = self.analyzer.analyze_trades(self.simulator.trades_history)
            print("\nPerformance Metrics:")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Maximum Drawdown: {metrics['largest_drawdown']:.2f}%")
            
        # Train ML model with collected data
        self.train_ml_model()

    def train_ml_model(self):
        """Train ML model for market predictions (supporting both up and down trends)"""
        try:
            if not self.ml_data['features'] or len(self.ml_data['features']) < 20:
                print(f"⚠️ Not enough training data: {len(self.ml_data['features'])} samples (need at least 20)")
                return False
            
            # Convert lists to numpy arrays for model training
            X = np.array(self.ml_data['features'])
            y = np.array(self.ml_data['targets'])
            
            print(f"🧠 Training ML model with {len(X)} samples")
            
            # Check for valid data
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print("⚠️ Training data contains NaN values - cleaning...")
                # Remove rows with NaN values
                valid_indices = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) < 20:
                    print(f"⚠️ Not enough valid training data after cleaning: {len(X)} samples")
                    return False
            
            # Normalize features for better convergence
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store the scaler for later predictions
            self.ml_scaler = scaler
            
            # Print target distribution to see if we have both positive and negative examples
            positives = sum(1 for target in y if target > 0.01)  # Clearly positive 
            negatives = sum(1 for target in y if target < -0.01)  # Clearly negative
            neutrals = sum(1 for target in y if abs(target) <= 0.01)  # Close to zero
            
            print(f"📊 Target distribution: {positives} positive, {negatives} negative, {neutrals} neutral")
            
            # Separate training for long and short models
            # For long model, focus on predicting positive returns
            long_mask = y >= 0
            long_X = X_scaled[long_mask]
            long_y = y[long_mask]
            
            # For short model, focus on predicting negative returns
            short_mask = y <= 0
            short_X = X_scaled[short_mask]
            short_y = abs(y[short_mask])  # Convert to positive values for training
            
            print(f"Long model: {len(long_X)} samples, Short model: {len(short_X)} samples")
            
            # Try different models based on available data
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
            from sklearn.linear_model import LinearRegression
            
            # Set up models
            if len(X) >= 100:
                long_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
                short_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            else:
                long_model = LinearRegression()
                short_model = LinearRegression()
            
            # Simple validation
            if len(long_X) >= 20 and len(short_X) >= 10:
                # Split for long model
                long_split_idx = int(len(long_X) * 0.7)
                train_long_X, val_long_X = long_X[:long_split_idx], long_X[long_split_idx:]
                train_long_y, val_long_y = long_y[:long_split_idx], long_y[long_split_idx:]
                
                # Split for short model (if enough data)
                short_split_idx = max(int(len(short_X) * 0.7), min(5, len(short_X)-1))
                if short_split_idx < len(short_X) - 1:
                    train_short_X = short_X[:short_split_idx]
                    val_short_X = short_X[short_split_idx:]
                    train_short_y = short_y[:short_split_idx]
                    val_short_y = short_y[short_split_idx:]
                    have_short_validation = True
                else:
                    # Not enough short data for validation split
                    train_short_X, train_short_y = short_X, short_y
                    have_short_validation = False
                
                # Train the long model
                long_model.fit(train_long_X, train_long_y)
                
                # Train the short model
                if len(train_short_X) > 0:
                    short_model.fit(train_short_X, train_short_y)
                
                # Evaluate long model
                long_preds = long_model.predict(val_long_X)
                long_rmse = np.sqrt(np.mean((val_long_y - long_preds) ** 2))
                
                # Evaluate short model if possible
                if have_short_validation and len(val_short_X) > 0:
                    short_preds = short_model.predict(val_short_X)
                    short_rmse = np.sqrt(np.mean((val_short_y - short_preds) ** 2))
                    print(f"🔍 Long model RMSE: {long_rmse:.4f}, Short model RMSE: {short_rmse:.4f}")
                else:
                    print(f"🔍 Long model RMSE: {long_rmse:.4f}, Short model: insufficient validation data")
                
                # If accuracy is poor for either model, try classification approach
                if long_rmse > 0.05 or (have_short_validation and short_rmse > 0.05):
                    print("⚠️ Poor regression performance, trying classification approach instead")
                    
                    # Convert to classification (profitable vs. unprofitable trades)
                    # For long trades: 1 for profit, 0 for loss
                    long_classes = (train_long_y > 0).astype(int)
                    val_long_classes = (val_long_y > 0).astype(int)
                    
                    # For short trades: 1 for profit, 0 for loss
                    if len(train_short_y) > 0:
                        short_classes = (train_short_y > 0).astype(int)
                        if have_short_validation:
                            val_short_classes = (val_short_y > 0).astype(int)
                    
                    # Train classifiers
                    long_clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                    long_clf.fit(train_long_X, long_classes)
                    
                    if len(train_short_y) > 0:
                        short_clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                        short_clf.fit(train_short_X, short_classes)
                    
                    # Validate long classifier
                    long_class_preds = long_clf.predict(val_long_X)
                    long_class_acc = np.mean(long_class_preds == val_long_classes) * 100
                    
                    # Validate short classifier if possible
                    if have_short_validation and len(val_short_X) > 0:
                        short_class_preds = short_clf.predict(val_short_X)
                        short_class_acc = np.mean(short_class_preds == val_short_classes) * 100
                        print(f"🔍 Long classifier: {long_class_acc:.1f}%, Short classifier: {short_class_acc:.1f}%")
                    else:
                        print(f"🔍 Long classifier: {long_class_acc:.1f}%, Short classifier: insufficient data")
                    
                    # Use classifiers if they perform better
                    self.ml_model_type = 'classifier'
                    self.long_ml_model = long_clf
                    if len(train_short_y) > 0:
                        self.short_ml_model = short_clf
                    else:
                        self.short_ml_model = None
                    return True
            
            # If we get here, use regression models
            if len(long_X) > 0:
                long_model.fit(long_X, long_y)
                self.long_ml_model = long_model
            else:
                self.long_ml_model = None
                
            if len(short_X) > 0:
                short_model.fit(short_X, short_y)
                self.short_ml_model = short_model
            else:
                self.short_ml_model = None
                
            self.ml_model_type = 'regressor'
            return True
            
        except Exception as e:
            print(f"❌ Error training ML model: {str(e)}")
            traceback.print_exc()
            return False
            
    def predict_with_ml(self, features, market_trend="neutral"):
        """
        Make predictions with ML model, supporting both regression and classification
        
        Parameters:
        - features: Feature vector for prediction
        - market_trend: Current market trend (bullish, bearish, neutral)
        
        Returns:
        - Float value indicating prediction (positive for long, negative for short)
        """
        if (not hasattr(self, 'long_ml_model') or self.long_ml_model is None) and \
           (not hasattr(self, 'short_ml_model') or self.short_ml_model is None):
            return None
            
        try:
            # Scale features
            if hasattr(self, 'ml_scaler') and self.ml_scaler is not None:
                features_scaled = self.ml_scaler.transform([features])
            else:
                # Fallback if no scaler is available
                features_scaled = np.array([features])
                
            # Determine which model to use based on market trend
            use_short_model = market_trend == "bearish" and hasattr(self, 'short_ml_model') and self.short_ml_model is not None
            use_long_model = (market_trend == "bullish" or market_trend == "neutral") and \
                             hasattr(self, 'long_ml_model') and self.long_ml_model is not None
                
            # Make prediction
            if hasattr(self, 'ml_model_type') and self.ml_model_type == 'classifier':
                # For classifier models
                if use_short_model:
                    # Short model predicts probability of profitable short
                    prediction_class = self.short_ml_model.predict(features_scaled)[0]
                    confidence = max(self.short_ml_model.predict_proba(features_scaled)[0])
                    
                    # Return negative value (for short) with confidence scaling
                    return -confidence * 0.3 if prediction_class == 1 else -0.05
                    
                elif use_long_model:
                    # Long model predicts probability of profitable long
                    prediction_class = self.long_ml_model.predict(features_scaled)[0]
                    confidence = max(self.long_ml_model.predict_proba(features_scaled)[0])
                    
                    # Return positive value (for long) with confidence scaling
                    return confidence * 0.3 if prediction_class == 1 else 0.05
                else:
                    return 0.0
            else:
                # For regression models
                if use_short_model:
                    # Short model predicts profit magnitude (return as negative)
                    pred = -self.short_ml_model.predict(features_scaled)[0]
                    # Amplify short signals in bearish markets
                    return pred * 1.5 if market_trend == "bearish" else pred
                    
                elif use_long_model:
                    # Long model predicts profit magnitude (return as positive)
                    pred = self.long_ml_model.predict(features_scaled)[0]
                    # Amplify long signals in bullish markets
                    return pred * 1.5 if market_trend == "bullish" else pred
                else:
                    return 0.0
                
        except Exception as e:
            print(f"❌ Error in ML prediction: {str(e)}")
            return None

def main():
    """Main function with improved configurability for both standard and short-selling strategies"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Backtester with Short-Selling Support')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest with both long and short strategies')
    backtest_parser.add_argument('--start', type=str, required=True, 
                        help='Start date in YYYYMMDD format')
    backtest_parser.add_argument('--end', type=str, required=True, 
                        help='End date in YYYYMMDD format')
    backtest_parser.add_argument('--balance', type=float, default=2000.0, 
                        help='Initial balance (default: 2000.0)')
    backtest_parser.add_argument('--symbols', type=str, default='BTC,ETH,XRP,ADA,SOL,DOT,AVAX,MATIC',
                        help='Comma-separated list of symbols to trade')
    
    # Legacy mode for backward compatibility
    legacy_parser = subparsers.add_parser('legacy', help='Run in legacy mode (for backward compatibility)')
    legacy_parser.add_argument('--balance', type=float, default=1000.0, 
                        help='Initial balance (default: 1000.0)')
    legacy_parser.add_argument('--days', type=int, default=90, 
                        help='Number of days for backtest (default: 90)')
    legacy_parser.add_argument('--symbols', type=str, default='BTC,ETH,XRP,ADA,SOL,DOT,AVAX,MATIC',
                        help='Comma-separated list of symbols to trade')
    legacy_parser.add_argument('--cap-min', type=int, default=5_000_000,
                        help='Minimum market cap in USD (default: 5M)')
    legacy_parser.add_argument('--cap-max', type=int, default=500_000_000,
                        help='Maximum market cap in USD (default: 500M)')
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = TradingBot(initial_balance=args.balance if hasattr(args, 'balance') else 2000.0)
    
    if args.command == 'backtest':
        print("🚀 Starting Advanced Crypto Trading Bot with Short-Selling Support")
        print("="*50)
        print(f"Initial Balance: ${args.balance:,.2f}")
        print(f"Backtest Period: {args.start} to {args.end}")
        symbols = args.symbols.split(',')
        print(f"Symbols: {args.symbols}")
        print("="*50)
        
        # Run the enhanced backtest
        bot.backtest(args.start, args.end, provided_symbols=symbols)
        
    elif args.command == 'legacy':
        print("🚀 Starting Crypto Trading Bot (Legacy Mode)")
        print("="*50)
        print(f"Initial Balance: ${args.balance:,.2f}")
        print(f"Backtest Period: {args.days} days")
        print(f"Symbols: {args.symbols}")
        print("="*50)
        
        # Configure dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Get symbols
        symbols = args.symbols.split(',')
        
        # Format dates for the backtest function
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # Run backtest
        bot.backtest(start_date_str, end_date_str, provided_symbols=symbols)
    
    else:
        # Default behavior if no command is specified
        parser.print_help()
        return
    
    print("\n🏁 Backtest completed")
    
    # Suggest next steps
    print("\n📋 Suggested Next Steps:")
    print("1. Try different date ranges to test strategy performance")
    print("2. Compare long-only vs short-selling in different market conditions")
    print("3. Adjust trading parameters in the code for better performance")
    print("4. Add more technical indicators to improve predictions")
    print("5. Implement real-time trading via exchange APIs")

if __name__ == "__main__":
    main()