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
        start = pd.to_datetime(start_date, format='%Y%m%d')
        end = pd.to_datetime(end_date, format='%Y%m%d')
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
        """Get current day data with concise debugging"""
        try:
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)
            
            daily_data = []
            symbols_found = []
            symbols_missing = []
            
            for symbol, df in historical_data.items():
                try:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    day_data = df[df.index.date == date.date()]
                    
                    if not day_data.empty:
                        daily_data.append(day_data)
                        symbols_found.append(symbol)
                    else:
                        symbols_missing.append(symbol)
                
                except Exception as e:
                    symbols_missing.append(symbol)
                    continue
            
            if daily_data:
                # Combine all symbol data for this date
                combined_data = pd.concat(daily_data)
                print(f"📊 Data available for {len(symbols_found)}/{len(historical_data)} symbols")
                return combined_data
            else:
                print("❌ No data available for this date")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error in get_current_day_data: {e}")
            return pd.DataFrame()
    
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

            # Create current prices dictionary for portfolio valuation
            current_prices = {
                row['symbol']: row['Close'] 
                for _, row in current_data.iterrows()
            }
            
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
            
            # Get representative symbol (e.g., BTC) to determine market trend
            trend_symbol = "BTC"
            if trend_symbol in historical_data:
                trend_data = historical_data[trend_symbol]
                # Get prices for the last 7 days
                recent_prices = []
                for i in range(7):
                    days_back = i + 1
                    check_date = date - pd.Timedelta(days=days_back)
                    if not isinstance(trend_data.index, pd.DatetimeIndex):
                        trend_data.index = pd.to_datetime(trend_data.index)
                    
                    day_data = trend_data[trend_data.index.date == check_date.date()]
                    if not day_data.empty:
                        recent_prices.append(day_data['Close'].iloc[-1])
                
                if recent_prices:
                    market_trend = self.risk_manager.detect_market_trend(recent_prices)
                    print(f"📈 Market trend: {market_trend.upper()}")

            # Calculate technical indicators and predictions
            ml_features = []
            symbols_analyzed = []

            # Process all symbols to gather data for prediction
            for symbol in current_data['symbol'].unique():
                try:
                    symbol_data = current_data[current_data['symbol'] == symbol].copy()
                    
                    if len(symbol_data) < 2:
                        continue

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

                except Exception as e:
                    continue

            # Make predictions
            predictions = []
            
            # Attempt ML predictions if model exists and we have features
            if hasattr(self, 'ml_model') and self.ml_model is not None and ml_features:
                try:
                    # Use our new prediction function
                    ml_predictions = []
                    for i, features in enumerate(ml_features):
                        ml_pred = self.predict_with_ml(features)
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
                                macd_signal=ml_features[i][5]
                            )
                            ml_predictions.append(rule_prediction)
                    
                    # Use ML predictions with some weight
                    ml_weight = min(0.7, len(self.ml_data['features']) / 1000)
                    
                    for i, symbol in enumerate(symbols_analyzed):
                        # Calculate rule-based prediction
                        rule_prediction = self.calculate_rule_based_prediction(
                            price=ml_features[i][0],
                            price_change=ml_features[i][1],
                            volume_change=ml_features[i][2],
                            rsi=ml_features[i][3],
                            macd=ml_features[i][4],
                            macd_signal=ml_features[i][5]
                        )
                        
                        # Combine rule-based and ML predictions
                        final_prediction = (rule_prediction * (1 - ml_weight)) + (ml_predictions[i] * ml_weight)
                        predictions.append((symbol, final_prediction))
                        
                except Exception as e:
                    print(f"⚠️ ML prediction error: {str(e)}")
                    # Fall back to rule-based predictions
                    for i, symbol in enumerate(symbols_analyzed):
                        prediction = self.calculate_rule_based_prediction(
                            price=ml_features[i][0],
                            price_change=ml_features[i][1],
                            volume_change=ml_features[i][2],
                            rsi=ml_features[i][3],
                            macd=ml_features[i][4],
                            macd_signal=ml_features[i][5]
                        )
                        predictions.append((symbol, prediction))
            else:
                # Use rule-based predictions
                for i, symbol in enumerate(symbols_analyzed):
                    prediction = self.calculate_rule_based_prediction(
                        price=ml_features[i][0],
                        price_change=ml_features[i][1],
                        volume_change=ml_features[i][2],
                        rsi=ml_features[i][3],
                        macd=ml_features[i][4],
                        macd_signal=ml_features[i][5]
                    )
                    predictions.append((symbol, prediction))

            # Execute trades based on predictions
            buys_executed = []
            sells_executed = []
            shorts_executed = []
            covers_executed = []
            
            for symbol, prediction in predictions:
                # Skip if prediction is too weak
                if abs(prediction) < 0.1:
                    continue
                
                price = current_prices.get(symbol, 0)
                if price == 0:
                    continue
                
                # Check existing positions
                has_long = symbol in self.simulator.portfolio
                has_short = symbol in self.simulator.short_portfolio
                
                # Long position logic (buy)
                if prediction > 0.1 and not has_long and not has_short and market_trend != "bearish":
                    # Don't open long positions in a bearish market
                    # Dynamic position sizing based on conviction
                    position_size = min(
                        self.simulator.balance * (0.03 + (prediction * 0.05)),  # Base 3% + up to 5% more
                        self.simulator.balance * 0.1  # Cap at 10%
                    )
                    
                    if position_size >= 10:  # Minimum $10 trade
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
                        
                        if ok_to_trade:
                            trade = self.simulator.execute_trade(
                                date, symbol, price, 'BUY',
                                quantity, prediction
                            )
                            
                            if trade:
                                self.learning_metrics['trades'] += 1
                                buys_executed.append(f"{symbol} (${price:.4f}, ${position_size:.2f})")
                                
                                # Store features and target for ML model
                                # Target will be updated when position is closed
                                for i, sym in enumerate(symbols_analyzed):
                                    if sym == symbol:
                                        self.ml_data['features'].append(ml_features[i])
                                        self.ml_data['targets'].append(0)  # Placeholder until we sell
                
                # Short position logic
                elif prediction < -0.1 and not has_short and not has_long and market_trend == "bearish":
                    # Only open short positions in a bearish market
                    # Be more conservative with short position sizing
                    position_size = min(
                        self.simulator.balance * (0.02 + (abs(prediction) * 0.03)),  # More conservative sizing
                        self.simulator.balance * 0.08  # Lower cap for shorts
                    )
                    
                    if position_size >= 10:  # Minimum $10 trade
                        quantity = position_size / price
                        
                        # Additional risk check for short positions
                        ok_to_trade = self.risk_manager.check_trade(
                            self.simulator.portfolio, 
                            self.simulator.short_portfolio,
                            symbol, 
                            price, 
                            quantity, 
                            'SHORT'
                        )
                        
                        if ok_to_trade:
                            trade = self.simulator.execute_trade(
                                date, symbol, price, 'SHORT',
                                quantity, prediction
                            )
                            
                            if trade:
                                self.learning_metrics['trades'] += 1
                                shorts_executed.append(f"{symbol} (${price:.4f}, ${position_size:.2f})")
                                
                                # Store features and target for ML model
                                for i, sym in enumerate(symbols_analyzed):
                                    if sym == symbol:
                                        self.ml_data['features'].append(ml_features[i])
                                        self.ml_data['targets'].append(0)  # Placeholder until we cover

                # Manage existing long positions
                elif has_long and (prediction < 0 or market_trend == "bearish"):
                    position = self.simulator.portfolio[symbol]
                    entry_price = position['entry_price']
                    profit_pct = (price - entry_price) / entry_price * 100
                    
                    # Dynamic take profit and stop loss
                    vol_factor = 1.0  # Could be calculated based on historical volatility
                    take_profit = 2.0 * vol_factor
                    stop_loss = -1.0 * vol_factor
                    
                    # Should we sell?
                    sell_signals = []
                    
                    if prediction < -0.1:
                        sell_signals.append(f"Signal ({prediction:.2f})")
                    
                    if market_trend == "bearish":
                        sell_signals.append("Bearish market")
                    
                    if profit_pct >= take_profit:
                        sell_signals.append(f"Profit {profit_pct:.1f}%")
                    
                    if profit_pct <= stop_loss:
                        sell_signals.append(f"Stop {profit_pct:.1f}%")
                    
                    # Calculate holding period
                    entry_date = position['entry_date']
                    if hasattr(entry_date, 'date'):
                        days_held = (date.date() - entry_date.date()).days
                    else:
                        days_held = 0
                    
                    # Sell if held for more than 7 days with minimal profit
                    if days_held >= 7 and profit_pct < 1.0:
                        sell_signals.append(f"Time ({days_held}d)")
                    
                    if sell_signals:
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'SELL',
                            position['quantity'], profit_pct/100
                        )
                        
                        if trade:
                            # Update ML training data with actual profit/loss
                            for i, features in enumerate(self.ml_data['features']):
                                # Simplified matching - in real system would need position ID
                                if self.ml_data['targets'][i] == 0:  # Placeholder target
                                    self.ml_data['targets'][i] = profit_pct / 100  # Update target with actual profit
                                    break
                            
                            sells_executed.append(f"{symbol} ({profit_pct:+.1f}%, reason: {', '.join(sell_signals)})")
                            
                            if profit_pct > 0:
                                self.learning_metrics['successful_trades'] += 1
                                self.learning_metrics['accuracy'].append(1)
                            else:
                                self.learning_metrics['accuracy'].append(0)
                
                # Manage existing short positions
                elif has_short and (prediction > 0 or market_trend != "bearish"):
                    position = self.simulator.short_portfolio[symbol]
                    entry_price = position['entry_price']
                    # For shorts, profit is when price goes down
                    profit_pct = (entry_price - price) / entry_price * 100
                    
                    # Dynamic take profit and stop loss for shorts
                    vol_factor = 1.0
                    take_profit = 2.0 * vol_factor
                    stop_loss = -1.0 * vol_factor
                    
                    # Should we cover the short?
                    cover_signals = []
                    
                    if prediction > 0.1:
                        cover_signals.append(f"Signal ({prediction:.2f})")
                    
                    if market_trend != "bearish":
                        cover_signals.append(f"Market trend ({market_trend})")
                    
                    if profit_pct >= take_profit:
                        cover_signals.append(f"Profit {profit_pct:.1f}%")
                    
                    if profit_pct <= stop_loss:
                        cover_signals.append(f"Stop {profit_pct:.1f}%")
                    
                    # Calculate holding period
                    entry_date = position['entry_date']
                    if hasattr(entry_date, 'date'):
                        days_held = (date.date() - entry_date.date()).days
                    else:
                        days_held = 0
                    
                    # Cover if held for more than 5 days (shorter for shorts)
                    if days_held >= 5 and profit_pct < 1.0:
                        cover_signals.append(f"Time ({days_held}d)")
                    
                    if cover_signals:
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'COVER',
                            position['quantity'], profit_pct/100
                        )
                        
                        if trade:
                            # Update ML training data with actual profit/loss
                            for i, features in enumerate(self.ml_data['features']):
                                if self.ml_data['targets'][i] == 0:  # Placeholder target
                                    self.ml_data['targets'][i] = profit_pct / 100  # Update target with actual profit
                                    break
                            
                            covers_executed.append(f"{symbol} ({profit_pct:+.1f}%, reason: {', '.join(cover_signals)})")
                            
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
            trading_activity = buys_executed or sells_executed or shorts_executed or covers_executed or portfolio_reduced
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
            
    def calculate_rule_based_prediction(self, price, price_change, volume_change, rsi, macd, macd_signal):
        """Calculate rule-based prediction score (positive for long, negative for short)"""
        prediction = 0.0
        
        # For long positions (positive signals)
        long_score = 0.0
        
        # Buy signals
        if price_change > 0.001:  # Price up 0.1%
            long_score += 0.05
        
        if volume_change > 0.01:  # Volume up 1%
            long_score += 0.05
        
        if rsi < 30:  # Oversold
            long_score += 0.1
        
        if macd > macd_signal:  # MACD crossover
            long_score += 0.1
            
        # For short positions (negative signals)
        short_score = 0.0
        
        # Short signals
        if price_change < -0.001:  # Price down 0.1%
            short_score += 0.05
        
        if volume_change < -0.01:  # Volume down 1%
            short_score += 0.05
        
        if rsi > 70:  # Overbought
            short_score += 0.1
        
        if macd < macd_signal:  # MACD crossover down
            short_score += 0.1
            
        # Determine final prediction
        if long_score > short_score:
            prediction = long_score  # Positive prediction (BUY/HOLD)
        elif short_score > long_score:
            prediction = -short_score  # Negative prediction (SHORT/SELL)
        
        return prediction

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
            positives = sum(1 for target in y if target > 0)
            negatives = sum(1 for target in y if target < 0)
            zeros = sum(1 for target in y if target == 0)
            
            print(f"📊 Target distribution: {positives} positive, {negatives} negative, {zeros} zero")
            
            # Try different models based on available data
            if len(X) >= 100:
                # Use a more complex model for larger datasets
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            else:
                # Use a simpler model for smaller datasets
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            
            # Fit the model with available data
            model.fit(X_scaled, y)
            
            # Simple validation
            if len(X) >= 30:
                # Use last 30% for validation
                split_idx = int(len(X) * 0.7)
                train_X, val_X = X_scaled[:split_idx], X_scaled[split_idx:]
                train_y, val_y = y[:split_idx], y[split_idx:]
                
                model.fit(train_X, train_y)
                
                # Make predictions on validation set
                val_preds = model.predict(val_X)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(val_y, val_preds)
                rmse = np.sqrt(mse)
                
                # Directional accuracy (positive/negative)
                directional_matches = sum(1 for i, pred in enumerate(val_preds) 
                                         if (pred > 0 and val_y[i] > 0) or 
                                            (pred < 0 and val_y[i] < 0) or
                                            (abs(pred) < 0.001 and abs(val_y[i]) < 0.001))
                directional_accuracy = directional_matches / len(val_preds) * 100
                
                print(f"🔍 Model validation: RMSE = {rmse:.4f}, Directional Accuracy = {directional_accuracy:.1f}%")
                
                # If accuracy is really poor, use a different approach
                if directional_accuracy < 45:
                    print("⚠️ Poor model performance, trying classification approach instead")
                    
                    # Convert to classification problem (up/down/neutral)
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # Convert targets to classes: -1 (down), 0 (neutral), 1 (up)
                    threshold = 0.01  # 1% threshold
                    train_classes = np.zeros(len(train_y))
                    train_classes[train_y > threshold] = 1
                    train_classes[train_y < -threshold] = -1
                    
                    val_classes = np.zeros(len(val_y))
                    val_classes[val_y > threshold] = 1
                    val_classes[val_y < -threshold] = -1
                    
                    # Train a classifier
                    clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                    clf.fit(train_X, train_classes)
                    
                    # Validate classifier
                    class_preds = clf.predict(val_X)
                    class_accuracy = sum(1 for i, pred in enumerate(class_preds) 
                                        if pred == val_classes[i]) / len(val_classes) * 100
                    
                    print(f"🔍 Classification model: Accuracy = {class_accuracy:.1f}%")
                    
                    # If classification works better, use it
                    if class_accuracy > directional_accuracy:
                        print("🔄 Switching to classification model")
                        # Train on full dataset
                        all_classes = np.zeros(len(y))
                        all_classes[y > threshold] = 1
                        all_classes[y < -threshold] = -1
                        
                        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                        clf.fit(X_scaled, all_classes)
                        self.ml_model = clf
                        self.ml_model_type = 'classifier'
                        return True
            
            # Final model training on all data
            model.fit(X_scaled, y)
            self.ml_model = model
            self.ml_model_type = 'regressor'
            return True
            
        except Exception as e:
            print(f"❌ Error training ML model: {str(e)}")
            traceback.print_exc()
            return False
            
    def predict_with_ml(self, features):
        """Make predictions with ML model, supporting both regression and classification"""
        if not hasattr(self, 'ml_model') or self.ml_model is None:
            return None
            
        try:
            # Scale features
            if hasattr(self, 'ml_scaler') and self.ml_scaler is not None:
                features_scaled = self.ml_scaler.transform([features])
            else:
                # Fallback if no scaler is available
                features_scaled = np.array([features])
                
            # Make prediction
            if hasattr(self, 'ml_model_type') and self.ml_model_type == 'classifier':
                # For classifier models
                prediction_class = self.ml_model.predict(features_scaled)[0]
                # Convert class (-1, 0, 1) to confidence score
                confidence_scores = self.ml_model.predict_proba(features_scaled)[0]
                max_confidence = max(confidence_scores)
                
                # Map class to prediction value with confidence
                if prediction_class == 1:  # Up trend
                    return max_confidence * 0.2  # Map to 0-0.2 range
                elif prediction_class == -1:  # Down trend
                    return -max_confidence * 0.2  # Map to -0.2-0 range
                else:  # Neutral
                    return 0.0
            else:
                # For regression models
                return self.ml_model.predict(features_scaled)[0]
                
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