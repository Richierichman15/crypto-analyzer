from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback
import argparse

from data.data_fetcher import DataFetcher
from data.historical_data import HistoricalDataFetcher
from trading.simulator import TradingSimulator
from trading.risk_manager import RiskManager
from analysis.performance import PerformanceTracker
from monitoring.monitor import TradingMonitor
from analysis.optimizer import StrategyOptimizer

class TradingBot:
    def __init__(self, initial_balance=1000):
        self.simulator = TradingSimulator(initial_balance)
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
        
    def backtest(self, start_date, end_date, provided_symbols=None):
        """Run backtest with learning status"""
        try:
            print("\n🚀 Starting backtest...")
            print(f"Period: {start_date} to {end_date}")
            
            self.learning_metrics = {
                'predictions': [],
                'accuracy': [],
                'trades': 0,
                'successful_trades': 0
            }
            
            # Use provided symbols if available, otherwise fetch from market data
            symbols = None
            if provided_symbols and len(provided_symbols) > 0:
                print(f"\n📊 Using {len(provided_symbols)} provided symbols: {', '.join(provided_symbols)}")
                symbols = provided_symbols
            else:
                # Fetch market data to find symbols
                data_fetcher = DataFetcher(
                    min_market_cap=5_000_000,    
                    max_market_cap=500_000_000   
                )
                market_data = data_fetcher.scrape_market_data()
                
                if market_data.empty:
                    print("❌ No market data available")
                    return
                
                print(f"\n📊 Found {len(market_data)} coins within market cap range")
                print("\nTop 5 coins by market cap:")
                print(market_data.head().to_string())
                
                symbols = market_data['symbol'].str.upper().tolist()
            
            # Verify we have symbols to process
            if not symbols or len(symbols) == 0:
                print("❌ No symbols to analyze. Please provide symbols or ensure market data fetch works.")
                return
            
            historical_fetcher = HistoricalDataFetcher(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
            
            historical_data = historical_fetcher.fetch_historical_data()
            
            if not historical_data:
                print("❌ No historical data available for backtesting")
                return
            
            # Enhanced progress tracking
            dates = pd.date_range(start=start_date, end=end_date)
            total_days = len(dates)
            print(f"\n📆 Simulation will process {total_days} trading days")
            print("=" * 50)
            print("🔄 STARTING SIMULATION PERIOD")
            print("=" * 50)
            
            # Track portfolio value over time for charting
            self.portfolio_history = []
            
            for i, date in enumerate(dates):
                try:
                    # Display progress
                    progress = (i + 1) / total_days * 100
                    print(f"\n📅 DAY {i+1}/{total_days} ({progress:.1f}% complete) - {date.date()}")
                    print("-" * 50)
                    
                    # Process trading day
                    self.execute_trading_day(date, historical_data)
                    
                    # Track portfolio value at end of day
                    if i % 5 == 0 or i == total_days - 1:  # Every 5 days or last day
                        # Get the last prices we have
                        current_prices = {}
                        for symbol, df in historical_data.items():
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            day_data = df[df.index.date == date.date()]
                            if not day_data.empty:
                                current_prices[symbol] = day_data['Close'].iloc[-1]
                        
                        portfolio_value = self.simulator.get_portfolio_value(current_prices)
                        total_value = portfolio_value + self.simulator.balance
                        self.portfolio_history.append({
                            'date': date,
                            'portfolio_value': portfolio_value,
                            'cash_balance': self.simulator.balance,
                            'total_value': total_value
                        })
                    
                except Exception as e:
                    print(f"❌ Error processing date {date}: {str(e)}")
                    continue
            
            print("\n" + "=" * 50)
            print("🏁 SIMULATION PERIOD COMPLETED")
            print("=" * 50)
            
            # Display portfolio value progression
            if self.portfolio_history:
                print("\n📈 Portfolio Value Progression:")
                print("-" * 50)
                print(f"{'Date':<12} {'Portfolio':<12} {'Cash':<12} {'Total':<12} {'Change %':<10}")
                print("-" * 50)
                
                initial_value = self.simulator.initial_balance
                prev_value = initial_value
                
                for i, record in enumerate(self.portfolio_history):
                    if i % 10 == 0 or i == len(self.portfolio_history) - 1:  # Show every 10th day and last day
                        date_str = record['date'].strftime('%Y-%m-%d')
                        portfolio = record['portfolio_value']
                        cash = record['cash_balance']
                        total = record['total_value']
                        
                        pct_change_total = ((total / initial_value) - 1) * 100
                        daily_change = ((total / prev_value) - 1) * 100 if prev_value > 0 else 0
                        
                        print(f"{date_str:<12} ${portfolio:<11.2f} ${cash:<11.2f} ${total:<11.2f} {pct_change_total:+.2f}%")
                        prev_value = total
            
            # Display final results
            self.display_results()
            
            print("\n📚 Learning Progress:")
            if self.learning_metrics['trades'] > 0:
                success_rate = (self.learning_metrics['successful_trades'] / 
                                self.learning_metrics['trades'] * 100)
                print(f"Total Trades: {self.learning_metrics['trades']}")
                print(f"Successful Trades: {self.learning_metrics['successful_trades']}")
                print(f"Success Rate: {success_rate:.2f}%")
                print(f"Prediction Accuracy Trend: {self.get_accuracy_trend()}")
            else:
                print("❌ No trades executed - Check prediction thresholds and data quality")
            
        except Exception as e:
            print(f"❌ Error in backtest: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
        """Execute trades with enhanced strategy including ML and improved risk management"""
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
            if self.simulator.portfolio:
                risk_status = self.risk_manager.check_risk_limits(
                    self.simulator.portfolio, current_prices
                )
                
                if any(risk_status.values()):
                    risk_issues = [k.replace('_', ' ').title() for k, v in risk_status.items() if v]
                    print(f"⚠️ Risk limits exceeded: {', '.join(risk_issues)}")
                    
                    # Force liquidate most risky positions if needed
                    if risk_status.get('daily_loss_exceeded', False):
                        positions_liquidated = []
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
                                        positions_liquidated.append(f"{symbol} ({profit_pct:.1f}%)")
                                        portfolio_reduced = True
                        
                        if positions_liquidated:
                            print(f"🔄 Risk management liquidated: {', '.join(positions_liquidated)}")

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
                    import numpy as np
                    ml_predictions = self.ml_model.predict(np.array(ml_features))
                    
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
            
            for symbol, prediction in predictions:
                # Skip if prediction is too weak
                if abs(prediction) < 0.1:
                    continue
                
                price = current_prices.get(symbol, 0)
                if price == 0:
                    continue
                
                # Buy logic
                if prediction > 0.1 and symbol not in self.simulator.portfolio:
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

                # Sell logic
                elif symbol in self.simulator.portfolio:
                    position = self.simulator.portfolio[symbol]
                    entry_price = position['entry_price']
                    profit_pct = (price - entry_price) / entry_price * 100
                    
                    # Dynamic take profit and stop loss based on volatility
                    vol_factor = 1.0  # Could be calculated based on historical volatility
                    take_profit = 2.0 * vol_factor
                    stop_loss = -1.0 * vol_factor
                    
                    # Should we sell?
                    sell_signals = []
                    
                    if prediction < -0.1:
                        sell_signals.append(f"Signal ({prediction:.2f})")
                    
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
                            symbol_index = -1
                            for i, (s, p) in enumerate(predictions):
                                if s == symbol:
                                    symbol_index = i
                                    break
                            
                            # Find the corresponding feature index for this symbol when we bought it
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

            # Show trading activity summary
            if buys_executed:
                print(f"🛒 Buys: {', '.join(buys_executed)}")
            if sells_executed:
                print(f"💰 Sells: {', '.join(sells_executed)}")
            
            # Only show portfolio summary if we had trading activity or risk management
            if buys_executed or sells_executed or portfolio_reduced:
                portfolio_value = self.simulator.get_portfolio_value(current_prices)
                total_value = portfolio_value + self.simulator.balance
                profit_pct = ((total_value / self.simulator.initial_balance) - 1) * 100
                
                print(f"📊 Portfolio: ${portfolio_value:.2f} | Cash: ${self.simulator.balance:.2f} | Total: ${total_value:.2f} ({profit_pct:+.2f}%)")
                print(f"Open positions: {len(self.simulator.portfolio)}")

        except Exception as e:
            print(f"❌ Error in execute_trading_day: {str(e)}")
            traceback.print_exc()
            
    def calculate_rule_based_prediction(self, price, price_change, volume_change, rsi, macd, macd_signal):
        """Calculate rule-based prediction score"""
        prediction = 0.0
        
        # Buy signals
        if price_change > 0.001:  # Price up 0.1%
            prediction += 0.05
        
        if volume_change > 0.01:  # Volume up 1%
            prediction += 0.05
        
        if rsi < 30:  # Oversold
            prediction += 0.1
        
        if macd > macd_signal:  # MACD crossover
            prediction += 0.1
        
        # Sell signals
        if price_change < -0.001:  # Price down 0.1%
            prediction -= 0.05
        
        if volume_change < -0.01:  # Volume down 1%
            prediction -= 0.05
        
        if rsi > 70:  # Overbought
            prediction -= 0.1
        
        if macd < macd_signal:  # MACD crossover down
            prediction -= 0.1
        
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
        """Train machine learning model using collected data from trades"""
        try:
            print("\n🧠 Training machine learning model...")
            
            # Check if we have enough data
            if not self.ml_data['features'] or len(self.ml_data['features']) < 20:
                print("⚠️ Not enough training data yet. Need at least 20 samples.")
                return
                
            # Import required libraries
            import numpy as np
            from xgboost import XGBRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Convert to numpy arrays
            X = np.array(self.ml_data['features'])
            y = np.array(self.ml_data['targets'])
            
            print(f"📊 Training data: {X.shape[0]} samples with {X.shape[1]} features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Define model
            self.ml_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Train model
            self.ml_model.fit(X_train, y_train)
            
            # Evaluate model
            train_preds = self.ml_model.predict(X_train)
            test_preds = self.ml_model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            print("\n📈 Model Performance:")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Testing RMSE: {test_rmse:.4f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Testing R²: {test_r2:.4f}")
            
            # Feature importance
            feature_names = [
                'Price', 'Price_Change', 'Volume_Change', 
                'RSI', 'MACD', 'MACD_Signal'
            ]
            
            importance = self.ml_model.feature_importances_
            print("\n🔍 Feature Importance:")
            for i, name in enumerate(feature_names):
                if i < len(importance):
                    print(f"{name}: {importance[i]:.4f}")
                    
            print("\n✅ ML model training complete!")
            
        except Exception as e:
            print(f"❌ Error training ML model: {str(e)}")
            traceback.print_exc()
            self.ml_model = None

def main():
    """Main function with improved configurability"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Backtester')
    parser.add_argument('--balance', type=float, default=1000.0, 
                        help='Initial balance (default: 1000.0)')
    parser.add_argument('--days', type=int, default=90, 
                        help='Number of days for backtest (default: 90)')
    parser.add_argument('--symbols', type=str, default='BTC,ETH,XRP,ADA,SOL,DOT,AVAX,MATIC',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--cap-min', type=int, default=5_000_000,
                        help='Minimum market cap in USD (default: 5M)')
    parser.add_argument('--cap-max', type=int, default=500_000_000,
                        help='Maximum market cap in USD (default: 500M)')
    args = parser.parse_args()
    
    print("🚀 Starting Crypto Trading Bot")
    print("="*50)
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Backtest Period: {args.days} days")
    print(f"Symbols: {args.symbols}")
    print("="*50)
    
    # Create bot instance
    bot = TradingBot(initial_balance=args.balance)
    
    # Configure dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Get symbols
    symbols = args.symbols.split(',')
    
    # Fetch historical data
    print("\n📈 Fetching historical data...")
    fetcher = HistoricalDataFetcher(start_date, end_date, symbols)
    historical_data = fetcher.fetch_historical_data()
    
    if not historical_data:
        print("❌ Failed to fetch historical data")
        return
        
    print(f"\n✅ Fetched data for {len(historical_data)} symbols")
    
    # Run backtest
    bot.backtest(start_date, end_date, provided_symbols=symbols)
    
    print("\n🏁 Backtest completed")
    
    # Suggest next steps
    print("\n📋 Suggested Next Steps:")
    print("1. Increase backtest period (--days) for more training data")
    print("2. Adjust trading parameters in the code for better performance")
    print("3. Add more technical indicators to improve predictions")
    print("4. Explore different ML models for prediction")
    print("5. Implement real-time trading via exchange APIs")

if __name__ == "__main__":
    main()