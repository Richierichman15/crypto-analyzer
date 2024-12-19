from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import traceback

# Custom modules
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
        
    def backtest(self, start_date, end_date):
        """Run backtest with learning status"""
        try:
            print("\n🚀 Starting backtest...")
            print(f"Period: {start_date} to {end_date}")
            
            # Initialize learning metrics
            self.learning_metrics = {
                'predictions': [],
                'accuracy': [],
                'trades': 0,
                'successful_trades': 0
            }
            
            # Initialize DataFetcher with market cap range
            data_fetcher = DataFetcher(
                min_market_cap=5_000_000,    # $5 million
                max_market_cap=500_000_000   # $500 million
            )
            market_data = data_fetcher.scrape_market_data()
            
            if market_data.empty:
                print("❌ No market data available")
                return
            
            print(f"\n📊 Found {len(market_data)} coins within market cap range")
            print("\nTop 5 coins by market cap:")
            print(market_data.head().to_string())
            
            # Get symbols from market data
            symbols = market_data['symbol'].str.upper().tolist()
            
            # Create historical data fetcher
            historical_fetcher = HistoricalDataFetcher(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
            
            # Fetch historical data
            historical_data = historical_fetcher.fetch_historical_data()
            
            if not historical_data:
                print("❌ No historical data available for backtesting")
                return
            
            # Run simulation day by day
            dates = pd.date_range(start=start_date, end=end_date)
            for date in dates:
                try:
                    self.execute_trading_day(date, historical_data)
                except Exception as e:
                    print(f"Error processing date {date}: {str(e)}")
                    continue
            
            # Display final results
            self.display_results()
            
            # Print learning progress
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
        """Get current day data with debugging"""
        try:
            print(f"\n🔍 DEBUG: Processing data for {date}")
            
            # Convert date to timestamp if it isn't already
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)
            
            # Create an empty list to store data for all symbols
            daily_data = []
            
            # Debug information
            print(f"Number of symbols in historical data: {len(historical_data)}")
            
            # Process each symbol's data
            for symbol, df in historical_data.items():
                try:
                    # Convert index to datetime if it's not already
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # Get data for the specific date
                    day_data = df[df.index.date == date.date()]
                    
                    if not day_data.empty:
                        daily_data.append(day_data)
                        print(f"✅ Found data for {symbol} on {date.date()}")
                    else:
                        print(f"⚠️ No data for {symbol} on {date.date()}")
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    continue
            
            if daily_data:
                # Combine all symbol data for this date
                combined_data = pd.concat(daily_data)
                print(f"📊 Combined data shape: {combined_data.shape}")
                return combined_data
            else:
                print("❌ No data available for this date")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error in get_current_day_data: {e}")
            return pd.DataFrame()
    
    def make_predictions(self, current_data):
        """Make predictions with detailed debugging"""
        predictions = []
        print("\n🔍 Making predictions:")
        print(f"Processing {len(current_data)} symbols")
        
        for _, row in current_data.iterrows():
            try:
                symbol = row['symbol']
                close_price = row['Close']
                volume = row['Volume']
                
                print(f"\n📈 Analyzing {symbol}:")
                print(f"Close Price: ${close_price:.4f}")
                print(f"Volume: {volume:.2f}")
                
                # Calculate prediction (simple momentum strategy)
                prediction = 0
                
                # Example: Buy signal if volume is above average and price is trending up
                volume_change = row.get('Volume_Change', 0)
                price_change = row.get('Price_Change', 0)
                
                if volume_change > 0.1 and price_change > 0:  # 10% volume increase and positive price change
                    prediction = 0.1  # Buy signal
                elif volume_change < -0.1 or price_change < 0:
                    prediction = -0.1  # Sell signal
                    
                print(f"Prediction: {prediction:.4f}")
                predictions.append(prediction)
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                predictions.append(0)
                
        return predictions
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators with error checking"""
        indicators = {}
        try:
            # Basic price data
            if 'Close' in data:
                indicators['Close'] = data['Close']
                print(f"Close price: {data['Close']}")
            else:
                print("❌ No Close price found!")
                
            # Volume data
            if 'Volume' in data:
                indicators['Volume'] = data['Volume']
                print(f"Volume: {data['Volume']}")
            else:
                print("❌ No Volume data found!")
                
            # Add more indicator calculations with debugging
            # ... (rest of your indicator calculations)
            
            return indicators
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {str(e)}")
            return {}
    
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
        """Execute trades with detailed debugging"""
        try:
            print(f"\n📅 Processing trades for {date}")
            
            current_data = self.get_current_day_data(historical_data, date)
            if current_data.empty:
                print("❌ No data available for this date")
                return
            
            print(f"Found {len(current_data)} symbols to analyze")
            predictions = self.make_predictions(current_data)
            
            for i, (_, row) in enumerate(current_data.iterrows()):
                symbol = str(row['symbol'])
                price = row['Close']
                prediction = predictions[i]
                
                print(f"\n🔄 Processing {symbol}:")
                print(f"Price: ${price:.4f}")
                print(f"Prediction: {prediction:.4f}")
                
                # Super aggressive trading (0.05 threshold)
                if prediction > 0.05:
                    print(f"🎯 Buy signal detected for {symbol}")
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_aggressive_position_size(
                        self.simulator.balance,
                        price,
                        row.get('Volatility', 0.1),
                        abs(prediction)
                    )
                    
                    # Calculate fees
                    fees = self.risk_manager.calculate_fees(price, position_size / price)
                    
                    print(f"Position size: ${position_size:.2f}")
                    print(f"Estimated fees: ${fees:.2f}")
                    
                    if position_size - fees >= 3:
                        quantity = (position_size - fees) / price
                        print(f"Executing buy: {quantity:.4f} units")
                        
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'BUY', quantity, prediction,
                            fees=fees
                        )
                        
                        if trade:
                            self.learning_metrics['trades'] += 1
                            print(f"✅ Buy executed: {symbol}")
                    else:
                        print(f"❌ Position size too small after fees")
                
                elif prediction < -0.05:
                    print(f"🎯 Sell signal detected for {symbol}")
                    if symbol in self.simulator.portfolio:
                        position = self.simulator.portfolio[symbol]
                        print(f"Current position: {position['quantity']:.4f} units")
                        
                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'SELL',
                            position['quantity'], prediction
                        )
                        
                        if trade:
                            self.learning_metrics['trades'] += 1
                            print(f"✅ Sell executed: {symbol}")
                    else:
                        print(f"❌ No position to sell")
                
            # Print daily summary
            print(f"\n📊 Daily Summary for {date}")
            print(f"Portfolio Value: ${self.simulator.get_portfolio_value(date, current_data['Close'].to_dict()):.2f}")
            print(f"Cash Balance: ${self.simulator.balance:.2f}")
            
        except Exception as e:
            print(f"❌ Error in execute_trading_day: {str(e)}")
            traceback.print_exc()
    
    def display_results(self):
        """Display backtest results"""
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
    
    def calculate_metrics(self, df):
        """Calculate additional metrics for trading decisions"""
        try:
            # Calculate price changes
            df['Price_Change'] = df['Close'].pct_change()
            
            # Calculate volume changes
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Calculate moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return df

def main():
    # Initialize the trading bot
    bot = TradingBot(initial_balance=1000)
    
    # Set up backtest parameters for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Run backtest
    bot.backtest(start_date, end_date)

if __name__ == "__main__":
    main()