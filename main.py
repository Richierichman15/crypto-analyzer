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
        """Make predictions with enhanced trading signals"""
        predictions = []
        print("\n🔮 Making predictions:")
        
        for symbol in current_data['symbol'].unique():
            try:
                # Get symbol data
                symbol_data = current_data[current_data['symbol'] == symbol]
                price = symbol_data['Close'].iloc[-1]
                
                # Calculate price and volume changes
                price_change = symbol_data['Close'].pct_change().iloc[-1]
                volume_change = symbol_data['Volume'].pct_change().iloc[-1]
                
                print(f"\n📊 Analysis for {symbol}:")
                print(f"Price: ${price:.4f}")
                print(f"Price Change: {price_change*100:.2f}%")
                print(f"Volume Change: {volume_change*100:.2f}%")
                
                # More sensitive trading signals
                prediction = 0.0
                
                # Buy signals
                if price_change > 0.001:  # Price up 0.1%
                    prediction += 0.05
                    print("✅ Positive price momentum")
                    
                if volume_change > 0.01:  # Volume up 1%
                    prediction += 0.05
                    print("✅ Positive volume momentum")
                    
                # Sell signals
                if price_change < -0.001:  # Price down 0.1%
                    prediction -= 0.05
                    print("🔻 Negative price momentum")
                    
                if volume_change < -0.01:  # Volume down 1%
                    prediction -= 0.05
                    print("🔻 Negative volume momentum")
                
                print(f"Final Prediction: {prediction:.4f}")
                predictions.append(prediction)
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                predictions.append(0)
                
        return predictions
    
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
        """Execute trades with enhanced strategy including MACD"""
        try:
            print(f"\n📅 Processing trades for {date}")

            current_data = self.get_current_day_data(historical_data, date)
            if current_data.empty:
                print("❌ No data available for this date")
                return

            # Create current prices dictionary for portfolio valuation
            current_prices = {
                row['symbol']: row['Close'] 
                for _, row in current_data.iterrows()
            }

            # Calculate technical indicators
            for symbol in current_data['symbol'].unique():
                symbol_data = current_data[current_data['symbol'] == symbol].copy()

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

                print(f"\n📊 Analysis for {symbol}:")
                print(f"Price: ${symbol_data['Close'].iloc[-1]:.4f}")
                print(f"RSI: {symbol_data['RSI'].iloc[-1]:.2f}")
                print(f"MACD: {symbol_data['MACD'].iloc[-1]:.4f}")
                print(f"MACD Signal: {symbol_data['MACD_Signal'].iloc[-1]:.4f}")

                # Enhanced buy conditions
                price = symbol_data['Close'].iloc[-1]
                rsi = symbol_data['RSI'].iloc[-1]
                macd = symbol_data['MACD'].iloc[-1]
                macd_signal = symbol_data['MACD_Signal'].iloc[-1]
                price_change = symbol_data['Close'].pct_change().iloc[-1]
                volume_change = symbol_data['Volume'].pct_change().iloc[-1]

                # Buy signals
                buy_signals = []
                if rsi < 30:  # Oversold
                    buy_signals.append("Oversold (RSI)")
                if price_change > 0.01:  # 1% price increase
                    buy_signals.append("Price momentum")
                if volume_change > 0.05:  # 5% volume increase
                    buy_signals.append("Volume surge")
                if macd > macd_signal:  # MACD crossover
                    buy_signals.append("MACD Crossover")

                if len(buy_signals) >= 2:  # Need at least 2 confirming signals
                    print(f"🎯 Buy signals: {', '.join(buy_signals)}")

                    # Position sizing (5% of balance per trade)
                    position_size = min(self.simulator.balance * 0.05, 50)

                    if position_size >= 10:  # Minimum $10 trade
                        quantity = position_size / price

                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'BUY',
                            quantity, len(buy_signals)/4  # Adjusted signal strength
                        )

                        if trade:
                            self.learning_metrics['trades'] += 1
                            print(f"✅ Buy executed: {symbol}")

                # Check existing positions for sell signals
                if symbol in self.simulator.portfolio:
                    position = self.simulator.portfolio[symbol]
                    entry_price = position['entry_price']
                    profit_pct = (price - entry_price) / entry_price * 100

                    # Enhanced sell conditions
                    sell_signals = []
                    if profit_pct >= 3.0:  # Take profit at 3%
                        sell_signals.append(f"Take profit {profit_pct:.2f}%")
                    if profit_pct <= -1.5:  # Stop loss at -1.5%
                        sell_signals.append(f"Stop loss {profit_pct:.2f}%")
                    if rsi > 70:  # Overbought
                        sell_signals.append("Overbought (RSI)")
                    if macd < macd_signal:  # MACD crossover down
                        sell_signals.append("MACD Crossover Down")

                    if sell_signals:
                        print(f"🎯 Sell signals: {', '.join(sell_signals)}")

                        trade = self.simulator.execute_trade(
                            date, symbol, price, 'SELL',
                            position['quantity'], profit_pct/100
                        )

                        if trade:
                            if profit_pct > 0:
                                self.learning_metrics['successful_trades'] += 1
                                self.learning_metrics['accuracy'].append(1)
                                print(f"✅ Profitable sell: {symbol} ({profit_pct:.2f}%)")
                            else:
                                self.learning_metrics['accuracy'].append(0)
                                print(f"🔻 Loss taken: {symbol} ({profit_pct:.2f}%)")

            # Print daily summary
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            print(f"\n📊 Daily Summary for {date}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Cash Balance: ${self.simulator.balance:.2f}")
            print(f"Open Positions: {len(self.simulator.portfolio)}")
            print(f"Success Rate: {self.get_success_rate():.2f}%")

        except Exception as e:
            print(f"❌ Error in execute_trading_day: {str(e)}")
            traceback.print_exc()

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

def main():
    # Initialize the trading bot
    bot = TradingBot(initial_balance=1000)
    
    # Set up backtest parameters for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']  # Added more symbols for better testing
    
    # Create fetcher instance and get historical data
    fetcher = HistoricalDataFetcher(start_date, end_date, symbols)
    historical_data = fetcher.fetch_historical_data()
    
    # Run backtest
    bot.backtest(start_date, end_date)

if __name__ == "__main__":
    main()