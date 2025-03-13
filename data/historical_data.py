import pandas as pd
import time
import requests  # Added for API calls
import os
import pickle
from datetime import datetime, timedelta
import numpy as np

class HistoricalDataFetcher:
    def __init__(self, cache_dir='data/cache'):
        """Initialize the historical data fetcher with cache directory"""
        self.delay = 0.25
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_filename(self, start_date, end_date):
        """Generate unique cache filename based on parameters"""
        if isinstance(start_date, pd.Timestamp) or isinstance(start_date, datetime):
            start = start_date.strftime('%Y%m%d')
        else:
            start = pd.Timestamp(start_date).strftime('%Y%m%d')
            
        if isinstance(end_date, pd.Timestamp) or isinstance(end_date, datetime):
            end = end_date.strftime('%Y%m%d')
        else:
            end = pd.Timestamp(end_date).strftime('%Y%m%d')
            
        return f"{self.cache_dir}/historical_data_{start}_{end}.pkl"
    
    def save_to_cache(self, data, start_date, end_date):
        """Save data to cache file"""
        try:
            cache_file = self.get_cache_filename(start_date, end_date)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✅ Data cached to {cache_file}")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def load_from_cache(self, start_date, end_date):
        """Load data from cache if available"""
        cache_file = self.get_cache_filename(start_date, end_date)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Loaded data from cache: {cache_file}")
                return data
            except Exception as e:
                print(f"❌ Error loading cache: {e}")
        return None

    def fetch_historical_data(self, symbol, start_date, end_date):
        """
        Fetch historical data for a single symbol
        
        Parameters:
        - symbol: The cryptocurrency symbol (e.g., 'BTC')
        - start_date: Start date (datetime, Timestamp, or date string)
        - end_date: End date (datetime, Timestamp, or date string)
        
        Returns:
        - DataFrame with historical data or None if not available
        """
        # Check cache for all symbols first
        all_data_cache = self.load_from_cache(start_date, end_date)
        if all_data_cache is not None and symbol in all_data_cache:
            return all_data_cache[symbol]
        
        # Convert dates to Unix timestamps
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
            
        from_timestamp = int(start_date.timestamp())
        to_timestamp = int(end_date.timestamp())
        
        # Get coin ID for the symbol
        try:
            response = requests.get("https://api.coingecko.com/api/v3/coins/list")
            all_coins = response.json()
            
            # Find the coin ID (case-insensitive match)
            coin_id = None
            symbol_upper = symbol.upper()
            for coin in all_coins:
                if coin['symbol'].upper() == symbol_upper:
                    coin_id = coin['id']
                    break
                    
            if not coin_id:
                print(f"❌ Could not find coin ID for symbol {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error finding coin ID: {str(e)}")
            return None
            
        # Fetch historical data using the coin ID
        try:
            print(f"📊 Fetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
            
            response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range", params={
                'vs_currency': 'usd',
                'from': from_timestamp,
                'to': to_timestamp
            })
            
            if response.status_code != 200:
                print(f"❌ API error: {response.status_code}")
                return None
                
            data = response.json()
            
            # Convert to DataFrame
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
            
            df = pd.merge(prices_df, volumes_df, on='timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Calculate additional metrics
            df['daily_return'] = df['Close'].pct_change()
            df['volume_change'] = df['Volume'].pct_change()
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            print(f"✅ Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None
            
    def fetch_historical_data_batch(self, symbols, start_date, end_date):
        """
        Fetch historical data for multiple symbols with caching
        
        Parameters:
        - symbols: List of symbols to fetch
        - start_date: Start date
        - end_date: End date
        
        Returns:
        - Dictionary with symbol as key and DataFrame as value
        """
        # Try to load from cache first
        cached_data = self.load_from_cache(start_date, end_date)
        if cached_data is not None:
            # Filter for requested symbols
            return {sym: df for sym, df in cached_data.items() if sym in symbols}
        
        # Fetch data for each symbol
        historical_data = {}
        for symbol in symbols:
            df = self.fetch_historical_data(symbol, start_date, end_date)
            if df is not None:
                historical_data[symbol] = df
            time.sleep(self.delay)  # Delay between requests
            
        # Save to cache
        self.save_to_cache(historical_data, start_date, end_date)
        
        return historical_data

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df

    def get_batch_market_data(self, symbols, chunk_size=100):
        """Get market data in batches"""
        all_data = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                response = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
                    'vs_currency': 'usd',
                    'symbols': ','.join(chunk),
                    'per_page': chunk_size,
                    'page': 1
                })
                data = response.json()  # Get the JSON response
                all_data.extend(data)
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error fetching batch {i//chunk_size + 1}: {e}")
        return all_data

    def generate_mock_data(self, symbols, start_date, end_date):
        """Generate mock data for testing with a bearish trend"""
        try:
            historical_data = {}
            
            # Add 30 days of historical data before the start date for indicator calculation
            extended_start_date = start_date - pd.Timedelta(days=30)
            
            # Generate mock data for each symbol
            for symbol in symbols:
                # Create a date range for the backtest period with extended history
                date_range = pd.date_range(start=extended_start_date, end=end_date, freq='D')
                
                # Create a dataframe with the date range
                df = pd.DataFrame(index=date_range)
                
                # Generate a starting price based on the symbol
                if symbol == 'BTC':
                    base_price = 90000  # Start high for BTC
                elif symbol == 'ETH':
                    base_price = 2000
                else:
                    base_price = 100
                
                # Generate price data with a bearish trend (downward bias)
                price_data = []
                current_price = base_price
                
                for i in range(len(date_range)):
                    # Add a downward bias to simulate a bear market (more likely to go down than up)
                    if i > 0:
                        # 65% chance of going down, 35% chance of going up (bearish)
                        if np.random.random() < 0.65:
                            # Down move with random magnitude between 0.5% and 3%
                            pct_change = -np.random.uniform(0.005, 0.03)
                        else:
                            # Up move with random magnitude between 0.1% and 2% (smaller up moves in bear market)
                            pct_change = np.random.uniform(0.001, 0.02)
                        
                        # Apply daily trend overlay (sustained downtrend)
                        day_factor = -0.005  # Slight downward bias overall
                        
                        # Apply the changes
                        current_price = current_price * (1 + pct_change + day_factor)
                    
                    price_data.append(current_price)
                
                # Add price data to dataframe
                df['Open'] = price_data
                df['High'] = df['Open'] * (1 + np.random.uniform(0, 0.02, len(df)))
                df['Low'] = df['Open'] * (1 - np.random.uniform(0, 0.02, len(df)))
                df['Close'] = df['Open'] * (1 + np.random.normal(0, 0.01, len(df)))
                
                # Ensure Close is within High and Low
                df['Close'] = np.minimum(df['High'], np.maximum(df['Low'], df['Close']))
                
                # Generate volume data
                volume_base = 1000000 if symbol == 'BTC' else 5000000 if symbol == 'ETH' else 10000000
                df['Volume'] = np.random.uniform(0.5, 1.5, len(df)) * volume_base
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Calculate some basic indicators for the data
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Moving averages
                df['SMA_5'] = df['Close'].rolling(window=5).mean()
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                
                # MACD
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Store in historical data dictionary
                historical_data[symbol] = df
            
            print(f"✅ Generated mock data for {len(symbols)} symbols")
            return historical_data
            
        except Exception as e:
            print(f"❌ Error generating mock data: {str(e)}")
            return {}
