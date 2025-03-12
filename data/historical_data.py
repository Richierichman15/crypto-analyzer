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
        """
        Generate mock data for testing when API access is limited
        
        Parameters:
        - symbols: List of symbols to generate data for
        - start_date: Start date
        - end_date: End date
        
        Returns:
        - Dictionary with symbol as key and DataFrame as value
        """
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
            
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate mock data for each symbol
        historical_data = {}
        
        # Different price trends and volatilities for each symbol
        trends = {
            'BTC': {'start_price': 60000, 'trend': -0.001, 'volatility': 0.02},  # Downward trend
            'ETH': {'start_price': 2200, 'trend': 0.0005, 'volatility': 0.015},  # Slight upward trend
            'SOL': {'start_price': 80, 'trend': -0.0015, 'volatility': 0.03},     # Stronger downward trend
            'ADA': {'start_price': 0.5, 'trend': 0.001, 'volatility': 0.025},     # Upward trend
            'DOT': {'start_price': 6, 'trend': -0.0005, 'volatility': 0.02},     # Slight downward trend
            'XRP': {'start_price': 0.6, 'trend': 0.0, 'volatility': 0.018},      # Neutral trend
            'AVAX': {'start_price': 20, 'trend': -0.002, 'volatility': 0.025},   # Strong downward trend
            'MATIC': {'start_price': 0.8, 'trend': 0.0008, 'volatility': 0.02}   # Moderate upward trend
        }
        
        # Default values for symbols not in the trends dictionary
        default_trend = {'start_price': 100, 'trend': 0.0, 'volatility': 0.02}
        
        for symbol in symbols:
            # Get trend parameters or use defaults
            params = trends.get(symbol, default_trend)
            
            # Generate price movement with random walk
            price = params['start_price']
            prices = []
            volumes = []
            
            for i in range(len(date_range)):
                # Apply trend and volatility
                daily_return = params['trend'] + np.random.normal(0, params['volatility'])
                price = price * (1 + daily_return)
                prices.append(price)
                
                # Generate random volume
                volume = price * (1 + np.random.normal(0, 0.5)) * 10000
                volumes.append(volume)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Close': prices,
                'Volume': volumes,
                'symbol': symbol
            }, index=date_range)
            
            # Calculate additional metrics
            df['daily_return'] = df['Close'].pct_change()
            df['volume_change'] = df['Volume'].pct_change()
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            historical_data[symbol] = df
            
        print(f"✅ Generated mock data for {len(historical_data)} symbols")
        return historical_data
