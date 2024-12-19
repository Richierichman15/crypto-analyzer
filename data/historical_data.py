from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import pickle
import json

class HistoricalDataFetcher:
    def __init__(self, start_date, end_date, symbols):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.cg = CoinGeckoAPI()
        self.delay = 0.25
        self.cache_dir = 'data/cache'  # Directory to store cached data
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_filename(self):
        """Generate unique cache filename based on parameters"""
        start = pd.Timestamp(self.start_date).strftime('%Y%m%d')
        end = pd.Timestamp(self.end_date).strftime('%Y%m%d')
        return f"{self.cache_dir}/historical_data_{start}_{end}.pkl"
    
    def save_to_cache(self, data):
        """Save data to cache file"""
        try:
            cache_file = self.get_cache_filename()
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"\n✅ Data cached successfully to {cache_file}")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def load_from_cache(self):
        """Load data from cache if available"""
        cache_file = self.get_cache_filename()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"\n✅ Loaded data from cache: {cache_file}")
                return data
            except Exception as e:
                print(f"❌ Error loading cache: {e}")
        return None

    def fetch_historical_data(self):
        """Fetch historical data with caching"""
        # Try to load from cache first
        cached_data = self.load_from_cache()
        if cached_data is not None:
            return cached_data
        
        print("\n🔄 No cache found, fetching fresh data...")
        historical_data = {}
        
        # Convert dates to Unix timestamps
        from_timestamp = int(pd.Timestamp(self.start_date).timestamp())
        to_timestamp = int(pd.Timestamp(self.end_date).timestamp())
        
        print(f"\nFetching historical data for {len(self.symbols)} symbols...")
        
        # Get all coin IDs in one batch request
        try:
            all_coins = self.cg.get_coins_markets(
                vs_currency='usd',
                per_page=250,
                page=1
            )
            symbol_to_id = {
                coin['symbol'].upper(): coin['id'] 
                for coin in all_coins
            }
        except Exception as e:
            print(f"Error fetching coin list: {e}")
            return {}

        # Process each symbol with progress bar
        for symbol in tqdm(self.symbols, desc="Fetching Historical Data"):
            try:
                if symbol not in symbol_to_id:
                    print(f"\n⚠️ No coin ID found for {symbol}")
                    continue
                
                coin_id = symbol_to_id[symbol]
                
                # Fetch historical data
                data = self.cg.get_coin_market_chart_range_by_id(
                    id=coin_id,
                    vs_currency='usd',
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp
                )
                
                # Convert to DataFrame efficiently
                prices_df = pd.DataFrame(data['prices'], 
                                      columns=['timestamp', 'Close'])
                volumes_df = pd.DataFrame(data['total_volumes'], 
                                       columns=['timestamp', 'Volume'])
                
                # Merge price and volume data
                df = pd.merge(prices_df, volumes_df, on='timestamp')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['symbol'] = symbol
                
                historical_data[symbol] = df
                
                # Shorter delay between requests
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"\n❌ Error fetching data for {symbol}: {str(e)}")
                continue
            
        print(f"\n✅ Successfully fetched data for {len(historical_data)} symbols")
        
        # Save to cache
        self.save_to_cache(historical_data)
        
        return historical_data

    def get_batch_market_data(self, symbols, chunk_size=100):
        """Get market data in batches"""
        all_data = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                data = self.cg.get_coins_markets(
                    vs_currency='usd',
                    symbols=chunk,
                    per_page=chunk_size,
                    page=1
                )
                all_data.extend(data)
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error fetching batch {i//chunk_size + 1}: {e}")
        return all_data

    def calculate_indicators(self, df):
        """Calculate technical indicators for trend following"""
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
