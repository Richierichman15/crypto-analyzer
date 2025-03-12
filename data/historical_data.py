import pandas as pd
import time
import requests  # Added for API calls
import os
import pickle

class HistoricalDataFetcher:
    def __init__(self, start_date, end_date, symbols):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
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
        """Fetch historical data with detailed coin information"""
        # Try to load from cache first
        cached_data = self.load_from_cache()
        if cached_data is not None:
            print("\n🔍 Analyzing cached data quality:")
            for symbol, df in cached_data.items():
                print(f"\n{symbol} Analysis:")
                print(f"Data points: {len(df)}")
                print(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
                print(f"Daily returns range: {df['Close'].pct_change().min():.2%} to {df['Close'].pct_change().max():.2%}")
                print(f"Volume range: ${df['Volume'].min():.2f} to ${df['Volume'].max():.2f}")
            return cached_data
        
        print("\n🔄 No cache found, fetching fresh data...")
        historical_data = {}
        
        # Convert dates to Unix timestamps
        from_timestamp = int(pd.Timestamp(self.start_date).timestamp())
        to_timestamp = int(pd.Timestamp(self.end_date).timestamp())
        
        # Get detailed coin information
        try:
            print("\n📊 Fetching coin details...")
            response = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
                'vs_currency': 'usd',
                'per_page': 250,
                'page': 1,
                'sparkline': 'false'
            })
            all_coins = response.json()  # Get the JSON response
            
            # Create detailed symbol mapping
            symbol_to_details = {}
            for coin in all_coins:
                symbol = coin['symbol'].upper()
                symbol_to_details[symbol] = {
                    'id': coin['id'],
                    'name': coin['name'],
                    'market_cap': coin['market_cap'],
                    'current_price': coin['current_price'],
                    'volume': coin['total_volume']
                }
                
            print(f"\nFound {len(symbol_to_details)} coins in total")
            
        except Exception as e:
            print(f"Error fetching coin list: {e}")
            return {}

        # Process each symbol without progress bar
        for symbol in self.symbols:
            try:
                if symbol not in symbol_to_details:
                    print(f"\n⚠️ No coin details found for {symbol}")
                    continue
                
                coin_details = symbol_to_details[symbol]
                print(f"\n📈 Processing {symbol} ({coin_details['name']}):")
                print(f"Market Cap: ${coin_details['market_cap']:,.2f}")
                print(f"Current Price: ${coin_details['current_price']:,.8f}")
                print(f"24h Volume: ${coin_details['volume']:,.2f}")
                
                # Fetch historical data using your API
                response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_details['id']}/market_chart/range", params={
                    'vs_currency': 'usd',
                    'from': from_timestamp,
                    'to': to_timestamp
                })
                data = response.json()  # Get the JSON response
                
                # Convert to DataFrame with additional information
                prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
                volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
                
                df = pd.merge(prices_df, volumes_df, on='timestamp')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add coin details
                df['symbol'] = symbol
                df['name'] = coin_details['name']
                df['market_cap'] = coin_details['market_cap']
                
                # Calculate additional metrics
                df['daily_return'] = df['Close'].pct_change()
                df['volume_change'] = df['Volume'].pct_change()
                
                historical_data[symbol] = df
                
                # Print data summary
                print(f"Data points: {len(df)}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Price range: ${df['Close'].min():.8f} to ${df['Close'].max():.8f}")
                
                time.sleep(self.delay)  # Delay between requests
                
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
