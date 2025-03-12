import pandas as pd
import requests

class DataFetcher:
    def __init__(self, min_market_cap=5_000_000, max_market_cap=500_000_000):
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        
    def fetch_market_data(self, symbols):
        """Fetch market data for the given symbols from CoinGecko."""
        try:
            print("Fetching market data from CoinGecko...")
            response = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
                'vs_currency': 'usd',
                'symbols': ','.join(symbols),
                'order': 'market_cap_desc',
                'per_page': len(symbols),
                'page': 1,
                'sparkline': 'false'
            })
            data = response.json() 
            
            if 'error' in data:
                print(f"Error fetching market data: {data['error']}")
                return None
            
            return data  

        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            return None

    def scrape_market_data(self):
        """Get coins within our market cap range with good trading volume"""
        try:
            print("\nFetching market data from CoinGecko...")
            
            coins = self.fetch_market_data([]) 
            
            if coins is None:
                return pd.DataFrame()  
            
            df = pd.DataFrame(coins)
            
            filtered_df = df[
                (df['market_cap'] >= self.min_market_cap) &
                (df['market_cap'] <= self.max_market_cap) &
                (df['total_volume'] > 1_000_000)  
            ]
            
            if filtered_df.empty:
                print("❌ No coins found matching criteria")
                return pd.DataFrame()
                
            print(f"\nReceived {len(df)} coins from API")
            print(f"\nSample coin data:")
            print(f"Name: {df.iloc[0]['name']}")
            print(f"Symbol: {df.iloc[0]['symbol']}")
            print(f"Market Cap: ${df.iloc[0]['market_cap']:,.2f}")
            print(f"\nFiltered down to {len(filtered_df)} coins")
            
            return filtered_df
            
        except Exception as e:
            print(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()