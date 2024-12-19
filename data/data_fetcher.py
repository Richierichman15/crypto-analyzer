from pycoingecko import CoinGeckoAPI
import pandas as pd

class DataFetcher:
    def __init__(self, min_market_cap=5_000_000, max_market_cap=500_000_000):
        self.cg = CoinGeckoAPI()  # Initialize CoinGecko client
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        
    def scrape_market_data(self):
        """Get coins within our market cap range with good trading volume"""
        try:
            print("\nFetching market data from CoinGecko...")
            
            # Get coin market data
            coins = self.cg.get_coins_markets(
                vs_currency='usd',
                order='volume_desc',  # Order by volume
                per_page=250,         # Get more coins
                sparkline=False,
                price_change_percentage='24h'
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(coins)
            
            # Filter by market cap
            filtered_df = df[
                (df['market_cap'] >= self.min_market_cap) &
                (df['market_cap'] <= self.max_market_cap) &
                (df['total_volume'] > 1_000_000)  # Minimum daily volume $1M
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