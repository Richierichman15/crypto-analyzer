import requests
import pandas as pd
from config.settings import COIN_API_URL, HEADERS, DEFAULT_PARAMS

class DataFetcher:
    def __init__(self, min_market_cap, max_market_cap):
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap

    def scrape_market_data(self):
        """Fetch low to mid market cap coins with specific filtering"""
        try:
            # Debug prints
            print("\nAPI Request Details:")
            print(f"URL: {COIN_API_URL}")
            print(f"Headers: {HEADERS}")
            print(f"Params: {DEFAULT_PARAMS}")

            response = requests.get(COIN_API_URL, headers=HEADERS, params=DEFAULT_PARAMS)
            
            # Debug response
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nReceived {len(data)} coins from API")  # Debug print
                
                if not data:
                    print("API returned empty data")
                    return pd.DataFrame()
                
                # Print first coin as sample
                if data:
                    print("\nSample coin data:")
                    print(f"Name: {data[0].get('name')}")
                    print(f"Symbol: {data[0].get('symbol')}")
                    print(f"Market Cap: ${data[0].get('market_cap', 0):,.2f}")

                filtered_coins = []
                for coin in data:
                    market_cap = coin.get('market_cap', 0)
                    total_volume = coin.get('total_volume', 0)
                    
                    if (self.min_market_cap <= market_cap <= self.max_market_cap and 
                        total_volume > 0):
                        
                        filtered_coins.append({
                            "name": coin['name'],
                            "symbol": coin['symbol'],
                            "market_cap": market_cap,
                            "total_volume": total_volume,
                            "price_change_24h": coin.get('price_change_percentage_24h', 0),
                            "current_price": coin.get('current_price', 0),
                            "market_cap_rank": coin.get('market_cap_rank', 0),
                            "price_change_7d": coin.get('price_change_percentage_7d_in_currency', 0)
                        })
                
                print(f"\nFiltered down to {len(filtered_coins)} coins")  # Debug print
                return pd.DataFrame(filtered_coins)
            else:
                print(f"Full Response Text: {response.text}")
                raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()