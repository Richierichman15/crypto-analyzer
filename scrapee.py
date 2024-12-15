import requests
import os
from dotenv import load_dotenv

load_dotenv()


def scrape_market_data():
    """Fetch low market cap coins and filter them based on certain criteria."""
    COIN_API_URL = "https://api.coingecko.com/api/v3/coins/markets"
    API_KEY = os.getenv("APIKEY")
    print(API_KEY)
    HEADERS = {
        'accept': 'application/json',
        "x-cg-pro-api-key": API_KEY 
    }

    params = {
        "vs_currency": "usd", 
        "order": "market_cap_asc",  
        "per_page": 20,  
        "page": 1,  
        "sparkline": "false",  
        "price_change_percentage": "1hr,24h,7d"  
    }

    try:
        
        response = requests.get(COIN_API_URL, headers=HEADERS, params=params)
        print("data", response)
        
        if response.status_code == 200:
            data = response.json()
            print("data", data)
            
            
            filtered_coins = []
            for coin in data:
                market_cap = coin.get('market_cap')
                total_volume = coin.get('total_volume')
                
                
                if total_volume:
                    volume_market_cap_ratio = total_volume
                    
                  
                    # if 0.5 <= volume_market_cap_ratio <= 1:
                    filtered_coins.append({
                        "name": coin['name'],
                        "symbol": coin['symbol'],
                        "market_cap": market_cap,
                        "total_volume": total_volume,
                        "volume_market_cap_ratio": round(volume_market_cap_ratio, 2),
                        "price_change_24h": coin.get('price_change_percentage_24h'),
                        "current_price": coin.get('current_price'),
                        "market_cap_rank": coin.get('market_cap_rank')
                    })
            
            return filtered_coins
        else:
            raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return []