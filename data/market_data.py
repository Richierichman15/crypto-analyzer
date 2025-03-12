import os
import json
import pandas as pd
import requests
from datetime import datetime
import time
import random

class MarketDataFetcher:
    """Fetches current market data for cryptocurrencies"""
    
    def __init__(self, cache_dir="data/cache"):
        """Initialize the market data fetcher"""
        self.cache_dir = cache_dir
        self.api_url = "https://api.coingecko.com/api/v3"
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def fetch_top_coins(self, limit=10, use_cache=True, cache_expiry_hours=24):
        """
        Fetch top cryptocurrencies by market cap
        
        Parameters:
        - limit: Number of top coins to fetch
        - use_cache: Whether to use cached data if available
        - cache_expiry_hours: Cache expiry in hours
        
        Returns:
        - List of dictionaries with coin data
        """
        cache_file = f"{self.cache_dir}/market_data_top_{limit}.json"
        
        # Check cache first
        if use_cache and os.path.exists(cache_file):
            file_age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 3600
            
            if file_age_hours < cache_expiry_hours:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        print(f"📊 Using cached market data ({file_age_hours:.1f} hours old)")
                        return cached_data
                except Exception as e:
                    print(f"⚠️ Error reading cache: {str(e)}")
        
        try:
            # Add a random delay to avoid API rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
            # Fetch market data from CoinGecko API
            response = requests.get(
                f"{self.api_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": limit,
                    "page": 1,
                    "sparkline": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format data for our use
                coins = []
                for coin in data:
                    coins.append({
                        "id": coin["id"],
                        "symbol": coin["symbol"].upper(),
                        "name": coin["name"],
                        "market_cap": coin["market_cap"],
                        "price": coin["current_price"],
                        "volume_24h": coin["total_volume"],
                        "price_change_24h": coin["price_change_percentage_24h"]
                    })
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump(coins, f)
                
                print(f"📊 Fetched {len(coins)} coins from API")
                return coins
            else:
                print(f"❌ API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching market data: {str(e)}")
            return []
    
    def fetch_coin_details(self, coin_id):
        """
        Fetch detailed information for a specific coin
        
        Parameters:
        - coin_id: CoinGecko coin ID (e.g., 'bitcoin')
        
        Returns:
        - Dictionary with coin details
        """
        cache_file = f"{self.cache_dir}/coin_details_{coin_id}.json"
        
        # Check cache first (only valid for 6 hours for details)
        if os.path.exists(cache_file):
            file_age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 3600
            
            if file_age_hours < 6:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        return cached_data
                except Exception:
                    pass
        
        try:
            # Add a random delay to avoid API rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
            # Fetch coin details from CoinGecko API
            response = requests.get(
                f"{self.api_url}/coins/{coin_id}",
                params={
                    "localization": False,
                    "tickers": False,
                    "market_data": True,
                    "community_data": False,
                    "developer_data": False,
                    "sparkline": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format data for our use
                details = {
                    "id": data["id"],
                    "symbol": data["symbol"].upper(),
                    "name": data["name"],
                    "market_cap": data["market_data"]["market_cap"]["usd"],
                    "price": data["market_data"]["current_price"]["usd"],
                    "volume_24h": data["market_data"]["total_volume"]["usd"],
                    "price_change_24h": data["market_data"]["price_change_percentage_24h"],
                    "price_change_7d": data["market_data"]["price_change_percentage_7d"],
                    "price_change_30d": data["market_data"]["price_change_percentage_30d"],
                    "ath": data["market_data"]["ath"]["usd"],
                    "ath_change_percentage": data["market_data"]["ath_change_percentage"]["usd"],
                    "market_cap_rank": data["market_data"]["market_cap_rank"]
                }
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump(details, f)
                
                return details
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error fetching coin details: {str(e)}")
            return None 