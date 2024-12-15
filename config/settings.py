import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug prints
API_KEY = os.getenv("APIKEY")
print("\nAPI Key Debug:")
print(f"First 5 chars of API key: {API_KEY[:5] if API_KEY else 'No Key Found'}")

# API Configuration
COIN_API_URL = "https://api.coingecko.com/api/v3/coins/markets"

# Market Cap Limits
MIN_MARKET_CAP = 5_000_000
MAX_MARKET_CAP = 1_000_000_000

# API Request Parameters
DEFAULT_PARAMS = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 250,
    "page": 1,
    "sparkline": "false",
    "price_change_percentage": "1hr,24h,7d"
}

# Headers
HEADERS = {
    'accept': 'application/json',
    'x-cg-demo-api-key': API_KEY
}