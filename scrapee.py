import requests
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

class CryptoAITrader:
    def __init__(self, min_market_cap=50_000, max_market_cap=1_000_000_000):
        self.model = None
        self.scaler = StandardScaler()
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap

    def scrape_market_data(self):
        """Fetch low to mid market cap coins with specific filtering"""
        COIN_API_URL = "https://api.coingecko.com/api/v3/coins/markets"
        API_KEY = os.getenv("APIKEY")
        
        HEADERS = {
            'accept': 'application/json',
            "x-cg-pro-api-key": API_KEY 
        }

        params = {
            "vs_currency": "usd", 
            "order": "market_cap_asc",  
            "per_page": 250,  # Increased to capture more potential coins
            "page": 1,  
            "sparkline": "false",  
            "price_change_percentage": "1hr,24h,7d"  
        }

        try:
            response = requests.get(COIN_API_URL, headers=HEADERS, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                filtered_coins = []
                for coin in data:
                    market_cap = coin.get('market_cap', 0)
                    total_volume = coin.get('total_volume', 0)
                    
                    # Filter for micro, small, and mid-cap coins
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
                
                return pd.DataFrame(filtered_coins)
            else:
                raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, df):
        """Advanced feature engineering for investment potential"""
        # Feature engineering
        df['volume_market_cap_ratio'] = df['total_volume'] / df['market_cap']
        df['price_momentum'] = (
            df['price_change_24h'] * 0.5 + 
            df['price_change_7d'] * 0.5
        )
        
        # Investment scoring mechanism
        df['investment_score'] = (
            # Volume to Market Cap Ratio (liquidity)
            df['volume_market_cap_ratio'] * 0.4 +
            
            # Price Momentum 
            (1 - abs(df['price_momentum'] / 100)) * 0.3 +
            
            # Inverse Market Cap Rank (smaller is better)
            (1 / df['market_cap_rank']) * 0.3
        )
        
        # Select features for training
        features = [
            'market_cap', 
            'total_volume', 
            'volume_market_cap_ratio', 
            'price_momentum'
        ]
        
        X = df[features]
        y = df['investment_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def train_ai_model(self, X, y):
        """Train XGBoost model for crypto investment prediction"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        score = self.model.score(X_test, y_test)
        print(f"Model Training Accuracy: {score}")
        
        return self.model

    def analyze_crypto_opportunities(self):
        """Advanced crypto opportunity analysis"""
        # Fetch market data
        market_data = self.scrape_market_data()
        
        if market_data.empty:
            print("No market data available")
            return []
        
        # Prepare training data
        X, y = self.prepare_training_data(market_data)
        
        # Train AI model
        self.train_ai_model(X, y)
        
        # Predict investment potential
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        market_data['investment_potential'] = predictions
        
        # Sort and recommend top opportunities
        top_opportunities = market_data.sort_values('investment_potential', ascending=False).head(10)
        
        print("\n🚀 Top Crypto Investment Opportunities 🚀")
        for _, coin in top_opportunities.iterrows():
            print(f"\n{coin['name']} ({coin['symbol'].upper()})")
            print(f"Market Cap: ${coin['market_cap']:,.2f}")
            print(f"Current Price: ${coin['current_price']:.4f}")
            print(f"24h Price Change: {coin['price_change_24h']:.2f}%")
            print(f"Investment Potential Score: {coin['investment_potential']:.4f}")
        
        return top_opportunities

    def risk_assessment(self, opportunities):
        """Additional risk assessment for top picks"""
        risk_analysis = []
        
        for _, coin in opportunities.iterrows():
            risk_level = self.calculate_risk(coin)
            risk_analysis.append({
                'name': coin['name'],
                'symbol': coin['symbol'],
                'risk_level': risk_level
            })
        
        print("\n⚠️ Risk Assessment:")
        for analysis in risk_analysis:
            print(f"{analysis['name']} ({analysis['symbol'].upper()}): Risk Level - {analysis['risk_level']}")
        
        return risk_analysis

    def calculate_risk(self, coin):
        """Detailed risk calculation"""
        # Multiple risk factors
        volume_risk = 1 / np.log(coin['total_volume'] + 1)
        market_cap_risk = 1 / np.log(coin['market_cap'] + 1)
        price_volatility = abs(coin['price_change_24h']) / 10
        
        # Composite risk score
        risk_score = (volume_risk * 0.3 + 
                      market_cap_risk * 0.4 + 
                      price_volatility * 0.3)
        
        # Risk categorization
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.5:
            return "Moderate Risk"
        else:
            return "High Risk"

# Main execution
if __name__ == "__main__":
    # Initialize trader with market cap range
    trader = CryptoAITrader(
        min_market_cap=50_000,      # $50,000 minimum
        max_market_cap=1_000_000_000  # $1 billion maximum
    )
    
    # Analyze crypto opportunities
    top_opportunities = trader.analyze_crypto_opportunities()
    
    # Perform risk assessment
    trader.risk_assessment(top_opportunities)