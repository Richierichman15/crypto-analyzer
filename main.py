from config.settings import MIN_MARKET_CAP, MAX_MARKET_CAP
from data.data_fetcher import DataFetcher
from data.news_fetcher import NewsFetcher
from models.ai_model import CryptoModel
from analysis.feature_engineering import FeatureEngineer
from analysis.risk_analysis import RiskAnalyzer
import os

class CryptoAITrader:
    def __init__(self):
        self.data_fetcher = DataFetcher(MIN_MARKET_CAP, MAX_MARKET_CAP)
        self.model = CryptoModel()
        self.feature_engineer = FeatureEngineer()
        self.risk_analyzer = RiskAnalyzer()
        self.news_fetcher = NewsFetcher(api_key=os.getenv("NEWS_API_KEY"))

    def analyze_crypto_opportunities(self):
        """Advanced crypto opportunity analysis"""
        market_data = self.data_fetcher.scrape_market_data()
        
        if market_data.empty:
            print("No market data available")
            return []
        
        X, y = self.feature_engineer.prepare_training_data(market_data)
        self.model.train_ai_model(X, y)
        predictions = self.model.predict(X)
        
        market_data['investment_potential'] = predictions
        top_opportunities = market_data.sort_values('investment_potential', ascending=False).head(10)
        
        print("\n🚀 Top Crypto Investment Opportunities 🚀")
        for _, coin in top_opportunities.iterrows():
            print(f"\n{coin['name']} ({coin['symbol'].upper()})")
            print(f"Market Cap: ${coin['market_cap']:,.2f}")
            print(f"Current Price: ${coin['current_price']:.4f}")
            print(f"24h Price Change: {coin['price_change_24h']:.2f}%")
            print(f"Investment Potential Score: {coin['investment_potential']:.4f}")
            
            # Fetch news using the News API
            self.news_fetcher.fetch_news(query=coin['name'])
        
        return top_opportunities

if __name__ == "__main__":
    print("Current directory:", os.getcwd())
    print("API Key present:", bool(os.getenv("APIKEY")))
    trader = CryptoAITrader()
    top_opportunities = trader.analyze_crypto_opportunities()
    trader.risk_analyzer.risk_assessment(top_opportunities)