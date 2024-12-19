import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class HistoricalDataFetcher:
    def __init__(self, start_date, end_date, symbols):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        
    def fetch_historical_data(self):
        """Fetch historical data for multiple cryptocurrencies"""
        all_data = {}
        
        for symbol in self.symbols:
            try:
                # Format symbol for Yahoo Finance (add -USD suffix)
                ticker = f"{symbol}-USD"
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                
                if not df.empty:
                    # Calculate additional metrics
                    df['Symbol'] = symbol
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Volatility'] = df['Daily_Return'].rolling(window=7).std()
                    df['Volume_MA'] = df['Volume'].rolling(window=7).mean()
                    
                    all_data[symbol] = df
                    print(f"Successfully fetched data for {symbol}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return all_data
