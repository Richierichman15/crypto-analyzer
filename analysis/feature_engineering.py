from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_training_data(self, market_data):
        """Prepare data for AI model training"""
        try:
            # Create features
            X = pd.DataFrame({
                'market_cap': market_data['market_cap'],
                'volume': market_data['total_volume'],
                'price_change_24h': market_data['price_change_24h'],
                'market_cap_rank': market_data['market_cap_rank']
            })
            
            # Create target variable (you can adjust this formula)
            y = (market_data['price_change_24h'] / 100.0)  # Convert percentage to decimal
            
            # Clean data
            X = X.fillna(0)  # Replace NaN with 0
            y = y.fillna(0)  # Replace NaN with 0
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale large numbers
            X['market_cap'] = np.log1p(X['market_cap'])
            X['volume'] = np.log1p(X['volume'])
            
            print("\nData Preparation Summary:")
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            print("\nFeature Statistics:")
            print(X.describe())
            
            return X, y
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return pd.DataFrame(), pd.Series()