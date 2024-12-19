from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        logging.basicConfig(level=logging.INFO)

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
            
            # Create target variable
            y = (market_data['price_change_24h'] / 100.0)
            
            # Clean data
            X = X.fillna(0)
            y = y.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X['market_cap'] = np.log1p(X['market_cap'])
            X['volume'] = np.log1p(X['volume'])
            X_scaled = self.scaler.fit_transform(X)
            
            logging.info("Data Preparation Summary:")
            logging.info(f"Features shape: {X.shape}")
            logging.info(f"Target shape: {y.shape}")
            logging.info("Feature Statistics:")
            logging.info(X.describe())
            
            return X_scaled, y
            
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")
            return pd.DataFrame(), pd.Series()

# Example usage
if __name__ == "__main__":
    # Mock data for testing
    mock_data = pd.DataFrame({
        'market_cap': [1e9, 5e8, 2e9],
        'total_volume': [1e7, 5e6, 2e7],
        'price_change_24h': [5, -3, 2],
        'market_cap_rank': [1, 2, 3]
    })
    engineer = FeatureEngineer()
    X, y = engineer.prepare_training_data(mock_data)
    print(X, y)