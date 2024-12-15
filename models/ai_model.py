import xgboost as xgb
from sklearn.model_selection import train_test_split

class CryptoModel:
    def __init__(self):
        self.model = None

    def train_ai_model(self, X, y):
        """Train XGBoost model for crypto investment prediction"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5
        )
        
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        print(f"Model Training Accuracy: {score}")
        
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise Exception("Model not trained yet! Call train_ai_model first.")
        return self.model.predict(X)
