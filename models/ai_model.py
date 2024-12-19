import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class CryptoModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train_ai_model(self, X, y):
        """Train XGBoost model for crypto investment prediction"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Define and train the model with fixed hyperparameters
        self.model = xgb.XGBRegressor(
            learning_rate=0.1,
            n_estimators=200,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n📊 Model Training Results")
        print("=" * 30)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = dict(zip(
            ['Market Cap', 'Volume', '24h Change', 'Market Rank'],
            self.model.feature_importances_
        ))
        
        print("\n🎯 Feature Importance")
        print("=" * 30)
        for feature, importance in sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"{feature}: {importance:.4f}")
        
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise Exception("Model not trained yet! Call train_ai_model first.")
        
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
