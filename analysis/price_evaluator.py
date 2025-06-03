import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import pickle

class McLarenPriceEvaluator:
    """Standalone McLaren Price Evaluator using Top 3 ML Models"""
    
    def __init__(self):
        self.models = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def train_models(self, data_path='mclaren_us_processed_final.csv'):
        """Train the top 3 models on the McLaren dataset"""
        print("🏎️  Training McLaren Price Evaluator...")
        
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Remove extreme outliers
        df = df[df['price'] <= 50000000]
        
        # Feature engineering
        df_features = self._feature_engineering(df)
        X, y = self._prepare_features(df_features)
        
        # Train top 3 models
        models_config = {
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                'mae_weight': 1/94343  # Inverse of MAE for weighting
            },
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
                'mae_weight': 1/124498
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                'mae_weight': 1/135466
            }
        }
        
        trained_models = {}
        for name, config in models_config.items():
            print(f"   Training {name}...")
            model = config['model']
            model.fit(X, y)
            trained_models[name] = {
                'model': model,
                'weight': config['mae_weight']
            }
        
        self.models = trained_models
        self.is_trained = True
        print("✅ Training complete!")
        
    def _feature_engineering(self, df):
        """Create features for price prediction"""
        df_features = df.copy()
        
        # Age calculation
        current_year = datetime.now().year
        df_features['age'] = current_year - df_features['year']
        df_features['age_squared'] = df_features['age'] ** 2
        
        # Mileage features
        df_features['mileage_per_year'] = df_features['mileage'] / (df_features['age'] + 1)
        df_features['low_mileage'] = (df_features['mileage'] < 5000).astype(int)
        df_features['high_mileage'] = (df_features['mileage'] > 30000).astype(int)
        
        # Model category features
        hypercar_models = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
        track_models = ['765LT', '675LT', '600LT', '620R']
        entry_models = ['MP4-12C', '12C Spider', '12C Coupe']
        grand_tourer = ['570GT', 'GT']
        
        df_features['is_hypercar'] = df_features['model'].isin(hypercar_models).astype(int)
        df_features['is_track_focused'] = df_features['model'].isin(track_models).astype(int)
        df_features['is_entry_level'] = df_features['model'].isin(entry_models).astype(int)
        df_features['is_grand_tourer'] = df_features['model'].isin(grand_tourer).astype(int)
        
        # Market timing features
        df_features['listing_date'] = pd.to_datetime(df_features['listing_date'], format='mixed', errors='coerce')
        df_features['listing_year'] = df_features['listing_date'].dt.year
        df_features['listing_month'] = df_features['listing_date'].dt.month
        df_features['days_since_launch'] = (df_features['listing_date'] - 
                                          pd.to_datetime('2011-01-01')).dt.days
        
        # State economic indicators
        high_income_states = ['CA', 'NY', 'FL', 'TX', 'IL']
        df_features['high_income_state'] = df_features['state'].isin(high_income_states).astype(int)
        
        # Seller reputation
        top_dealers = df_features['seller_name'].value_counts().head(10).index
        df_features['top_dealer'] = df_features['seller_name'].isin(top_dealers).astype(int)
        
        # Market status
        df_features['is_sold'] = (df_features['sold'] == 'y').astype(int)
        
        return df_features
    
    def _prepare_features(self, df_features):
        """Prepare features for ML models"""
        feature_columns = [
            'age', 'age_squared', 'mileage', 'mileage_per_year', 'year',
            'low_mileage', 'high_mileage', 'is_hypercar', 'is_track_focused',
            'is_entry_level', 'is_grand_tourer', 'listing_year', 'listing_month',
            'days_since_launch', 'high_income_state', 'top_dealer', 'is_sold'
        ]
        
        categorical_features = ['model', 'state', 'transmission']
        
        X = df_features[feature_columns].copy()
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df_features.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df_features[cat_feature].fillna('Unknown'))
                X[f'{cat_feature}_encoded'] = encoded_values
                self.label_encoders[cat_feature] = le
        
        X = X.fillna(X.median())
        self.feature_names = X.columns.tolist()
        
        return X, df_features['price']
    
    def predict_price(self, year, model, mileage, state='CA', transmission='Automatic', is_sold=False):
        """Predict McLaren price using ensemble of top 3 models"""
        
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        # Create feature vector
        current_year = datetime.now().year
        age = current_year - year
        
        # Basic features
        features = {
            'age': age,
            'age_squared': age ** 2,
            'mileage': mileage,
            'mileage_per_year': mileage / (age + 1) if age >= 0 else mileage,
            'year': year,
            'low_mileage': 1 if mileage < 5000 else 0,
            'high_mileage': 1 if mileage > 30000 else 0,
            'listing_year': current_year,
            'listing_month': datetime.now().month,
            'days_since_launch': (datetime.now() - datetime(2011, 1, 1)).days,
            'high_income_state': 1 if state in ['CA', 'NY', 'FL', 'TX', 'IL'] else 0,
            'top_dealer': 0,  # Conservative assumption
            'is_sold': 1 if is_sold else 0
        }
        
        # Model category features
        hypercar_models = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
        track_models = ['765LT', '675LT', '600LT', '620R']
        entry_models = ['MP4-12C', '12C Spider', '12C Coupe']
        grand_tourer = ['570GT', 'GT']
        
        features['is_hypercar'] = 1 if model in hypercar_models else 0
        features['is_track_focused'] = 1 if model in track_models else 0
        features['is_entry_level'] = 1 if model in entry_models else 0
        features['is_grand_tourer'] = 1 if model in grand_tourer else 0
        
        # Encode categorical features
        for cat_feature in ['model', 'state', 'transmission']:
            if cat_feature in self.label_encoders:
                le = self.label_encoders[cat_feature]
                value = locals()[cat_feature]
                try:
                    encoded_value = le.transform([value])[0]
                except ValueError:
                    encoded_value = 0  # Handle unseen categories
                features[f'{cat_feature}_encoded'] = encoded_value
        
        # Create feature vector
        X_pred = np.array([features[fname] for fname in self.feature_names]).reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        total_weight = 0
        weighted_sum = 0
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)  # Ensure non-negative
            
            predictions[model_name] = pred
            weighted_sum += pred * weight
            total_weight += weight
        
        # Calculate ensemble prediction
        ensemble_prediction = weighted_sum / total_weight
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'confidence_range': {
                'low': min(predictions.values()),
                'high': max(predictions.values())
            },
            'model_agreement': (max(predictions.values()) - min(predictions.values())) / ensemble_prediction
        }
    
    def save_model(self, filename='mclaren_price_evaluator.pkl'):
        """Save the trained evaluator"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename='mclaren_price_evaluator.pkl'):
        """Load a trained evaluator"""
        with open(filename, 'rb') as f:
            evaluator = pickle.load(f)
        print(f"📂 Model loaded from {filename}")
        return evaluator

def demonstrate_evaluator():
    """Demonstrate the price evaluator"""
    print("🎯 McLaren Price Evaluator Demonstration")
    print("=" * 50)
    
    # Create and train evaluator
    evaluator = McLarenPriceEvaluator()
    evaluator.train_models()
    
    # Test cases
    test_cases = [
        {'year': 2020, 'model': '720S', 'mileage': 5000, 'state': 'CA'},
        {'year': 2019, 'model': 'Senna', 'mileage': 1000, 'state': 'FL'},
        {'year': 2015, 'model': 'P1', 'mileage': 2000, 'state': 'NY'},
        {'year': 2024, 'model': '750S', 'mileage': 500, 'state': 'TX'},
        {'year': 2018, 'model': '570S', 'mileage': 8000, 'state': 'CA'},
    ]
    
    print("\n🔮 Price Predictions:")
    for i, case in enumerate(test_cases, 1):
        result = evaluator.predict_price(**case)
        
        print(f"\n{i}. {case['year']} McLaren {case['model']} - {case['mileage']:,} miles ({case['state']}):")
        print(f"   💰 Predicted Price: ${result['ensemble_prediction']:,.0f}")
        print(f"   📊 Individual Models:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"      {model_name}: ${pred:,.0f}")
        print(f"   📈 Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")
        print(f"   🎯 Agreement: {(1-result['model_agreement'])*100:.1f}%")
    
    # Save the model
    evaluator.save_model()
    
    return evaluator

if __name__ == "__main__":
    evaluator = demonstrate_evaluator()
    
    print("\n🎉 McLaren Price Evaluator ready!")
    print("   Use evaluator.predict_price(year, model, mileage, state) to get predictions!") 