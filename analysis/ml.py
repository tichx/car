import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class McLarenPricePredictor:
    """Comprehensive McLaren Price Prediction System with Multiple ML Algorithms"""
    
    def __init__(self, data_path='mclaren_us_processed_final.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.model_scores = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_explore_data(self):
        """Load and perform initial data exploration"""
        print("🏎️  McLaren ML Price Prediction System")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        
        # Remove extreme price outliers (likely data errors)
        print(f"\n🧹 Data Cleaning:")
        initial_count = len(self.df)
        
        # Remove prices above $50M (likely parsing errors)
        self.df = self.df[self.df['price'] <= 50000000]
        outliers_removed = initial_count - len(self.df)
        print(f"   Removed {outliers_removed} extreme price outliers (>$50M)")
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Date range: {self.df['listing_date'].min()} to {self.df['listing_date'].max()}")
        print(f"   Price range: ${self.df['price'].min():,.0f} - ${self.df['price'].max():,.0f}")
        print(f"   Models: {self.df['model'].nunique()} different McLaren models")
        
        # Basic statistics
        print(f"\n📈 Price Statistics:")
        print(f"   Mean: ${self.df['price'].mean():,.0f}")
        print(f"   Median: ${self.df['price'].median():,.0f}")
        print(f"   Std Dev: ${self.df['price'].std():,.0f}")
        
        return self.df
    
    def feature_engineering(self):
        """Advanced feature engineering for automotive price prediction"""
        print(f"\n🔧 Feature Engineering:")
        
        df_features = self.df.copy()
        
        # 1. Age calculation
        current_year = datetime.now().year
        df_features['age'] = current_year - df_features['year']
        df_features['age_squared'] = df_features['age'] ** 2
        
        # 2. Mileage features
        df_features['mileage_per_year'] = df_features['mileage'] / (df_features['age'] + 1)
        df_features['low_mileage'] = (df_features['mileage'] < 5000).astype(int)
        df_features['high_mileage'] = (df_features['mileage'] > 30000).astype(int)
        
        # 3. Model category features
        hypercar_models = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
        track_models = ['765LT', '675LT', '600LT', '620R']
        entry_models = ['MP4-12C', '12C Spider', '12C Coupe']
        grand_tourer = ['570GT', 'GT']
        
        df_features['is_hypercar'] = df_features['model'].isin(hypercar_models).astype(int)
        df_features['is_track_focused'] = df_features['model'].isin(track_models).astype(int)
        df_features['is_entry_level'] = df_features['model'].isin(entry_models).astype(int)
        df_features['is_grand_tourer'] = df_features['model'].isin(grand_tourer).astype(int)
        
        # 4. Market timing features
        df_features['listing_date'] = pd.to_datetime(df_features['listing_date'], format='mixed', errors='coerce')
        df_features['listing_year'] = df_features['listing_date'].dt.year
        df_features['listing_month'] = df_features['listing_date'].dt.month
        df_features['days_since_launch'] = (df_features['listing_date'] - 
                                          pd.to_datetime('2011-01-01')).dt.days
        
        # 5. State economic indicators (simplified)
        high_income_states = ['CA', 'NY', 'FL', 'TX', 'IL']
        df_features['high_income_state'] = df_features['state'].isin(high_income_states).astype(int)
        
        # 6. Seller reputation
        top_dealers = df_features['seller_name'].value_counts().head(10).index
        df_features['top_dealer'] = df_features['seller_name'].isin(top_dealers).astype(int)
        
        # 7. Market status
        df_features['is_sold'] = (df_features['sold'] == 'y').astype(int)
        
        print(f"   Created {len(df_features.columns) - len(self.df.columns)} new features")
        
        return df_features
    
    def prepare_features(self, df_features):
        """Prepare features for ML models"""
        print(f"\n🎯 Feature Preparation:")
        
        # Select and encode features
        feature_columns = [
            'age', 'age_squared', 'mileage', 'mileage_per_year', 'year',
            'low_mileage', 'high_mileage', 'is_hypercar', 'is_track_focused',
            'is_entry_level', 'is_grand_tourer', 'listing_year', 'listing_month',
            'days_since_launch', 'high_income_state', 'top_dealer', 'is_sold'
        ]
        
        # Categorical features to encode
        categorical_features = ['model', 'state', 'transmission']
        
        X = df_features[feature_columns].copy()
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in df_features.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df_features[cat_feature].fillna('Unknown'))
                X[f'{cat_feature}_encoded'] = encoded_values
                self.label_encoders[cat_feature] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        self.X = X
        self.y = df_features['price']
        
        print(f"   Final feature count: {len(self.feature_names)}")
        print(f"   Features: {self.feature_names}")
        
        return self.X, self.y
    
    def create_models(self):
        """Create multiple ML models from basic to advanced"""
        print(f"\n🤖 Creating ML Models:")
        
        # 1. Linear Models (Basic)
        self.models['Linear Regression'] = {
            'model': LinearRegression(),
            'complexity': 'Basic',
            'explanation': 'Simple linear relationship, highly interpretable but may miss non-linear patterns'
        }
        
        self.models['Ridge Regression'] = {
            'model': Ridge(alpha=1.0),
            'complexity': 'Basic',
            'explanation': 'Linear with L2 regularization, prevents overfitting, good baseline'
        }
        
        self.models['Lasso Regression'] = {
            'model': Lasso(alpha=1.0),
            'complexity': 'Basic',
            'explanation': 'Linear with L1 regularization, performs feature selection automatically'
        }
        
        self.models['Elastic Net'] = {
            'model': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'complexity': 'Basic',
            'explanation': 'Combines Ridge and Lasso, balanced regularization approach'
        }
        
        # 2. Tree-based Models (Intermediate)
        self.models['Decision Tree'] = {
            'model': DecisionTreeRegressor(max_depth=10, random_state=42),
            'complexity': 'Intermediate',
            'explanation': 'Highly interpretable, captures non-linear patterns, prone to overfitting'
        }
        
        self.models['Random Forest'] = {
            'model': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'complexity': 'Intermediate',
            'explanation': 'Ensemble of trees, reduces overfitting, good feature importance, robust'
        }
        
        # 3. Boosting Models (Advanced)
        self.models['Gradient Boosting'] = {
            'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'complexity': 'Advanced',
            'explanation': 'Sequential learning, excellent performance, captures complex patterns'
        }
        
        self.models['XGBoost'] = {
            'model': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'complexity': 'Advanced',
            'explanation': 'Optimized gradient boosting, state-of-the-art performance, handles missing values'
        }
        
        self.models['AdaBoost'] = {
            'model': AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42),
            'complexity': 'Advanced',
            'explanation': 'Adaptive boosting, focuses on difficult examples, good for weak learners'
        }
        
        # 4. Neural Network (Advanced)
        self.models['Neural Network'] = {
            'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'complexity': 'Advanced',
            'explanation': 'Deep learning approach, captures complex non-linear patterns, black box'
        }
        
        # 5. Support Vector Machine (Advanced)
        self.models['Support Vector Regression'] = {
            'model': SVR(kernel='rbf', C=1000, gamma=0.1),
            'complexity': 'Advanced',
            'explanation': 'Kernel-based method, good for high-dimensional data, memory efficient'
        }
        
        print(f"   Created {len(self.models)} models ranging from basic to advanced")
        
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print(f"\n📊 Model Evaluation:")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            try:
                # Use scaled data for neural networks and SVR
                if name in ['Neural Network', 'Support Vector Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                              scoring='neg_mean_absolute_error')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                              scoring='neg_mean_absolute_error')
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Calculate percentage error
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results.append({
                    'Model': name,
                    'Complexity': model_info['complexity'],
                    'MAE': mae,
                    'RMSE': rmse,
                    'R²': r2,
                    'MAPE': mape,
                    'CV_MAE': cv_mae,
                    'CV_Std': cv_std,
                    'Explanation': model_info['explanation']
                })
                
                self.model_scores[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'complexity': model_info['complexity'],
                    'explanation': model_info['explanation']
                }
                
            except Exception as e:
                print(f"   Error with {name}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('MAE')
        
        print("\n🏆 Model Performance Ranking (by Mean Absolute Error):")
        print("=" * 100)
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<25} | MAE: ${row['MAE']:>8,.0f} | R²: {row['R²']:>6.3f} | "
                  f"MAPE: {row['MAPE']:>5.1f}% | {row['Complexity']}")
        
        return results_df
    
    def analyze_feature_importance(self, top_models=3):
        """Analyze feature importance for tree-based models"""
        print(f"\n🎯 Feature Importance Analysis:")
        
        tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        
        for model_name in tree_based_models:
            if model_name in self.model_scores:
                model = self.model_scores[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n{model_name} - Top 10 Important Features:")
                    for _, row in importance_df.head(10).iterrows():
                        print(f"   {row['feature']:<20}: {row['importance']:.4f}")
    
    def time_series_validation(self):
        """Time series cross-validation for temporal data"""
        print(f"\n⏰ Time Series Validation:")
        
        # Sort by listing date
        df_sorted = self.df.copy()
        df_sorted['listing_date'] = pd.to_datetime(df_sorted['listing_date'], format='mixed', errors='coerce')
        df_sorted = df_sorted.sort_values('listing_date')
        
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        top_3_models = sorted(self.model_scores.items(), key=lambda x: x[1]['mae'])[:3]
        
        print("   Time-based validation results:")
        for model_name, model_info in top_3_models:
            model = model_info['model']
            
            ts_scores = []
            for train_idx, test_idx in tscv.split(self.X):
                X_train_ts, X_test_ts = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train_ts, y_test_ts = self.y.iloc[train_idx], self.y.iloc[test_idx]
                
                if model_name in ['Neural Network', 'Support Vector Regression']:
                    X_train_ts_scaled = self.scaler.fit_transform(X_train_ts)
                    X_test_ts_scaled = self.scaler.transform(X_test_ts)
                    model.fit(X_train_ts_scaled, y_train_ts)
                    y_pred_ts = model.predict(X_test_ts_scaled)
                else:
                    model.fit(X_train_ts, y_train_ts)
                    y_pred_ts = model.predict(X_test_ts)
                
                mae_ts = mean_absolute_error(y_test_ts, y_pred_ts)
                ts_scores.append(mae_ts)
            
            avg_ts_mae = np.mean(ts_scores)
            print(f"   {model_name:<25}: MAE ${avg_ts_mae:,.0f} (±${np.std(ts_scores):,.0f})")
    
    def create_price_evaluator(self):
        """Create a price evaluator using top 3 models"""
        print(f"\n🎪 Creating Price Evaluator with Top 3 Models:")
        
        # Get top 3 models by performance
        top_3 = sorted(self.model_scores.items(), key=lambda x: x[1]['mae'])[:3]
        
        print("   Selected models:")
        for i, (name, info) in enumerate(top_3, 1):
            print(f"   {i}. {name}: MAE ${info['mae']:,.0f}, R² {info['r2']:.3f}")
        
        class McLarenPriceEvaluator:
            def __init__(self, models, feature_names, scaler, label_encoders):
                self.models = models
                self.feature_names = feature_names
                self.scaler = scaler
                self.label_encoders = label_encoders
                self.model_names = [name for name, _ in models]
            
            def predict_price(self, year, model, mileage, state='CA', transmission='Automatic', 
                            is_sold=False):
                """Predict McLaren price using top 3 models"""
                
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
                            # Handle unseen categories
                            encoded_value = 0
                        features[f'{cat_feature}_encoded'] = encoded_value
                
                # Create feature vector in correct order
                X_pred = np.array([features[fname] for fname in self.feature_names]).reshape(1, -1)
                
                # Get predictions from all models
                predictions = {}
                for i, (model_name, model_info) in enumerate(self.models):
                    model = model_info['model']
                    
                    if model_name in ['Neural Network', 'Support Vector Regression']:
                        X_pred_scaled = self.scaler.transform(X_pred)
                        pred = model.predict(X_pred_scaled)[0]
                    else:
                        pred = model.predict(X_pred)[0]
                    
                    predictions[model_name] = max(0, pred)  # Ensure non-negative
                
                # Calculate ensemble prediction (weighted average)
                weights = [1/info['mae'] for _, info in self.models]  # Inverse of MAE as weights
                total_weight = sum(weights)
                weighted_avg = sum(pred * weight for pred, weight in zip(predictions.values(), weights)) / total_weight
                
                return {
                    'individual_predictions': predictions,
                    'ensemble_prediction': weighted_avg,
                    'confidence_range': {
                        'low': min(predictions.values()),
                        'high': max(predictions.values())
                    }
                }
        
        return McLarenPriceEvaluator(top_3, self.feature_names, self.scaler, self.label_encoders)
    
    def run_complete_analysis(self):
        """Run the complete ML analysis pipeline"""
        # Load and explore data
        self.load_and_explore_data()
        
        # Feature engineering
        df_features = self.feature_engineering()
        
        # Prepare features
        self.prepare_features(df_features)
        
        # Create models
        self.create_models()
        
        # Evaluate models
        results_df = self.evaluate_models()
        
        # Feature importance analysis
        self.analyze_feature_importance()
        
        # Time series validation
        self.time_series_validation()
        
        # Create price evaluator
        evaluator = self.create_price_evaluator()
        
        return results_df, evaluator

def demonstrate_price_evaluator(evaluator):
    """Demonstrate the price evaluator with example predictions"""
    print(f"\n🎯 PRICE EVALUATOR DEMONSTRATION:")
    print("=" * 60)
    
    test_cases = [
        {'year': 2020, 'model': '720S', 'mileage': 5000, 'state': 'CA'},
        {'year': 2019, 'model': 'Senna', 'mileage': 1000, 'state': 'FL'},
        {'year': 2015, 'model': 'P1', 'mileage': 2000, 'state': 'NY'},
        {'year': 2017, 'model': '570S', 'mileage': 15000, 'state': 'TX'},
        {'year': 2013, 'model': 'MP4-12C', 'mileage': 25000, 'state': 'IL'}
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = evaluator.predict_price(**case)
        
        print(f"\n{i}. {case['year']} McLaren {case['model']} - {case['mileage']:,} miles ({case['state']}):")
        print(f"   🎯 Ensemble Prediction: ${result['ensemble_prediction']:,.0f}")
        print(f"   📊 Individual Models:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"      {model_name}: ${pred:,.0f}")
        print(f"   📈 Confidence Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")

if __name__ == "__main__":
    # Initialize and run analysis
    predictor = McLarenPricePredictor()
    results_df, evaluator = predictor.run_complete_analysis()
    
    # Demonstrate the evaluator
    demonstrate_price_evaluator(evaluator)
    
    print(f"\n🎉 McLaren ML Analysis Complete!")
    print("   Use the evaluator object to predict prices for any McLaren configuration!")
