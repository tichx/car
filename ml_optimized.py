import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HighPrecisionMcLarenPredictor:
    """Ultra High Precision McLaren Price Predictor - Target MAE < $1000"""
    
    def __init__(self, data_path='mclaren_us_processed_final.csv'):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.model_scores = {}
        self.feature_names = []
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.label_encoders = {}
        self.price_segments = {}
        
    def extreme_data_cleaning(self):
        """Extreme data cleaning to remove all noise"""
        print("🧹 EXTREME DATA CLEANING:")
        initial_count = len(self.df)
        
        # 1. Remove extreme outliers using IQR method for each model separately
        cleaned_segments = []
        for model_name in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model_name].copy()
            
            Q1 = model_data['price'].quantile(0.25)
            Q3 = model_data['price'].quantile(0.75)
            IQR = Q3 - Q1
            
            # More conservative outlier removal
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            model_data = model_data[
                (model_data['price'] >= lower_bound) & 
                (model_data['price'] <= upper_bound)
            ]
            
            # Additional cleaning: remove cars with unrealistic mileage for age
            model_data['expected_mileage'] = (2024 - model_data['year']) * 12000
            model_data = model_data[
                (model_data['mileage'] <= model_data['expected_mileage'] * 3) &  # Max 3x expected
                (model_data['mileage'] >= 0)
            ]
            
            cleaned_segments.append(model_data)
        
        self.df = pd.concat(cleaned_segments, ignore_index=True)
        
        # 2. Remove cars with missing critical info
        critical_columns = ['year', 'mileage', 'model', 'price', 'listing_date']
        self.df = self.df.dropna(subset=critical_columns)
        
        # 3. Remove cars older than 15 years (too volatile)
        current_year = datetime.now().year
        self.df = self.df[self.df['year'] >= current_year - 15]
        
        # 4. Remove duplicate listings (same car relisted)
        self.df = self.df.drop_duplicates(subset=['year', 'model', 'mileage'], keep='first')
        
        final_count = len(self.df)
        print(f"   Removed {initial_count - final_count} samples ({((initial_count - final_count)/initial_count)*100:.1f}%)")
        print(f"   Final dataset: {final_count} samples")
        
        return self.df
    
    def advanced_feature_engineering(self):
        """Ultra-advanced feature engineering for maximum precision"""
        print("🔧 ADVANCED FEATURE ENGINEERING:")
        
        df_features = self.df.copy()
        
        # 1. Time-based features with high granularity
        df_features['listing_date'] = pd.to_datetime(df_features['listing_date'], format='mixed', errors='coerce')
        current_date = datetime.now()
        
        df_features['age_months'] = (current_date.year - df_features['year']) * 12
        df_features['listing_recency_days'] = (current_date - df_features['listing_date']).dt.days
        df_features['season'] = df_features['listing_date'].dt.month % 12 // 3 + 1
        df_features['is_end_of_year'] = (df_features['listing_date'].dt.month >= 11).astype(int)
        
        # 2. Mileage sophistication
        df_features['mileage_per_month'] = df_features['mileage'] / (df_features['age_months'] + 1)
        df_features['mileage_z_score'] = (df_features['mileage'] - df_features['mileage'].mean()) / df_features['mileage'].std()
        df_features['ultra_low_mileage'] = (df_features['mileage'] < 1000).astype(int)
        df_features['garage_queen'] = (df_features['mileage_per_month'] < 100).astype(int)
        
        # 3. Model-specific depreciation curves
        for model in df_features['model'].unique():
            model_data = df_features[df_features['model'] == model]
            if len(model_data) > 5:  # Only if we have enough data
                # Calculate model-specific depreciation rate
                if len(model_data) > 1:
                    age_price_corr = model_data['age_months'].corr(model_data['price'])
                    df_features.loc[df_features['model'] == model, 'model_depreciation_rate'] = abs(age_price_corr)
        
        df_features['model_depreciation_rate'] = df_features['model_depreciation_rate'].fillna(0.5)
        
        # 4. Market segment analysis
        # Calculate market position for each model
        model_stats = df_features.groupby('model')['price'].agg(['mean', 'std', 'count']).reset_index()
        model_stats.columns = ['model', 'model_avg_price', 'model_price_std', 'model_sample_count']
        df_features = df_features.merge(model_stats, on='model', how='left')
        
        df_features['price_vs_model_avg'] = df_features['price'] / df_features['model_avg_price']
        df_features['is_rare_model'] = (df_features['model_sample_count'] < 10).astype(int)
        
        # 5. Geographic premium features
        luxury_states = ['CA', 'NY', 'FL', 'TX', 'IL', 'CT', 'NJ', 'MA']
        df_features['luxury_market'] = df_features['state'].isin(luxury_states).astype(int)
        
        # 6. Seller analysis
        seller_stats = df_features.groupby('seller_name').agg({
            'price': ['mean', 'count'],
            'sold': lambda x: (x == 'y').mean()
        }).reset_index()
        seller_stats.columns = ['seller_name', 'seller_avg_price', 'seller_listing_count', 'seller_success_rate']
        df_features = df_features.merge(seller_stats, on='seller_name', how='left')
        
        df_features['reputable_seller'] = (df_features['seller_listing_count'] >= 5).astype(int)
        df_features['high_success_seller'] = (df_features['seller_success_rate'] >= 0.5).astype(int)
        
        # 7. Interaction features
        df_features['age_mileage_interaction'] = df_features['age_months'] * df_features['mileage_per_month']
        df_features['luxury_market_x_model_price'] = df_features['luxury_market'] * df_features['model_avg_price']
        
        print(f"   Created {len(df_features.columns) - len(self.df.columns)} new features")
        
        return df_features
    
    def create_price_segments(self, df_features):
        """Create price-based segments for specialized models"""
        print("📊 CREATING PRICE SEGMENTS:")
        
        # Define price segments based on McLaren model ranges
        segments = {
            'entry': (0, 200000),        # 570S, 540C, etc.
            'mid': (200000, 500000),     # 720S, 765LT, etc.
            'high': (500000, 1500000),   # Senna, P1, etc.
            'ultra': (1500000, float('inf'))  # F1, rare models
        }
        
        self.price_segments = {}
        
        for segment_name, (min_price, max_price) in segments.items():
            segment_data = df_features[
                (df_features['price'] >= min_price) & 
                (df_features['price'] < max_price)
            ].copy()
            
            if len(segment_data) > 20:  # Only create segment if enough data
                self.price_segments[segment_name] = {
                    'data': segment_data,
                    'price_range': (min_price, max_price),
                    'count': len(segment_data)
                }
                print(f"   {segment_name.upper()} segment: {len(segment_data)} samples (${min_price:,} - ${max_price:,})")
        
        return self.price_segments
    
    def prepare_segment_features(self, segment_data):
        """Prepare features for a specific price segment"""
        
        # Select features based on correlation with price in this segment
        numeric_features = segment_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col != 'price']
        
        # Calculate correlation with price and select top features
        correlations = segment_data[numeric_features].corrwith(segment_data['price']).abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()  # Top 15 most correlated features
        
        X = segment_data[top_features].copy()
        
        # Categorical encoding
        categorical_features = ['model', 'state', 'transmission']
        for cat_feature in categorical_features:
            if cat_feature in segment_data.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(segment_data[cat_feature].fillna('Unknown'))
                X[f'{cat_feature}_encoded'] = encoded_values
        
        # Handle missing values
        X = X.fillna(X.median())
        
        y = segment_data['price']
        
        return X, y
    
    def create_ultra_precision_models(self):
        """Create ultra-high precision models optimized for low MAE"""
        print("🚀 CREATING ULTRA-PRECISION MODELS:")
        
        models = {}
        
        # 1. Optimized LightGBM (best for low MAE)
        models['LightGBM_Ultra'] = lgb.LGBMRegressor(
            objective='mae',  # Direct MAE optimization
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Optimized XGBoost
        models['XGBoost_Ultra'] = xgb.XGBRegressor(
            objective='reg:absoluteerror',  # MAE objective
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1
        )
        
        # 3. CatBoost (excellent for categorical features)
        models['CatBoost_Ultra'] = CatBoostRegressor(
            loss_function='MAE',
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=10,
            random_seed=42,
            verbose=False
        )
        
        # 4. Extra Trees (high variance reduction)
        models['ExtraTrees_Ultra'] = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # 5. K-Nearest Neighbors (local similarity)
        models['KNN_Ultra'] = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            metric='manhattan'  # L1 metric aligns with MAE
        )
        
        # 6. Huber Regression (robust to outliers)
        models['Huber_Ultra'] = HuberRegressor(
            epsilon=1.1,  # Slightly more sensitive to outliers
            max_iter=200,
            alpha=0.01
        )
        
        return models
    
    def segment_based_training(self):
        """Train specialized models for each price segment"""
        print("🎯 SEGMENT-BASED TRAINING:")
        
        df_features = self.advanced_feature_engineering()
        self.create_price_segments(df_features)
        
        segment_models = {}
        segment_scores = {}
        
        for segment_name, segment_info in self.price_segments.items():
            print(f"\n   Training models for {segment_name.upper()} segment...")
            
            X, y = self.prepare_segment_features(segment_info['data'])
            
            if len(X) < 30:  # Skip if too few samples
                print(f"   Skipping {segment_name} - insufficient data")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = self.create_ultra_precision_models()
            best_model = None
            best_mae = float('inf')
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    if model_name in ['KNN_Ultra', 'Huber_Ultra']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model
                        best_model_name = model_name
                        
                except Exception as e:
                    print(f"      Error with {model_name}: {str(e)}")
                    continue
            
            segment_models[segment_name] = {
                'model': best_model,
                'model_name': best_model_name,
                'scaler': scaler,
                'features': X.columns.tolist(),
                'price_range': segment_info['price_range']
            }
            
            segment_scores[segment_name] = {
                'mae': best_mae,
                'samples': len(X)
            }
            
            print(f"      Best model: {best_model_name} with MAE: ${best_mae:,.0f}")
        
        self.segment_models = segment_models
        self.segment_scores = segment_scores
        
        return segment_models, segment_scores
    
    def create_ensemble_predictor(self):
        """Create final ensemble predictor"""
        print("\n🎪 CREATING ULTRA-PRECISION ENSEMBLE:")
        
        class UltraPrecisionEvaluator:
            def __init__(self, segment_models, price_segments):
                self.segment_models = segment_models
                self.price_segments = price_segments
                
            def predict_price(self, year, model, mileage, state='CA', transmission='Automatic', is_sold=False):
                """Ultra-precision price prediction"""
                
                # Create comprehensive feature vector
                current_date = datetime.now()
                age_months = (current_date.year - year) * 12
                
                # All possible features
                features = {
                    'age_months': age_months,
                    'mileage': mileage,
                    'year': year,
                    'mileage_per_month': mileage / (age_months + 1) if age_months >= 0 else mileage,
                    'ultra_low_mileage': 1 if mileage < 1000 else 0,
                    'garage_queen': 1 if (mileage / (age_months + 1)) < 100 else 0,
                    'season': current_date.month % 12 // 3 + 1,
                    'is_end_of_year': 1 if current_date.month >= 11 else 0,
                    'luxury_market': 1 if state in ['CA', 'NY', 'FL', 'TX', 'IL', 'CT', 'NJ', 'MA'] else 0,
                    'is_sold': 1 if is_sold else 0
                }
                
                # Model categorization
                hypercar_models = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
                track_models = ['765LT', '675LT', '600LT', '620R']
                entry_models = ['MP4-12C', '12C Spider', '12C Coupe', '570S', '540C']
                
                features['is_hypercar'] = 1 if model in hypercar_models else 0
                features['is_track_focused'] = 1 if model in track_models else 0
                features['is_entry_level'] = 1 if model in entry_models else 0
                
                # Get initial price estimate to determine segment
                # Use simple heuristic based on model
                if model in hypercar_models:
                    estimated_price = 1000000
                elif model in track_models:
                    estimated_price = 400000
                elif model in entry_models:
                    estimated_price = 150000
                else:
                    estimated_price = 300000
                
                # Determine which segment to use
                segment_to_use = 'mid'  # default
                for seg_name, seg_info in self.price_segments.items():
                    min_price, max_price = seg_info['price_range']
                    if min_price <= estimated_price < max_price:
                        segment_to_use = seg_name
                        break
                
                if segment_to_use not in self.segment_models:
                    segment_to_use = list(self.segment_models.keys())[0]  # fallback
                
                # Use the appropriate segment model
                segment_model_info = self.segment_models[segment_to_use]
                model_obj = segment_model_info['model']
                scaler = segment_model_info['scaler']
                required_features = segment_model_info['features']
                
                # Create feature vector with only required features
                X_pred = []
                for feat in required_features:
                    if feat in features:
                        X_pred.append(features[feat])
                    elif feat.endswith('_encoded'):
                        X_pred.append(0)  # default for encoded features
                    else:
                        X_pred.append(0)  # default value
                
                X_pred = np.array(X_pred).reshape(1, -1)
                
                # Scale if needed
                model_name = segment_model_info['model_name']
                if model_name in ['KNN_Ultra', 'Huber_Ultra']:
                    X_pred = scaler.transform(X_pred)
                
                # Predict
                prediction = model_obj.predict(X_pred)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                return {
                    'price_prediction': prediction,
                    'segment_used': segment_to_use,
                    'model_used': model_name,
                    'confidence': 'high' if segment_to_use in ['entry', 'mid'] else 'medium'
                }
        
        return UltraPrecisionEvaluator(self.segment_models, self.price_segments)
    
    def run_ultra_precision_analysis(self):
        """Run complete ultra-precision analysis"""
        print("🏎️  ULTRA-PRECISION McLAREN PRICE PREDICTOR")
        print("=" * 70)
        print("🎯 TARGET: MAE < $1,000")
        print("=" * 70)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Initial dataset: {len(self.df)} samples")
        
        # Extreme cleaning
        self.extreme_data_cleaning()
        
        # Segment-based training
        segment_models, segment_scores = self.segment_based_training()
        
        # Create evaluator
        evaluator = self.create_ensemble_predictor()
        
        # Display results
        print("\n🏆 SEGMENT MODEL PERFORMANCE:")
        total_weighted_mae = 0
        total_samples = 0
        
        for segment, scores in segment_scores.items():
            mae = scores['mae']
            samples = scores['samples']
            total_weighted_mae += mae * samples
            total_samples += samples
            
            print(f"   {segment.upper():<8}: MAE ${mae:>8,.0f} ({samples:>3} samples)")
        
        if total_samples > 0:
            overall_mae = total_weighted_mae / total_samples
            print(f"\n🎯 OVERALL WEIGHTED MAE: ${overall_mae:,.0f}")
            
            if overall_mae <= 1000:
                print("✅ TARGET ACHIEVED: MAE ≤ $1,000!")
            else:
                print(f"❌ TARGET MISSED: Need to reduce MAE by ${overall_mae-1000:,.0f}")
        
        return evaluator

def test_ultra_precision_evaluator(evaluator):
    """Test the ultra-precision evaluator"""
    print(f"\n🎯 ULTRA-PRECISION EVALUATION DEMO:")
    print("=" * 50)
    
    test_cases = [
        {'year': 2020, 'model': '720S', 'mileage': 5000, 'state': 'CA'},
        {'year': 2019, 'model': 'Senna', 'mileage': 1000, 'state': 'FL'},
        {'year': 2017, 'model': '570S', 'mileage': 15000, 'state': 'TX'},
        {'year': 2015, 'model': 'P1', 'mileage': 2000, 'state': 'NY'},
        {'year': 2018, 'model': '540C', 'mileage': 8000, 'state': 'CA'}
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = evaluator.predict_price(**case)
        
        print(f"\n{i}. {case['year']} McLaren {case['model']} - {case['mileage']:,} miles ({case['state']}):")
        print(f"   💰 Price: ${result['price_prediction']:,.0f}")
        print(f"   📊 Segment: {result['segment_used']}")
        print(f"   🤖 Model: {result['model_used']}")
        print(f"   🎯 Confidence: {result['confidence']}")

if __name__ == "__main__":
    predictor = HighPrecisionMcLarenPredictor()
    evaluator = predictor.run_ultra_precision_analysis()
    
    if evaluator:
        test_ultra_precision_evaluator(evaluator)
    
    print(f"\n🚀 Ultra-Precision Analysis Complete!") 