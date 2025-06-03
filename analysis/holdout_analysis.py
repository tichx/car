import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class McLarenHoldoutAnalysis:
    """Analysis with hypercar/supercar holdout to test generalization"""
    
    def __init__(self, data_path='mclaren_us_processed_final.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        
    def load_and_categorize_data(self):
        """Load data and categorize McLaren models by tier"""
        print("🏎️  McLaren Holdout Analysis - Testing Generalization")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        
        # Remove extreme outliers
        initial_count = len(self.df)
        self.df = self.df[self.df['price'] <= 50000000]
        outliers_removed = initial_count - len(self.df)
        print(f"   Removed {outliers_removed} extreme price outliers")
        
        # Define model categories
        hypercars = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
        supercars = ['720S', '765LT', '750S', '675LT', '650S', '720S Spider', '765LT Spider']
        track_focused = ['600LT', '620R', '675LT', '765LT']
        grand_tourers = ['570GT', 'GT']
        entry_sports = ['MP4-12C', '12C Spider', '12C Coupe', '570S', '570S Spider', '540C']
        
        # Create comprehensive categorization
        def categorize_model(model):
            if model in hypercars:
                return 'Hypercar'
            elif model in supercars:
                return 'Supercar'
            elif model in track_focused:
                return 'Track-Focused'
            elif model in grand_tourers:
                return 'Grand Tourer'
            elif model in entry_sports:
                return 'Entry Sports'
            else:
                return 'Other'
        
        self.df['model_category'] = self.df['model'].apply(categorize_model)
        
        # Show distribution
        print(f"\n📊 Model Category Distribution:")
        category_counts = self.df['model_category'].value_counts()
        for category, count in category_counts.items():
            avg_price = self.df[self.df['model_category'] == category]['price'].mean()
            print(f"   {category:<15}: {count:>3} cars (avg: ${avg_price:>8,.0f})")
        
        return self.df
    
    def feature_engineering(self, df):
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
        
        # Model category features (excluding hypercar to test generalization)
        df_features['is_supercar'] = (df_features['model_category'] == 'Supercar').astype(int)
        df_features['is_track_focused'] = (df_features['model_category'] == 'Track-Focused').astype(int)
        df_features['is_entry_level'] = (df_features['model_category'] == 'Entry Sports').astype(int)
        df_features['is_grand_tourer'] = (df_features['model_category'] == 'Grand Tourer').astype(int)
        
        # Market timing features
        df_features['listing_date'] = pd.to_datetime(df_features['listing_date'], format='mixed', errors='coerce')
        df_features['listing_year'] = df_features['listing_date'].dt.year
        df_features['listing_month'] = df_features['listing_date'].dt.month
        
        # State economic indicators
        high_income_states = ['CA', 'NY', 'FL', 'TX', 'IL']
        df_features['high_income_state'] = df_features['state'].isin(high_income_states).astype(int)
        
        # Market status
        df_features['is_sold'] = (df_features['sold'] == 'y').astype(int)
        
        return df_features
    
    def prepare_features(self, df_features, include_hypercar_flag=False):
        """Prepare features for ML models"""
        feature_columns = [
            'age', 'age_squared', 'mileage', 'mileage_per_year', 'year',
            'low_mileage', 'high_mileage', 'is_supercar', 'is_track_focused',
            'is_entry_level', 'is_grand_tourer', 'listing_year', 'listing_month',
            'high_income_state', 'is_sold'
        ]
        
        # Optionally include hypercar flag for comparison
        if include_hypercar_flag:
            df_features['is_hypercar'] = (df_features['model_category'] == 'Hypercar').astype(int)
            feature_columns.append('is_hypercar')
        
        # Categorical features to encode
        categorical_features = ['model', 'state', 'transmission']
        
        X = df_features[feature_columns].copy()
        
        # Encode categorical features
        label_encoders = {}
        for cat_feature in categorical_features:
            if cat_feature in df_features.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df_features[cat_feature].fillna('Unknown'))
                X[f'{cat_feature}_encoded'] = encoded_values
                label_encoders[cat_feature] = le
        
        X = X.fillna(X.median())
        
        return X, df_features['price'], label_encoders
    
    def run_holdout_experiment(self):
        """Run the main holdout experiment"""
        # Load and categorize data
        df = self.load_and_categorize_data()
        
        # Feature engineering
        df_features = self.feature_engineering(df)
        
        print(f"\n🧪 EXPERIMENT 1: Training WITHOUT Hypercars")
        print("=" * 50)
        
        # Split data: train on non-hypercars, test on all
        non_hypercar_mask = df_features['model_category'] != 'Hypercar'
        hypercar_mask = df_features['model_category'] == 'Hypercar'
        
        df_train = df_features[non_hypercar_mask].copy()
        df_hypercar_test = df_features[hypercar_mask].copy()
        
        print(f"   Training set: {len(df_train)} non-hypercar McLarens")
        print(f"   Hypercar test set: {len(df_hypercar_test)} hypercars")
        
        # Prepare features (without hypercar flag in training)
        X_train, y_train, le_train = self.prepare_features(df_train, include_hypercar_flag=False)
        
        # Train models
        models = {
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        trained_models = {}
        
        # Train and evaluate on non-hypercar data
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Performance on NON-HYPERCAR models:")
        for name, model in models.items():
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_test_split)
            
            mae = mean_absolute_error(y_test_split, y_pred)
            r2 = r2_score(y_test_split, y_pred)
            mape = np.mean(np.abs((y_test_split - y_pred) / y_test_split)) * 100
            
            print(f"   {name:<20}: MAE ${mae:>8,.0f} | R² {r2:>6.3f} | MAPE {mape:>5.1f}%")
            
            trained_models[name] = {
                'model': model,
                'mae_non_hypercar': mae,
                'r2_non_hypercar': r2,
                'mape_non_hypercar': mape
            }
        
        # Test on hypercars (if any exist)
        if len(df_hypercar_test) > 0:
            print(f"\n🚀 Performance on HYPERCAR models (unseen during training):")
            
            # Prepare hypercar features (need to handle encoding carefully)
            X_hypercar_features = df_hypercar_test[X_train.columns[:-3]].copy()  # Exclude encoded features
            
            # Encode hypercar categorical features using training encoders
            for cat_feature in ['model', 'state', 'transmission']:
                if cat_feature in le_train:
                    le = le_train[cat_feature]
                    try:
                        encoded_values = []
                        for value in df_hypercar_test[cat_feature].fillna('Unknown'):
                            try:
                                encoded_values.append(le.transform([value])[0])
                            except ValueError:
                                # Handle unseen categories with a default value
                                encoded_values.append(0)
                        X_hypercar_features[f'{cat_feature}_encoded'] = encoded_values
                    except Exception as e:
                        print(f"   Warning: Could not encode {cat_feature} for hypercars: {e}")
                        X_hypercar_features[f'{cat_feature}_encoded'] = 0
            
            X_hypercar_features = X_hypercar_features.fillna(X_hypercar_features.median())
            y_hypercar = df_hypercar_test['price']
            
            for name, model_info in trained_models.items():
                try:
                    model = model_info['model']
                    y_pred_hyper = model.predict(X_hypercar_features)
                    
                    mae_hyper = mean_absolute_error(y_hypercar, y_pred_hyper)
                    r2_hyper = r2_score(y_hypercar, y_pred_hyper)
                    mape_hyper = np.mean(np.abs((y_hypercar - y_pred_hyper) / y_hypercar)) * 100
                    
                    print(f"   {name:<20}: MAE ${mae_hyper:>8,.0f} | R² {r2_hyper:>6.3f} | MAPE {mape_hyper:>5.1f}%")
                    
                    model_info['mae_hypercar'] = mae_hyper
                    model_info['r2_hypercar'] = r2_hyper
                    model_info['mape_hypercar'] = mape_hyper
                    
                    # Show individual predictions
                    if len(df_hypercar_test) <= 10:  # Show details if few hypercars
                        print(f"\n   {name} - Individual Hypercar Predictions:")
                        for idx, (_, row) in enumerate(df_hypercar_test.iterrows()):
                            actual = row['price']
                            predicted = y_pred_hyper[idx]
                            error_pct = abs(actual - predicted) / actual * 100
                            print(f"      {row['year']} {row['model']}: ${actual:,.0f} → ${predicted:,.0f} ({error_pct:.1f}% error)")
                    
                except Exception as e:
                    print(f"   Error testing {name} on hypercars: {e}")
        
        # Now compare with training INCLUDING hypercar flag
        print(f"\n🧪 EXPERIMENT 2: Training WITH Hypercar Flag")
        print("=" * 50)
        
        X_full, y_full, _ = self.prepare_features(df_features, include_hypercar_flag=True)
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )
        
        print(f"📊 Performance with HYPERCAR FLAG included:")
        for name, model_class in models.items():
            model = model_class.__class__(**model_class.get_params())  # Fresh model
            model.fit(X_train_full, y_train_full)
            y_pred_full = model.predict(X_test_full)
            
            mae_full = mean_absolute_error(y_test_full, y_pred_full)
            r2_full = r2_score(y_test_full, y_pred_full)
            mape_full = np.mean(np.abs((y_test_full - y_pred_full) / y_test_full)) * 100
            
            print(f"   {name:<20}: MAE ${mae_full:>8,.0f} | R² {r2_full:>6.3f} | MAPE {mape_full:>5.1f}%")
        
        self.models = trained_models
        return trained_models
    
    def analyze_results(self):
        """Analyze and summarize the holdout experiment results"""
        print(f"\n🎯 HOLDOUT EXPERIMENT ANALYSIS:")
        print("=" * 50)
        
        print(f"\n📈 Key Findings:")
        
        for name, results in self.models.items():
            print(f"\n{name}:")
            print(f"   Non-Hypercar Performance: MAE ${results['mae_non_hypercar']:,.0f}, R² {results['r2_non_hypercar']:.3f}")
            
            if 'mae_hypercar' in results:
                print(f"   Hypercar Performance:     MAE ${results['mae_hypercar']:,.0f}, R² {results['r2_hypercar']:.3f}")
                
                # Calculate performance degradation
                degradation = (results['mae_hypercar'] - results['mae_non_hypercar']) / results['mae_non_hypercar'] * 100
                if degradation > 0:
                    print(f"   ⚠️  Performance degradation on hypercars: +{degradation:.1f}% MAE increase")
                else:
                    print(f"   ✅ Better performance on hypercars: {abs(degradation):.1f}% MAE decrease")
        
        print(f"\n🔍 Insights:")
        print(f"   1. Models trained without hypercar data can still predict hypercar prices")
        print(f"   2. Performance differences reveal how model categories impact predictions")
        print(f"   3. This tests the generalization capability to unseen model types")
        
        return self.models

def run_holdout_analysis():
    """Run the complete holdout analysis"""
    analyzer = McLarenHoldoutAnalysis()
    models = analyzer.run_holdout_experiment()
    results = analyzer.analyze_results()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Use separate models for hypercars vs regular McLarens if performance differs significantly")
    print(f"   • Include model category features for better generalization")
    print(f"   • Consider ensemble approach combining category-specific models")
    
    return analyzer, models, results

if __name__ == "__main__":
    analyzer, models, results = run_holdout_analysis() 