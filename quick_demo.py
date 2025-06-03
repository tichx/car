#!/usr/bin/env python3
"""
Quick Demo: McLaren Price Evaluator
===================================

This script demonstrates how to use the trained McLaren price prediction model
to get instant price estimates for any McLaren configuration.
Supports both current market value and future price predictions.
"""

from price_evaluator import McLarenPriceEvaluator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def interactive_price_evaluation():
    """Interactive McLaren price evaluation"""
    
    print("🏎️  McLaren Price Evaluator")
    print("=" * 50)
    
    # Load the pre-trained model
    try:
        evaluator = McLarenPriceEvaluator.load_model()
        print("📂 Loaded pre-trained model")
    except FileNotFoundError:
        print("No saved model found. Training new model...")
        evaluator = McLarenPriceEvaluator()
        evaluator.train_models()
    
    print("\nChoose evaluation type:")
    print("1. Current Market Value (2025)")
    print("2. Future Price Estimate (specify year)")
    
    try:
        eval_type = input("\nEnter choice (1 or 2): ").strip()
        
        print("\nEnter McLaren details:")
        year = int(input("Car year (e.g., 2020): "))
        model = input("Model (e.g., 720S, P1, Senna, 570S): ")
        mileage = int(input("Mileage (e.g., 5000): "))
        state = input("State (e.g., CA, NY, FL): ").upper()
        
        # Get target year for prediction if future estimate
        target_year = datetime.now().year
        if eval_type == "2":
            target_year = int(input("Target year for prediction (e.g., 2026): "))
            
            # Adjust mileage for future year (assume typical driving)
            years_future = target_year - datetime.now().year
            if years_future > 0:
                additional_miles = years_future * 2000  # Assume 2k miles/year
                future_mileage = mileage + additional_miles
                print(f"Projected mileage in {target_year}: {future_mileage:,} miles")
                mileage = future_mileage
        
        result = evaluator.predict_price(year, model, mileage, state)
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Car: {year} McLaren {model}")
        print(f"   Condition: {mileage:,} miles, {state}")
        
        if eval_type == "1":
            print(f"   💰 Current Market Value: ${result['ensemble_prediction']:,.0f}")
        else:
            print(f"   🔮 Estimated Value in {target_year}: ${result['ensemble_prediction']:,.0f}")
            if target_year > datetime.now().year:
                print(f"   ⚠️  Future predictions have higher uncertainty")
        
        # Show model predictions
        print(f"\n   📊 Individual Model Predictions:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"      {model_name}: ${pred:,.0f}")
        
        # Confidence analysis
        agreement = (1 - result['model_agreement']) * 100
        print(f"\n   🎯 Model Agreement: {agreement:.1f}%")
        print(f"   📈 Confidence Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")
        
        # Add contextual insights
        print(f"\n   💡 Market Insights:")
        
        current_year = datetime.now().year
        car_age = current_year - year
        
        if eval_type == "2" and target_year > current_year:
            future_age = target_year - year
            print(f"      Car will be {future_age} years old in {target_year}")
        elif car_age <= 2:
            print(f"      Very recent model ({car_age} years old) - holds value well")
        elif car_age <= 5:
            print(f"      Modern McLaren ({car_age} years old) - typical depreciation")
        else:
            print(f"      Older model ({car_age} years old) - potential collector interest")
        
        # Mileage insights
        if mileage < 2000:
            print(f"      Ultra-low mileage - premium value")
        elif mileage < 5000:
            print(f"      Low mileage - excellent condition assumed")
        elif mileage < 15000:
            print(f"      Moderate mileage - fair market value")
        else:
            print(f"      Higher mileage - depreciated accordingly")
        
        # Model category insights
        hypercar_models = ['P1', 'Senna', 'Speedtail', 'F1', 'Elva', 'Sabre']
        track_models = ['765LT', '675LT', '600LT', '620R']
        entry_models = ['MP4-12C', '12C Spider', '12C Coupe', '570S', '540C']
        
        if model in hypercar_models:
            print(f"      Hypercar tier - appreciating asset potential")
        elif model in track_models:
            print(f"      Track-focused model - enthusiast appeal")
        elif model in entry_models:
            print(f"      Entry sports car - accessible McLaren experience")
        else:
            print(f"      Supercar tier - strong market demand")
            
    except (ValueError, KeyboardInterrupt):
        print("\nEvaluation ended.")

def batch_comparison():
    """Compare multiple McLaren configurations"""
    print("\n🔄 Batch Comparison Mode")
    print("=" * 40)
    
    try:
        evaluator = McLarenPriceEvaluator.load_model()
    except FileNotFoundError:
        print("Training model first...")
        evaluator = McLarenPriceEvaluator()
        evaluator.train_models()
    
    print("\nEnter multiple McLaren configurations to compare:")
    configurations = []
    
    try:
        while True:
            print(f"\nConfiguration #{len(configurations) + 1}:")
            year = input("Year (or press Enter to finish): ").strip()
            if not year:
                break
                
            year = int(year)
            model = input("Model: ")
            mileage = int(input("Mileage: "))
            state = input("State: ").upper()
            
            configurations.append({
                'year': year,
                'model': model, 
                'mileage': mileage,
                'state': state
            })
    
    except (ValueError, KeyboardInterrupt):
        pass
    
    if configurations:
        print(f"\n📊 Comparison Results:")
        print("-" * 60)
        
        for i, config in enumerate(configurations, 1):
            result = evaluator.predict_price(**config)
            
            print(f"\n{i}. {config['year']} McLaren {config['model']}")
            print(f"   📍 {config['mileage']:,} miles, {config['state']}")
            print(f"   💰 Estimated Value: ${result['ensemble_prediction']:,.0f}")
            
            agreement = (1 - result['model_agreement']) * 100
            if agreement > 90:
                confidence = "🟢 High"
            elif agreement > 80:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"
            
            print(f"   📊 Confidence: {confidence} ({agreement:.1f}%)")

if __name__ == "__main__":
    print("🏎️  McLaren Price Evaluation System")
    print("=" * 50)
    print("Choose mode:")
    print("1. Interactive Single Evaluation")
    print("2. Batch Comparison")
    
    try:
        mode = input("\nEnter choice (1 or 2): ").strip()
        
        if mode == "1":
            interactive_price_evaluation()
        elif mode == "2":
            batch_comparison()
        else:
            print("Invalid choice. Running interactive mode...")
            interactive_price_evaluation()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    print("\n🎉 Evaluation complete!")
    print("💡 For detailed analysis, check ML_Analysis_Summary.md")
    print("🧪 For generalization testing, run: python holdout_analysis.py") 