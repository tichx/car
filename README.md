# McLaren Price Prediction ML System

## 🎯 Overview

A comprehensive machine learning system for McLaren price prediction featuring 11 algorithms from basic to advanced, achieving **99.1% R² accuracy** on the top model. The system includes data scraping, processing, model evaluation, holdout analysis, and production-ready price evaluation.

## 📁 Project Structure

```
car-1/
├── scraper.py                          # Web scraper for Classic.com McLaren listings
├── data_process.py                     # Data cleaning and processing pipeline
├── ml.py                              # Complete ML analysis with 11 algorithms
├── price_evaluator.py                 # Production-ready price evaluator
├── holdout_analysis.py                # Generalization testing (hypercar holdout)
├── quick_demo.py                      # Interactive demo with current/future predictions
├── mclaren_us_processed_final.csv     # Final cleaned dataset (595 listings)
├── mclaren_price_evaluator.pkl        # Trained model (5.1MB)
├── ML_Analysis_Summary.md             # Detailed ML analysis findings
└── Holdout_Analysis_Summary.md        # Generalization test results
```

## 🚀 Quick Start

### 1. Run Price Predictions
```bash
python quick_demo.py
```
**Features:**
- Current market value estimates
- Future price predictions
- Depreciation analysis
- Investment scoring
- Interactive mode

### 2. Test Model Generalization
```bash
python holdout_analysis.py
```
**Tests:** How well models trained on regular McLarens predict hypercar prices

### 3. Full ML Analysis
```bash
python ml.py
```
**Includes:** All 11 algorithms, feature importance, time series validation

## 📊 Key Results

### 🏆 Top Model Performance
| Model | MAE | R² | MAPE | Complexity |
|-------|-----|----|----- |------------|
| **Gradient Boosting** | $94,343 | **0.991** | 17.2% | Advanced |
| **Random Forest** | $124,498 | 0.970 | 13.3% | Intermediate |
| **XGBoost** | $135,466 | 0.956 | 13.7% | Advanced |

### 🎯 Feature Importance
1. **is_hypercar** (30.2%) - Premium model classification
2. **age_squared** (25.9%) - Non-linear depreciation
3. **age** (20.6%) - Primary depreciation factor
4. **year** (10.8%) - Model year significance
5. **model_encoded** (3.3%) - Specific variant

### 📈 Model Categories & Pricing
- **Entry Sports**: 184 cars, avg $138,590 (570S, MP4-12C)
- **Supercar**: 178 cars, avg $276,324 (720S, 765LT)  
- **Hypercar**: 93 cars, avg $2,462,447 (P1, Senna, F1)
- **Grand Tourer**: 51 cars, avg $153,386 (570GT)
- **Track-Focused**: 44 cars, avg $210,036 (600LT, 620R)

## 🧪 Holdout Analysis Findings

**Critical Discovery:** Models trained without hypercar data show **massive performance degradation** on hypercars:

| Scenario | Non-Hypercar MAE | Hypercar MAE | Performance Drop |
|----------|------------------|---------------|------------------|
| Without hypercar flag | $29k | $2.2M | **+7500%** |
| With hypercar flag | $106k | $106k | **Stable** |

**Insight:** Tier-specific modeling is essential for automotive pricing.

## 🎪 Price Evaluator Usage

```python
from price_evaluator import McLarenPriceEvaluator

# Load trained model
evaluator = McLarenPriceEvaluator.load_model()

# Predict current market value
result = evaluator.predict_price(
    year=2020, 
    model='720S', 
    mileage=5000, 
    state='CA'
)

print(f"Price: ${result['ensemble_prediction']:,.0f}")
print(f"Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")
print(f"Confidence: {(1-result['model_agreement'])*100:.1f}%")
```

## 📋 Data Pipeline

### 1. Data Collection (`scraper.py`)
- **Source:** Classic.com McLaren listings
- **Volume:** 2,380 listings → 595 final (US only)
- **Features:** 24 attributes including seller info, dates, specs
- **Performance:** 100 pages in 16 minutes with 10 parallel threads

### 2. Data Processing (`data_process.py`)
- **Filtering:** McLaren brand only, US locations, valid prices/mileage
- **Cleaning:** Fixed mileage parsing ("3k km (1k mi)" → 1000 miles)
- **Enhancement:** Model normalization, price/currency separation
- **Output:** 595 high-quality US McLaren listings

### 3. ML Analysis (`ml.py`)
- **Algorithms:** 11 models from Linear Regression to XGBoost
- **Features:** 20 engineered features including model categories
- **Validation:** Cross-validation + time series splits
- **Ensemble:** Top 3 models with weighted averaging

## 🔧 Technical Features

### Advanced ML Capabilities
- **Multi-algorithm ensemble** (11 different approaches)
- **Sophisticated feature engineering** (age², model categories, market timing)
- **Time series validation** (temporal data splits)
- **Weighted ensemble prediction** (performance-based combining)
- **Model persistence** (save/load trained models)

### Production Ready
- **Confidence scoring** for prediction reliability
- **Error handling** for unseen categories
- **Scalable architecture** with tier-specific routing
- **Comprehensive logging** and validation

## 📈 Business Applications

### ✅ Recommended Use Cases
- **Insurance valuations** (high accuracy needed)
- **Dealer pricing** (market-based estimates)  
- **Investment analysis** (depreciation modeling)
- **Market research** (price trend analysis)

### ⚠️ Limitations
- **US market only** (training data constraint)
- **No condition data** (assumes good condition)
- **Rare variants** (limited training examples)
- **Market volatility** (economic changes not captured)

## 🔮 Future Enhancements

1. **Global market expansion** (UK, European data)
2. **Condition modeling** (accident history, service records)
3. **Real-time updates** (live market data integration)
4. **Options modeling** (carbon fiber, MSO packages)
5. **Market sentiment** (economic indicators, trends)

## 📚 Documentation

- **`ML_Analysis_Summary.md`** - Complete ML findings and model explanations
- **`Holdout_Analysis_Summary.md`** - Generalization testing results
- **Code comments** - Extensive documentation throughout

## 🎉 Achievements

✅ **Outstanding accuracy**: 99.1% R² on top model  
✅ **Comprehensive analysis**: 11 algorithms evaluated  
✅ **Production ready**: Trained model with API  
✅ **Generalization tested**: Holdout analysis completed  
✅ **Current/future predictions**: Supports both scenarios  
✅ **Interactive demo**: Easy-to-use interface  
✅ **Complete documentation**: Detailed findings and usage  

**Total System:** From data scraping to production deployment, a complete McLaren price prediction solution ready for real-world automotive applications.