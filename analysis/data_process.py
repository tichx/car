# process the data

# 1. remove the listings that are not from brand mclaren
# 3. remove the listings that are not in the US
# 4. process the price raw data to update the price column
# 5. process the mileage raw data to update the mileage column in integer format
# filter data that does not have a price
# filter data that does not have mileage data
# show how many listings are left, and how many listings are removed and why
# show data summary by model, year, price, mileage, location, country, seller name, listing date, sold, active listing

# Do this in the above order.

# data is in the file mclaren_listings_final_20250601_201026.csv

# save the processed data to a new file

import pandas as pd
import numpy as np
import re
from datetime import datetime

def process_mclaren_data():
    """Process McLaren listings data according to the specified plan"""
    
    print("🏎️  McLaren Data Processing Pipeline")
    print("=" * 60)
    
    # Load the data
    print("\n📁 Loading data from mclaren_listings_final_20250601_201026.csv...")
    df = pd.read_csv('mclaren_listings_final_20250601_201026.csv')
    print(f"Initial dataset: {len(df):,} listings")
    
    # Track removals
    removal_stats = {}
    
    # 1. Remove listings that are not from brand McLaren
    print("\n🔍 Step 1: Filtering McLaren brand only...")
    initial_count = len(df)
    df_mclaren = df[df['brand'].str.contains('McLaren', case=False, na=False)].copy()
    non_mclaren_removed = initial_count - len(df_mclaren)
    removal_stats['Non-McLaren brands'] = non_mclaren_removed
    
    print(f"   Removed {non_mclaren_removed:,} non-McLaren listings")
    print(f"   Remaining: {len(df_mclaren):,} McLaren listings")
    
    # Show what brands were removed
    if non_mclaren_removed > 0:
        other_brands = df[~df['brand'].str.contains('McLaren', case=False, na=False)]['brand'].value_counts()
        print(f"   Removed brands: {dict(other_brands)}")
    
    # 2. Remove listings that are not in the US
    print("\n🇺🇸 Step 2: Filtering US listings only...")
    before_us_filter = len(df_mclaren)
    df_us = df_mclaren[df_mclaren['country_code'] == 'US'].copy()
    non_us_removed = before_us_filter - len(df_us)
    removal_stats['Non-US locations'] = non_us_removed
    
    print(f"   Removed {non_us_removed:,} non-US listings")
    print(f"   Remaining: {len(df_us):,} US McLaren listings")
    
    # Show what countries were removed
    if non_us_removed > 0:
        other_countries = df_mclaren[df_mclaren['country_code'] != 'US']['country_code'].value_counts()
        print(f"   Removed countries: {dict(other_countries)}")
    
    # 3. Process the price raw data to update the price column
    print("\n💰 Step 3: Processing price data...")
    def clean_price(price_raw):
        """Extract numeric price from raw price text"""
        if pd.isna(price_raw):
            return None
        
        price_str = str(price_raw)
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[^\d.]', '', price_str)
        
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    # Apply price cleaning
    df_us['price_processed'] = df_us['price_raw'].apply(clean_price)
    
    # Update the price column with processed prices, fallback to original
    df_us['price_final'] = df_us['price_processed'].fillna(df_us['price'])
    
    price_improvements = df_us['price_processed'].notna().sum() - df_us['price'].notna().sum()
    print(f"   Processed {df_us['price_raw'].notna().sum():,} raw price entries")
    print(f"   Price data improvements: {max(0, price_improvements)} additional valid prices")
    
    # 4. Process the mileage raw data to update the mileage column in integer format
    print("\n🛣️  Step 4: Processing mileage data (FIXED)...")
    def clean_mileage(mileage_text):
        """Extract numeric mileage from text - CORRECTLY handles km/mi formats"""
        if pd.isna(mileage_text):
            return None
        
        mileage_str = str(mileage_text).lower()
        
        # First, check if there's a miles value in parentheses (like "3k km (1k mi)")
        miles_in_parens = re.search(r'\((\d+(?:,\d+)*(?:\.\d+)?)\s*k?\s*mi', mileage_str)
        if miles_in_parens:
            try:
                value = float(miles_in_parens.group(1).replace(',', ''))
                # If it contains 'k', multiply by 1000
                if 'k' in miles_in_parens.group(0):
                    value *= 1000
                return int(value)
            except ValueError:
                pass
        
        # If no miles in parentheses, look for direct miles value
        miles_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k?\s*mi', mileage_str)
        if miles_match:
            try:
                value = float(miles_match.group(1).replace(',', ''))
                # If it contains 'k', multiply by 1000
                if 'k' in miles_match.group(0):
                    value *= 1000
                return int(value)
            except ValueError:
                pass
        
        # If no miles found, look for any number with k (treat as thousands)
        number_k_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*k', mileage_str)
        if number_k_match:
            try:
                value = float(number_k_match.group(1).replace(',', '')) * 1000
                return int(value)
            except ValueError:
                pass
        
        # Last resort: extract any number
        number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', mileage_str)
        if number_match:
            try:
                value = float(number_match.group(1).replace(',', ''))
                return int(value)
            except ValueError:
                pass
        
        return None
    
    # Apply mileage cleaning
    df_us['mileage_processed'] = df_us['mileage'].apply(clean_mileage)
    
    mileage_improvements = df_us['mileage_processed'].notna().sum()
    print(f"   Processed {df_us['mileage'].notna().sum():,} mileage entries")
    print(f"   Successfully parsed: {mileage_improvements:,} mileage values")
    
    # Show some examples of mileage parsing
    print("\n   Mileage parsing examples:")
    sample_mileage = df_us[['mileage', 'mileage_processed']].dropna().head(10)
    for _, row in sample_mileage.iterrows():
        print(f"     '{row['mileage']}' → {row['mileage_processed']:,} miles")
    
    # 5. Filter data that does not have a price
    print("\n💸 Step 5: Filtering entries with valid prices...")
    before_price_filter = len(df_us)
    df_with_price = df_us[df_us['price_final'].notna() & (df_us['price_final'] > 0)].copy()
    no_price_removed = before_price_filter - len(df_with_price)
    removal_stats['No valid price'] = no_price_removed
    
    print(f"   Removed {no_price_removed:,} listings without valid prices")
    print(f"   Remaining: {len(df_with_price):,} listings")
    
    # 6. Filter data that does not have mileage data
    print("\n🚗 Step 6: Filtering entries with valid mileage...")
    before_mileage_filter = len(df_with_price)
    df_final = df_with_price[df_with_price['mileage_processed'].notna()].copy()
    no_mileage_removed = before_mileage_filter - len(df_final)
    removal_stats['No valid mileage'] = no_mileage_removed
    
    print(f"   Removed {no_mileage_removed:,} listings without valid mileage")
    print(f"   Final dataset: {len(df_final):,} listings")
    
    # 7. Show how many listings are left and removal summary
    print(f"\n📊 REMOVAL SUMMARY:")
    print(f"   Initial dataset: {initial_count:,} listings")
    for reason, count in removal_stats.items():
        print(f"   Removed - {reason}: {count:,}")
    print(f"   Final dataset: {len(df_final):,} listings")
    print(f"   Retention rate: {len(df_final)/initial_count*100:.1f}%")
    
    # 8. Show data summary
    print(f"\n📈 DATA SUMMARY BY CATEGORIES:")
    
    # Summary by model
    print(f"\n🚗 BY MODEL:")
    model_summary = df_final.groupby('model').agg({
        'price_final': ['count', 'mean', 'median'],
        'year': ['min', 'max'],
        'mileage_processed': 'median'
    }).round(0)
    model_summary.columns = ['Count', 'Avg_Price', 'Med_Price', 'Min_Year', 'Max_Year', 'Med_Mileage']
    model_summary = model_summary.sort_values('Count', ascending=False)
    print(model_summary.head(15))
    
    # Summary by year
    print(f"\n📅 BY YEAR:")
    year_summary = df_final.groupby('year').agg({
        'price_final': ['count', 'mean'],
        'mileage_processed': 'median'
    }).round(0)
    year_summary.columns = ['Count', 'Avg_Price', 'Med_Mileage']
    year_summary = year_summary.sort_values('year', ascending=False)
    print(year_summary.head(10))
    
    # Price ranges
    print(f"\n💰 PRICE ANALYSIS:")
    print(f"   Price range: ${df_final['price_final'].min():,.0f} - ${df_final['price_final'].max():,.0f}")
    print(f"   Average price: ${df_final['price_final'].mean():,.0f}")
    print(f"   Median price: ${df_final['price_final'].median():,.0f}")
    
    # Mileage analysis
    print(f"\n🛣️  MILEAGE ANALYSIS:")
    mileage_data = df_final['mileage_processed'].dropna()
    if len(mileage_data) > 0:
        print(f"   Mileage range: {mileage_data.min():,.0f} - {mileage_data.max():,.0f} miles")
        print(f"   Average mileage: {mileage_data.mean():,.0f} miles")
        print(f"   Median mileage: {mileage_data.median():,.0f} miles")
    
    # Location analysis
    print(f"\n🏛️  BY STATE:")
    state_summary = df_final['state'].value_counts().head(10)
    for state, count in state_summary.items():
        percentage = count / len(df_final) * 100
        print(f"   {state}: {count:,} ({percentage:.1f}%)")
    
    # Seller analysis
    print(f"\n👥 TOP SELLERS:")
    seller_summary = df_final['seller_name'].value_counts().head(10)
    for seller, count in seller_summary.items():
        print(f"   {seller}: {count:,} listings")
    
    # Listing status
    print(f"\n📋 LISTING STATUS:")
    sold_counts = df_final['sold'].value_counts()
    print(f"   Sold: {sold_counts.get('y', 0):,}")
    print(f"   Available: {sold_counts.get('n', 0):,}")
    
    active_counts = df_final['active_listing'].value_counts()
    print(f"   Active: {active_counts.get('y', 0):,}")
    print(f"   Inactive: {active_counts.get('n', 0):,}")
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"mclaren_us_processed_{timestamp}.csv"
    
    # Prepare final columns
    final_columns = [
        'brand', 'model', 'model_raw', 'year', 'price_final', 'currency', 'price_raw',
        'mileage_processed', 'mileage', 'location', 'country_code', 'state', 
        'listing_url', 'listing_date', 'engine', 'transmission', 
        'sold', 'active_listing', 'seller_name', 'seller_url', 'seller_verified',
        'title', 'image_url', 'scraped_at'
    ]
    
    # Rename columns for clarity
    df_output = df_final[final_columns].copy()
    df_output = df_output.rename(columns={
        'price_final': 'price',
        'mileage_processed': 'mileage'
    })
    
    df_output.to_csv(output_filename, index=False)
    print(f"\n💾 Processed data saved to: {output_filename}")
    print(f"📊 Final dataset: {len(df_output):,} US McLaren listings ready for analysis")
    
    # Print all models and their counts
    print(f"\n🏎️  ALL MCLAREN MODELS AND COUNTS:")
    print("=" * 50)
    all_models = df_final['model'].value_counts().sort_values(ascending=False)
    for model, count in all_models.items():
        percentage = count / len(df_final) * 100
        print(f"   {model}: {count:,} ({percentage:.1f}%)")
    
    return df_output, output_filename

if __name__ == "__main__":
    processed_df, filename = process_mclaren_data()





