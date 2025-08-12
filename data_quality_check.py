# Data Quality Check Script for Lab 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_data_quality(df):
    """
    Basic data quality check to identify potential issues
    Student-friendly implementation
    """
    print("DATA QUALITY ANALYSIS FOR LAB 4")
    print("=" * 50)
    
    # Check basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()
    
    # Check for extreme values that might be data errors
    problematic_features = []
    
    print("CHECKING FOR EXTREME VALUES...")
    print("-" * 30)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Bankrupt?':
            continue
            
        values = df[col].dropna()
        
        if len(values) == 0:
            continue
            
        # Check for extremely large values that might be errors
        q99 = values.quantile(0.99)
        max_val = values.max()
        mean_val = values.mean()
        
        # If max is way bigger than 99th percentile, might be an error
        if max_val > q99 * 100 and max_val > 1000:
            problematic_features.append({
                'feature': col,
                'max': max_val,
                'q99': q99,
                'mean': mean_val,
                'ratio': max_val / q99 if q99 > 0 else float('inf')
            })
            
            print(f"⚠️  {col[:35]}...")
            print(f"   Max: {max_val:,.0f}")
            print(f"   99th percentile: {q99:.2f}")
            print(f"   Mean: {mean_val:.4f}")
            print(f"   Ratio (max/99th): {max_val/q99:.0f}x")
            print()
    
    if not problematic_features:
        print("✅ No extreme outliers detected")
    
    return problematic_features

def analyze_class_distribution(df, target_col='Bankrupt?'):
    """
    Analyze the target variable distribution
    """
    print("\nCLASS DISTRIBUTION ANALYSIS:")
    print("-" * 30)
    
    class_counts = df[target_col].value_counts().sort_index()
    total = len(df)
    
    print(f"Non-bankrupt (0): {class_counts[0]:,} ({class_counts[0]/total*100:.1f}%)")
    print(f"Bankrupt (1): {class_counts[1]:,} ({class_counts[1]/total*100:.1f}%)")
    print(f"Imbalance ratio: 1:{class_counts[0]//class_counts[1]}")
    
    # Visual representation
    print(f"\nVisual representation:")
    print(f"Non-bankrupt: {'█' * int(class_counts[0]/total*50)}")
    print(f"Bankrupt:     {'█' * max(1, int(class_counts[1]/total*50))}")
    
    return class_counts

def quick_feature_stats(df, num_features=5):
    """
    Show stats for a few key features
    """
    print(f"\nQUICK STATS FOR {num_features} KEY FEATURES:")
    print("-" * 40)
    
    # Pick some important-looking financial features
    key_features = [
        ' ROA(C) before interest and depreciation before interest',
        ' Debt ratio %', 
        ' Current Ratio',
        ' Operating Gross Margin',
        ' Total Asset Turnover'
    ]
    
    for feature in key_features[:num_features]:
        if feature in df.columns:
            values = df[feature].dropna()
            print(f"\n{feature[:30]}...")
            print(f"  Range: {values.min():.4f} to {values.max():.4f}")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std: {values.std():.4f}")
            
            # Count potential outliers using IQR method
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = values[(values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)]
            print(f"  Outliers: {len(outliers)} ({len(outliers)/len(values)*100:.1f}%)")

# Main execution
if __name__ == "__main__":
    print("Running Data Quality Check for Lab 4...")
    print("Loading data.csv...")
    
    try:
        df = pd.read_csv('data.csv')
        
        # Run all checks
        issues = check_data_quality(df)
        class_counts = analyze_class_distribution(df)
        quick_feature_stats(df)
        
        print(f"\n" + "="*50)
        print("SUMMARY FOR LAB 4:")
        print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"⚠️  Features with extreme values: {len(issues)}")
        print(f"⚠️  Class imbalance severity: {class_counts[0]//class_counts[1]}:1")
        
        if issues:
            print(f"\n🔧 PREPROCESSING NEEDED:")
            print(f"   - Fix extreme outliers in {len(issues)} features")
            print(f"   - Handle class imbalance (use SMOTE)")
            print(f"   - Use stratified train/test split")
        
    except FileNotFoundError:
        print("❌ Error: data.csv not found!")
        print("Make sure the data file is in the same directory.")
    except Exception as e:
        print(f"❌ Error: {e}")