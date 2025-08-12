# PSI Implementation for Lab 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def calculate_psi(train_data, test_data, bins=10):
    """
    Calculate PSI between train and test data
    Simple implementation that I understood from research
    """
    # Remove any missing values
    train_clean = train_data.dropna()
    test_clean = test_data.dropna()
    
    if len(train_clean) == 0 or len(test_clean) == 0:
        return None
    
    # Create bins based on training data
    try:
        # Use quantiles for binning to get equal-sized bins
        _, bin_edges = pd.qcut(train_clean, bins, retbins=True, duplicates='drop')
        
        # Make sure the edges cover the full range
        bin_edges[0] = min(train_clean.min(), test_clean.min()) - 0.001
        bin_edges[-1] = max(train_clean.max(), test_clean.max()) + 0.001
        
        # Calculate distributions
        train_dist = pd.cut(train_clean, bin_edges).value_counts().sort_index()
        test_dist = pd.cut(test_clean, bin_edges).value_counts().sort_index()
        
        # Convert to percentages
        train_pct = train_dist / len(train_clean)
        test_pct = test_dist / len(test_clean)
        
        # Replace zeros with small number to avoid log(0)
        train_pct = train_pct.replace(0, 0.0001)
        test_pct = test_pct.replace(0, 0.0001)
        
        # Calculate PSI
        psi = sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        
        return psi
        
    except Exception as e:
        print(f"Error calculating PSI: {e}")
        return None

def analyze_psi_for_features(df, target_col='Bankrupt?', random_state=42):
    """
    Analyze PSI for key features after train/test split
    """
    # Split the data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    # Select some key financial features to analyze
    financial_features = [
        ' ROA(C) before interest and depreciation before interest',
        ' Debt ratio %',
        ' Current Ratio',
        ' Operating Gross Margin',
        ' Total Asset Turnover',
        ' Net Worth Turnover Rate (times)',
        ' Cash Flow to Total Assets',
        ' Interest Coverage Ratio (Interest expense to EBIT)',
        ' Total debt/Total net worth',
        ' Working Capital to Total Assets'
    ]
    
    # Calculate PSI for each feature
    psi_results = {}
    
    print("PSI Analysis Results:")
    print("=" * 50)
    
    for feature in financial_features:
        if feature in X_train.columns:
            psi_value = calculate_psi(X_train[feature], X_test[feature])
            psi_results[feature] = psi_value
            
            # Interpret PSI value
            if psi_value is None:
                interpretation = "Error in calculation"
            elif psi_value < 0.1:
                interpretation = "No drift"
            elif psi_value < 0.2:
                interpretation = "Slight drift"
            else:
                interpretation = "Significant drift"
            
            print(f"{feature[:40]}: {psi_value:.4f} ({interpretation})")
    
    return psi_results, X_train, X_test, y_train, y_test

def plot_psi_results(psi_results):
    """
    Create a simple bar plot of PSI values
    """
    # Filter out None values
    clean_results = {k: v for k, v in psi_results.items() if v is not None}
    
    if not clean_results:
        print("No valid PSI results to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    features = list(clean_results.keys())
    psi_values = list(clean_results.values())
    
    # Shorten feature names for better display
    short_names = [f[:20] + "..." if len(f) > 20 else f for f in features]
    
    bars = plt.bar(range(len(short_names)), psi_values)
    
    # Color code bars based on PSI value
    for i, (bar, psi_val) in enumerate(zip(bars, psi_values)):
        if psi_val < 0.1:
            bar.set_color('green')
        elif psi_val < 0.2:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Slight drift threshold')
    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Significant drift threshold')
    
    plt.xlabel('Features')
    plt.ylabel('PSI Value')
    plt.title('Population Stability Index (PSI) Analysis\nTrain vs Test Data')
    plt.xticks(range(len(short_names)), short_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Bankrupt?'].value_counts()}")
    
    # Run PSI analysis
    psi_results, X_train, X_test, y_train, y_test = analyze_psi_for_features(df)
    
    # Plot results
    plot_psi_results(psi_results)
    
    print("\nSummary:")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features analyzed: {len([v for v in psi_results.values() if v is not None])}")