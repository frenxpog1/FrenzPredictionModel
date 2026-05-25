import pandas as pd
import numpy as np

matches_path = '/Users/fritzhelrosenacario/Desktop/Predictive analysis practice/csv_data/matches.csv'
feature_matrix_path = '/Users/fritzhelrosenacario/Desktop/Predictive analysis practice/csv_data/ML_Feature_Matrix.csv'

try:
    df_matches = pd.read_csv(matches_path)
    print(f"Loaded matches.csv: {df_matches.shape[0]} rows, {df_matches.shape[1]} columns")
except Exception as e:
    print(f"Error loading matches.csv: {e}")
    df_matches = None

try:
    df_features = pd.read_csv(feature_matrix_path)
    print(f"Loaded ML_Feature_Matrix.csv: {df_features.shape[0]} rows, {df_features.shape[1]} columns")
except Exception as e:
    print(f"Error loading ML_Feature_Matrix.csv: {e}")
    df_features = None

if df_matches is not None:
    print("\n--- MATCHES AUDIT ---")
    missing_matches = df_matches.isnull().sum()
    missing_cols = missing_matches[missing_matches > 0]
    if not missing_cols.empty:
        print("Missing Values:\n", missing_cols)
    else:
        print("No missing values found.")
        
    dupes = df_matches.duplicated().sum()
    print(f"Exact Duplicates: {dupes}")

if df_features is not None:
    print("\n--- FEATURE MATRIX AUDIT ---")
    missing_features = df_features.isnull().sum()
    missing_cols = missing_features[missing_features > 0]
    if not missing_cols.empty:
        print("Missing Values:\n", missing_cols)
    else:
        print("No missing values found.")
        
    dupes = df_features.duplicated().sum()
    print(f"Exact Duplicates: {dupes}")
    
    print("\nFeatures with <= 1 unique value (Zero Variance / Constant):")
    constant_features = [col for col in df_features.columns if df_features[col].nunique() <= 1]
    print(len(constant_features), "features.")
    if len(constant_features) > 0 and len(constant_features) < 20:
        print(constant_features)
    elif len(constant_features) >= 20:
        print(constant_features[:10], "... and more")

