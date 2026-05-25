import pandas as pd
import numpy as np

matches_path = '/Users/fritzhelrosenacario/Desktop/Predictive analysis practice/csv_data/matches.csv'
feature_matrix_path = '/Users/fritzhelrosenacario/Desktop/Predictive analysis practice/csv_data/ML_Feature_Matrix.csv'

df_matches = pd.read_csv(matches_path)
df_features = pd.read_csv(feature_matrix_path)

print("Matches Columns:")
print(list(df_matches.columns))

print("\nFeature Matrix Columns:")
print(list(df_features.columns))

# Check for shadow duplicates in team names if they exist
if 'team1' in df_matches.columns and 'team2' in df_matches.columns:
    teams = set(df_matches['team1'].dropna()).union(set(df_matches['team2'].dropna()))
    print(f"\nUnique teams count: {len(teams)}")
    print("Sorted teams:")
    print(sorted(list(teams))[:20]) # Print first 20 to check for variations like Team Liquid / Team Liquid PH

