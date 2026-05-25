import pandas as pd

# Load the dataset
df = pd.read_csv('csv_data/season_rosters.csv')

# Standardize team names
df.replace('SGD Omega', 'Omega Esports', inplace=True)
df.replace('ONIC Esports PH', 'ONIC PH', inplace=True)
df.replace('Onic Philippines', 'ONIC PH', inplace=True)
df.replace('FNATIC Onic PH', 'ONIC PH', inplace=True)


# Save the cleaned data
df.to_csv('csv_data/season_rosters.csv', index=False)

print("Roster team names cleaned successfully.")
