import pandas as pd

# Load the dataset
df = pd.read_csv('csv_data/matches.csv')

# Standardize team names
df.replace('SGD Omega', 'Omega Esports', inplace=True)
# Add more replacements here if needed

# Save the cleaned data
df.to_csv('csv_data/matches.csv', index=False)

print("Team names cleaned successfully.")
