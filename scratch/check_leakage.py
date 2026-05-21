import pandas as pd
import numpy as np

# Load games.csv
df = pd.read_csv('csv_data/games.csv')

# Group by match_id and look at map_winner in chronological order of game_number
print("Checking first 10 matches in games.csv to see game_number and map_winner sequence:\n")

match_ids = df['match_id'].unique()[:10]

for mid in match_ids:
    match_games = df[df['match_id'] == mid].sort_values('game_number')
    print(f"Match ID: {mid}")
    print(match_games[['game_number', 'blue_side_team', 'red_side_team', 'map_winner']])
    print("-" * 50)
