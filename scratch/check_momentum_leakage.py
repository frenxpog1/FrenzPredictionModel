import pandas as pd

# Load games and matches
games_df = pd.read_csv('csv_data/games.csv')
matches_df = pd.read_csv('csv_data/matches.csv')

# Merge to get series scores
df = pd.merge(games_df, matches_df[['match_id', 'series_score_a', 'series_score_b', 'team_a_name', 'team_b_name']], on='match_id')

# Find the overall series winner
def get_series_winner(row):
    if row['series_score_a'] > row['series_score_b']:
        return row['team_a_name']
    else:
        return row['team_b_name']

df['series_winner'] = df.apply(get_series_winner, axis=1)

# Check if the map winner of Game 1 is always the overall series winner
g1_df = df[df['game_number'] == 1]
always_winner_wins_g1 = (g1_df['map_winner'] == g1_df['series_winner']).mean()

print(f"Total series analyzed: {len(g1_df)}")
print(f"Percentage of matches where Game 1 map winner in the database is the overall match winner: {always_winner_wins_g1 * 100:.2f}%")

# Let's look at games where series_winner is not map_winner of game 1, if any
diff_df = g1_df[g1_df['map_winner'] != g1_df['series_winner']]
print(f"Number of series where Game 1 map winner is NOT the series winner: {len(diff_df)}")
if len(diff_df) > 0:
    print(diff_df[['match_id', 'game_number', 'blue_side_team', 'red_side_team', 'map_winner', 'series_winner']].head(5))
