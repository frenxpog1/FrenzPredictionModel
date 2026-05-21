import pandas as pd

# Load the prediction tracker
df_tracker = pd.read_csv('csv_data/prediction_tracker.csv')

# Filter for Game 2 matches
df_g2 = df_tracker[df_tracker['game_number'] == 2].copy()
print(f"Total Game 2 matches in test set: {len(df_g2)}")

if len(df_g2) > 0:
    correct_count = df_g2['correct'].sum()
    print(f"Game 2 Accuracy in Test Set: {correct_count / len(df_g2) * 100:.2f}% ({correct_count}/{len(df_g2)})")
    
    # Load games and matches from the CSV and match them
    games_df = pd.read_csv('csv_data/games.csv')
    matches_df = pd.read_csv('csv_data/matches.csv')
    df_m = pd.merge(games_df, matches_df[['match_id', 'series_score_a', 'series_score_b', 'team_a_name', 'team_b_name']], on='match_id')
    df_m['series_winner'] = df_m.apply(lambda r: r['team_a_name'] if r['series_score_a'] > r['series_score_b'] else r['team_b_name'], axis=1)
    
    # Let's keep unique game rows with series_winner
    df_keys = df_m[['season', 'game_number', 'blue_side_team', 'red_side_team', 'series_winner']].drop_duplicates()
    
    # Merge back into our Game 2 test set
    df_g2_merged = pd.merge(df_g2, df_keys, on=['season', 'game_number', 'blue_side_team', 'red_side_team'], how='left')
    
    # Check how many times the actual game winner is the series winner
    actual_is_series_winner = (df_g2_merged['actual_winner'] == df_g2_merged['series_winner']).sum()
    print(f"Actual Game 2 winner is the overall SERIES winner: {actual_is_series_winner}/{len(df_g2_merged)} ({actual_is_series_winner/len(df_g2_merged)*100:.1f}%)")
else:
    print("No Game 2 matches found in the test set!")
