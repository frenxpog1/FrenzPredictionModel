import pandas as pd

# Load the prediction tracker
df_tracker = pd.read_csv('csv_data/prediction_tracker.csv')

# Filter for Game 1 matches
df_g1 = df_tracker[df_tracker['game_number'] == 1].copy()
print(f"Total Game 1 matches in test set: {len(df_g1)}")

if len(df_g1) > 0:
    correct_count = df_g1['correct'].sum()
    print(f"Game 1 Accuracy in Test Set: {correct_count / len(df_g1) * 100:.2f}% ({correct_count}/{len(df_g1)})")
    
    # Load games and matches from the CSV and match them
    games_df = pd.read_csv('csv_data/games.csv')
    matches_df = pd.read_csv('csv_data/matches.csv')
    df_m = pd.merge(games_df, matches_df[['match_id', 'series_score_a', 'series_score_b', 'team_a_name', 'team_b_name']], on='match_id')
    df_m['series_winner'] = df_m.apply(lambda r: r['team_a_name'] if r['series_score_a'] > r['series_score_b'] else r['team_b_name'], axis=1)
    
    # Let's keep unique game rows with team_a_name and series_winner
    df_keys = df_m[['season', 'game_number', 'blue_side_team', 'red_side_team', 'team_a_name', 'series_winner']].drop_duplicates()
    
    # Merge back into our Game 1 test set
    df_g1_merged = pd.merge(df_g1, df_keys, on=['season', 'game_number', 'blue_side_team', 'red_side_team'], how='left')
    
    # Check how many times the actual game winner is Team A (which corresponds to Team 1)
    actual_is_team_a = (df_g1_merged['actual_winner'] == df_g1_merged['team_a_name']).sum()
    print(f"Actual Game 1 winner is TEAM A (Team 1): {actual_is_team_a}/{len(df_g1_merged)} ({actual_is_team_a/len(df_g1_merged)*100:.1f}%)")
    
    # Check how many times the actual game winner is the overall SERIES winner
    actual_is_series_winner = (df_g1_merged['actual_winner'] == df_g1_merged['series_winner']).sum()
    print(f"Actual Game 1 winner is the overall SERIES winner: {actual_is_series_winner}/{len(df_g1_merged)} ({actual_is_series_winner/len(df_g1_merged)*100:.1f}%)")
else:
    print("No Game 1 matches found in the test set!")
