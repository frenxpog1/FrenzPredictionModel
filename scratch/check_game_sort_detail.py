import pandas as pd

# Load games and matches
games_df = pd.read_csv('csv_data/games.csv')
matches_df = pd.read_csv('csv_data/matches.csv')

# Merge to get series scores and names
df = pd.merge(games_df, matches_df[['match_id', 'series_score_a', 'series_score_b', 'team_a_name', 'team_b_name']], on='match_id')

def get_series_winner_loser(row):
    if row['series_score_a'] > row['series_score_b']:
        return pd.Series([row['team_a_name'], row['team_b_name']])
    else:
        return pd.Series([row['team_b_name'], row['team_a_name']])

df[['series_winner', 'series_loser']] = df.apply(get_series_winner_loser, axis=1)

# Group by match_id and analyze the order of map winners for matches that are 2-1 (best of 3, non-sweep)
non_sweeps = df[(df['series_score_a'] + df['series_score_b'] == 3) & ((df['series_score_a'] == 2) | (df['series_score_b'] == 2))]
match_groups = non_sweeps.groupby('match_id')

print(f"Total non-sweep 2-1 matches: {len(match_groups)}")

patterns = {}
for mid, group in match_groups:
    group_sorted = group.sort_values('game_number')
    if len(group_sorted) != 3:
        # Skip if game count is not 3
        continue
    map_winners = group_sorted['map_winner'].tolist()
    winner = group_sorted['series_winner'].iloc[0]
    loser = group_sorted['series_loser'].iloc[0]
    
    # Map map_winners to 'Winner' or 'Loser'
    pattern = []
    for w in map_winners:
        if w == winner:
            pattern.append('Winner')
        elif w == loser:
            pattern.append('Loser')
        else:
            pattern.append('Other')
    pattern_tuple = tuple(pattern)
    patterns[pattern_tuple] = patterns.get(pattern_tuple, 0) + 1

print("\nMap winner sequences in database for 2-1 matches (Winner vs Loser):")
for pat, count in patterns.items():
    print(f"Pattern {list(pat)}: {count} matches ({count/sum(patterns.values())*100:.2f}%)")
