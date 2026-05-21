import pandas as pd
import sqlite3

# Load from SQLite database
conn = sqlite3.connect("mlbb_data.db")

# Read matches and games
matches = pd.read_sql_query("SELECT * FROM matches", conn)
games = pd.read_sql_query("SELECT * FROM games", conn)
conn.close()

print(f"Matches count: {len(matches)}")
print(f"Games count: {len(games)}")

# Let's inspect series scores
print("\nSeries score distributions:")
print(matches.groupby(['series_score_a', 'series_score_b']).size())

# Let's merge matches and games
df = pd.merge(games, matches[['match_id', 'series_score_a', 'series_score_b', 'team_a_name', 'team_b_name', 'stage']], on='match_id')

# Check game_number distribution
print("\nGame number distribution:")
print(df['game_number'].value_counts().sort_index())

# Filter for Bo3 matches (Regular Season has Bo3, Playoffs has Bo5/Bo7)
df_bo3 = df[df['stage'] != 'Playoffs']
print(f"\nBo3 games count: {len(df_bo3)}")

# Group by match_id and check patterns of map_winner
groups = df_bo3.groupby('match_id')
patterns = {}
for mid, group in groups:
    group = group.sort_values('game_number')
    score_a = group['series_score_a'].iloc[0]
    score_b = group['series_score_b'].iloc[0]
    
    # map winner sequence
    winners = []
    for _, row in group.iterrows():
        w = row['map_winner']
        t1 = row['team_a_name']
        t2 = row['team_b_name']
        if w == t1:
            winners.append('t1')
        elif w == t2:
            winners.append('t2')
        else:
            winners.append('other')
            
    pattern_key = (score_a, score_b, tuple(winners))
    patterns[pattern_key] = patterns.get(pattern_key, 0) + 1

print("\nBo3 match patterns (series_score_a, series_score_b, map_winners_sequence):")
for pat, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
    print(f"Score {pat[0]}-{pat[1]} | Sequence {pat[2]}: {count} matches")

# Let's check target_blue_win in Game 1
df['blue_won_g1'] = (df['map_winner'] == df['blue_side_team']).astype(int)
g1_df = df[df['game_number'] == 1]
print("\nGame 1 target_blue_win by series score:")
print(g1_df.groupby(['series_score_a', 'series_score_b'])['blue_won_g1'].mean())
