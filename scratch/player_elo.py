"""
Individual Player ELO with Time Decay — MPL PH Seasons 1-17

Time decay mechanisms:
1. Season Regression: Between each season, ELO regresses 30% back toward 1500
   (new season = partial fresh start, old reputation fades)
2. Inactivity Decay: If a player skips a season, extra 15% regression per missed season
3. Recency K-factor: K scales with season recency so S17 results matter ~2x more than S1
   K_effective = base_K * (0.6 + 0.04 * season_number)  →  S1=0.64x, S17=1.28x
"""
import sqlite3, pandas as pd, json
from collections import defaultdict

conn = sqlite3.connect('mlbb_data.db')
matches = pd.read_sql("""
    SELECT match_id, season, stage, match_timestamp, team_a_name, team_b_name,
           series_score_a, series_score_b
    FROM matches ORDER BY match_timestamp
""", conn)
rosters_raw = pd.read_sql("SELECT season, team_name, players FROM season_rosters", conn)
conn.close()

# Parse rosters
roster_map = {}
for _, row in rosters_raw.iterrows():
    players = json.loads(row.players) if isinstance(row.players, str) else row.players
    roster_map[(int(row.season), row.team_name)] = players

def get_roster(season, team):
    if (season, team) in roster_map:
        return roster_map[(season, team)]
    for s in [season, season-1, season+1]:
        if (s, team) in roster_map:
            return roster_map[(s, team)]
    return []

# Role base K-factors
ROLE_K = {
    'S': 40, 'Jungler': 40,
    'Middle': 32, 'Mid': 32,
    'EXP Lane': 28, 'Gold Lane': 28,
    'Roamer': 24, 'Support': 24,
}
DEFAULT_K = 30

SEASON_REGRESSION   = 0.30   # regress 30% toward 1500 each new season
INACTIVITY_PENALTY  = 0.15   # extra 15% regression per missed season

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo, expected, actual, k):
    return elo + k * (actual - expected)

def recency_k_multiplier(season):
    # S1 → 0.64x, S9 → 0.96x, S17 → 1.28x
    return 0.60 + 0.04 * season

# State tracking
player_elo           = defaultdict(lambda: 1500.0)
player_role          = {}
player_team          = {}
player_seasons_active= defaultdict(set)
player_games         = defaultdict(int)
player_wins          = defaultdict(int)
player_last_season   = {}

# Group matches by season so we can apply between-season decay
matches['season'] = matches['season'].astype(int)
all_seasons = sorted(matches['season'].unique())

# Get all players who ever appeared in any roster
all_players_by_season = defaultdict(set)  # season → set of player igns
for (season, team), players in roster_map.items():
    for p in players:
        all_players_by_season[season].add(p['ign'])

# ─── MAIN LOOP ───
prev_season = None

for season in all_seasons:
    season_matches = matches[matches['season'] == season]

    # ── BETWEEN-SEASON DECAY ──
    if prev_season is not None and season != prev_season:
        active_this_season = all_players_by_season.get(season, set())
        active_prev_season = all_players_by_season.get(prev_season, set())

        for ign in list(player_elo.keys()):
            missed = 0
            # Count how many seasons they've been absent
            last = player_last_season.get(ign, prev_season)
            missed = season - last  # 1 = normal gap, 2+ = skipped seasons

            # Baseline season regression (everyone)
            regress = SEASON_REGRESSION + (missed - 1) * INACTIVITY_PENALTY
            regress = min(regress, 0.80)  # cap at 80% regression

            old = player_elo[ign]
            player_elo[ign] = old + (1500.0 - old) * regress

    # ── PROCESS MATCHES IN THIS SEASON ──
    for _, match in season_matches.iterrows():
        team_a   = match.team_a_name
        team_b   = match.team_b_name
        score_a  = match.series_score_a
        score_b  = match.series_score_b

        if score_a > score_b:
            winner, loser = team_a, team_b
        elif score_b > score_a:
            winner, loser = team_b, team_a
        else:
            continue

        roster_a = get_roster(season, team_a)
        roster_b = get_roster(season, team_b)
        if not roster_a or not roster_b:
            continue

        # Register players
        for p in roster_a + roster_b:
            team = team_a if p in roster_a else team_b
            ign = p['ign']
            player_role[ign] = p.get('role', 'Unknown')
            player_team[ign] = team_a if p in roster_a else team_b
            player_seasons_active[ign].add(season)
            player_last_season[ign] = season

        # Recency multiplier for this season
        k_mult = recency_k_multiplier(season)

        # Average team ELO for expected score
        elos_a = [player_elo[p['ign']] for p in roster_a]
        elos_b = [player_elo[p['ign']] for p in roster_b]
        avg_a = sum(elos_a) / len(elos_a)
        avg_b = sum(elos_b) / len(elos_b)
        exp_a = expected_score(avg_a, avg_b)
        exp_b = 1 - exp_a

        for side, roster, expected in [(team_a, roster_a, exp_a), (team_b, roster_b, exp_b)]:
            actual = 1.0 if side == winner else 0.0
            for p in roster:
                ign  = p['ign']
                role = p.get('role', 'Unknown')
                k    = ROLE_K.get(role, DEFAULT_K) * k_mult
                player_elo[ign] = update_elo(player_elo[ign], expected, actual, k)
                player_games[ign] += 1
                if actual == 1.0:
                    player_wins[ign] += 1

    prev_season = season

# ─── BUILD OUTPUT ───
rows = []
for ign, elo in player_elo.items():
    gp = player_games[ign]
    if gp < 1:
        continue
    seasons = sorted(player_seasons_active[ign])
    rows.append({
        'player':       ign,
        'role':         player_role.get(ign, 'Unknown'),
        'last_team':    player_team.get(ign, 'Unknown'),
        'seasons_list': seasons,
        'last_season':  max(seasons),
        'games_played': gp,
        'wins':         player_wins[ign],
        'win_rate':     round(player_wins[ign] / gp * 100, 1),
        'elo':          round(elo, 1),
    })

df = pd.DataFrame(rows).sort_values('elo', ascending=False).reset_index(drop=True)
df.index += 1

print("=" * 95)
print("🎮 INDIVIDUAL PLAYER ELO RANKINGS (WITH TIME DECAY) — MPL PH S1-S17")
print("   Season regression: 30% | Inactivity penalty: +15%/season | Recency K: S1=0.64x → S17=1.28x")
print("=" * 95)
print(f"Total players ranked: {len(df)}")
print()

def print_table(title, sub_df, n=15):
    print(f"\n{'─'*95}")
    print(f"  {title}")
    print(f"  {'#':<4} {'Player':<18} {'Team':<22} {'ELO':>6}  {'W':<5} {'G':<5} {'WR%':<7} {'Last Season'}")
    print("  " + "-" * 75)
    for rank, (_, row) in enumerate(sub_df.head(n).iterrows(), 1):
        bar = "█" * int(row['elo'] / 100 - 9) if row['elo'] > 900 else ""
        print(f"  {rank:<4} {row['player']:<18} {row['last_team']:<22} {row['elo']:>6.0f}  {row['wins']:<5} {row['games_played']:<5} {row['win_rate']:<7} S{row['last_season']}")

print_table("🏆 TOP 20 OVERALL (All Roles, Time-Decay ELO)", df, 20)

roles_config = [
    ('S',        'Jungler',  '⚔️  JUNGLER'),
    ('Middle',   'Mid',      '🔮 MIDLANER'),
    ('EXP Lane', None,       '🛡️  EXP LANE'),
    ('Gold Lane',None,       '💰 GOLD LANE'),
    ('Roamer',   'Support',  '🌀 ROAMER'),
]
for r1, r2, label in roles_config:
    mask = df['role'] == r1
    if r2:
        mask = mask | (df['role'] == r2)
    print_table(f"TOP 10 {label}", df[mask], 10)

# Save CSV
df_out = df.copy()
df_out['seasons'] = df_out['seasons_list'].apply(lambda x: ', '.join(f'S{s}' for s in x))
df_out = df_out.drop(columns=['seasons_list'])
df_out.to_csv('csv_data/player_elo_rankings.csv', index=True, index_label='rank')
print(f"\n{'='*95}")
print(f"✅ Saved to csv_data/player_elo_rankings.csv")
print()
print("📌 TIME DECAY EXPLANATION:")
print("   • Between every season, ALL player ELOs regress 30% toward 1500 (fresh start effect)")
print("   • Missing a season adds extra 15% regression per missed season (inactivity penalty)")
print("   • K-factor scales by recency: S1 games count at 0.64x, S17 games count at 1.28x")
print("   • This means S1 legends need recent S15-S17 performances to stay highly ranked")
