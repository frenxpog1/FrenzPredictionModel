# ═══════════════════════════════════════════════════════════════════
# 🎮 INDIVIDUAL PLAYER ELO RANKINGS (WITH TIME DECAY)
# MPL PH Seasons 1-17
#
# Time decay mechanisms:
#   1. Season Regression: ELO regresses 30% toward 1500 each new season
#   2. Inactivity Penalty: +15% regression per missed season
#   3. Recency K-factor: S1 games count at 0.64x, S17 games at 1.28x
# ═══════════════════════════════════════════════════════════════════
import sqlite3, pandas as pd, json, os
from collections import defaultdict

# Resolve paths (works from both root and 1_NoteBook folder)
db_path    = "mlbb_data.db"    if os.path.exists("mlbb_data.db")    else "../mlbb_data.db"
csv_dir_pe = "csv_data"        if os.path.exists("csv_data")        else "../csv_data"

conn = sqlite3.connect(db_path)
matches = pd.read_sql("""
    SELECT match_id, season, stage, match_timestamp, team_a_name, team_b_name,
           series_score_a, series_score_b
    FROM matches ORDER BY match_timestamp
""", conn)
rosters_raw = pd.read_sql("SELECT season, team_name, players FROM season_rosters", conn)
conn.close()

# Parse rosters → {(season, team): [{"ign": ..., "role": ...}, ...]}
roster_map = {}
for _, row in rosters_raw.iterrows():
    players = json.loads(row.players) if isinstance(row.players, str) else row.players
    roster_map[(int(row.season), row.team_name)] = players

## ── IGN ALIAS MAP ──
# (Handles cases where Liquipedia wiki editors failed to create redirects)
IGN_ALIASES = {
    # ── SAME TEAM (certain) ──
    "3Mar":         "3MarTzy",          # TNC Pro Team
    "BON CHAN":     "Bon Chan",         # Blacklist Intl.
    "DEX STAR":     "Dex Star",         # Blacklist Intl.
    "Dlar":         "Dlarskie",         # ONIC PH
    "Domeng":       "Domengkite",       # Minana/Aurora Gold Lane
    "DomengDR":     "Domengkite",       # Nexplay (same player, DR tag era)
    "E2Max":        "E2MAX",            # Execration
    "E2max":        "E2MAX",            # Execration
    "EDWARD":       "Edward",           # Blacklist → Aurora EXP Lane
    "ESON":         "Eson",             # Blacklist Intl.
    "Flap":         "FlapTzy",          # AP.Bren → Team Falcons
    "H2WO":         "H2wo",             # Nexplay EVOS
    "Imbadeejay":   "ImbaDeejay",       # Cignal Ultra
    "Karl":         "KarlTzy",          # SGD Omega → AP.Bren → Team Liquid
    "KyleTzy":      "Kyle",             # AP.Bren → Team Falcons Jungler
    "Lancecy":      "LanceCy",          # Minana EVOS → TNC Pro Team Mid
    "Lord Malikk":  "Malik",            # AP.Bren Jungler
    "Mico":         "Micophobia",       # ONIC → Nexplay
    "Netskie":      "Nets",             # ONIC → Omega Gold Lane
    "Nova":         "xNova",            # AP.Bren Roamer
    "OHEB":         "Oheb",             # Blacklist → Team Liquid Gold Lane
    "Pando":        "Pandora",          # AP.Bren EXP Lane
    "RENEJAY":      "Renejay",          # Nexplay → Blacklist → Aurora
    "rTzy":         "RTZY",             # RSG PH Middle (NOT 3MarTzy)
    "SUPER MARCO":  "Super Marco",      # AP.Bren → Team Falcons Gold Lane
    "YellyHazeDR":  "YellyHaze",        # Nexplay Middle
    "yNot":         "YnoT",             # Omega Esports
    # ── LEET-SPEAK / NUMBER SUBSTITUTION (confirmed same player) ──
    "1rrad":        "Irrad",            # RSG PH Jungler S10→S11 (1 vs I)
    "Ukir":         "Uk1r",             # Omega Esports Middle S13→S15 (i vs 1)
    "P4kbet":       "Pakbet",           # Execration/Omega Middle (4 vs a)
    # ── USER CONFIRMED ADDITIONAL ALIASES ──
    "Chuuu":        "SDzyz",            # TNC Pro Team Jungler S8→S9
    "Kousei":       "Kouzen",           # RSG PH → TNC Gold Lane (same player, IGN change)
    "RTzy":         "RTZY",             # Work-Auster S7 → RSG PH S12 (same player, NOT 3MarTzy)
    "Exort":        "Bornok",           # Omega Esports Middle S13→S14 (most likely same)
    "ynoT":         "YnoT",             # GeekFam S4-S5 (lowercase 'y')
    "Bon Chon":     "Bon Chan",         # SxC/EVOS S2-S3 player name -> Coach Bon Chan
    "Kekedoot":     "Kekedot",          # ONIC S9-S11 Roamer
    "KKDot":        "Kekedot",          # AP.Bren S15 Roamer
    "Pandaaa":      "Panda",            # ArkAngel S2 Jungler -> Coach Panda
    # ── CASING / PUNCTUATION FIXES ──
    "JeffQT4ever":  "JeffQt4ever",
    "Jeffqt4ever":  "JeffQt4ever",
    "Shaiderqt":    "ShaiderQT",
}

def resolve_ign(ign):
    """Return canonical IGN, following alias chain."""
    seen = set()
    while ign in IGN_ALIASES and ign not in seen:
        seen.add(ign)
        ign = IGN_ALIASES[ign]
    return ign

# Apply aliases to all rosters in-place
for key in roster_map:
    for p in roster_map[key]:
        p["ign"] = resolve_ign(p["ign"])

def get_roster(season, team):
    if (season, team) in roster_map:
        return roster_map[(season, team)]
    for s in [season, season - 1, season + 1]:
        if (s, team) in roster_map:
            return roster_map[(s, team)]
    return []

# Role base K-factors (higher = more individual carry impact)
ROLE_K = {
    "S": 40, "Jungler": 40,
    "Middle": 32, "Mid": 32,
    "EXP Lane": 28, "Gold Lane": 28,
    "Roamer": 24, "Support": 24,
}
DEFAULT_K          = 30
SEASON_REGRESSION  = 0.30   # 30% regress toward 1500 each new season
INACTIVITY_PENALTY = 0.15   # extra 15% regression per missed season

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo, expected, actual, k):
    return elo + k * (actual - expected)

def recency_k_multiplier(season):
    # S1 → 0.64x weight, S9 → 0.96x, S17 → 1.28x
    return 0.60 + 0.04 * season

# State trackers
player_elo            = defaultdict(lambda: 1500.0)
player_role           = {}
player_team           = {}
player_seasons_active = defaultdict(set)
player_games          = defaultdict(int)
player_wins           = defaultdict(int)
player_last_season    = {}

matches["season"] = matches["season"].astype(int)

# ── SEASON CUTOFF ── change this number to include more/fewer seasons
SEASON_CUTOFF = 17   # Include up to S17
matches = matches[matches["season"] <= SEASON_CUTOFF]
all_seasons = sorted(matches["season"].unique())

all_players_by_season = defaultdict(set)
for (season, team), players in roster_map.items():
    for p in players:
        all_players_by_season[season].add(p["ign"])

# ── MAIN LOOP ──
prev_season = None
for season in all_seasons:
    season_matches = matches[matches["season"] == season]

    # Between-season decay
    if prev_season is not None and season != prev_season:
        for ign in list(player_elo.keys()):
            last    = player_last_season.get(ign, prev_season)
            missed  = season - last
            regress = min(SEASON_REGRESSION + (missed - 1) * INACTIVITY_PENALTY, 0.80)
            player_elo[ign] = player_elo[ign] + (1500.0 - player_elo[ign]) * regress

    for _, match in season_matches.iterrows():
        team_a, team_b   = match.team_a_name, match.team_b_name
        score_a, score_b = match.series_score_a, match.series_score_b
        if   score_a > score_b: winner = team_a
        elif score_b > score_a: winner = team_b
        else: continue

        roster_a = get_roster(season, team_a)
        roster_b = get_roster(season, team_b)
        if not roster_a or not roster_b:
            continue

        for p in roster_a + roster_b:
            ign = p["ign"]
            player_role[ign] = p.get("role", "Unknown")
            player_team[ign] = team_a if p in roster_a else team_b
            player_seasons_active[ign].add(season)
            player_last_season[ign] = season

        k_mult = recency_k_multiplier(season)
        elos_a = [player_elo[p["ign"]] for p in roster_a]
        elos_b = [player_elo[p["ign"]] for p in roster_b]
        exp_a  = expected_score(sum(elos_a) / len(elos_a), sum(elos_b) / len(elos_b))
        exp_b  = 1 - exp_a

        for side, roster, expected in [(team_a, roster_a, exp_a), (team_b, roster_b, exp_b)]:
            actual = 1.0 if side == winner else 0.0
            for p in roster:
                ign  = p["ign"]
                role = p.get("role", "Unknown")
                k    = ROLE_K.get(role, DEFAULT_K) * k_mult
                player_elo[ign] = update_elo(player_elo[ign], expected, actual, k)
                player_games[ign] += 1
                if actual == 1.0:
                    player_wins[ign] += 1

    prev_season = season

# ── BUILD OUTPUT ──
rows = []
for ign, elo in player_elo.items():
    gp = player_games[ign]
    if gp < 1:
        continue
    seasons = sorted(player_seasons_active[ign])
    rows.append({
        "player":       ign,
        "role":         player_role.get(ign, "Unknown"),
        "last_team":    player_team.get(ign, "Unknown"),
        "seasons":      ", ".join("S" + str(s) for s in seasons),
        "last_season":  max(seasons),
        "games_played": gp,
        "wins":         player_wins[ign],
        "win_rate":     round(player_wins[ign] / gp * 100, 1),
        "elo":          round(elo, 1),
    })

df_players = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
df_players.index += 1

# Save CSV
df_players.to_csv(csv_dir_pe + "/player_elo_rankings.csv", index=True, index_label="rank")

# ── DISPLAY ALL PLAYERS ──
HDR = "{:<5} {:<20} {:<12} {:<22} {:>6}  {:<5} {:<5} {:<7} {}"
ROW = "{:<5} {:<20} {:<12} {:<22} {:>6.0f}  {:<5} {:<5} {:<7} S{}..S{}"
SEP = "=" * 95

print(SEP)
print("INDIVIDUAL PLAYER ELO RANKINGS — MPL PH S1-S" + str(SEASON_CUTOFF) + " (Time Decay)")
print("Season regression: 30% | Inactivity: +15%/season | Recency K: S1=0.64x -> S17=1.28x")
print("Total players: " + str(len(df_players)))
print(SEP)

roles_order = [
    ("S",         "Jungler",   "JUNGLER"),
    ("Middle",    "Mid",       "MIDLANER"),
    ("EXP Lane",  None,        "EXP LANE"),
    ("Gold Lane", None,        "GOLD LANE"),
    ("Roamer",    "Support",   "ROAMER / SUPPORT"),
    (None,        None,        "OTHER / FLEX"),
]

for r1, r2, label in roles_order:
    if r1 is None:
        # catch-all for unlabelled/flex roles
        known = ["S","Jungler","Middle","Mid","EXP Lane","Gold Lane","Roamer","Support"]
        mask = ~df_players["role"].isin(known)
    else:
        mask = df_players["role"] == r1
        if r2:
            mask = mask | (df_players["role"] == r2)

    sub = df_players[mask].reset_index(drop=True)
    if sub.empty:
        continue

    print("\n" + "-" * 95)
    print("  " + label + "  (" + str(len(sub)) + " players)")
    print("  " + HDR.format("Rank", "Player", "Role", "Last Team", "ELO", "W", "G", "WR%", "Seasons"))
    print("  " + "-" * 88)
    for rank, (_, r) in enumerate(sub.iterrows(), 1):
        seasons_list = sorted(int(s.strip()[1:]) for s in r["seasons"].split(",") if s.strip())
        s_min = seasons_list[0] if seasons_list else 0
        s_max = seasons_list[-1] if seasons_list else 0
        print("  " + ROW.format(
            rank,
            str(r["player"]),
            str(r["role"]),
            str(r["last_team"]),
            float(r["elo"]),
            int(r["wins"]),
            int(r["games_played"]),
            float(r["win_rate"]),
            s_min, s_max
        ))

print("\n" + SEP)
print("Full rankings saved to: " + csv_dir_pe + "/player_elo_rankings.csv")
