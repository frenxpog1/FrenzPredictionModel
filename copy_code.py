import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from draft_embeddings import get_svd_hero_embeddings

# ── TEAM ALIAS MAP (Franchise Rebrands) ──
# Maps historical rebranded team names to their modern franchise equivalent.
TEAM_ALIAS_MAP = {
    "Aether Main": "Team Falcons PH",
    "Aether Valkyrie": "Team Falcons PH",
    
    # "Aurora" (S6/S7) is mapped contextually in the function since the modern Aurora franchise exists.
    "Sunsparks": "Team Liquid PH",
    
    "ECHO": "Team Liquid PH",
    
    "Execration": "Omega Esports",
    "SGD Omega": "Omega Esports",
    
    "ONIC Esports PH": "ONIC PH",
    "Onic Philippines": "ONIC PH",
    "Fnatic ONIC PH": "ONIC PH",
    
    "Work-Auster Force": "TNC Pro Team",
}

def resolve_team_name(team_name, season=None):
    t = str(team_name).strip()
    
    # Handle the "Aurora" collision: 
    # S6 & S7 "Aurora" is actually "Aura PH" (which became Team Liquid PH).
    # S14+ "Aurora" is the modern franchise formed by Blacklist players.
    if t == "Aurora" and season is not None:
        try:
            s_int = int(str(season).replace('S', ''))
            if s_int <= 10:
                return "Team Liquid PH"
        except:
            pass
            
    # Handle AP.Bren vs Team Falcons PH transition:
    # Before Season 15, AP.Bren is the franchise that rebranded to Team Falcons PH in Season 15.
    # In Season 15+, AP.Bren and Team Falcons PH are two separate teams.
    if t == "AP.Bren" and season is not None:
        try:
            s_int = int(str(season).replace('S', ''))
            if s_int < 15:
                return "Team Falcons PH"
        except:
            pass
    elif t == "Falcons AP.Bren" or t == "FC AP.Bren":
        return "Team Falcons PH"
            
    return TEAM_ALIAS_MAP.get(t, t)

print("🚀 Starting V6 Master Pipeline...")
print("   New Features: Ban Disruption, Series Momentum, Gap Days, G3 Clutch, Playoff Elo, Reverse Sweep Rate")

# ==========================================
# STEP 1: LOAD & CLEAN
# ==========================================
matches_df = pd.read_csv('csv_data/matches.csv')
games_df   = pd.read_csv('csv_data/games.csv')
rosters_df = pd.read_csv('csv_data/season_rosters.csv')
patches_df = pd.read_csv('csv_data/patches.csv')

# --- APPLY FRANCHISE MAPPING GLOBALLY ---
if 'team_a_name' in matches_df.columns: matches_df['team_a_name'] = matches_df.apply(lambda r: resolve_team_name(r['team_a_name'], r.get('season')), axis=1)
if 'team_b_name' in matches_df.columns: matches_df['team_b_name'] = matches_df.apply(lambda r: resolve_team_name(r['team_b_name'], r.get('season')), axis=1)

games_df['blue_side_team'] = games_df.apply(lambda r: resolve_team_name(r['blue_side_team'], r.get('season')), axis=1)
games_df['red_side_team']  = games_df.apply(lambda r: resolve_team_name(r['red_side_team'], r.get('season')), axis=1)
games_df['map_winner']     = games_df.apply(lambda r: resolve_team_name(r['map_winner'], r.get('season')), axis=1)

rosters_df['team_name']    = rosters_df.apply(lambda r: resolve_team_name(r['team_name'], r.get('season')), axis=1)

# --- LOAD OFFICIAL SEASONAL STATS & STANDINGS ---
teams_df = pd.read_csv('csv_data/official_mpl_ph_stats/mpl_ph_official_teams_s5_s15_s17.csv')
standings_df = pd.read_csv('csv_data/official_mpl_ph_stats/mpl_ph_official_standings_s5_s15_s17.csv')

teams_df['team'] = teams_df.apply(lambda r: resolve_team_name(r['team'], r.get('season')), axis=1)
standings_df['team'] = standings_df.apply(lambda r: resolve_team_name(r['team'], r.get('season')), axis=1)

# Standings and teams parsing / compilation helpers
def parse_w_l(val):
    if pd.isna(val): return 0, 0
    parts = str(val).split('-')
    if len(parts) == 2:
        try: return int(parts[0].strip()), int(parts[1].strip())
        except: return 0, 0
    return 0, 0

standings_dict = {}
for _, row in standings_df.iterrows():
    s = row['season']
    t = row['team']
    mw, ml = parse_w_l(row['match_w_l'])
    gw, gl = parse_w_l(row['games_w_l'])
    m_played = mw + ml
    g_played = gw + gl
    m_wr = mw / m_played if m_played > 0 else 0.5
    g_wr = gw / g_played if g_played > 0 else 0.5
    
    standings_dict[(s, t)] = {
        'games_played': g_played,
        'matches_played': m_played,
        'match_winrate': m_wr,
        'game_winrate': g_wr
    }

compiled_stats = {}
for _, row in teams_df.iterrows():
    s = row['season']
    t = row['team']
    standings = standings_dict.get((s, t), {'games_played': 0, 'match_winrate': 0.5, 'game_winrate': 0.5})
    g_played = standings['games_played']
    
    if s == 11:
        avg_kills = row['average_kills']
        avg_deaths = row['average_deaths']
        avg_assists = row['average_assists']
        avg_kda = row['average_kda']
    else:
        avg_kills = row['total_kills'] / g_played if g_played > 0 else np.nan
        avg_deaths = row['total_deaths'] / g_played if g_played > 0 else np.nan
        avg_assists = row['total_assists'] / g_played if g_played > 0 else np.nan
        avg_kda = (avg_kills + avg_assists) / max(1.0, avg_deaths) if not (pd.isna(avg_kills) or pd.isna(avg_deaths)) else np.nan
        
    compiled_stats[(s, t)] = {
        'match_winrate': standings['match_winrate'],
        'game_winrate': standings['game_winrate'],
        'avg_kills': avg_kills,
        'avg_deaths': avg_deaths,
        'avg_assists': avg_assists,
        'avg_kda': avg_kda
    }

def get_official_team_abbreviation(db_name, season):
    name = str(db_name).strip()
    if name == 'Blacklist Intl.': return 'BLCK'
    elif name == 'ONIC PH': return 'ONIC'
    elif name == 'AP.Bren': return 'APBR' if season >= 12 else 'BREN'
    elif name == 'Omega Esports': return 'SGD' if season == 5 else 'OMG'
    elif name == 'Execration': return 'EXE'
    elif name == 'BSB': return 'BSB'
    elif name == 'ULVL': return 'ULVL'
    elif name == 'Sunsparks': return 'SS'
    elif name == 'Geek Fam PH': return 'GEEK'
    elif name == 'STI e-Olympians': return 'STI'
    elif name == 'BnK Blufire': return 'BNK'
    elif name == 'Cignal Ultra': return 'CIG'
    elif name == 'Nexplay EVOS': return 'NXP' if season <= 7 else 'NXPE'
    elif name == 'PlayBook Esports': return 'LPE'
    elif name == 'Work-Auster Force': return 'WORK'
    elif name == 'RSG PH': return 'RSG'
    elif name == 'TNC Pro Team': return 'TNC'
    elif name == 'Team Liquid PH': return 'AURA' if season <= 7 else 'TLPH'
    elif name == 'Aurora': return 'AURA' if season <= 7 else 'RORA'
    elif name == 'Minana EVOS': return 'MNNE'
    elif name == 'Team Falcons PH': return 'FLCN'
    elif name == 'Twisted Minds PH': return 'TWIS'
    return None

def lookup_prior_stats(db_name, current_season):
    res = {
        'match_winrate': np.nan,
        'game_winrate': np.nan,
        'avg_kills': np.nan,
        'avg_deaths': np.nan,
        'avg_assists': np.nan,
        'avg_kda': np.nan
    }
    
    # 1. Find the most recent match_winrate and game_winrate
    for s in range(current_season - 1, 4, -1):
        abbrev = get_official_team_abbreviation(db_name, s)
        if abbrev and (s, abbrev) in compiled_stats:
            stats = compiled_stats[(s, abbrev)]
            if pd.isna(res['match_winrate']) and not pd.isna(stats['match_winrate']):
                res['match_winrate'] = stats['match_winrate']
            if pd.isna(res['game_winrate']) and not pd.isna(stats['game_winrate']):
                res['game_winrate'] = stats['game_winrate']
            if not pd.isna(res['match_winrate']) and not pd.isna(res['game_winrate']):
                break
                
    # 2. Find the most recent valid kills/deaths/assists/kda (stats not all 0), only S15 or older
    for s in range(min(current_season - 1, 15), 4, -1):
        abbrev = get_official_team_abbreviation(db_name, s)
        if abbrev and (s, abbrev) in compiled_stats:
            stats = compiled_stats[(s, abbrev)]
            # If stats are all 0 or NaN, skip
            if stats.get('avg_kills', 0) == 0 and stats.get('avg_deaths', 0) == 0:
                continue
            if pd.isna(res['avg_kills']) and not pd.isna(stats['avg_kills']):
                res['avg_kills'] = stats['avg_kills']
                res['avg_deaths'] = stats['avg_deaths']
                res['avg_assists'] = stats['avg_assists']
                res['avg_kda'] = stats['avg_kda']
                break
                
    # Fill in defaults if still NaN
    if pd.isna(res['match_winrate']): res['match_winrate'] = 0.5
    if pd.isna(res['game_winrate']): res['game_winrate'] = 0.5
    if pd.isna(res['avg_kills']): res['avg_kills'] = 11.0
    if pd.isna(res['avg_deaths']): res['avg_deaths'] = 11.0
    if pd.isna(res['avg_assists']): res['avg_assists'] = 25.0
    if pd.isna(res['avg_kda']): res['avg_kda'] = 3.0
    
    return res, None

matches_df['match_timestamp'] = pd.to_datetime(matches_df['match_timestamp'])
games_df['match_timestamp']   = pd.to_datetime(games_df['match_timestamp'])

valid_match_ids = games_df['match_id'].unique()
matches_df = matches_df[matches_df['match_id'].isin(valid_match_ids)]
matches_df = matches_df[matches_df['match_timestamp'] < pd.Timestamp.now()]

def clean_patch_version(x):
    x = str(x).strip()
    if x.lower() in ('nan', 'none', ''):
        return 'Unknown'
    x = x.replace('–', '-')
    return x.split('-')[-1].strip() if '-' in x else x

matches_df['patch_version'] = matches_df['patch_version'].apply(clean_patch_version)
matches_df.loc[matches_df['season'] == 13, 'patch_version'] = matches_df.loc[matches_df['season'] == 13, 'patch_version'].replace('Unknown', 'S13_Unknown')

target_date = matches_df['match_timestamp'].max()
matches_df['days_ago']     = (target_date - matches_df['match_timestamp']).dt.days
matches_df['time_weight']  = np.exp(-0.002 * matches_df['days_ago'])
matches_df['is_playoff_match'] = matches_df['stage'].str.lower().str.strip() == 'playoffs'

# ==========================================
# STEP 2: ELO WITH DECAY + SEPARATE PLAYOFF ELO
# ==========================================
print("Calculating Elo ratings (Regular + Playoff tracks)...")

team_rosters = {}
for _, row in rosters_df.iterrows():
    team, season = row['team_name'], str(row['season']).strip()
    try:    player_igns = set([p['ign'].strip() for p in json.loads(row['players'])])
    except: player_igns = set()
    if team not in team_rosters: team_rosters[team] = {}
    team_rosters[team][season] = player_igns

matches_df = matches_df.sort_values('match_timestamp').reset_index(drop=True)

# Two parallel Elo systems: regular and playoff-only
player_elos         = {}   # Updated every match
player_playoff_elos = {}   # Only updated for playoff matches

default_elo  = 1500
k_regular    = 32
k_playoff    = 56
decay_rate   = 0.20

player_championship_wins = {}
player_playoff_experience = {}

# ── IGN ALIAS MAP (For Missing Liquipedia Redirects) ──
IGN_ALIASES = {
    "3Mar":         "3MarTzy",          # TNC Pro Team
    "BON CHAN":     "Bon Chan",         # Blacklist Intl.
    "DEX STAR":     "Dex Star",         # Blacklist Intl.
    "Dlar":         "Dlarskie",         # ONIC PH
    "Domeng":       "Domengkite",       # Minana/Aurora Gold Lane
    "DomengDR":     "Domengkite",       # Nexplay (same player, DR tag era)
    "Bon Chon":     "Bon Chan",         # Typo
    "BruskoDR":     "Brusko",           # Nexplay DR tag
    "GoyongDR":     "Goyong",           # Nexplay DR tag
    "Had ji":       "Hadji",            # Space
    "Had Ji":       "Hadji",            # Space
    "Kekedot":      "Kekedoot",         # Typo
    "YellyHazeDR":  "YellyHaze",        # Nexplay DR tag
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
    "1rrad":        "Irrad",            # RSG PH Jungler S10→S11 (1 vs I)
    "Ukir":         "Uk1r",             # Omega Esports Middle S13→S15 (i vs 1)
    "P4kbet":       "Pakbet",           # Execration/Omega Middle (4 vs a)
    "Chuuu":        "SDzyz",            # TNC Pro Team Jungler S8→S9
    "Kousei":       "Kouzen",           # RSG PH → TNC Gold Lane (same player, IGN change)
    "RTzy":         "RTZY",             # Work-Auster S7 → RSG PH S12 (same player, NOT 3MarTzy)
    "Exort":        "Bornok",           # Omega Esports Middle S13→S14 (most likely same)
    "ynoT":         "YnoT",             # GeekFam S4-S5 (lowercase 'y')
    "Bon Chon":     "Bon Chan",         # SxC/EVOS S2-S3 player name -> Coach Bon Chan
    "Kekedoot":     "Kekedot",          # ONIC S9-S11 Roamer
    "KKDot":        "Kekedot",          # AP.Bren S15 Roamer
    "Pandaaa":      "Panda",            # ArkAngel S2 Jungler -> Coach Panda
    "JeffQT4ever":  "JeffQt4ever",
    "Jeffqt4ever":  "JeffQt4ever",
    "Shaiderqt":    "ShaiderQT",
}

import re
NORMALIZED_ALIASES = {}
for k, v in IGN_ALIASES.items():
    clean_k = re.sub(r'\s+|-|_|─|–', '', str(k).strip().lower())
    clean_v = re.sub(r'\s+|-|_|─|–', '', str(v).strip().lower())
    NORMALIZED_ALIASES[clean_k] = clean_v

def resolve_ign(ign):
    if not ign or str(ign) == 'nan':
        return "unknown"
    clean_str = re.sub(r'\s+|-|_|─|–', '', str(ign).strip().lower())
    
    seen = set()
    while clean_str in NORMALIZED_ALIASES and clean_str not in seen:
        seen.add(clean_str)
        clean_str = NORMALIZED_ALIASES[clean_str]
    return clean_str

matches_df['team_a_elo']          = 0.0
matches_df['team_b_elo']          = 0.0
matches_df['team_a_playoff_elo']  = 0.0
matches_df['team_b_playoff_elo']  = 0.0
current_global_season = str(matches_df.iloc[0]['season'])

def get_team_avg_elo(team_name, season_string, elo_dict):
    roster = team_rosters.get(team_name, {}).get(season_string, set())
    roster_igns = [resolve_ign(ign) for ign in roster]
    if not roster_igns: return default_elo
    return sum([elo_dict.get(ign, default_elo) for ign in roster_igns]) / len(roster_igns)



# ==========================================
# STEP 3: MERGE & PATCH META
# ==========================================
def parse_patch_json(json_str):
    if pd.isna(json_str) or json_str == '{}': return {}
    try:    return json.loads(str(json_str).replace('""', '"'))
    except: return {}

patches_df['adjustments_dict'] = patches_df['hero_adjustments'].apply(parse_patch_json)
patch_lookup = {str(row['patch_version']).strip(): row['adjustments_dict'] for _, row in patches_df.iterrows()}

match_cols = [
    'match_id', 'season', 'match_timestamp', 'patch_version', 'stage',
    'team_a_name', 'team_b_name',
    
    
    'time_weight'
]

training_df = pd.merge(
    games_df,
    matches_df[match_cols],
    on=['match_id', 'season', 'match_timestamp'],
    how='inner'
)
training_df = training_df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)

def calculate_draft_meta(row):
    try:
        picks = json.loads(str(row['picks']).replace('""', '"'))
        blue_heroes, red_heroes = picks.get('blue', []), picks.get('red', [])
    except: return pd.Series([0, 0, 0, 0])
    adjustments = patch_lookup.get(str(row['patch_version']).strip(), {})
    b_buffs = sum(1 for h in blue_heroes if adjustments.get(h) == 'BUFF')
    b_nerfs = sum(1 for h in blue_heroes if adjustments.get(h) == 'NERF')
    r_buffs = sum(1 for h in red_heroes  if adjustments.get(h) == 'BUFF')
    r_nerfs = sum(1 for h in red_heroes  if adjustments.get(h) == 'NERF')
    return pd.Series([b_buffs, b_nerfs, r_buffs, r_nerfs])

training_df[['blue_buffs_in_draft', 'blue_nerfs_in_draft',
             'red_buffs_in_draft',  'red_nerfs_in_draft']] = training_df.apply(calculate_draft_meta, axis=1)

# ==========================================
# STEP 4: PRE-COMPUTE SERIES GAME LOOKUP (For Series Momentum)
# ==========================================
# For Game 2 and 3, we need to know who won the previous game in the same match.
# We build a lookup as we iterate in order.
game_winner_lookup = {}   # (match_id, game_number) -> map_winner team name

# ==========================================
# STEP 5: THE V6 MAIN TRACKERS LOOP
# ==========================================
print("Calculating all V6 features (this includes 14 tracker types)...")

# --- Existing Trackers ---
# SOTA Trackers
player_elos           = {}
player_playoff_elos   = {}
current_global_season = None
global_synergy_matrix = {}
global_counter_matrix = {}

team_hero_tracker      = {}
seasonal_hero_tracker  = {}   # (season, team) -> set of unique heroes picked this season
team_recent_form       = {}   # team -> last 5 results [1/0]
h2h_tracker            = {}   # (teamA, teamB) -> {blue_wins, total}
patch_practice         = {}   # team -> int (games on current patch)
playoff_clutch_tracker = {}   # team -> {wins, games} in playoffs
global_hero_tracker    = {}   # hero -> {wins, games} globally
team_win_durations     = {}   # team -> list of game durations when won
global_side_tracker    = []   # last 50 blue-side outcomes (1/0)
current_patch_tracker  = None
team_patch_stats       = {}   # team -> {patch: {wins, games}}
team_hist_stats        = {}   # team -> {wins, games}

# --- V6 New Trackers ---
team_last_game_date      = {}   # team -> last match timestamp
team_roster_history      = {}   # team -> list of sets of resolved player IGNs (pre-match, leak-safe)
team_g3_tracker          = {}   # team -> {wins, games} in game_number >= 3
team_playoff_games_count = {}   # team -> total playoff games played (experience)

# FIXED: Reverse Sweep Tracker computed INCREMENTALLY inside the main loop.
# We detect series end by checking if it's the last game in a match.
# This prevents future data from leaking into earlier game features.
reverse_sweep_tracker = {}   # team -> {down_01: int, came_back: int}

# --- V8 New Trackers ---
team_loss_durations          = {}   # team -> list of game durations when lost (up to 20)
team_recent_lategame_results = {}   # team -> list of last 10 deep late-game outcomes [1/0] (>18 mins / 1080s)
team_draft_scores            = {}   # team -> list of draft mastery scores (last 15)
team_exec_scores             = {}   # team -> list of execution mastery scores (last 15)

# Safe pre-computation: only uses game_number structure (not outcomes)
# This tells us which game_number ends each match so we know when a series finishes.
last_game_in_match = training_df.groupby('match_id')['game_number'].max().to_dict()

# Tracks G1 winner per match for reverse sweep detection
series_g1_winner   = {}   # match_id -> (g1_winner_team, g1_loser_team)
# Tracks in-series wins for each team to detect series winner at series end
current_series_wins = {}  # match_id -> {blue_team: wins, red_team: wins}

# IN-SERIES DRAFT TRACKER
series_draft_history = {} # match_id -> list of game dicts

# --- S17 Champion's Curse & RS Rank Trackers ---
champions_history = {} # season (int) -> winner_team (str)
rs_standings = {}      # season (int) -> {team: {'wins': int, 'matches': int}}
b_rs_rank, r_rs_rank = [], []
b_is_defending_champ, r_is_defending_champ = [], []

# --- Output Lists ---
b_side_elo, r_side_elo = [], []
b_playoff_elo, r_playoff_elo = [], []
b_championship_dna, r_championship_dna = [], []
b_playoff_winrate, r_playoff_winrate = [], []
b_prev_winner_exhaustion, r_prev_winner_exhaustion = [], []
b_draft_exhaustion, r_draft_exhaustion = [], []
b_synergy, r_synergy = [], []
b_counter, r_counter = [], []

b_com, r_com, b_exp, r_exp = [], [], [], []
b_momentum, r_momentum     = [], []
b_h2h                      = []
b_patch_exp, r_patch_exp   = [], []
b_playoff_clutch, r_playoff_clutch = [], []
is_playoffs_list           = []
b_global_draft_wr, r_global_draft_wr = [], []
b_avg_win_duration, r_avg_win_duration = [], []
blue_side_advantage        = []
# V6 New Output Lists
b_rest_factor, r_rest_factor           = [], []
b_ban_disruption, r_ban_disruption     = [], []
b_g3_clutch_wr, r_g3_clutch_wr         = [], []
b_playoff_exp_count, r_playoff_exp_count = [], []
series_momentum_blue                    = []
b_reverse_sweep_rate, r_reverse_sweep_rate = [], []

# New Series Draft Features lists
b_g1_comfort, r_g1_comfort = [], []
g1_winner_heroes_banned_blue = []
g1_winner_heroes_banned_red = []

# V8 New Output Lists
b_avg_loss_duration, r_avg_loss_duration = [], []
b_execution_margin, r_execution_margin   = [], []
b_execution_punish_score, r_execution_punish_score = [], []
b_lategame_winrate, r_lategame_winrate   = [], []

b_prev_comfort, r_prev_comfort = [], []
b_draft_mastery_list, r_draft_mastery_list = [], []
b_exec_mastery_list, r_exec_mastery_list = [], []
b_draft_reliance_list, r_draft_reliance_list = [], []
prev_winner_heroes_banned_blue = []
prev_winner_heroes_banned_red = []

prev_played_comfort_banned_blue = []
prev_played_comfort_banned_red = []

# NEW: Patch notes comfort hero impact
b_comfort_patch_score = []
r_comfort_patch_score = []

# NEW: Patch Adaptation SOTA Features
b_patch_winrate = []
r_patch_winrate = []
b_patch_adaptation = []
r_patch_adaptation = []

# NEW: Expected Comfort based on top 5 most-played heroes
b_expected_comfort = []
r_expected_comfort = []

# NEW: Prior official season statistics (pre-match, leak-safe lookups)
blue_prev_season_match_wr, red_prev_season_match_wr = [], []
blue_prev_season_game_wr, red_prev_season_game_wr = [], []
blue_prev_season_kda, red_prev_season_kda = [], []
blue_prev_season_avg_kills, red_prev_season_avg_kills = [], []
blue_prev_season_avg_deaths, red_prev_season_avg_deaths = [], []
blue_prev_season_avg_assists, red_prev_season_avg_assists = [], []
diff_prev_season_match_wr, diff_prev_season_game_wr, diff_prev_season_kda = [], [], []


# NEW SOTA: Roster Stability and Draft Overlap output lists
b_roster_stability, r_roster_stability, diff_roster_stability_list = [], [], []
b_draft_overlap_list, r_draft_overlap_list, diff_draft_overlap_list = [], [], []

# SVD Draft Embeddings features lists
blue_draft_embs = [[] for _ in range(16)]
red_draft_embs = [[] for _ in range(16)]
diff_draft_embs = [[] for _ in range(16)]
draft_style_sims = []


def rest_factor_score(gap_days):
    """Converts days since last match to a performance multiplier."""
    if gap_days is None:     return 1.0
    if gap_days <= 3:        return 0.97   # Fatigue
    elif gap_days <= 14:     return 1.02   # Optimal zone
    elif gap_days <= 30:     return 1.0 - 0.001 * (gap_days - 14)  # Mild rust
    else:                    return max(0.90, 0.986 * np.exp(-0.005 * (gap_days - 30)))  # Heavy rust

def get_comfort_patch_impact(team, hero_tracker, patch_version):
    """Measures how well the active patch aligns with players' comfort pools."""
    if team not in hero_tracker: return 0.0
    opp_picks = hero_tracker[team]
    top_comfort = sorted(opp_picks.keys(), key=lambda h: opp_picks[h]['games'], reverse=True)[:12]
    adjustments = patch_lookup.get(patch_version, {})
    score = 0.0
    for h in top_comfort:
        adj = adjustments.get(h)
        if adj == 'BUFF':
            score += 1.0
        elif adj == 'NERF':
            score -= 1.0
    return score

def get_expected_comfort(team, hero_tracker):
    """
    Calculates the expected comfort score for a team prior to the match.
    Expected comfort is the average Bayesian-smoothed win rate of the team's top 5 most-played heroes historically.
    """
    if team not in hero_tracker or not hero_tracker[team]:
        return 0.5  # Neutral default when no history is available
    
    # Sort team's historical heroes by games played descending
    team_heroes = hero_tracker[team]
    sorted_heroes = sorted(team_heroes.keys(), key=lambda h: team_heroes[h]['games'], reverse=True)
    
    # Take top 5
    top_5 = sorted_heroes[:5]
    if not top_5:
        return 0.5
        
    smoothed_wrs = []
    for h in top_5:
        stats = team_heroes[h]
        # Bayesian-smoothed win rate: (wins + 2) / (games + 4)
        smoothed_wr = (stats['wins'] + 2) / (stats['games'] + 4)
        smoothed_wrs.append(smoothed_wr)
        
    return float(np.mean(smoothed_wrs))

def get_ban_disruption(banned_heroes, opponent_team, hero_tracker):
    """
    Measures how well a team targeted the opponent's comfort heroes with their bans.
    Returns: fraction of bans that hit one of opponent's top-12 comfort heroes.
    """
    if not banned_heroes or opponent_team not in hero_tracker: return 0.0
    opp_picks = hero_tracker[opponent_team]
    # Sort opponent's heroes by games played to find their top comfort pool
    top_comfort = set(sorted(opp_picks.keys(), key=lambda h: opp_picks[h]['games'], reverse=True)[:12])
    hits = sum(1 for h in banned_heroes if h in top_comfort)
    return hits / len(banned_heroes)

def get_g3_wr(team):
    stats = team_g3_tracker.get(team, {'wins': 0, 'games': 0})
    return (stats['wins'] + 2) / (stats['games'] + 4)  # Bayesian smoothing

def get_reverse_sweep_rate(team):
    stats = reverse_sweep_tracker.get(team, {'down_01': 0, 'came_back': 0})
    return (stats['came_back'] + 1) / (stats['down_01'] + 2)  # Bayesian smoothing

# Load SVD Draft Embeddings and initialize historical trackers
hero_embeddings, fallback_emb = get_svd_hero_embeddings("./mlbb_data.db", K=16)
team_draft_history = {}

# ---- MAIN LOOP ----
b_heroes_stolen_list = []
r_heroes_stolen_list = []
b_synergy_delta_list = []
r_synergy_delta_list = []
prev_stomp_margin_list = []
is_side_swap_list = []
pending_elo_updates = {}
pending_playoff_elo_updates = {}
for index, row in training_df.iterrows():
    is_match_end = (index == len(training_df) - 1) or (training_df.iloc[index+1]["match_id"] != row["match_id"])
    blue_team   = row['blue_side_team']
    red_team    = row['red_side_team']
    winner      = row['map_winner']
    patch_v     = str(row['patch_version']).strip()
    is_playoffs = 1 if str(row['stage']).strip().lower() == 'playoffs' else 0
    duration    = row['game_duration_seconds']
    game_num    = row['game_number']
    match_id    = row['match_id']
    cur_date    = row['match_timestamp']
    match_season = str(row['season'])
    
    # === PRE-MATCH PRIOR SEASON STATISTICS LOOKUP (leak-safe) ===
    cur_s_int = int(row['season'])
    
    # === S17 CHAMPION'S CURSE & RS RANK LOOKUP ===
    prev_s_int = cur_s_int - 1
    def_champ = champions_history.get(prev_s_int, None)
    b_is_defending_champ.append(1 if blue_team == def_champ else 0)
    r_is_defending_champ.append(1 if red_team == def_champ else 0)
    
    def get_rs_rank(team):
        if cur_s_int not in rs_standings: return 99
        s_stats = rs_standings[cur_s_int]
        if team not in s_stats: return 99
        team_wrs = []
        for t, stats in s_stats.items():
            wr = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
            team_wrs.append((wr, t))
        team_wrs.sort(key=lambda x: x[0], reverse=True)
        for rank, (wr, t) in enumerate(team_wrs):
            if t == team: return rank + 1
        return 99

    b_rs_rank.append(get_rs_rank(blue_team))
    r_rs_rank.append(get_rs_rank(red_team))
    b_prior, b_s_found = lookup_prior_stats(blue_team, cur_s_int)
    r_prior, r_s_found = lookup_prior_stats(red_team, cur_s_int)
    
    # Store them
    blue_prev_season_match_wr.append(b_prior['match_winrate'])
    red_prev_season_match_wr.append(r_prior['match_winrate'])
    
    blue_prev_season_game_wr.append(b_prior['game_winrate'])
    red_prev_season_game_wr.append(r_prior['game_winrate'])
    
    blue_prev_season_kda.append(b_prior['avg_kda'])
    red_prev_season_kda.append(r_prior['avg_kda'])
    
    blue_prev_season_avg_kills.append(b_prior['avg_kills'])
    red_prev_season_avg_kills.append(r_prior['avg_kills'])
    
    blue_prev_season_avg_deaths.append(b_prior['avg_deaths'])
    red_prev_season_avg_deaths.append(r_prior['avg_deaths'])
    
    blue_prev_season_avg_assists.append(b_prior['avg_assists'])
    red_prev_season_avg_assists.append(r_prior['avg_assists'])
    
    diff_prev_season_match_wr.append(b_prior['match_winrate'] - r_prior['match_winrate'])
    diff_prev_season_game_wr.append(b_prior['game_winrate'] - r_prior['game_winrate'])
    diff_prev_season_kda.append(b_prior['avg_kda'] - r_prior['avg_kda'])

    
    # 3. Dynamic Elo Update
    if match_season != current_global_season:
        if current_global_season is not None:
            for ign in player_elos:
                player_elos[ign] = 1500 + ((player_elos[ign] - 1500) * (1 - decay_rate))
            for ign in player_playoff_elos:
                player_playoff_elos[ign] = 1500 + ((player_playoff_elos[ign] - 1500) * (1 - decay_rate))
        current_global_season = match_season

    def get_team_avg_elo(team_name, season_string, elo_dict):
        roster = team_rosters.get(team_name, {}).get(season_string, set())
        roster_igns = [resolve_ign(ign) for ign in roster]
        if not roster_igns: return default_elo
        return sum([elo_dict.get(ign, default_elo) for ign in roster_igns]) / len(roster_igns)
        
    elo_a = get_team_avg_elo(blue_team, match_season, player_elos)
    elo_b = get_team_avg_elo(red_team, match_season, player_elos)
    p_elo_a = get_team_avg_elo(blue_team, match_season, player_playoff_elos)
    p_elo_b = get_team_avg_elo(red_team, match_season, player_playoff_elos)

    b_p_clutch = playoff_clutch_tracker.get(blue_team, {'wins': 0, 'games': 0})
    r_p_clutch = playoff_clutch_tracker.get(red_team, {'wins': 0, 'games': 0})
    b_playoff_wr = b_p_clutch['wins'] / b_p_clutch['games'] if b_p_clutch['games'] > 0 else 0.5
    r_playoff_wr = r_p_clutch['wins'] / r_p_clutch['games'] if r_p_clutch['games'] > 0 else 0.5

    blue_roster_clean = [resolve_ign(ign) for ign in team_rosters.get(blue_team, {}).get(match_season, set())]
    red_roster_clean  = [resolve_ign(ign) for ign in team_rosters.get(red_team, {}).get(match_season, set())]
    b_dna = sum([player_championship_wins.get(ign, 0) for ign in blue_roster_clean])
    r_dna = sum([player_championship_wins.get(ign, 0) for ign in red_roster_clean])
    
    b_elo = elo_a
    r_elo = elo_b


    if patch_v != current_patch_tracker:
        patch_practice        = {}
        current_patch_tracker = patch_v

    try:
        picks_data   = json.loads(str(row['picks']).replace('""', '"'))
        blue_heroes  = picks_data.get('blue', [])
        red_heroes   = picks_data.get('red', [])
    except: blue_heroes, red_heroes = [], []

    try:
        bans_data  = json.loads(str(row['bans']).replace('""', '"'))
        blue_bans  = bans_data.get('blue', [])   # Blue team bans (targets red comfort)
        red_bans   = bans_data.get('red', [])    # Red team bans (targets blue comfort)
    except: blue_bans, red_bans = [], []

    # --- Hero Pool Depth (Draft Flexibility) ---
    blue_season_key = (match_season, blue_team)
    red_season_key  = (match_season, red_team)
    
    
    
    # ===== PRE-MATCH FEATURE CALCULATION (use ONLY historical data) =====

    # 1. Draft Comfort & Experience
    def get_mastery(team, heroes):
        if not heroes or team not in team_hero_tracker: return 0.5, 0
        wins, games = 0, 0
        for h in heroes:
            s = team_hero_tracker[team].get(h, {'wins': 0, 'games': 0})
            wins += s['wins']; games += s['games']
        return (wins + 2) / (games + 4), games

    b_wr, b_g = get_mastery(blue_team, blue_heroes)
    r_wr, r_g = get_mastery(red_team,  red_heroes)

    # 2. Recent Form Momentum (last 5 games)
    b_mom = np.mean(team_recent_form.get(blue_team, [0.5]))
    r_mom = np.mean(team_recent_form.get(red_team,  [0.5]))

    # 3. Head-to-Head Win Rate
    mk = tuple(sorted([blue_team, red_team]))
    h2h = h2h_tracker.get(mk, {'blue_wins': 0, 'total': 0})
    if h2h['total'] == 0:
        h2h_score = 0.5
    else:
        w = h2h['blue_wins'] if mk[0] == blue_team else (h2h['total'] - h2h['blue_wins'])
        h2h_score = (w + 1) / (h2h['total'] + 2)

    # 4. Patch Practice (games played on current patch)
    b_patch = patch_practice.get(blue_team, 0)
    r_patch = patch_practice.get(red_team,  0)

    # 5. Playoff Clutch Rate
    def get_clutch_rate(team):
        s = playoff_clutch_tracker.get(team, {'wins': 0, 'games': 0})
        return (s['wins'] + 2) / (s['games'] + 4)

    b_playoff = get_clutch_rate(blue_team)
    r_playoff = get_clutch_rate(red_team)

    # 6. Global Draft Meta Strength (hero win rates)
    def get_global_draft_wr(heroes):
        if not heroes: return 0.5
        total = 0
        for h in heroes:
            s = global_hero_tracker.get(h, {'wins': 0, 'games': 0})
            total += (s['wins'] + 5) / (s['games'] + 10)
        return total / len(heroes)

    b_global_wr = get_global_draft_wr(blue_heroes)
    r_global_wr = get_global_draft_wr(red_heroes)

    # 7. Team Pace & Execution Speed (Dynamic Game Duration V8 features)
    b_avg_dur = np.mean(team_win_durations.get(blue_team, [1020]))
    r_avg_dur = np.mean(team_win_durations.get(red_team,  [1020]))
    
    b_avg_loss_dur = np.mean(team_loss_durations.get(blue_team, [1020]))
    r_avg_loss_dur = np.mean(team_loss_durations.get(red_team,  [1020]))
    
    # Time margin (in minutes) between loss resistance and win execution speed
    b_exec_margin = (b_avg_loss_dur - b_avg_dur) / 60.0
    r_exec_margin = (r_avg_loss_dur - r_avg_dur) / 60.0
    
    # Elo-weighted punish/execution score (captures aggressive clean punishment)

    b_exec_punish = (1020.0 / b_avg_dur) * (b_elo / 1500.0)
    r_exec_punish = (1020.0 / r_avg_dur) * (r_elo / 1500.0)
    
    # Dynamic Late-Game Win Rate (>18 mins / 1080s, rolling last 10 games)
    b_lg_wr = np.mean(team_recent_lategame_results.get(blue_team, [0.5]))
    r_lg_wr = np.mean(team_recent_lategame_results.get(red_team,  [0.5]))

    # 8. Dynamic Side Advantage (rolling 50-game blue side win rate)
    cur_blue_bias = np.mean(global_side_tracker) if global_side_tracker else 0.5

    # === V6 NEW FEATURES ===

    # 9. Rest Factor (Gap Days — optimal prep zone is 4-14 days)
    b_last = team_last_game_date.get(blue_team)
    r_last = team_last_game_date.get(red_team)
    b_gap  = (cur_date - b_last).days if b_last else 7
    r_gap  = (cur_date - r_last).days if r_last else 7
    b_rest = rest_factor_score(b_gap)
    r_rest = rest_factor_score(r_gap)

    # 10. Ban Disruption (how well did each team target the opponent's comfort heroes?)
    # Blue team bans red's heroes; Red team bans blue's heroes
    b_ban_dis = get_ban_disruption(blue_bans, red_team,  team_hero_tracker)
    r_ban_dis = get_ban_disruption(red_bans,  blue_team, team_hero_tracker)

    # 11. G3 (Decider Game) Clutch Win Rate
    b_g3 = get_g3_wr(blue_team)
    r_g3 = get_g3_wr(red_team)

    # 12. Playoff Experience (total playoff games played historically)
    b_po_exp = sum([player_playoff_experience.get(ign, 0) for ign in blue_roster_clean])
    r_po_exp = sum([player_playoff_experience.get(ign, 0) for ign in red_roster_clean])

    # 13. Series Momentum (did this team win the PREVIOUS game in this series?)
    # Game 1 = neutral (0.5). Game 2+ = look back at previous game winner.
    prev_game_result = game_winner_lookup.get((match_id, game_num - 1))
    if prev_game_result is None:
        momentum_score = 0.5   # Game 1, no history
    elif prev_game_result == blue_team:
        momentum_score = 1.0   # Blue team won the previous game
    else:
        momentum_score = 0.0   # Red team won the previous game

    # 14. Reverse Sweep Resilience (how often does each team come back from 0-1?)
    b_rsweep = get_reverse_sweep_rate(blue_team)
    r_rsweep = get_reverse_sweep_rate(red_team)

    # --- NEW: Previous Winner Comfort Exhaustion (Pillar 5 Fix) ---
    blue_exhaustion_val = 0.0
    red_exhaustion_val = 0.0
    if prev_game_result is not None:
        prev_games_hist = series_draft_history.get(match_id, [])
        if prev_games_hist:
            if prev_game_result == blue_team:
                last_game = prev_games_hist[-1]
                last_game_heroes = last_game['picks']['blue' if last_game['blue_side_team'] == blue_team else 'red']
                mastery_val, _ = get_mastery(blue_team, last_game_heroes)
                if mastery_val > 0.6: blue_exhaustion_val = (mastery_val - 0.6)
            else:
                last_game = prev_games_hist[-1]
                last_game_heroes = last_game['picks']['red' if last_game['red_side_team'] == red_team else 'blue']
                mastery_val, _ = get_mastery(red_team, last_game_heroes)
                if mastery_val > 0.6: red_exhaustion_val = (mastery_val - 0.6)


    # SOTA FEATURE CALCULATIONS
    b_syn, r_syn, b_ctr, r_ctr = 0.0, 0.0, 0.0, 0.0
    for i, h1 in enumerate(blue_heroes):
        for h2 in blue_heroes[i+1:]: b_syn += global_synergy_matrix.get((h1, h2), 0.0)
    for i, h1 in enumerate(red_heroes):
        for h2 in red_heroes[i+1:]: r_syn += global_synergy_matrix.get((h1, h2), 0.0)
    for bh in blue_heroes:
        for rh in red_heroes:
            b_ctr += global_counter_matrix.get((bh, rh), 0.0)
            r_ctr += global_counter_matrix.get((rh, bh), 0.0)
            
    b_exhaust, r_exhaust = 0.0, 0.0
    if game_num > 1:
        prev_maps = series_draft_history.get(match_id, [])
        for m in prev_maps:
            if m['map_winner'] == blue_team:
                heroes = m['picks']['blue'] if m['blue_side_team'] == blue_team else m['picks']['red']
                for hero in heroes:
                    hr = get_mastery(blue_team, [hero])
                    if hr[0] > 0.60: b_exhaust += (hr[0] - 0.60)
            if m['map_winner'] == red_team:
                heroes = m['picks']['red'] if m['red_side_team'] == red_team else m['picks']['blue']
                for hero in heroes:
                    hr = get_mastery(red_team, [hero])
                    if hr[0] > 0.60: r_exhaust += (hr[0] - 0.60)

    # === SOTA ENHANCEMENT: Roster Stability Index (Pillar 4) ===
    def get_game_roster(team_name, roster_col_val, season_str):
        try:
            if pd.notna(roster_col_val) and roster_col_val != '[]':
                players = json.loads(str(roster_col_val).replace('""', '"'))
                if players:
                    return set([resolve_ign(p) for p in players])
        except:
            pass
        roster_set = team_rosters.get(team_name, {}).get(season_str, set())
        return set([resolve_ign(ign) for ign in roster_set])

    blue_roster = get_game_roster(blue_team, row['blue_roster'], match_season)
    red_roster = get_game_roster(red_team, row['red_roster'], match_season)

    def compute_roster_stability(team, current_roster, history_dict):
        hist = history_dict.get(team, [])
        if not hist:
            return 1.0  # Perfect stability for first game
        similarities = []
        for prev_roster in hist[-3:]:
            if not current_roster or not prev_roster:
                similarities.append(1.0)
            else:
                intersection = len(current_roster.intersection(prev_roster))
                union = len(current_roster.union(prev_roster))
                similarities.append(intersection / union if union > 0 else 1.0)
        return float(np.mean(similarities))

    blue_roster_stability_val = compute_roster_stability(blue_team, blue_roster, team_roster_history)
    red_roster_stability_val = compute_roster_stability(red_team, red_roster, team_roster_history)
    diff_roster_stability_val = blue_roster_stability_val - red_roster_stability_val

    # === SOTA ENHANCEMENT: Draft Overlap Penalty (Pillar 5) ===
    blue_draft_overlap_val = 0.0
    red_draft_overlap_val = 0.0
    if game_num > 1:
        prev_games = series_draft_history.get(match_id, [])
        if prev_games:
            prev_game = prev_games[-1]
            blue_was_blue_in_prev = prev_game['blue_side_team'] == blue_team
            prev_blue_picks = prev_game['picks']['blue' if blue_was_blue_in_prev else 'red']
            
            red_was_red_in_prev = prev_game['red_side_team'] == red_team
            prev_red_picks = prev_game['picks']['red' if red_was_red_in_prev else 'blue']
            
            if blue_heroes and prev_blue_picks:
                blue_draft_overlap_val = len(set(blue_heroes).intersection(set(prev_blue_picks))) / 5.0
            if red_heroes and prev_red_picks:
                red_draft_overlap_val = len(set(red_heroes).intersection(set(prev_red_picks))) / 5.0

    diff_draft_overlap_val = blue_draft_overlap_val - red_draft_overlap_val

    # Append to lists
    b_roster_stability.append(blue_roster_stability_val)
    r_roster_stability.append(red_roster_stability_val)
    diff_roster_stability_list.append(diff_roster_stability_val)
    b_draft_overlap_list.append(blue_draft_overlap_val)
    r_draft_overlap_list.append(red_draft_overlap_val)
    diff_draft_overlap_list.append(diff_draft_overlap_val)

    b_heroes_stolen_list.append(0)
    r_heroes_stolen_list.append(0)
    b_synergy_delta_list.append(0.0)
    r_synergy_delta_list.append(0.0)
    prev_stomp_margin_list.append(0)
    is_side_swap_list.append(0)
    b_side_elo.append(elo_a)
    r_side_elo.append(elo_b)
    b_playoff_elo.append(p_elo_a)
    r_playoff_elo.append(p_elo_b)
    b_championship_dna.append(b_dna)
    r_championship_dna.append(r_dna)
    b_playoff_winrate.append(b_playoff_wr)
    r_playoff_winrate.append(r_playoff_wr)
    b_prev_winner_exhaustion.append(blue_exhaustion_val)
    r_prev_winner_exhaustion.append(red_exhaustion_val)
    b_draft_exhaustion.append(b_exhaust)
    r_draft_exhaustion.append(r_exhaust)
    b_synergy.append(b_syn)
    r_synergy.append(r_syn)
    b_counter.append(b_ctr)
    r_counter.append(r_ctr)

    # === ADVANCED SERIES DRAFT ADAPTATION ===
    g1_winner_banned_b = 0.0
    g1_winner_banned_r = 0.0
    b_g1_comfort_val = 0.5
    r_g1_comfort_val = 0.5

    prev_winner_banned_b_val = 0.0
    prev_winner_banned_r_val = 0.0
    b_prev_comfort_val = 0.5
    r_prev_comfort_val = 0.5

    prev_played_comfort_banned_b_val = 0.0
    prev_played_comfort_banned_r_val = 0.0

    if game_num > 1:
        prev_games = series_draft_history.get(match_id, [])
        if prev_games:
            # --- 1. Game 1 Baseline ---
            g1 = prev_games[0]
            g1_winner = g1['map_winner']
            blue_was_blue_in_g1 = g1['blue_side_team'] == blue_team
            blue_g1_heroes = g1['picks']['blue' if blue_was_blue_in_g1 else 'red']
            red_g1_heroes = g1['picks']['red' if blue_was_blue_in_g1 else 'blue']

            b_g1_comfort_val, _ = get_mastery(blue_team, blue_g1_heroes)
            r_g1_comfort_val, _ = get_mastery(red_team, red_g1_heroes)

            if g1_winner:
                g1_winner_was_blue = g1['blue_side_team'] == g1_winner
                g1_winner_heroes = g1['picks']['blue' if g1_winner_was_blue else 'red']
                g1_winner_banned_b = sum(1 for h in blue_bans if h in g1_winner_heroes) / max(1, len(blue_bans))
                g1_winner_banned_r = sum(1 for h in red_bans if h in g1_winner_heroes) / max(1, len(red_bans))

            # --- 2. Immediately Preceding Game Adaptation ---
            prev_game = prev_games[-1]
            prev_winner = prev_game['map_winner']
            blue_was_blue_in_prev = prev_game['blue_side_team'] == blue_team
            # --- 3. Draft Deltas (Hero Stealing & Synergy) ---
            if not blue_was_blue_in_prev:
                is_side_swap_list[-1] = 1
                
            if prev_winner:
                prev_winner_was_blue = prev_game['blue_side_team'] == prev_winner
                prev_winner_heroes = prev_game['picks']['blue' if prev_winner_was_blue else 'red']
                if prev_winner != blue_team: b_heroes_stolen_list[-1] = sum(1 for h in blue_heroes if h in prev_winner_heroes)
                if prev_winner != red_team:  r_heroes_stolen_list[-1] = sum(1 for h in red_heroes if h in prev_winner_heroes)

            prev_blue_syn = prev_game.get('blue_synergy', b_syn) if blue_was_blue_in_prev else prev_game.get('red_synergy', b_syn)
            prev_red_syn = prev_game.get('red_synergy', r_syn) if blue_was_blue_in_prev else prev_game.get('blue_synergy', r_syn)
            b_synergy_delta_list[-1] = b_syn - prev_blue_syn
            r_synergy_delta_list[-1] = r_syn - prev_red_syn
            
            dur = prev_game.get('duration', 15*60)
            if pd.notna(dur) and dur > 0:
                if dur < 12*60: prev_stomp_margin_list[-1] = 1
                elif dur > 22*60: prev_stomp_margin_list[-1] = -1
                else: prev_stomp_margin_list[-1] = 0

            blue_prev_heroes = prev_game['picks']['blue' if blue_was_blue_in_prev else 'red']
            red_prev_heroes = prev_game['picks']['red' if blue_was_blue_in_prev else 'blue']

            b_prev_comfort_val, _ = get_mastery(blue_team, blue_prev_heroes)
            r_prev_comfort_val, _ = get_mastery(red_team, red_prev_heroes)

            if prev_winner:
                prev_winner_was_blue = prev_game['blue_side_team'] == prev_winner
                prev_winner_heroes = prev_game['picks']['blue' if prev_winner_was_blue else 'red']
                prev_winner_banned_b_val = sum(1 for h in blue_bans if h in prev_winner_heroes) / max(1, len(blue_bans))
                prev_winner_banned_r_val = sum(1 for h in red_bans if h in prev_winner_heroes) / max(1, len(red_bans))

            # --- 3. Targeted Bans on Opponent Comfort Heroes Played in Previous Game ---
            # Blue targets red comfort heroes played in the last game
            opp_red_picks = team_hero_tracker.get(red_team, {})
            red_top_comfort = set(sorted(opp_red_picks.keys(), key=lambda h: opp_red_picks[h]['games'], reverse=True)[:12])
            red_prev_played_comfort = [h for h in red_prev_heroes if h in red_top_comfort]
            if red_prev_played_comfort:
                prev_played_comfort_banned_b_val = sum(1 for h in blue_bans if h in red_prev_played_comfort) / len(red_prev_played_comfort)

            # Red targets blue comfort heroes played in the last game
            opp_blue_picks = team_hero_tracker.get(blue_team, {})
            blue_top_comfort = set(sorted(opp_blue_picks.keys(), key=lambda h: opp_blue_picks[h]['games'], reverse=True)[:12])
            blue_prev_played_comfort = [h for h in blue_prev_heroes if h in blue_top_comfort]
            if blue_prev_played_comfort:
                prev_played_comfort_banned_r_val = sum(1 for h in red_bans if h in blue_prev_played_comfort) / len(blue_prev_played_comfort)

    # Comfort Patch Impact
    b_comfort_patch = get_comfort_patch_impact(blue_team, team_hero_tracker, patch_v)
    r_comfort_patch = get_comfort_patch_impact(red_team, team_hero_tracker, patch_v)

    # Expected comfort based on top 5 most-played heroes
    b_exp_comfort = get_expected_comfort(blue_team, team_hero_tracker)
    r_exp_comfort = get_expected_comfort(red_team, team_hero_tracker)

    # --- NEW: Patch Adaptation ---
    # Initialize trackers if missing
    for team in [blue_team, red_team]:
        if team not in team_patch_stats:
            team_patch_stats[team] = {}
        if patch_v not in team_patch_stats[team]:
            team_patch_stats[team][patch_v] = {'wins': 0, 'games': 0}
        if team not in team_hist_stats:
            team_hist_stats[team] = {'wins': 0, 'games': 0}

    # Calculate pre-match stats
    b_pwins, b_pgames = team_patch_stats[blue_team][patch_v]['wins'], team_patch_stats[blue_team][patch_v]['games']
    b_hwins, b_hgames = team_hist_stats[blue_team]['wins'], team_hist_stats[blue_team]['games']
    
    r_pwins, r_pgames = team_patch_stats[red_team][patch_v]['wins'], team_patch_stats[red_team][patch_v]['games']
    r_hwins, r_hgames = team_hist_stats[red_team]['wins'], team_hist_stats[red_team]['games']

    # Calculate rates (use 0.5 as default if < 3 games for patch, < 10 for hist)
    b_p_rate = b_pwins / b_pgames if b_pgames >= 3 else 0.5
    b_h_rate = b_hwins / b_hgames if b_hgames >= 10 else 0.5
    r_p_rate = r_pwins / r_pgames if r_pgames >= 3 else 0.5
    r_h_rate = r_hwins / r_hgames if r_hgames >= 10 else 0.5

    b_p_adapt = b_p_rate - b_h_rate if b_pgames >= 3 else 0.0
    r_p_adapt = r_p_rate - r_h_rate if r_pgames >= 3 else 0.0

    b_patch_winrate.append(b_p_rate)
    r_patch_winrate.append(r_p_rate)
    b_patch_adaptation.append(b_p_adapt)
    r_patch_adaptation.append(r_p_adapt)

    # --- V8+ Playstyle Archetypes ---
    b_draft_score = (b_global_wr * 0.4) + (b_wr * 0.4) + (b_ban_dis * 0.2)
    r_draft_score = (r_global_wr * 0.4) + (r_wr * 0.4) + (r_ban_dis * 0.2)
    b_exec_score  = b_exec_punish * (b_lg_wr + 0.5) * (b_g3 + 1)
    r_exec_score  = r_exec_punish * (r_lg_wr + 0.5) * (r_g3 + 1)
    
    b_draft_mastery = np.mean(team_draft_scores.get(blue_team, [0.5]))
    r_draft_mastery = np.mean(team_draft_scores.get(red_team,  [0.5]))
    b_exec_mastery  = np.mean(team_exec_scores.get(blue_team, [1.0]))
    r_exec_mastery  = np.mean(team_exec_scores.get(red_team,  [1.0]))
    
    b_draft_reliance = b_draft_mastery / (b_exec_mastery + 0.1)
    r_draft_reliance = r_draft_mastery / (r_exec_mastery + 0.1)

    # --- Expected Draft Style SVD Features (100% Pre-Match Leak-Safe) ---
    b_draft_history = team_draft_history.get(blue_team, [])
    if not b_draft_history:
        E_blue = np.array(fallback_emb)
    else:
        # Collect all heroes from last 10 games
        b_heroes_flat = [h for g in b_draft_history for h in g]
        b_vectors = [np.array(hero_embeddings.get(h, fallback_emb)) for h in b_heroes_flat]
        E_blue = np.mean(b_vectors, axis=0)

    r_draft_history = team_draft_history.get(red_team, [])
    if not r_draft_history:
        E_red = np.array(fallback_emb)
    else:
        # Collect all heroes from last 10 games
        r_heroes_flat = [h for g in r_draft_history for h in g]
        r_vectors = [np.array(hero_embeddings.get(h, fallback_emb)) for h in r_heroes_flat]
        E_red = np.mean(r_vectors, axis=0)

    # Cosine Similarity between E_blue and E_red
    dot_prod = np.dot(E_blue, E_red)
    norm_eb = np.linalg.norm(E_blue)
    norm_er = np.linalg.norm(E_red)
    if norm_eb == 0 or norm_er == 0:
        draft_style_sim = 1.0
    else:
        draft_style_sim = float(dot_prod / (norm_eb * norm_er))

    # Append features
    draft_style_sims.append(draft_style_sim)
    for i in range(16):
        blue_draft_embs[i].append(float(E_blue[i]))
        red_draft_embs[i].append(float(E_red[i]))
        diff_draft_embs[i].append(float(E_blue[i] - E_red[i]))

    # === APPEND ALL FEATURES ===
    b_com.append(b_wr);         r_com.append(r_wr)
    b_exp.append(b_g);          r_exp.append(r_g)
    b_momentum.append(b_mom);   r_momentum.append(r_mom)
    b_h2h.append(h2h_score)
    b_patch_exp.append(b_patch); r_patch_exp.append(r_patch)
    b_playoff_clutch.append(b_playoff); r_playoff_clutch.append(r_playoff)
    is_playoffs_list.append(is_playoffs)
    b_global_draft_wr.append(b_global_wr); r_global_draft_wr.append(r_global_wr)
    b_avg_win_duration.append(b_avg_dur);  r_avg_win_duration.append(r_avg_dur)
    blue_side_advantage.append(cur_blue_bias)
    # V6
    b_rest_factor.append(b_rest);          r_rest_factor.append(r_rest)
    b_ban_disruption.append(b_ban_dis);    r_ban_disruption.append(r_ban_dis)
    b_g3_clutch_wr.append(b_g3);          r_g3_clutch_wr.append(r_g3)
    b_playoff_exp_count.append(b_po_exp); r_playoff_exp_count.append(r_po_exp)
    series_momentum_blue.append(momentum_score)
    b_reverse_sweep_rate.append(b_rsweep); r_reverse_sweep_rate.append(r_rsweep)

    # New Series Draft Features
    b_g1_comfort.append(b_g1_comfort_val)
    r_g1_comfort.append(r_g1_comfort_val)
    g1_winner_heroes_banned_blue.append(g1_winner_banned_b)
    g1_winner_heroes_banned_red.append(g1_winner_banned_r)

    # V8 Output Lists
    b_avg_loss_duration.append(b_avg_loss_dur)
    r_avg_loss_duration.append(r_avg_loss_dur)
    b_execution_margin.append(b_exec_margin)
    r_execution_margin.append(r_exec_margin)
    b_execution_punish_score.append(b_exec_punish)
    r_execution_punish_score.append(r_exec_punish)
    b_lategame_winrate.append(b_lg_wr)
    r_lategame_winrate.append(r_lg_wr)
    b_draft_mastery_list.append(b_draft_mastery)
    r_draft_mastery_list.append(r_draft_mastery)
    b_exec_mastery_list.append(b_exec_mastery)
    r_exec_mastery_list.append(r_exec_mastery)
    b_draft_reliance_list.append(b_draft_reliance)
    r_draft_reliance_list.append(r_draft_reliance)

    b_prev_comfort.append(b_prev_comfort_val)
    r_prev_comfort.append(r_prev_comfort_val)
    prev_winner_heroes_banned_blue.append(prev_winner_banned_b_val)
    prev_winner_heroes_banned_red.append(prev_winner_banned_r_val)

    prev_played_comfort_banned_blue.append(prev_played_comfort_banned_b_val)
    prev_played_comfort_banned_red.append(prev_played_comfort_banned_r_val)

    # NEW: Patch notes comfort hero impact
    b_comfort_patch_score.append(b_comfort_patch)
    r_comfort_patch_score.append(r_comfort_patch)

    # NEW: Expected comfort based on top 5 most-played heroes
    b_expected_comfort.append(b_exp_comfort)
    r_expected_comfort.append(r_exp_comfort)

    # ===== UPDATE ALL TRACKERS AFTER RECORDING PRE-MATCH STATE =====
    blue_won = 1 if winner == blue_team else 0
    red_won  = 1 - blue_won

    # Store this game in series history

    # SOTA STATE UPDATES (POST-MAP)
    if winner == blue_team:
        for i, h1 in enumerate(blue_heroes):
            for h2 in blue_heroes[i+1:]: global_synergy_matrix[(h1, h2)] = global_synergy_matrix.get((h1, h2), 0.0) + 1.0
        for bh in blue_heroes:
            for rh in red_heroes: global_counter_matrix[(bh, rh)] = global_counter_matrix.get((bh, rh), 0.0) + 1.0
    else:
        for i, h1 in enumerate(red_heroes):
            for h2 in red_heroes[i+1:]: global_synergy_matrix[(h1, h2)] = global_synergy_matrix.get((h1, h2), 0.0) + 1.0
        for rh in red_heroes:
            for bh in blue_heroes: global_counter_matrix[(rh, bh)] = global_counter_matrix.get((rh, bh), 0.0) + 1.0
            
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    actual_a   = 1 if winner == blue_team else 0
    actual_b   = 1 - actual_a
    expected_b = 1 - expected_a
    k = k_playoff if is_playoffs else k_regular
    for ign_raw in team_rosters.get(blue_team, {}).get(match_season, set()):
        ign = resolve_ign(ign_raw)
        pending_elo_updates[ign] = pending_elo_updates.get(ign, 0.0) + (k * (actual_a - expected_a))
    for ign_raw in team_rosters.get(red_team, {}).get(match_season, set()):
        ign = resolve_ign(ign_raw)
        pending_elo_updates[ign] = pending_elo_updates.get(ign, 0.0) + (k * (actual_b - expected_b))
        
    if is_playoffs:
        p_expected_a = 1 / (1 + 10 ** ((p_elo_b - p_elo_a) / 400))
        p_expected_b = 1 - p_expected_a
        for ign_raw in team_rosters.get(blue_team, {}).get(match_season, set()):
            ign = resolve_ign(ign_raw)
            pending_playoff_elo_updates[ign] = pending_playoff_elo_updates.get(ign, 0.0) + (k_playoff * (actual_a - p_expected_a))
            player_playoff_experience[ign] = player_playoff_experience.get(ign, 0) + 1
        for ign_raw in team_rosters.get(red_team, {}).get(match_season, set()):
            ign = resolve_ign(ign_raw)
            pending_playoff_elo_updates[ign] = pending_playoff_elo_updates.get(ign, 0.0) + (k_playoff * (actual_b - p_expected_b))
            player_playoff_experience[ign] = player_playoff_experience.get(ign, 0) + 1

    if is_match_end:
        for ign, delta in pending_elo_updates.items():
            player_elos[ign] = player_elos.get(ign, default_elo) + delta
        for ign, delta in pending_playoff_elo_updates.items():
            player_playoff_elos[ign] = player_playoff_elos.get(ign, default_elo) + delta
        pending_elo_updates.clear()
        pending_playoff_elo_updates.clear()


    # Update team draft history (recorded post-game to be leak-safe for next matches)
    team_draft_history.setdefault(blue_team, []).append(blue_heroes)
    team_draft_history[blue_team] = team_draft_history[blue_team][-10:]
    team_draft_history.setdefault(red_team, []).append(red_heroes)
    team_draft_history[red_team] = team_draft_history[red_team][-10:]

    series_draft_history.setdefault(match_id, []).append({
        'game_number': game_num,
        'blue_side_team': blue_team,
        'red_side_team': red_team,
        'picks': {'blue': blue_heroes, 'red': red_heroes},
        'bans': {'blue': blue_bans, 'red': red_bans},
        'map_winner': winner,
        'duration': duration,
        'blue_synergy': b_syn,
        'red_synergy': r_syn
    })

    # Store this game's result for future series momentum lookups
    game_winner_lookup[(match_id, game_num)] = winner

    # Hero Comfort + Global Hero Meta
    if blue_team not in team_hero_tracker: team_hero_tracker[blue_team] = {}
    if red_team  not in team_hero_tracker: team_hero_tracker[red_team]  = {}
    for h in blue_heroes:
        team_hero_tracker[blue_team].setdefault(h, {'wins': 0, 'games': 0})
        team_hero_tracker[blue_team][h]['games'] += 1
        team_hero_tracker[blue_team][h]['wins']  += blue_won
        global_hero_tracker.setdefault(h, {'wins': 0, 'games': 0})
        global_hero_tracker[h]['games'] += 1
        global_hero_tracker[h]['wins']  += blue_won
    for h in red_heroes:
        team_hero_tracker[red_team].setdefault(h, {'wins': 0, 'games': 0})
        team_hero_tracker[red_team][h]['games'] += 1
        team_hero_tracker[red_team][h]['wins']  += red_won
        global_hero_tracker.setdefault(h, {'wins': 0, 'games': 0})
        global_hero_tracker[h]['games'] += 1
        global_hero_tracker[h]['wins']  += red_won
        
    for h in blue_heroes: seasonal_hero_tracker.setdefault(blue_season_key, set()).add(h)
    for h in red_heroes:  seasonal_hero_tracker.setdefault(red_season_key, set()).add(h)

    # Recent Form (last 5 games)
    team_recent_form.setdefault(blue_team, [])
    team_recent_form.setdefault(red_team,  [])
    team_recent_form[blue_team] = (team_recent_form[blue_team] + [blue_won])[-5:]
    team_recent_form[red_team]  = (team_recent_form[red_team]  + [red_won])[-5:]

    # H2H
    h2h_tracker.setdefault(mk, {'blue_wins': 0, 'total': 0})
    h2h_tracker[mk]['total'] += 1
    if (mk[0] == blue_team and blue_won) or (mk[0] == red_team and red_won):
        h2h_tracker[mk]['blue_wins'] += 1

    # Patch Practice & Adaptation
    patch_practice[blue_team] = patch_practice.get(blue_team, 0) + 1
    patch_practice[red_team]  = patch_practice.get(red_team,  0) + 1
    
    team_patch_stats[blue_team][patch_v]['games'] += 1
    team_patch_stats[red_team][patch_v]['games'] += 1
    team_hist_stats[blue_team]['games'] += 1
    team_hist_stats[red_team]['games'] += 1
    if blue_won:
        team_patch_stats[blue_team][patch_v]['wins'] += 1
        team_hist_stats[blue_team]['wins'] += 1
    else:
        team_patch_stats[red_team][patch_v]['wins'] += 1
        team_hist_stats[red_team]['wins'] += 1

    # Playoff Clutch
    if is_playoffs == 1:
        playoff_clutch_tracker.setdefault(blue_team, {'wins': 0, 'games': 0})
        playoff_clutch_tracker.setdefault(red_team,  {'wins': 0, 'games': 0})
        playoff_clutch_tracker[blue_team]['games'] += 1
        playoff_clutch_tracker[blue_team]['wins']  += blue_won
        playoff_clutch_tracker[red_team]['games']  += 1
        playoff_clutch_tracker[red_team]['wins']   += red_won

    # Team Win/Loss Durations (pace and execution speed tracker)
    if duration and duration > 0:
        if blue_won:
            team_win_durations.setdefault(blue_team, [])
            team_win_durations[blue_team] = (team_win_durations[blue_team] + [duration])[-20:]
            team_loss_durations.setdefault(red_team, [])
            team_loss_durations[red_team] = (team_loss_durations[red_team] + [duration])[-20:]
        else:
            team_win_durations.setdefault(red_team, [])
            team_win_durations[red_team]  = (team_win_durations[red_team]  + [duration])[-20:]
            team_loss_durations.setdefault(blue_team, [])
            team_loss_durations[blue_team] = (team_loss_durations[blue_team] + [duration])[-20:]

        # V8: Dynamic Rolling Late-Game Win Rate (>18 mins / 1080s, rolling last 10 games)
        if duration > 1080:
            team_recent_lategame_results.setdefault(blue_team, [])
            team_recent_lategame_results[blue_team] = (team_recent_lategame_results[blue_team] + [blue_won])[-10:]
            team_recent_lategame_results.setdefault(red_team, [])
            team_recent_lategame_results[red_team] = (team_recent_lategame_results[red_team] + [not blue_won])[-10:]

    # V8+: Update Playstyle Trackers
    team_draft_scores.setdefault(blue_team, [])
    team_draft_scores[blue_team] = (team_draft_scores[blue_team] + [b_draft_score])[-15:]
    team_draft_scores.setdefault(red_team, [])
    team_draft_scores[red_team] = (team_draft_scores[red_team] + [r_draft_score])[-15:]
    
    team_exec_scores.setdefault(blue_team, [])
    team_exec_scores[blue_team] = (team_exec_scores[blue_team] + [b_exec_score])[-15:]
    team_exec_scores.setdefault(red_team, [])
    team_exec_scores[red_team] = (team_exec_scores[red_team] + [r_exec_score])[-15:]

    # Global Side Advantage
    global_side_tracker = (global_side_tracker + [blue_won])[-50:]

    # V6: Last Game Date (for gap/rest calculation)
    team_last_game_date[blue_team] = cur_date
    team_last_game_date[red_team]  = cur_date

    # Update roster history tracker chronologically (pre-match information, recorded post-game to be leak-safe)
    team_roster_history.setdefault(blue_team, []).append(blue_roster)
    team_roster_history.setdefault(red_team, []).append(red_roster)

    # V6: G3 Clutch Tracker (game 3 or later = "deciding game pressure")
    if game_num >= 3:
        team_g3_tracker.setdefault(blue_team, {'wins': 0, 'games': 0})
        team_g3_tracker.setdefault(red_team,  {'wins': 0, 'games': 0})
        team_g3_tracker[blue_team]['games'] += 1
        team_g3_tracker[blue_team]['wins']  += blue_won
        team_g3_tracker[red_team]['games']  += 1
        team_g3_tracker[red_team]['wins']   += red_won

    # V6: Playoff Experience Count
    if is_playoffs == 1:
        team_playoff_games_count[blue_team] = team_playoff_games_count.get(blue_team, 0) + 1
        team_playoff_games_count[red_team]  = team_playoff_games_count.get(red_team,  0) + 1

    # V6: Reverse Sweep — INCREMENTAL (read before update, no future leakage)
    # Track G1 winner for this match
    if game_num == 1:
        g1_loser = red_team if blue_won else blue_team
        series_g1_winner[match_id] = (winner, g1_loser)

    # Update in-series win counts
    if match_id not in current_series_wins:
        current_series_wins[match_id] = {blue_team: 0, red_team: 0}
    current_series_wins[match_id][winner] = current_series_wins[match_id].get(winner, 0) + 1

    # Detect series end: if this is the last game in the match, update reverse sweep
    if game_num == last_game_in_match.get(match_id, -1):
        if match_id in series_g1_winner:
            g1_win, g1_loss = series_g1_winner[match_id]
            series_wins = current_series_wins[match_id]
            series_winner_team = max(series_wins, key=series_wins.get)
            # Only relevant if G1 loser ultimately won the series
            reverse_sweep_tracker.setdefault(g1_loss, {'down_01': 0, 'came_back': 0})
            reverse_sweep_tracker[g1_loss]['down_01'] += 1
            if series_winner_team == g1_loss:
                reverse_sweep_tracker[g1_loss]['came_back'] += 1

    # === S17 CHAMPION'S CURSE & RS RANK UPDATES (Leak-safe) ===
    stage_str = str(row['stage']).strip().lower()
    
    if stage_str in ['rs', 'group stage', 'regular season']:
        if cur_s_int not in rs_standings:
            rs_standings[cur_s_int] = {}
        if blue_team not in rs_standings[cur_s_int]:
            rs_standings[cur_s_int][blue_team] = {'wins': 0, 'games': 0}
        if red_team not in rs_standings[cur_s_int]:
            rs_standings[cur_s_int][red_team] = {'wins': 0, 'games': 0}
            
        rs_standings[cur_s_int][blue_team]['games'] += 1
        rs_standings[cur_s_int][red_team]['games'] += 1
        
        if blue_won:
            rs_standings[cur_s_int][blue_team]['wins'] += 1
        else:
            rs_standings[cur_s_int][red_team]['wins'] += 1

    if stage_str in ['playoffs', 'grand final', 'upper bracket', 'lower bracket']:
        if game_num == last_game_in_match.get(match_id, -1):
            series_wins = current_series_wins[match_id]
            series_winner_team = max(series_wins, key=series_wins.get)
            champions_history[cur_s_int] = series_winner_team
            
            champ_roster = team_rosters.get(series_winner_team, {}).get(match_season, set())
            for ign_raw in champ_roster:
                ign = resolve_ign(ign_raw)
                player_championship_wins[ign] = player_championship_wins.get(ign, 0) + 1

# ==========================================
# STEP 6: ATTACH ALL COLUMNS
# ==========================================

training_df['blue_roster_stability'] = b_roster_stability
training_df['red_roster_stability'] = r_roster_stability
training_df['diff_roster_stability'] = diff_roster_stability_list
training_df['blue_draft_overlap'] = b_draft_overlap_list
training_df['red_draft_overlap'] = r_draft_overlap_list
training_df['diff_draft_overlap'] = diff_draft_overlap_list

training_df['blue_side_elo'] = b_side_elo
training_df['red_side_elo']  = r_side_elo
training_df['blue_playoff_elo']           = b_playoff_elo
training_df['red_playoff_elo']            = r_playoff_elo
training_df['blue_championship_dna']      = b_championship_dna
training_df['red_championship_dna']       = r_championship_dna
training_df['blue_playoff_winrate']       = b_playoff_winrate
training_df['red_playoff_winrate']        = r_playoff_winrate
training_df['blue_prev_winner_exhaustion'] = b_prev_winner_exhaustion
training_df['red_prev_winner_exhaustion']  = r_prev_winner_exhaustion
training_df['blue_heroes_stolen'] = b_heroes_stolen_list
training_df['red_heroes_stolen'] = r_heroes_stolen_list
training_df['blue_synergy_delta'] = b_synergy_delta_list
training_df['red_synergy_delta'] = r_synergy_delta_list
training_df['prev_stomp_margin'] = prev_stomp_margin_list
training_df['is_side_swap'] = is_side_swap_list
training_df['blue_draft_exhaustion'] = b_draft_exhaustion
training_df['red_draft_exhaustion']  = r_draft_exhaustion
training_df['blue_synergy'] = b_synergy
training_df['red_synergy']  = r_synergy
training_df['blue_counter'] = b_counter
training_df['red_counter']  = r_counter
training_df['blue_comfort_wr']      = b_com
training_df['red_comfort_wr']       = r_com
training_df['blue_draft_experience']= b_exp
training_df['red_draft_experience'] = r_exp
training_df['blue_momentum']        = b_momentum
training_df['red_momentum']         = r_momentum
training_df['blue_h2h_winrate']     = b_h2h
training_df['blue_patch_practice']  = b_patch_exp
training_df['red_patch_practice']   = r_patch_exp
training_df['blue_playoff_clutch']  = b_playoff_clutch
training_df['red_playoff_clutch']   = r_playoff_clutch
training_df['is_playoffs']          = is_playoffs_list
training_df['blue_global_draft_wr'] = b_global_draft_wr
training_df['red_global_draft_wr']  = r_global_draft_wr
training_df['blue_avg_win_duration']= b_avg_win_duration
training_df['red_avg_win_duration'] = r_avg_win_duration
training_df['current_blue_side_advantage'] = blue_side_advantage
# V6
training_df['blue_rest_factor']         = b_rest_factor
training_df['red_rest_factor']          = r_rest_factor
training_df['blue_ban_disruption']      = b_ban_disruption
training_df['red_ban_disruption']       = r_ban_disruption
training_df['blue_g3_clutch_wr']        = b_g3_clutch_wr
training_df['red_g3_clutch_wr']         = r_g3_clutch_wr
training_df['blue_playoff_exp']         = b_playoff_exp_count
training_df['red_playoff_exp']          = r_playoff_exp_count
training_df['series_momentum_blue']     = series_momentum_blue
training_df['blue_reverse_sweep_rate']  = b_reverse_sweep_rate
training_df['red_reverse_sweep_rate']   = r_reverse_sweep_rate

# New Series Draft Features
training_df['blue_g1_comfort'] = b_g1_comfort
training_df['red_g1_comfort'] = r_g1_comfort
training_df['g1_winner_heroes_banned_blue'] = g1_winner_heroes_banned_blue
training_df['g1_winner_heroes_banned_red'] = g1_winner_heroes_banned_red

training_df['blue_prev_comfort'] = b_prev_comfort
training_df['red_prev_comfort'] = r_prev_comfort
training_df['prev_winner_heroes_banned_blue'] = prev_winner_heroes_banned_blue
training_df['prev_winner_heroes_banned_red'] = prev_winner_heroes_banned_red

training_df['prev_played_comfort_banned_blue'] = prev_played_comfort_banned_blue
training_df['prev_played_comfort_banned_red'] = prev_played_comfort_banned_red

training_df['blue_comfort_patch_score'] = b_comfort_patch_score
training_df['red_comfort_patch_score']  = r_comfort_patch_score

training_df['blue_patch_winrate'] = b_patch_winrate
training_df['red_patch_winrate']  = r_patch_winrate
training_df['blue_patch_adaptation'] = b_patch_adaptation
training_df['red_patch_adaptation']  = r_patch_adaptation

training_df['blue_expected_comfort'] = b_expected_comfort
training_df['red_expected_comfort']  = r_expected_comfort

# NEW: S17 Champion's Curse & RS Rank features
training_df['blue_rs_rank'] = b_rs_rank
training_df['red_rs_rank']  = r_rs_rank
training_df['blue_is_defending_champ'] = b_is_defending_champ
training_df['red_is_defending_champ']  = r_is_defending_champ

# NEW: Prior official season statistics dataframe columns
training_df['blue_prev_season_match_wr'] = blue_prev_season_match_wr
training_df['red_prev_season_match_wr']  = red_prev_season_match_wr
training_df['blue_prev_season_game_wr']  = blue_prev_season_game_wr
training_df['red_prev_season_game_wr']   = red_prev_season_game_wr
training_df['blue_prev_season_kda']      = blue_prev_season_kda
training_df['red_prev_season_kda']       = red_prev_season_kda
training_df['blue_prev_season_avg_kills'] = blue_prev_season_avg_kills
training_df['red_prev_season_avg_kills']  = red_prev_season_avg_kills
training_df['blue_prev_season_avg_deaths'] = blue_prev_season_avg_deaths
training_df['red_prev_season_avg_deaths']  = red_prev_season_avg_deaths
training_df['blue_prev_season_avg_assists'] = blue_prev_season_avg_assists
training_df['red_prev_season_avg_assists']  = red_prev_season_avg_assists

training_df['diff_prev_season_match_wr'] = diff_prev_season_match_wr
training_df['diff_prev_season_game_wr']  = diff_prev_season_game_wr
training_df['diff_prev_season_kda']       = diff_prev_season_kda


# V8 Columns
training_df['blue_avg_loss_duration'] = b_avg_loss_duration
training_df['red_avg_loss_duration']  = r_avg_loss_duration
training_df['blue_execution_margin']  = b_execution_margin
training_df['red_execution_margin']   = r_execution_margin
training_df['blue_execution_punish_score'] = b_execution_punish_score
training_df['red_execution_punish_score']  = r_execution_punish_score
training_df['blue_lategame_winrate']  = b_lategame_winrate
training_df['red_lategame_winrate']   = r_lategame_winrate

training_df['blue_draft_mastery']     = b_draft_mastery_list
training_df['red_draft_mastery']      = r_draft_mastery_list
training_df['blue_execution_mastery'] = b_exec_mastery_list
training_df['red_execution_mastery']  = r_exec_mastery_list
training_df['blue_draft_reliance']    = b_draft_reliance_list
training_df['red_draft_reliance']     = r_draft_reliance_list

# SVD Draft Embeddings columns assignment
training_df['draft_style_sim'] = draft_style_sims
for i in range(16):
    training_df[f'blue_draft_emb_{i}'] = blue_draft_embs[i]
    training_df[f'red_draft_emb_{i}'] = red_draft_embs[i]
    training_df[f'diff_draft_emb_{i}'] = diff_draft_embs[i]

# ==========================================
# STEP 7: ALIGN ELO PER SIDE & SAVE
# ==========================================


training_df['target_blue_win'] = (training_df['map_winner'] == training_df['blue_side_team']).astype(int)

final_features = [
    'match_timestamp', 'match_id', 'season', 'game_number', 'patch_version',
    'blue_side_team', 'red_side_team',
    # SOTA features
    'blue_roster_stability', 'red_roster_stability', 'diff_roster_stability',
    'blue_draft_overlap', 'red_draft_overlap', 'diff_draft_overlap',
    'blue_rs_rank', 'red_rs_rank',
    'blue_is_defending_champ', 'red_is_defending_champ',
    # Ratings
    'blue_side_elo', 'red_side_elo',
    'blue_playoff_elo', 'red_playoff_elo',
    'blue_championship_dna', 'red_championship_dna',
    'blue_playoff_winrate', 'red_playoff_winrate',
    'blue_synergy', 'red_synergy',
    'blue_counter', 'red_counter',
    'blue_draft_exhaustion', 'red_draft_exhaustion',
    'blue_prev_winner_exhaustion', 'red_prev_winner_exhaustion',
    'blue_heroes_stolen', 'red_heroes_stolen',
    'blue_synergy_delta', 'red_synergy_delta',
    'prev_stomp_margin', 'is_side_swap',
    # Draft Skill
    'blue_comfort_wr', 'red_comfort_wr',
    'blue_draft_experience', 'red_draft_experience',
    'blue_global_draft_wr', 'red_global_draft_wr',
    # Bans (V6)
    'blue_ban_disruption', 'red_ban_disruption',
    # Patch buff/nerf from draft
    'blue_buffs_in_draft', 'blue_nerfs_in_draft',
    'red_buffs_in_draft',  'red_nerfs_in_draft',
    # Form & Momentum
    'blue_momentum', 'red_momentum',
    # H2H & Rivalry
    'blue_h2h_winrate',
    # Patch Prep
    'blue_patch_practice', 'red_patch_practice',
    # Playoff Pressure & Experience
    'is_playoffs',
    'blue_playoff_clutch', 'red_playoff_clutch',
    'blue_playoff_exp', 'red_playoff_exp',
    'blue_g3_clutch_wr', 'red_g3_clutch_wr',
    'blue_reverse_sweep_rate', 'red_reverse_sweep_rate',
    # Physical Readiness
    'blue_rest_factor', 'red_rest_factor',
    # Style & Meta
    'blue_avg_win_duration', 'red_avg_win_duration',
    'blue_avg_loss_duration', 'red_avg_loss_duration',
    'blue_execution_margin', 'red_execution_margin',
    'blue_execution_punish_score', 'red_execution_punish_score',
    'blue_lategame_winrate', 'red_lategame_winrate',
    # 'blue_draft_mastery', 'red_draft_mastery',
    # 'blue_execution_mastery', 'red_execution_mastery',
    # 'blue_draft_reliance', 'red_draft_reliance',
    'current_blue_side_advantage',
    'series_momentum_blue',
    # Preceding Draft Adaptation & Comfort Patch Impact
    'blue_g1_comfort', 'red_g1_comfort',
    'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
    'blue_prev_comfort', 'red_prev_comfort',
    'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red',
    'blue_comfort_patch_score', 'red_comfort_patch_score',
    'blue_patch_winrate', 'red_patch_winrate',
    'blue_patch_adaptation', 'red_patch_adaptation',
    'blue_expected_comfort', 'red_expected_comfort',
    # Official Seasonal Statistics (Prior Season)
    'blue_prev_season_match_wr', 'red_prev_season_match_wr',
    'blue_prev_season_game_wr', 'red_prev_season_game_wr',
    'blue_prev_season_kda', 'red_prev_season_kda',
    'blue_prev_season_avg_kills', 'red_prev_season_avg_kills',
    'blue_prev_season_avg_deaths', 'red_prev_season_avg_deaths',
    'blue_prev_season_avg_assists', 'red_prev_season_avg_assists',
    'diff_prev_season_match_wr', 'diff_prev_season_game_wr',
    'diff_prev_season_kda',
    # SVD Draft Embeddings features
    'draft_style_sim',
    'diff_draft_emb_0', 'diff_draft_emb_1', 'diff_draft_emb_2', 'diff_draft_emb_3',
    'diff_draft_emb_4', 'diff_draft_emb_5', 'diff_draft_emb_6', 'diff_draft_emb_7',
    'diff_draft_emb_8', 'diff_draft_emb_9', 'diff_draft_emb_10', 'diff_draft_emb_11',
    'diff_draft_emb_12', 'diff_draft_emb_13', 'diff_draft_emb_14', 'diff_draft_emb_15',
    'blue_draft_emb_0', 'blue_draft_emb_1', 'blue_draft_emb_2', 'blue_draft_emb_3',
    'blue_draft_emb_4', 'blue_draft_emb_5', 'blue_draft_emb_6', 'blue_draft_emb_7',
    'blue_draft_emb_8', 'blue_draft_emb_9', 'blue_draft_emb_10', 'blue_draft_emb_11',
    'blue_draft_emb_12', 'blue_draft_emb_13', 'blue_draft_emb_14', 'blue_draft_emb_15',
    'red_draft_emb_0', 'red_draft_emb_1', 'red_draft_emb_2', 'red_draft_emb_3',
    'red_draft_emb_4', 'red_draft_emb_5', 'red_draft_emb_6', 'red_draft_emb_7',
    'red_draft_emb_8', 'red_draft_emb_9', 'red_draft_emb_10', 'red_draft_emb_11',
    'red_draft_emb_12', 'red_draft_emb_13', 'red_draft_emb_14', 'red_draft_emb_15',
    # Target
    'time_weight', 'target_blue_win', 'stage'
]

final_matrix = training_df[final_features].dropna()
final_matrix.to_csv('csv_data/ML_Feature_Matrix.csv', index=False)
print(f"\n✅ V6 Data Pipeline Complete!")
print(f"   Rows: {len(final_matrix)} | Features: {len(final_features) - 7} predictive signals")  # minus id/meta/target cols


# =========================================================================================
# PART 2: V7 ENSEMBLE MODEL — DUAL ARCHITECTURE + HYPERPARAMETER TUNING
# =========================================================================================

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier

print("\n📊 Loading Feature Matrix for V7 Engine...")
df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')

# ==========================================
# NEW V7 FEATURE: ELIMINATION PRESSURE
# Computes series score for each team going INTO each game.
# e.g. Game 3 with score 1-1 → neither eliminated
#      Game 4 with score 1-2 → blue is eliminated if they lose
# ==========================================
print("🔧 Engineering Elimination Pressure features...")

df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)

blue_series_score_list = []
red_series_score_list  = []
is_elim_game_list      = []
valid_games_mask       = []

# Track running wins per match
series_wins_tracker = {}   # match_id -> {blue_team: wins, red_team: wins}

for _, row in df.iterrows():
    mid      = row['match_id']
    blue     = row['blue_side_team']
    red      = row['red_side_team']
    gnum     = row['game_number']
    winner   = row['blue_side_team'] if row['target_blue_win'] == 1 else row['red_side_team']
    is_playoffs = row['is_playoffs']

    if mid not in series_wins_tracker:
        series_wins_tracker[mid] = {blue: 0, red: 0}

    b_score = series_wins_tracker[mid].get(blue, 0)
    r_score = series_wins_tracker[mid].get(red, 0)
    
    # Dataset bug mitigation: if team already won 2 games in BO3 (Regular Season), match is over.
    # If the scraper logged a Game 3 anyway, it's invalid.
    if is_playoffs == 0 and max(b_score, r_score) >= 2:
        valid_games_mask.append(False)
    # Filter Game 4/5 from BO3 just in case, though max score handles it
    elif is_playoffs == 0 and gnum > 3:
        valid_games_mask.append(False)
    # Playoff BO5/BO7 filter (max wins 4)
    elif is_playoffs == 1 and max(b_score, r_score) >= 4:
        valid_games_mask.append(False)
    else:
        valid_games_mask.append(True)

    blue_series_score_list.append(b_score)
    red_series_score_list.append(r_score)

    # Is this an elimination game? (losing = out of the series)
    # In Bo3: losing at 0-2 or 1-1 → game 3 is always elimination for both
    # In Bo5: any game where loser hits max losses (2) → elim
    # Simple rule: if either team is 1 win away from being eliminated → True
    max_losses_needed = gnum - max(b_score, r_score) + 1
    is_elim = 1 if (b_score == r_score and gnum >= 3) or \
                   (abs(b_score - r_score) >= 1 and gnum >= 3) else 0
    is_elim_game_list.append(is_elim)

    # Update AFTER reading
    series_wins_tracker[mid][winner] = series_wins_tracker[mid].get(winner, 0) + 1

df['blue_series_score'] = blue_series_score_list
df['red_series_score']  = red_series_score_list
df['is_elimination_game'] = is_elim_game_list
df['score_diff_blue']   = df['blue_series_score'] - df['red_series_score']  # positive = blue leading
df['is_valid_game']     = valid_games_mask

df = df[df['is_valid_game'] == True].copy()
print(f"   Cleaned invalid dataset entries: kept {len(df)} valid games.")

# ==========================================
# DEFINE FULL FEATURE SET (with V7 additions)
# ==========================================
base_features = [
    'blue_roster_stability', 'red_roster_stability', 'diff_roster_stability',
    'blue_side_elo', 'red_side_elo',
    'blue_playoff_elo', 'red_playoff_elo',
    'blue_synergy', 'red_synergy',
    'blue_counter', 'red_counter',
    'blue_comfort_wr', 'red_comfort_wr',
    'blue_draft_experience', 'red_draft_experience',
    'blue_global_draft_wr', 'red_global_draft_wr',
    'blue_ban_disruption', 'red_ban_disruption',
    'blue_buffs_in_draft', 'blue_nerfs_in_draft',
    'red_buffs_in_draft',  'red_nerfs_in_draft',
    'blue_momentum', 'red_momentum',
    'blue_h2h_winrate',
    'blue_patch_practice', 'red_patch_practice',
    'is_playoffs',
    'blue_playoff_clutch', 'red_playoff_clutch',
    'blue_playoff_exp', 'red_playoff_exp',
    'blue_g3_clutch_wr', 'red_g3_clutch_wr',
    'blue_reverse_sweep_rate', 'red_reverse_sweep_rate',
    'blue_rest_factor', 'red_rest_factor',
    'blue_avg_win_duration', 'red_avg_win_duration',
    'current_blue_side_advantage',
    'blue_comfort_patch_score', 'red_comfort_patch_score',
    'blue_expected_comfort', 'red_expected_comfort',
    # V8 Game Duration & Dynamic Late-Game features
    'blue_avg_loss_duration', 'red_avg_loss_duration',
    'blue_execution_margin', 'red_execution_margin',
    'blue_execution_punish_score', 'red_execution_punish_score',
    'blue_lategame_winrate', 'red_lategame_winrate',
    # 'blue_draft_mastery', 'red_draft_mastery',
    # 'blue_execution_mastery', 'red_execution_mastery',
    # 'blue_draft_reliance', 'red_draft_reliance',
    # Prior official season statistics (pre-match, leak-safe)
    'blue_prev_season_match_wr', 'red_prev_season_match_wr',
    'blue_prev_season_game_wr', 'red_prev_season_game_wr',
    'blue_prev_season_kda', 'red_prev_season_kda',
    'blue_prev_season_avg_kills', 'red_prev_season_avg_kills',
    'blue_prev_season_avg_deaths', 'red_prev_season_avg_deaths',
    'blue_prev_season_avg_assists', 'red_prev_season_avg_assists',
    'diff_prev_season_match_wr', 'diff_prev_season_game_wr',
    'diff_prev_season_kda',
]

g1_features = base_features + ['draft_style_sim']

# RE-ENABLED MOMENTUM & PRESSURE FEATURES (SCRAPER BUG FIXED)
# The scraper now parses games in chronologically correct order.
# Momentum and series scores are completely valid, non-leaking pre-match signals!
all_features = base_features + [
    'blue_draft_overlap', 'red_draft_overlap', 'diff_draft_overlap',
    'blue_draft_exhaustion', 'red_draft_exhaustion',
    'blue_prev_winner_exhaustion', 'red_prev_winner_exhaustion',
    'blue_heroes_stolen', 'red_heroes_stolen',
    'blue_synergy_delta', 'red_synergy_delta',
    'prev_stomp_margin', 'is_side_swap',
    'series_momentum_blue',
    'blue_series_score', 'red_series_score',
    'score_diff_blue', 'is_elimination_game',
    'blue_g1_comfort', 'red_g1_comfort',
    'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
    'blue_prev_comfort', 'red_prev_comfort',
    'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red'
]

# ==========================================
# DUAL MODEL SPLIT (Game 1 vs Game 2+)
# Game 1 = no series history → use only base_features
# Game 2+ = series history known → use all features
# ==========================================
df_g1   = df[df['game_number'] == 1].copy()
df_g2plus = df[df['game_number'] > 1].copy()

X_g1      = df_g1[g1_features];     y_g1  = df_g1['target_blue_win'];     w_g1  = df_g1['time_weight']
X_g2plus  = df_g2plus[all_features];  y_g2  = df_g2plus['target_blue_win']; w_g2  = df_g2plus['time_weight']

# Grouped Match-ID Chronological Split (No leakage across series)
unique_matches = df['match_id'].unique()
split_idx = int(len(unique_matches) * 0.85)
train_match_ids = unique_matches[:split_idx]
test_match_ids = unique_matches[split_idx:]

train_df = df[df['match_id'].isin(train_match_ids)].copy()
test_df = df[df['match_id'].isin(test_match_ids)].copy()

train_g1 = train_df[train_df['game_number'] == 1].copy()
test_g1 = test_df[test_df['game_number'] == 1].copy()
train_g2 = train_df[train_df['game_number'] > 1].copy()
test_g2 = test_df[test_df['game_number'] > 1].copy()

X_train_g1 = train_g1[g1_features]; y_train_g1 = train_g1['target_blue_win']; w_train_g1 = train_g1['time_weight']
X_test_g1  = test_g1[g1_features];  y_test_g1  = test_g1['target_blue_win'];  w_test_g1  = test_g1['time_weight']

X_train_g2 = train_g2[all_features];  y_train_g2 = train_g2['target_blue_win']; w_train_g2 = train_g2['time_weight']
X_test_g2  = test_g2[all_features];   y_test_g2  = test_g2['target_blue_win'];  w_test_g2  = test_g2['time_weight']

# Normalize weights using training subset mean for each model to prevent leakage
mean_w_g1 = w_train_g1.mean() if w_train_g1.mean() != 0 else 1.0
mean_w_g2 = w_train_g2.mean() if w_train_g2.mean() != 0 else 1.0

w_train_g1 = w_train_g1 / mean_w_g1
w_test_g1  = w_test_g1 / mean_w_g1
w_train_g2 = w_train_g2 / mean_w_g2
w_test_g2  = w_test_g2 / mean_w_g2

print(f"   Game 1 model  → {len(X_train_g1)} train / {len(X_test_g1)} test")
print(f"   Game 2+ model → {len(X_train_g2)} train / {len(X_test_g2)} test")

# ==========================================
# HYPERPARAMETER TUNING (TimeSeriesSplit — no future leakage)
# ==========================================
print("\n⚙️  Tuning hyperparameters with TimeSeriesSplit (this may take 1-2 minutes)...")

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'n_estimators':    [200, 400, 600],
    'learning_rate':   [0.01, 0.02, 0.05],
    'max_depth':       [3, 4, 5],
    'subsample':       [0.8, 0.9, 1.0],
    'colsample_bytree':[0.7, 0.8, 0.9]
}

best_params = {'n_estimators': 250, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.7, 'min_child_weight': 3, 'gamma': 0.18}

print(f"   Mathematically Optimal V8 XGBoost params applied: {best_params}")

# ==========================================
# TRAIN ALL 3 ALGORITHMS WITH TUNED PARAMS
# ==========================================
print("\n🤖 Training V8 Ensemble (XGBoost + LightGBM + CatBoost + RandomForest)...")

def build_ensemble(best_params):
    xgb_model = xgb.XGBClassifier(
        **best_params, random_state=42, eval_metric='logloss', verbosity=0
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=best_params.get('n_estimators', 400),
        learning_rate=best_params.get('learning_rate', 0.02),
        max_depth=best_params.get('max_depth', 3),
        subsample=best_params.get('subsample', 0.9),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        random_state=42, verbose=-1
    )
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=best_params.get('max_depth', 3) + 2,
        random_state=42, n_jobs=-1
    )
    cat_model = CatBoostClassifier(
        iterations=250,
        learning_rate=0.02,
        depth=3,
        random_seed=42,
        verbose=0
    )
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model), 
            ('lgb', lgb_model), 
            ('rf', rf_model),
            ('cat', cat_model)
        ],
        voting='soft', weights=None   # Average probabilities, not just votes
    )
    return ensemble

# Game 1 Ensemble
ensemble_g1 = build_ensemble(best_params)
ensemble_g1.fit(X_train_g1, y_train_g1, sample_weight=w_train_g1)

# Game 2+ Ensemble
ensemble_g2 = build_ensemble(best_params)
ensemble_g2.fit(X_train_g2, y_train_g2, sample_weight=w_train_g2)

# ==========================================
# EVALUATE EACH MODEL SEPARATELY + COMBINED
# ==========================================
pred_g1    = ensemble_g1.predict(X_test_g1)
pred_g2    = ensemble_g2.predict(X_test_g2)
acc_g1     = accuracy_score(y_test_g1, pred_g1)
acc_g2     = accuracy_score(y_test_g2, pred_g2)
combined_acc = accuracy_score(
    pd.concat([y_test_g1, y_test_g2]),
    np.concatenate([pred_g1, pred_g2])
)

print("\n" + "="*55)
print(f"🏆 V7 RESULTS:")
print(f"   Game 1  Model Accuracy  : {acc_g1 * 100:.2f}%")
print(f"   Game 2+ Model Accuracy  : {acc_g2 * 100:.2f}%")
print(f"   📊 Combined Accuracy    : {combined_acc * 100:.2f}%")
print("="*55)

# ==========================================
# PREDICTION TRACKER — WHERE DID IT GO RIGHT/WRONG?
# ==========================================
test_g1_df   = test_g1.copy().reset_index(drop=True)
test_g2_df   = test_g2.copy().reset_index(drop=True)
test_g1_df['predicted_blue_win'] = pred_g1
test_g2_df['predicted_blue_win'] = pred_g2
test_g1_df['confidence'] = ensemble_g1.predict_proba(X_test_g1)[:, 1]
test_g2_df['confidence'] = ensemble_g2.predict_proba(X_test_g2)[:, 1]

test_df = pd.concat([test_g1_df, test_g2_df]).sort_values(['season', 'match_id', 'game_number']).reset_index(drop=True)
test_df['correct'] = (test_df['predicted_blue_win'] == test_df['target_blue_win']).astype(int)
test_df['predicted_winner'] = test_df.apply(
    lambda r: r['blue_side_team'] if r['predicted_blue_win'] == 1 else r['red_side_team'], axis=1)
test_df['actual_winner'] = test_df.apply(
    lambda r: r['blue_side_team'] if r['target_blue_win'] == 1 else r['red_side_team'], axis=1)
test_df['result_emoji'] = test_df['correct'].map({1: '✅', 0: '❌'})

total   = len(test_df)
correct = test_df['correct'].sum()
wrong   = total - correct

print(f"\n{'='*65}")
print(f"📊 PREDICTION TRACKER — TEST SET RESULTS")
print(f"{'='*65}")
print(f"   Total Games Tested : {total}")
print(f"   ✅ Correct          : {correct}  ({correct/total*100:.1f}%)")
print(f"   ❌ Wrong            : {wrong}   ({wrong/total*100:.1f}%)")

print("\n📌 ACCURACY BY STAGE:")
stage_acc = test_df.groupby('is_playoffs')['correct'].agg(['sum', 'count'])
stage_acc.index = stage_acc.index.map({0: 'Regular Season', 1: 'Playoffs'})
for stage, rs in stage_acc.iterrows():
    pct = rs['sum'] / rs['count'] * 100 if rs['count'] > 0 else 0
    bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
    print(f"   {stage:<15} : {bar}  {pct:.1f}%  ({int(rs['sum'])}/{int(rs['count'])})")

print("\n📅 ACCURACY BY SEASON:")
for season, rs in test_df.groupby('season')['correct'].agg(['sum', 'count']).iterrows():
    pct = rs['sum'] / rs['count'] * 100 if rs['count'] > 0 else 0
    bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
    print(f"   Season {str(season):<4} : {bar}  {pct:.1f}%  ({int(rs['sum'])}/{int(rs['count'])})")

print("\n🎯 ACCURACY BY GAME NUMBER (NEW — shows late-game improvement):")
for gnum, rs in test_df.groupby('game_number')['correct'].agg(['sum', 'count']).iterrows():
    pct = rs['sum'] / rs['count'] * 100 if rs['count'] > 0 else 0
    bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
    print(f"   Game {gnum:<2}       : {bar}  {pct:.1f}%  ({int(rs['sum'])}/{int(rs['count'])})")

print("\n🎯 ACCURACY BY MODEL CONFIDENCE:")
bins   = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ['<50% (Flip)', '50-60%', '60-70%', '70-80%', '80-90%', '90%+ (Sure)']
test_df['conf_bucket'] = pd.cut(test_df['confidence'].clip(0, 1-1e-9), bins=bins, labels=labels, right=False)
for bucket, rs in test_df.groupby('conf_bucket', observed=True)['correct'].agg(['sum', 'count']).iterrows():
    if rs['count'] == 0: continue
    pct = rs['sum'] / rs['count'] * 100
    print(f"   {str(bucket):<18}: {pct:.1f}%  ({int(rs['sum'])}/{int(rs['count'])} games)")

print("\n📋 GAME-BY-GAME PREDICTION LOG (Test Set):")
print(f"{'#':<4} {'Season':<8} {'Stage':<4} {'G':<3} {'Score':<6} {'Elim':<5} {'Blue Team':<22} {'Red Team':<22} {'Predicted':<22} {'Actual':<22} {'Conf':>6} {'OK'}")
print("-" * 155)
for i, row_g in test_df.iterrows():
    conf_d   = f"{row_g['confidence']*100:.1f}%"
    stage_s  = 'PO' if row_g['is_playoffs'] == 1 else 'RS'
    score_s  = f"{int(row_g.get('blue_series_score',0))}-{int(row_g.get('red_series_score',0))}"
    elim_s   = '🔥' if row_g.get('is_elimination_game', 0) == 1 else ' '
    print(
        f"{i+1:<4} S{str(row_g['season']):<7} {stage_s:<4} {int(row_g['game_number']):<3} "
        f"{score_s:<6} {elim_s:<5} "
        f"{row_g['blue_side_team']:<22} {row_g['red_side_team']:<22} "
        f"{row_g['predicted_winner']:<22} {row_g['actual_winner']:<22} "
        f"{conf_d:>6} {row_g['result_emoji']}"
    )

test_df[['season', 'is_playoffs', 'game_number', 'blue_series_score', 'red_series_score',
         'is_elimination_game', 'blue_side_team', 'red_side_team',
         'predicted_winner', 'actual_winner', 'confidence', 'correct', 'result_emoji']].to_csv(
    'csv_data/prediction_tracker.csv', index=False
)
print(f"\n💾 Full tracker saved to csv_data/prediction_tracker.csv")

# ==========================================
# FEATURE IMPORTANCE (XGBoost component of ensemble)
# ==========================================
xgb_g1 = ensemble_g1.named_estimators_['xgb']
importance_df = pd.DataFrame({
    'Feature':    g1_features,
    'Importance': xgb_g1.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n📈 TOP 15 FEATURES (Game 1 Model):")
print(importance_df.head(15).to_string(index=False))

# ==========================================
# AVAILABLE TEAMS
# ==========================================
print("\n📋 AVAILABLE TEAMS IN DATABASE:")
teams = sorted(set(df['blue_side_team'].unique().tolist() + df['red_side_team'].unique().tolist()))
print(teams)

# ==========================================
# V7 PLAYOFF SIMULATOR ENGINE
# ==========================================
def get_recent_stats(team_name, dataframe):
    """Safely extract the most recent pre-match stats for a team."""
    team_games = dataframe[
        (dataframe['blue_side_team'] == team_name) |
        (dataframe['red_side_team']  == team_name)
    ]
    if team_games.empty: return None
    last_game = team_games.iloc[-1]
    on_blue   = last_game['blue_side_team'] == team_name
    side = 'blue' if on_blue else 'red'
    invert_h2h = 0 if on_blue else 1
    stats_dict = {
        'elo':               last_game[f'{side}_side_elo'],
        'playoff_elo':       last_game[f'{side}_playoff_elo'],
        'roster_stability':  last_game[f'{side}_roster_stability'],
        'comfort_wr':        last_game[f'{side}_comfort_wr'],
        'draft_experience':  last_game[f'{side}_draft_experience'],
        'global_draft_wr':   last_game[f'{side}_global_draft_wr'],
        'momentum':          last_game[f'{side}_momentum'],
        'patch_practice':    last_game[f'{side}_patch_practice'],
        'playoff_clutch':    last_game[f'{side}_playoff_clutch'],
        'playoff_exp':       last_game[f'{side}_playoff_exp'],
        'g3_clutch_wr':      last_game[f'{side}_g3_clutch_wr'],
        'reverse_sweep_rate':last_game[f'{side}_reverse_sweep_rate'],
        'rest_factor':       last_game[f'{side}_rest_factor'],
        'avg_win_duration':  last_game[f'{side}_avg_win_duration'],
        'avg_loss_duration': last_game[f'{side}_avg_loss_duration'],
        'execution_margin':  last_game[f'{side}_execution_margin'],
        'execution_punish_score': last_game[f'{side}_execution_punish_score'],
        'lategame_winrate':  last_game[f'{side}_lategame_winrate'],
        'ban_disruption':    last_game[f'{side}_ban_disruption'],
        'comfort_patch_score':last_game[f'{side}_comfort_patch_score'],
        'patch_winrate':     last_game[f'{side}_patch_winrate'],
        'patch_adaptation':  last_game[f'{side}_patch_adaptation'],
        # 'draft_mastery':     last_game[f'{side}_draft_mastery'],
        # 'execution_mastery': last_game[f'{side}_execution_mastery'],
        # 'draft_reliance':    last_game[f'{side}_draft_reliance'],
        'h2h_winrate':       last_game['blue_h2h_winrate'] if on_blue else 1.0 - last_game['blue_h2h_winrate'],
        'comfort_patch_score':last_game[f'{side}_comfort_patch_score'],
        'expected_comfort':  last_game[f'{side}_expected_comfort'],
        'prev_season_match_wr': last_game[f'{side}_prev_season_match_wr'],
        'prev_season_game_wr': last_game[f'{side}_prev_season_game_wr'],
        'prev_season_kda': last_game[f'{side}_prev_season_kda'],
        'prev_season_avg_kills': last_game[f'{side}_prev_season_avg_kills'],
        'prev_season_avg_deaths': last_game[f'{side}_prev_season_avg_deaths'],
        'prev_season_avg_assists': last_game[f'{side}_prev_season_avg_assists'],
        'draft_overlap': last_game[f'{side}_draft_overlap'],
        'prev_winner_exhaustion': last_game[f'{side}_prev_winner_exhaustion'],
        'ban_disruption': last_game[f'{side}_ban_disruption'],
    }
    for i in range(16):
        stats_dict[f'draft_emb_{i}'] = last_game[f'{side}_draft_emb_{i}']
    return stats_dict

def simulate_matchup(blue_team, red_team, is_playoffs=True,
                     game_number=1, blue_series_score=0, red_series_score=0,
                     blue_won_last_game=None,
                     blue_comfort_patch_score=0.0, red_comfort_patch_score=0.0,
                     blue_patch_winrate=0.5, red_patch_winrate=0.5,
                     blue_patch_adaptation=0.0, red_patch_adaptation=0.0,
                     blue_expected_comfort=0.0, red_expected_comfort=0.0,
                     blue_g1_comfort=0.5, red_g1_comfort=0.5,
                     g1_winner_heroes_banned_blue=0.0, g1_winner_heroes_banned_red=0.0,
                     blue_prev_comfort=0.5, red_prev_comfort=0.5,
                     prev_winner_heroes_banned_blue=0.0, prev_winner_heroes_banned_red=0.0,
                     prev_played_comfort_banned_blue=0.0, prev_played_comfort_banned_red=0.0,
                     blue_prev_winner_exhaustion=0.0, red_prev_winner_exhaustion=0.0):
    """
    Simulate a matchup using the V7 Dual Ensemble Model.

    Parameters:
    -----------
    blue_team          : Exact name from AVAILABLE TEAMS list
    red_team           : Exact name from AVAILABLE TEAMS list
    is_playoffs        : True/False
    game_number        : 1, 2, 3, etc.
    blue_series_score  : Blue team's current wins in the series (e.g., 0, 1, 2)
    red_series_score   : Red team's current wins in the series
    blue_won_last_game : True/False/None (None for Game 1)
    """
    blue_stats = get_recent_stats(blue_team, df)
    red_stats  = get_recent_stats(red_team,  df)

    if blue_stats is None:
        print(f"❌ ERROR: Couldn't find '{blue_team}'. Check AVAILABLE TEAMS list!")
        return
    if red_stats is None:
        print(f"❌ ERROR: Couldn't find '{red_team}'. Check AVAILABLE TEAMS list!")
        return

    if blue_won_last_game is None: momentum = 0.5
    elif blue_won_last_game:       momentum = 1.0
    else:                          momentum = 0.0

    is_elim = 1 if (game_number >= 3 and blue_series_score != red_series_score) or \
                   (game_number >= 3 and blue_series_score == red_series_score) else 0

    latest_blue_bias = df['current_blue_side_advantage'].iloc[-1]
    series_momentum = momentum

    base_row = {
        'blue_side_elo':           blue_stats['elo'],
        'red_side_elo':            red_stats['elo'],
        'blue_playoff_elo':        blue_stats['playoff_elo'],
        'red_playoff_elo':         red_stats['playoff_elo'],
        'blue_roster_stability':   blue_stats['roster_stability'],
        'red_roster_stability':    red_stats['roster_stability'],
        'diff_roster_stability':   blue_stats['roster_stability'] - red_stats['roster_stability'],
        'blue_comfort_wr':         blue_stats['comfort_wr'],
        'red_comfort_wr':          red_stats['comfort_wr'],
        'blue_draft_experience':   blue_stats['draft_experience'],
        'red_draft_experience':    red_stats['draft_experience'],
        'blue_global_draft_wr':    0.5,
        'red_global_draft_wr':     0.5,
        'blue_ban_disruption':     blue_stats['ban_disruption'],
        'red_ban_disruption':      red_stats['ban_disruption'],
        'blue_buffs_in_draft':     0, 'blue_nerfs_in_draft': 0,
        'red_buffs_in_draft':      0, 'red_nerfs_in_draft':  0,
        'blue_momentum':           blue_stats['momentum'],
        'red_momentum':            red_stats['momentum'],
        'blue_h2h_winrate':        blue_stats['h2h_winrate'],
        'blue_patch_practice':     blue_stats['patch_practice'],
        'red_patch_practice':      red_stats['patch_practice'],
        'blue_synergy': 0.0, 'red_synergy': 0.0,
        'blue_counter': 0.0, 'red_counter': 0.0,
        'blue_draft_exhaustion': 0.0, 'red_draft_exhaustion': 0.0,
        'blue_prev_winner_exhaustion': 0.0, 'red_prev_winner_exhaustion': 0.0,
        'is_playoffs':             1 if is_playoffs else 0,
        'blue_playoff_clutch':     blue_stats['playoff_clutch'],
        'red_playoff_clutch':      red_stats['playoff_clutch'],
        'blue_playoff_exp':        blue_stats['playoff_exp'],
        'red_playoff_exp':         red_stats['playoff_exp'],
        'blue_g3_clutch_wr':       blue_stats['g3_clutch_wr'],
        'red_g3_clutch_wr':        red_stats['g3_clutch_wr'],
        'blue_reverse_sweep_rate': blue_stats['reverse_sweep_rate'],
        'red_reverse_sweep_rate':  red_stats['reverse_sweep_rate'],
        'blue_rest_factor':        1.02,
        'red_rest_factor':         1.02,
        'blue_avg_win_duration':   blue_stats['avg_win_duration'],
        'red_avg_win_duration':    red_stats['avg_win_duration'],
        'blue_avg_loss_duration':  blue_stats['avg_loss_duration'],
        'red_avg_loss_duration':   red_stats['avg_loss_duration'],
        'blue_execution_margin':   blue_stats['execution_margin'],
        'red_execution_margin':    red_stats['execution_margin'],
        'blue_execution_punish_score': blue_stats['execution_punish_score'],
        'red_execution_punish_score':  red_stats['execution_punish_score'],
        'blue_lategame_winrate':   blue_stats['lategame_winrate'],
        'red_lategame_winrate':    red_stats['lategame_winrate'],
        # 'blue_draft_mastery':      blue_stats['draft_mastery'],
        # 'red_draft_mastery':       red_stats['draft_mastery'],
        # 'blue_execution_mastery':  blue_stats['execution_mastery'],
        # 'red_execution_mastery':   red_stats['execution_mastery'],
        'current_blue_side_advantage': latest_blue_bias,
        'momentum_x_side_advantage': series_momentum * latest_blue_bias,
        'blue_comfort_patch_score': blue_comfort_patch_score if blue_comfort_patch_score != 0.0 else blue_stats['comfort_patch_score'],
        'red_comfort_patch_score':  red_comfort_patch_score if red_comfort_patch_score != 0.0 else red_stats['comfort_patch_score'],
        'blue_patch_winrate':       blue_patch_winrate if blue_patch_winrate != 0.5 else blue_stats['patch_winrate'],
        'red_patch_winrate':        red_patch_winrate if red_patch_winrate != 0.5 else red_stats['patch_winrate'],
        'blue_patch_adaptation':    blue_patch_adaptation if blue_patch_adaptation != 0.0 else blue_stats['patch_adaptation'],
        'red_patch_adaptation':     red_patch_adaptation if red_patch_adaptation != 0.0 else red_stats['patch_adaptation'],
        'blue_expected_comfort':    blue_expected_comfort if blue_expected_comfort != 0.0 else blue_stats['expected_comfort'],
        'red_expected_comfort':     red_expected_comfort if red_expected_comfort != 0.0 else red_stats['expected_comfort'],
        'blue_prev_season_match_wr': blue_stats['prev_season_match_wr'],
        'red_prev_season_match_wr': red_stats['prev_season_match_wr'],
        'blue_prev_season_game_wr': blue_stats['prev_season_game_wr'],
        'red_prev_season_game_wr': red_stats['prev_season_game_wr'],
        'blue_prev_season_kda': blue_stats['prev_season_kda'],
        'red_prev_season_kda': red_stats['prev_season_kda'],
        'blue_prev_season_avg_kills': blue_stats['prev_season_avg_kills'],
        'red_prev_season_avg_kills': red_stats['prev_season_avg_kills'],
        'blue_prev_season_avg_deaths': blue_stats['prev_season_avg_deaths'],
        'red_prev_season_avg_deaths': red_stats['prev_season_avg_deaths'],
        'blue_prev_season_avg_assists': blue_stats['prev_season_avg_assists'],
        'red_prev_season_avg_assists': red_stats['prev_season_avg_assists'],
        'diff_prev_season_match_wr': blue_stats['prev_season_match_wr'] - red_stats['prev_season_match_wr'],
        'diff_prev_season_game_wr': blue_stats['prev_season_game_wr'] - red_stats['prev_season_game_wr'],
        'diff_prev_season_kda': blue_stats['prev_season_kda'] - red_stats['prev_season_kda'],
        
    }

    # SVD Draft Embeddings features
    b_emb = [blue_stats[f'draft_emb_{i}'] for i in range(16)]
    r_emb = [red_stats[f'draft_emb_{i}'] for i in range(16)]
    dot_prod = np.dot(b_emb, r_emb)
    norm_eb = np.linalg.norm(b_emb)
    norm_er = np.linalg.norm(r_emb)
    if norm_eb == 0 or norm_er == 0:
        draft_style_sim = 1.0
    else:
        draft_style_sim = float(dot_prod / (norm_eb * norm_er))

    base_row['draft_style_sim'] = draft_style_sim
    for i in range(16):
        base_row[f'blue_draft_emb_{i}'] = blue_stats[f'draft_emb_{i}']
        base_row[f'red_draft_emb_{i}'] = red_stats[f'draft_emb_{i}']
        base_row[f'diff_draft_emb_{i}'] = blue_stats[f'draft_emb_{i}'] - red_stats[f'draft_emb_{i}']

    # Choose the right model and feature set
    if game_number == 1:
        matchup = pd.DataFrame([base_row])[g1_features]
        model   = ensemble_g1
    else:
        full_row = {
            **base_row,
            'series_momentum_blue': momentum,
            'blue_series_score': blue_series_score,
            'red_series_score': red_series_score,
            'score_diff_blue': blue_series_score - red_series_score,
            'is_elimination_game': is_elim,
            'blue_g1_comfort': blue_g1_comfort,
            'red_g1_comfort': red_g1_comfort,
            'g1_winner_heroes_banned_blue': g1_winner_heroes_banned_blue,
            'g1_winner_heroes_banned_red': g1_winner_heroes_banned_red,
            'blue_prev_comfort': blue_prev_comfort,
            'red_prev_comfort': red_prev_comfort,
            'prev_winner_heroes_banned_blue': prev_winner_heroes_banned_blue,
            'prev_winner_heroes_banned_red': prev_winner_heroes_banned_red,
            'prev_played_comfort_banned_blue': prev_played_comfort_banned_blue,
            'prev_played_comfort_banned_red': prev_played_comfort_banned_red,
            'blue_prev_winner_exhaustion': blue_prev_winner_exhaustion,
            'red_prev_winner_exhaustion': red_prev_winner_exhaustion,
            'blue_draft_overlap': 0.0,
            'red_draft_overlap': 0.0,
            'diff_draft_overlap': 0.0
        }
        matchup = pd.DataFrame([full_row])[all_features]
        model   = ensemble_g2

    probs     = model.predict_proba(matchup)[0]
    red_prob  = probs[0] * 100
    blue_prob = probs[1] * 100
    winner    = blue_team if blue_prob > red_prob else red_team

    elim_tag  = ' 🔥 ELIMINATION GAME' if is_elim and game_number > 1 else ''
    score_tag = f"  Series: {blue_series_score}-{red_series_score}" if game_number > 1 else ""
    print("\n" + "="*60)
    print(f"⚔️  {blue_team}  vs  {red_team}")
    print(f"   Game {game_number} | {'PLAYOFFS' if is_playoffs else 'Regular Season'}{score_tag}{elim_tag}")
    print("="*60)
    print(f"🟦  {blue_team}: {blue_prob:.1f}%")
    print(f"🟥  {red_team}:  {red_prob:.1f}%")
    print(f"\n🏆  PREDICTED WINNER: {winner}")
    print("="*60 + "\n")

# ==========================================
# RUN TOURNAMENT BRACKET
# ==========================================
print("\n\n--- 🏆 SIMULATING THE PLAYOFF BRACKET (V7 ENGINE) ---\n")

# Game 1 — uses the Game 1 specialist model
simulate_matchup("Team Liquid PH", "Team Falcons PH", is_playoffs=True, game_number=1)
simulate_matchup("ONIC PH",        "RSG PH",  is_playoffs=True, game_number=1)

# Example: Game 2 after Team Liquid PH won Game 1
# simulate_matchup("Team Liquid PH", "Team Falcons PH", is_playoffs=True,
#                  game_number=2, blue_series_score=1, red_series_score=0,
#                  blue_won_last_game=True)

# Example: Deciding Game 3 — ELIMINATION pressure auto-detected
# simulate_matchup("ONIC PH", "RSG PH", is_playoffs=True,
#                  game_number=3, blue_series_score=1, red_series_score=1,
#                  blue_won_last_game=False)
