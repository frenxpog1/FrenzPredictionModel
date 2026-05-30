import pandas as pd
import numpy as np
import json
import os
import re
import math
import sys
from draft_embeddings import get_svd_hero_embeddings

# ── TEAM ALIAS MAP (Franchise Rebrands) ──
# Maps historical rebranded team names to their modern franchise equivalent.
TEAM_ALIAS_MAP = {
    "Aether Main": "Team Falcons PH",
    "Aether Valkyrie": "Team Falcons PH",
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
    if t == "Aurora" and season is not None:
        try:
            s_int = int(str(season).replace('S', ''))
            if s_int <= 10:
                return "Team Liquid PH"
        except:
            pass
            
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

def main():
    print("Starting Feature Generation Pipeline...")
    
    # Check that required CSV files are present
    required_files = [
        'csv_data/matches.csv',
        'csv_data/games.csv',
        'csv_data/season_rosters.csv',
        'csv_data/patches.csv',
        'csv_data/official_mpl_ph_stats/mpl_ph_official_teams_s5_s15_s17.csv',
        'csv_data/official_mpl_ph_stats/mpl_ph_official_standings_s5_s15_s17.csv'
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Missing required file '{f}'.")
            sys.exit(1)

    matches_df = pd.read_csv('csv_data/matches.csv')
    games_df   = pd.read_csv('csv_data/games.csv')
    rosters_df = pd.read_csv('csv_data/season_rosters.csv')
    patches_df = pd.read_csv('csv_data/patches.csv')

    # Apply franchise mapping
    if 'team_a_name' in matches_df.columns: matches_df['team_a_name'] = matches_df.apply(lambda r: resolve_team_name(r['team_a_name'], r.get('season')), axis=1)
    if 'team_b_name' in matches_df.columns: matches_df['team_b_name'] = matches_df.apply(lambda r: resolve_team_name(r['team_b_name'], r.get('season')), axis=1)

    games_df['blue_side_team'] = games_df.apply(lambda r: resolve_team_name(r['blue_side_team'], r.get('season')), axis=1)
    games_df['red_side_team']  = games_df.apply(lambda r: resolve_team_name(r['red_side_team'], r.get('season')), axis=1)
    games_df['map_winner']     = games_df.apply(lambda r: resolve_team_name(r['map_winner'], r.get('season')), axis=1)

    rosters_df['team_name']    = rosters_df.apply(lambda r: resolve_team_name(r['team_name'], r.get('season')), axis=1)

    # Load official seasonal stats
    teams_df = pd.read_csv('csv_data/official_mpl_ph_stats/mpl_ph_official_teams_s5_s15_s17.csv')
    standings_df = pd.read_csv('csv_data/official_mpl_ph_stats/mpl_ph_official_standings_s5_s15_s17.csv')

    teams_df['team'] = teams_df.apply(lambda r: resolve_team_name(r['team'], r.get('season')), axis=1)
    standings_df['team'] = standings_df.apply(lambda r: resolve_team_name(r['team'], r.get('season')), axis=1)

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
                    
        for s in range(min(current_season - 1, 15), 4, -1):
            abbrev = get_official_team_abbreviation(db_name, s)
            if abbrev and (s, abbrev) in compiled_stats:
                stats = compiled_stats[(s, abbrev)]
                if stats.get('avg_kills', 0) == 0 and stats.get('avg_deaths', 0) == 0:
                    continue
                if pd.isna(res['avg_kills']) and not pd.isna(stats['avg_kills']):
                    res['avg_kills'] = stats['avg_kills']
                    res['avg_deaths'] = stats['avg_deaths']
                    res['avg_assists'] = stats['avg_assists']
                    res['avg_kda'] = stats['avg_kda']
                    break
                    
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

    print("Calculating Elo ratings (Regular + Playoff tracks)...")

    team_rosters = {}
    for _, row in rosters_df.iterrows():
        team, season = row['team_name'], str(row['season']).strip()
        try:    player_igns = set([p['ign'].strip() for p in json.loads(row['players'])])
        except: player_igns = set()
        if team not in team_rosters: team_rosters[team] = {}
        team_rosters[team][season] = player_igns

    matches_df = matches_df.sort_values('match_timestamp').reset_index(drop=True)

    player_elos         = {}
    player_playoff_elos = {}

    default_elo  = 1500
    k_regular    = 32
    k_playoff    = 56
    decay_rate   = 0.20

    player_championship_wins = {}
    player_playoff_experience = {}

    IGN_ALIASES = {
        "3Mar":         "3MarTzy",
        "BON CHAN":     "Bon Chan",
        "DEX STAR":     "Dex Star",
        "Dlar":         "Dlarskie",
        "Domeng":       "Domengkite",
        "DomengDR":     "Domengkite",
        "Bon Chon":     "Bon Chan",
        "BruskoDR":     "Brusko",
        "GoyongDR":     "Goyong",
        "Had ji":       "Hadji",
        "Had Ji":       "Hadji",
        "Kekedot":      "Kekedoot",
        "YellyHazeDR":  "YellyHaze",
        "E2Max":        "E2MAX",
        "E2max":        "E2MAX",
        "EDWARD":       "Edward",
        "ESON":         "Eson",
        "Flap":         "FlapTzy",
        "H2WO":         "H2wo",
        "Imbadeejay":   "ImbaDeejay",
        "Karl":         "KarlTzy",
        "KyleTzy":      "Kyle",
        "Lancecy":      "LanceCy",
        "Lord Malikk":  "Malik",
        "Mico":         "Micophobia",
        "Netskie":      "Nets",
        "Nova":         "xNova",
        "OHEB":         "Oheb",
        "Pando":        "Pandora",
        "RENEJAY":      "Renejay",
        "rTzy":         "RTZY",
        "SUPER MARCO":  "Super Marco",
        "ynoT":         "YnoT",
        "ynoT":         "YnoT",
        "1rrad":        "Irrad",
        "Ukir":         "Uk1r",
        "P4kbet":       "Pakbet",
        "Chuuu":        "SDzyz",
        "Kousei":       "Kouzen",
        "RTzy":         "RTZY",
        "Exort":        "Bornok",
        "Pandaaa":      "Panda",
        "JeffQT4ever":  "JeffQt4ever",
        "Jeffqt4ever":  "JeffQt4ever",
        "Shaiderqt":    "ShaiderQT",
    }

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

    def get_team_avg_elo(team_name, season_string, elo_dict):
        roster = team_rosters.get(team_name, {}).get(season_string, set())
        roster_igns = [resolve_ign(ign) for ign in roster]
        if not roster_igns: return default_elo
        return sum([elo_dict.get(ign, default_elo) for ign in roster_igns]) / len(roster_igns)


    matches_df['team_a_elo']          = 0.0
    matches_df['team_b_elo']          = 0.0
    matches_df['team_a_playoff_elo']  = 0.0
    matches_df['team_b_playoff_elo']  = 0.0
    current_global_season = str(matches_df.iloc[0]['season'])

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

    game_winner_lookup = {}

    print("Calculating all features (this includes 14 tracker types)...")

    # Trackers
    player_elos           = {}
    player_playoff_elos   = {}
    current_global_season = None
    global_synergy_matrix = {}
    global_counter_matrix = {}

    team_hero_tracker      = {}
    seasonal_hero_tracker  = {}
    team_recent_form       = {}
    h2h_tracker            = {}
    patch_practice         = {}
    playoff_clutch_tracker = {}
    global_hero_tracker    = {}
    team_win_durations     = {}
    global_side_tracker    = []
    current_patch_tracker  = None
    team_patch_stats       = {}
    team_hist_stats        = {}

    team_last_game_date      = {}
    team_roster_history      = {}
    team_g3_tracker          = {}
    team_playoff_games_count = {}
    reverse_sweep_tracker = {}

    team_loss_durations          = {}
    team_recent_lategame_results = {}
    team_draft_scores            = {}
    team_exec_scores             = {}

    last_game_in_match = training_df.groupby('match_id')['game_number'].max().to_dict()
    series_g1_winner   = {}
    current_series_wins = {}
    series_draft_history = {}

    champions_history = {}
    rs_standings = {}
    b_rs_rank, r_rs_rank = [], []
    b_is_defending_champ, r_is_defending_champ = [], []

    # Output Lists
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
    
    b_rest_factor, r_rest_factor           = [], []
    b_ban_disruption, r_ban_disruption     = [], []
    b_g3_clutch_wr, r_g3_clutch_wr         = [], []
    b_playoff_exp_count, r_playoff_exp_count = [], []
    series_momentum_blue                    = []
    b_reverse_sweep_rate, r_reverse_sweep_rate = [], []

    b_g1_comfort, r_g1_comfort = [], []
    g1_winner_heroes_banned_blue = []
    g1_winner_heroes_banned_red = []

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

    b_comfort_patch_score = []
    r_comfort_patch_score = []

    b_patch_winrate = []
    r_patch_winrate = []
    b_patch_adaptation = []
    r_patch_adaptation = []

    b_expected_comfort = []
    r_expected_comfort = []

    blue_prev_season_match_wr, red_prev_season_match_wr = [], []
    blue_prev_season_game_wr, red_prev_season_game_wr = [], []
    blue_prev_season_kda, red_prev_season_kda = [], []
    blue_prev_season_avg_kills, red_prev_season_avg_kills = [], []
    blue_prev_season_avg_deaths, red_prev_season_avg_deaths = [], []
    blue_prev_season_avg_assists, red_prev_season_avg_assists = [], []
    diff_prev_season_match_wr, diff_prev_season_game_wr, diff_prev_season_kda = [], [], []

    b_roster_stability, r_roster_stability, diff_roster_stability_list = [], [], []
    b_draft_overlap_list, r_draft_overlap_list, diff_draft_overlap_list = [], [], []

    blue_draft_embs = [[] for _ in range(16)]
    red_draft_embs = [[] for _ in range(16)]
    diff_draft_embs = [[] for _ in range(16)]
    draft_style_sims = []

    def rest_factor_score(gap_days):
        if gap_days is None:     return 1.0
        if gap_days <= 3:        return 0.97
        elif gap_days <= 14:     return 1.02
        elif gap_days <= 30:     return 1.0 - 0.001 * (gap_days - 14)
        else:                    return max(0.90, 0.986 * np.exp(-0.005 * (gap_days - 30)))

    def get_comfort_patch_impact(team, hero_tracker, patch_version):
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
        if team not in hero_tracker or not hero_tracker[team]:
            return 0.5
        team_heroes = hero_tracker[team]
        sorted_heroes = sorted(team_heroes.keys(), key=lambda h: team_heroes[h]['games'], reverse=True)
        top_5 = sorted_heroes[:5]
        if not top_5:
            return 0.5
        smoothed_wrs = []
        for h in top_5:
            stats = team_heroes[h]
            smoothed_wr = (stats['wins'] + 2) / (stats['games'] + 4)
            smoothed_wrs.append(smoothed_wr)
        return float(np.mean(smoothed_wrs))

    def get_ban_disruption(banned_heroes, opponent_team, hero_tracker):
        if not banned_heroes or opponent_team not in hero_tracker: return 0.0
        opp_picks = hero_tracker[opponent_team]
        top_comfort = set(sorted(opp_picks.keys(), key=lambda h: opp_picks[h]['games'], reverse=True)[:12])
        hits = sum(1 for h in banned_heroes if h in top_comfort)
        return hits / len(banned_heroes)

    def get_g3_wr(team):
        stats = team_g3_tracker.get(team, {'wins': 0, 'games': 0})
        return (stats['wins'] + 2) / (stats['games'] + 4)

    def get_reverse_sweep_rate(team):
        stats = reverse_sweep_tracker.get(team, {'down_01': 0, 'came_back': 0})
        return (stats['came_back'] + 1) / (stats['down_01'] + 2)

    hero_embeddings, fallback_emb = get_svd_hero_embeddings("mlbb_data.db", K=16)
    team_draft_history = {}

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
        cur_date    = pd.to_datetime(row['match_timestamp'])
        cur_s_int   = int(row['season'])
        match_season = str(row['season'])
        
        # Champ curse & RS rank
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
        b_prior, _ = lookup_prior_stats(blue_team, cur_s_int)
        r_prior, _ = lookup_prior_stats(red_team, cur_s_int)
        
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

        # Season transition Elo Decay
        if match_season != current_global_season:
            if current_global_season is not None:
                for ign in player_elos:
                    player_elos[ign] = 1500 + ((player_elos[ign] - 1500) * (1 - decay_rate))
                for ign in player_playoff_elos:
                    player_playoff_elos[ign] = 1500 + ((player_playoff_elos[ign] - 1500) * (1 - decay_rate))
            current_global_season = match_season

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
            blue_bans  = bans_data.get('blue', [])
            red_bans   = bans_data.get('red', [])
        except: blue_bans, red_bans = [], []

        blue_season_key = (match_season, blue_team)
        red_season_key  = (match_season, red_team)
        
        # 1. Draft Mastery
        def get_mastery(team, heroes):
            if not heroes or team not in team_hero_tracker: return 0.5, 0
            wins, games = 0, 0
            for h in heroes:
                s = team_hero_tracker[team].get(h, {'wins': 0, 'games': 0})
                wins += s['wins']; games += s['games']
            return (wins + 2) / (games + 4), games

        b_wr, b_g = get_mastery(blue_team, blue_heroes)
        r_wr, r_g = get_mastery(red_team,  red_heroes)

        b_mom = np.mean(team_recent_form.get(blue_team, [0.5]))
        r_mom = np.mean(team_recent_form.get(red_team,  [0.5]))

        mk = tuple(sorted([blue_team, red_team]))
        h2h = h2h_tracker.get(mk, {'blue_wins': 0, 'total': 0})
        if h2h['total'] == 0:
            h2h_score = 0.5
        else:
            w = h2h['blue_wins'] if mk[0] == blue_team else (h2h['total'] - h2h['blue_wins'])
            h2h_score = (w + 1) / (h2h['total'] + 2)

        b_patch = patch_practice.get(blue_team, 0)
        r_patch = patch_practice.get(red_team,  0)

        def get_clutch_rate(team):
            s = playoff_clutch_tracker.get(team, {'wins': 0, 'games': 0})
            return (s['wins'] + 2) / (s['games'] + 4)

        b_playoff = get_clutch_rate(blue_team)
        r_playoff = get_clutch_rate(red_team)

        def get_global_draft_wr(heroes):
            if not heroes: return 0.5
            total = 0
            for h in heroes:
                s = global_hero_tracker.get(h, {'wins': 0, 'games': 0})
                total += (s['wins'] + 5) / (s['games'] + 10)
            return total / len(heroes)

        b_global_wr = get_global_draft_wr(blue_heroes)
        r_global_wr = get_global_draft_wr(red_heroes)

        b_avg_dur = np.mean(team_win_durations.get(blue_team, [1020]))
        r_avg_dur = np.mean(team_win_durations.get(red_team,  [1020]))
        b_avg_loss_dur = np.mean(team_loss_durations.get(blue_team, [1020]))
        r_avg_loss_dur = np.mean(team_loss_durations.get(red_team,  [1020]))
        
        b_exec_margin = (b_avg_loss_dur - b_avg_dur) / 60.0
        r_exec_margin = (r_avg_loss_dur - r_avg_dur) / 60.0
        b_exec_bias = (1020.0 / b_avg_dur) * (b_elo / 1500.0)
        r_exec_bias = (1020.0 / r_avg_dur) * (r_elo / 1500.0)
        
        b_lg_wr = np.mean(team_recent_lategame_results.get(blue_team, [0.5]))
        r_lg_wr = np.mean(team_recent_lategame_results.get(red_team,  [0.5]))
        cur_blue_bias = np.mean(global_side_tracker) if global_side_tracker else 0.5

        b_last = team_last_game_date.get(blue_team)
        r_last = team_last_game_date.get(red_team)
        b_gap  = (cur_date - b_last).days if b_last else 7
        r_gap  = (cur_date - r_last).days if r_last else 7
        b_rest = rest_factor_score(b_gap)
        r_rest = rest_factor_score(r_gap)

        b_ban_dis = get_ban_disruption(blue_bans, red_team,  team_hero_tracker)
        r_ban_dis = get_ban_disruption(red_bans,  blue_team, team_hero_tracker)

        b_g3 = get_g3_wr(blue_team)
        r_g3 = get_g3_wr(red_team)

        b_po_exp = sum([player_playoff_experience.get(ign, 0) for ign in blue_roster_clean])
        r_po_exp = sum([player_playoff_experience.get(ign, 0) for ign in red_roster_clean])

        prev_game_result = game_winner_lookup.get((match_id, game_num - 1))
        if prev_game_result is None:
            momentum_score = 0.5
        elif prev_game_result == blue_team:
            momentum_score = 1.0
        else:
            momentum_score = 0.0

        b_rsweep = get_reverse_sweep_rate(blue_team)
        r_rsweep = get_reverse_sweep_rate(red_team)

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

        def get_game_roster(team_name, roster_col_val, season_str):
            try:
                if pd.notna(roster_col_val) and roster_col_val != '[]':
                    players = json.loads(str(roster_col_val).replace('""', '"'))
                    if players:
                        return set([resolve_ign(p) for p in players])
            except: pass
            roster_set = team_rosters.get(team_name, {}).get(season_str, set())
            return set([resolve_ign(ign) for ign in roster_set])

        blue_roster = get_game_roster(blue_team, row['blue_roster'], match_season)
        red_roster = get_game_roster(red_team, row['red_roster'], match_season)

        def compute_roster_stability(team, current_roster, history_dict):
            hist = history_dict.get(team, [])
            if not hist: return 1.0
            similarities = []
            for prev_roster in hist[-3:]:
                if not current_roster or not prev_roster: similarities.append(1.0)
                else:
                    intersection = len(current_roster.intersection(prev_roster))
                    union = len(current_roster.union(prev_roster))
                    similarities.append(intersection / union if union > 0 else 1.0)
            return float(np.mean(similarities))

        blue_roster_stability_val = compute_roster_stability(blue_team, blue_roster, team_roster_history)
        red_roster_stability_val = compute_roster_stability(red_team, red_roster, team_roster_history)
        diff_roster_stability_val = blue_roster_stability_val - red_roster_stability_val

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

                prev_game = prev_games[-1]
                prev_winner = prev_game['map_winner']
                blue_was_blue_in_prev = prev_game['blue_side_team'] == blue_team
                
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

                opp_red_picks = team_hero_tracker.get(red_team, {})
                red_top_comfort = set(sorted(opp_red_picks.keys(), key=lambda h: opp_red_picks[h]['games'], reverse=True)[:12])
                red_prev_played_comfort = [h for h in red_prev_heroes if h in red_top_comfort]
                if red_prev_played_comfort:
                    prev_played_comfort_banned_b_val = sum(1 for h in blue_bans if h in red_prev_played_comfort) / len(red_prev_played_comfort)

                opp_blue_picks = team_hero_tracker.get(blue_team, {})
                blue_top_comfort = set(sorted(opp_blue_picks.keys(), key=lambda h: opp_blue_picks[h]['games'], reverse=True)[:12])
                blue_prev_played_comfort = [h for h in blue_prev_heroes if h in blue_top_comfort]
                if blue_prev_played_comfort:
                    prev_played_comfort_banned_r_val = sum(1 for h in red_bans if h in blue_prev_played_comfort) / len(blue_prev_played_comfort)

        b_comfort_patch = get_comfort_patch_impact(blue_team, team_hero_tracker, patch_v)
        r_comfort_patch = get_comfort_patch_impact(red_team, team_hero_tracker, patch_v)

        b_exp_comfort = get_expected_comfort(blue_team, team_hero_tracker)
        r_exp_comfort = get_expected_comfort(red_team, team_hero_tracker)

        # Patch Stats
        for team in [blue_team, red_team]:
            if team not in team_patch_stats: team_patch_stats[team] = {}
            if patch_v not in team_patch_stats[team]: team_patch_stats[team][patch_v] = {'wins': 0, 'games': 0}
            if team not in team_hist_stats: team_hist_stats[team] = {'wins': 0, 'games': 0}

        b_pwins, b_pgames = team_patch_stats[blue_team][patch_v]['wins'], team_patch_stats[blue_team][patch_v]['games']
        b_hwins, b_hgames = team_hist_stats[blue_team]['wins'], team_hist_stats[blue_team]['games']
        r_pwins, r_pgames = team_patch_stats[red_team][patch_v]['wins'], team_patch_stats[red_team][patch_v]['games']
        r_hwins, r_hgames = team_hist_stats[red_team]['wins'], team_hist_stats[red_team]['games']

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

        b_draft_score = (b_global_wr * 0.4) + (b_wr * 0.4) + (b_ban_dis * 0.2)
        r_draft_score = (r_global_wr * 0.4) + (r_wr * 0.4) + (r_ban_dis * 0.2)
        b_exec_score  = b_exec_bias * (b_lg_wr + 0.5) * (b_g3 + 1)
        r_exec_score  = r_exec_bias * (r_lg_wr + 0.5) * (r_g3 + 1)
        
        b_draft_mastery = np.mean(team_draft_scores.get(blue_team, [0.5]))
        r_draft_mastery = np.mean(team_draft_scores.get(red_team,  [0.5]))
        b_exec_mastery  = np.mean(team_exec_scores.get(blue_team, [1.0]))
        r_exec_mastery  = np.mean(team_exec_scores.get(red_team,  [1.0]))
        
        b_draft_reliance = b_draft_mastery / (b_exec_mastery + 0.1)
        r_draft_reliance = r_draft_mastery / (r_exec_mastery + 0.1)

        b_draft_history = team_draft_history.get(blue_team, [])
        if not b_draft_history: E_blue = np.array(fallback_emb)
        else:
            b_heroes_flat = [h for g in b_draft_history for h in g]
            b_vectors = [np.array(hero_embeddings.get(h, fallback_emb)) for h in b_heroes_flat]
            E_blue = np.mean(b_vectors, axis=0)

        r_draft_history = team_draft_history.get(red_team, [])
        if not r_draft_history: E_red = np.array(fallback_emb)
        else:
            r_heroes_flat = [h for g in r_draft_history for h in g]
            r_vectors = [np.array(hero_embeddings.get(h, fallback_emb)) for h in r_heroes_flat]
            E_red = np.mean(r_vectors, axis=0)

        dot_prod = np.dot(E_blue, E_red)
        norm_eb = np.linalg.norm(E_blue)
        norm_er = np.linalg.norm(E_red)
        if norm_eb == 0 or norm_er == 0: draft_style_sim = 1.0
        else: draft_style_sim = float(dot_prod / (norm_eb * norm_er))

        draft_style_sims.append(draft_style_sim)
        for i in range(16):
            blue_draft_embs[i].append(float(E_blue[i]))
            red_draft_embs[i].append(float(E_red[i]))
            diff_draft_embs[i].append(float(E_blue[i] - E_red[i]))

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
        
        b_rest_factor.append(b_rest);          r_rest_factor.append(r_rest)
        b_ban_disruption.append(b_ban_dis);    r_ban_disruption.append(r_ban_dis)
        b_g3_clutch_wr.append(b_g3);          r_g3_clutch_wr.append(r_g3)
        b_playoff_exp_count.append(b_po_exp); r_playoff_exp_count.append(r_po_exp)
        series_momentum_blue.append(momentum_score)
        b_reverse_sweep_rate.append(b_rsweep); r_reverse_sweep_rate.append(r_rsweep)

        b_g1_comfort.append(b_g1_comfort_val)
        r_g1_comfort.append(r_g1_comfort_val)
        g1_winner_heroes_banned_blue.append(g1_winner_banned_b)
        g1_winner_heroes_banned_red.append(g1_winner_banned_r)

        b_avg_loss_duration.append(b_avg_loss_dur)
        r_avg_loss_duration.append(r_avg_loss_dur)
        b_execution_margin.append(b_exec_margin)
        r_execution_margin.append(r_exec_margin)
        b_execution_punish_score.append(b_exec_bias)
        r_execution_punish_score.append(r_exec_bias)
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

        b_comfort_patch_score.append(b_comfort_patch)
        r_comfort_patch_score.append(r_comfort_patch)
        b_expected_comfort.append(b_exp_comfort)
        r_expected_comfort.append(r_exp_comfort)

        # STATE UPDATES AFTER PRE-MATCH STATE RECORDED
        blue_won = 1 if winner == blue_team else 0
        red_won  = 1 - blue_won

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

        game_winner_lookup[(match_id, game_num)] = winner

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

        team_recent_form.setdefault(blue_team, [])
        team_recent_form.setdefault(red_team,  [])
        team_recent_form[blue_team] = (team_recent_form[blue_team] + [blue_won])[-5:]
        team_recent_form[red_team]  = (team_recent_form[red_team]  + [red_won])[-5:]

        h2h_tracker.setdefault(mk, {'blue_wins': 0, 'total': 0})
        h2h_tracker[mk]['total'] += 1
        if (mk[0] == blue_team and blue_won) or (mk[0] == red_team and red_won):
            h2h_tracker[mk]['blue_wins'] += 1

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

        if is_playoffs == 1:
            playoff_clutch_tracker.setdefault(blue_team, {'wins': 0, 'games': 0})
            playoff_clutch_tracker.setdefault(red_team,  {'wins': 0, 'games': 0})
            playoff_clutch_tracker[blue_team]['games'] += 1
            playoff_clutch_tracker[blue_team]['wins']  += blue_won
            playoff_clutch_tracker[red_team]['games']  += 1
            playoff_clutch_tracker[red_team]['wins']   += red_won

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

            if duration > 1080:
                team_recent_lategame_results.setdefault(blue_team, [])
                team_recent_lategame_results[blue_team] = (team_recent_lategame_results[blue_team] + [blue_won])[-10:]
                team_recent_lategame_results.setdefault(red_team, [])
                team_recent_lategame_results[red_team] = (team_recent_lategame_results[red_team] + [not blue_won])[-10:]

        team_draft_scores.setdefault(blue_team, [])
        team_draft_scores[blue_team] = (team_draft_scores[blue_team] + [b_draft_score])[-15:]
        team_draft_scores.setdefault(red_team, [])
        team_draft_scores[red_team] = (team_draft_scores[red_team] + [r_draft_score])[-15:]
        
        team_exec_scores.setdefault(blue_team, [])
        team_exec_scores[blue_team] = (team_exec_scores[blue_team] + [b_exec_score])[-15:]
        team_exec_scores.setdefault(red_team, [])
        team_exec_scores[red_team] = (team_exec_scores[red_team] + [r_exec_score])[-15:]

        global_side_tracker = (global_side_tracker + [blue_won])[-50:]
        team_last_game_date[blue_team] = cur_date
        team_last_game_date[red_team]  = cur_date

        team_roster_history.setdefault(blue_team, []).append(blue_roster)
        team_roster_history.setdefault(red_team, []).append(red_roster)

        if game_num >= 3:
            team_g3_tracker.setdefault(blue_team, {'wins': 0, 'games': 0})
            team_g3_tracker.setdefault(red_team,  {'wins': 0, 'games': 0})
            team_g3_tracker[blue_team]['games'] += 1
            team_g3_tracker[blue_team]['wins']  += blue_won
            team_g3_tracker[red_team]['games']  += 1
            team_g3_tracker[red_team]['wins']   += red_won

        if is_playoffs == 1:
            team_playoff_games_count[blue_team] = team_playoff_games_count.get(blue_team, 0) + 1
            team_playoff_games_count[red_team]  = team_playoff_games_count.get(red_team,  0) + 1

        if game_num == 1:
            g1_loser = red_team if blue_won else blue_team
            series_g1_winner[match_id] = (winner, g1_loser)

        if match_id not in current_series_wins:
            current_series_wins[match_id] = {blue_team: 0, red_team: 0}
        current_series_wins[match_id][winner] = current_series_wins[match_id].get(winner, 0) + 1

        if game_num == last_game_in_match.get(match_id, -1):
            if match_id in series_g1_winner:
                g1_win, g1_loss = series_g1_winner[match_id]
                series_wins = current_series_wins[match_id]
                series_winner_team = max(series_wins, key=series_wins.get)
                reverse_sweep_tracker.setdefault(g1_loss, {'down_01': 0, 'came_back': 0})
                reverse_sweep_tracker[g1_loss]['down_01'] += 1
                if series_winner_team == g1_loss:
                    reverse_sweep_tracker[g1_loss]['came_back'] += 1

        stage_str = str(row['stage']).strip().lower()
        if stage_str in ['rs', 'group stage', 'regular season']:
            if cur_s_int not in rs_standings: rs_standings[cur_s_int] = {}
            if blue_team not in rs_standings[cur_s_int]: rs_standings[cur_s_int][blue_team] = {'wins': 0, 'games': 0}
            if red_team not in rs_standings[cur_s_int]: rs_standings[cur_s_int][red_team] = {'wins': 0, 'games': 0}
            rs_standings[cur_s_int][blue_team]['games'] += 1
            rs_standings[cur_s_int][red_team]['games'] += 1
            if blue_won: rs_standings[cur_s_int][blue_team]['wins'] += 1
            else: rs_standings[cur_s_int][red_team]['wins'] += 1

        if stage_str in ['playoffs', 'grand final', 'upper bracket', 'lower bracket']:
            if game_num == last_game_in_match.get(match_id, -1):
                series_wins = current_series_wins[match_id]
                series_winner_team = max(series_wins, key=series_wins.get)
                champions_history[cur_s_int] = series_winner_team
                
                champ_roster = team_rosters.get(series_winner_team, {}).get(match_season, set())
                for ign_raw in champ_roster:
                    ign = resolve_ign(ign_raw)
                    player_championship_wins[ign] = player_championship_wins.get(ign, 0) + 1

    # Attach Columns
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

    training_df['blue_rs_rank'] = b_rs_rank
    training_df['red_rs_rank']  = r_rs_rank
    training_df['blue_is_defending_champ'] = b_is_defending_champ
    training_df['red_is_defending_champ']  = r_is_defending_champ

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

    training_df['draft_style_sim'] = draft_style_sims
    for i in range(16):
        training_df[f'blue_draft_emb_{i}'] = blue_draft_embs[i]
        training_df[f'red_draft_emb_{i}'] = red_draft_embs[i]
        training_df[f'diff_draft_emb_{i}'] = diff_draft_embs[i]

    training_df['target_blue_win'] = (training_df['map_winner'] == training_df['blue_side_team']).astype(int)

    final_features = [
        'match_timestamp', 'match_id', 'season', 'game_number', 'patch_version',
        'blue_side_team', 'red_side_team',
        'blue_roster_stability', 'red_roster_stability', 'diff_roster_stability',
        'blue_draft_overlap', 'red_draft_overlap', 'diff_draft_overlap',
        'blue_rs_rank', 'red_rs_rank',
        'blue_is_defending_champ', 'red_is_defending_champ',
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
        'blue_avg_loss_duration', 'red_avg_loss_duration',
        'blue_execution_margin', 'red_execution_margin',
        'blue_execution_punish_score', 'red_execution_punish_score',
        'blue_lategame_winrate', 'red_lategame_winrate',
        'current_blue_side_advantage',
        'series_momentum_blue',
        'blue_g1_comfort', 'red_g1_comfort',
        'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
        'blue_prev_comfort', 'red_prev_comfort',
        'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
        'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red',
        'blue_comfort_patch_score', 'red_comfort_patch_score',
        'blue_patch_winrate', 'red_patch_winrate',
        'blue_patch_adaptation', 'red_patch_adaptation',
        'blue_expected_comfort', 'red_expected_comfort',
        'blue_prev_season_match_wr', 'red_prev_season_match_wr',
        'blue_prev_season_game_wr', 'red_prev_season_game_wr',
        'blue_prev_season_kda', 'red_prev_season_kda',
        'blue_prev_season_avg_kills', 'red_prev_season_avg_kills',
        'blue_prev_season_avg_deaths', 'red_prev_season_avg_deaths',
        'blue_prev_season_avg_assists', 'red_prev_season_avg_assists',
        'diff_prev_season_match_wr', 'diff_prev_season_game_wr',
        'diff_prev_season_kda',
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
        'time_weight', 'target_blue_win', 'stage'
    ]

    final_matrix = training_df[final_features].dropna()
    os.makedirs('csv_data', exist_ok=True)
    final_matrix.to_csv('csv_data/ML_Feature_Matrix.csv', index=False)
    
    print("✅ Feature Matrix Re-generation Complete!")
    print(f"   Saved {len(final_matrix)} rows to 'csv_data/ML_Feature_Matrix.csv'.")
    print(f"   Features: {len(final_features) - 7} predictive signals compiled successfully.")

if __name__ == '__main__':
    main()
