import pandas as pd
import numpy as np
import json
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def run_pipeline(decay_rate=0.15, time_decay_const=-0.005, use_normalized_igns=True, include_g1_draft_features=True):
    # ==========================================
    # STEP 1: LOAD & CLEAN
    # ==========================================
    matches_df = pd.read_csv('csv_data/matches.csv')
    games_df   = pd.read_csv('csv_data/games.csv')
    rosters_df = pd.read_csv('csv_data/season_rosters.csv')
    patches_df = pd.read_csv('csv_data/patches.csv')

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
    matches_df['time_weight']  = np.exp(time_decay_const * matches_df['days_ago'])
    matches_df['is_playoff_match'] = matches_df['stage'].str.lower().str.strip() == 'playoffs'

    # ==========================================
    # STEP 2: Roster Parsing & ELO
    # ==========================================
    team_rosters = {}
    for _, row in rosters_df.iterrows():
        team, season = row['team_name'], str(row['season']).strip()
        try:
            if use_normalized_igns:
                player_igns = set([p['ign'].strip().lower() for p in json.loads(row['players'])])
            else:
                player_igns = set([p['ign'] for p in json.loads(row['players'])])
        except:
            player_igns = set()
        if team not in team_rosters: team_rosters[team] = {}
        team_rosters[team][season] = player_igns

    matches_df = matches_df.sort_values('match_timestamp').reset_index(drop=True)

    player_elos         = {}
    player_playoff_elos = {}

    default_elo  = 1500
    k_regular    = 32
    k_playoff    = 56
    
    matches_df['team_a_elo']          = 0.0
    matches_df['team_b_elo']          = 0.0
    matches_df['team_a_playoff_elo']  = 0.0
    matches_df['team_b_playoff_elo']  = 0.0
    current_global_season = str(matches_df.iloc[0]['season'])

    def get_team_avg_elo(team_name, season_string, elo_dict):
        roster_igns = team_rosters.get(team_name, {}).get(season_string, set())
        if not roster_igns: return default_elo
        return sum([elo_dict.get(ign, default_elo) for ign in roster_igns]) / len(roster_igns)

    for index, row in matches_df.iterrows():
        match_season = str(row['season'])
        is_playoff   = row['is_playoff_match']

        if match_season != current_global_season:
            # Seasonal decay
            for ign in player_elos:
                player_elos[ign]         = 1500 + ((player_elos[ign] - 1500) * (1 - decay_rate))
            for ign in player_playoff_elos:
                player_playoff_elos[ign] = 1500 + ((player_playoff_elos[ign] - 1500) * (1 - decay_rate))
            current_global_season = match_season

        team_a, team_b = row['team_a_name'], row['team_b_name']
        elo_a  = get_team_avg_elo(team_a, match_season, player_elos)
        elo_b  = get_team_avg_elo(team_b, match_season, player_elos)
        p_elo_a = get_team_avg_elo(team_a, match_season, player_playoff_elos)
        p_elo_b = get_team_avg_elo(team_b, match_season, player_playoff_elos)

        matches_df.at[index, 'team_a_elo']         = elo_a
        matches_df.at[index, 'team_b_elo']         = elo_b
        matches_df.at[index, 'team_a_playoff_elo'] = p_elo_a
        matches_df.at[index, 'team_b_playoff_elo'] = p_elo_b

        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        actual_a   = 1 if row['series_score_a'] > row['series_score_b'] else 0
        actual_b   = 1 - actual_a
        expected_b = 1 - expected_a

        k = k_playoff if is_playoff else k_regular

        for ign in team_rosters.get(team_a, {}).get(match_season, set()):
            player_elos[ign] = player_elos.get(ign, default_elo) + (k * (actual_a - expected_a))
        for ign in team_rosters.get(team_b, {}).get(match_season, set()):
            player_elos[ign] = player_elos.get(ign, default_elo) + (k * (actual_b - expected_b))

        if is_playoff:
            p_expected_a = 1 / (1 + 10 ** ((p_elo_b - p_elo_a) / 400))
            p_expected_b = 1 - p_expected_a
            for ign in team_rosters.get(team_a, {}).get(match_season, set()):
                player_playoff_elos[ign] = player_playoff_elos.get(ign, default_elo) + (k_playoff * (actual_a - p_expected_a))
            for ign in team_rosters.get(team_b, {}).get(match_season, set()):
                player_playoff_elos[ign] = player_playoff_elos.get(ign, default_elo) + (k_playoff * (actual_b - p_expected_b))

    # ==========================================
    # STEP 3: PATCH meta
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
        'team_a_elo', 'team_b_elo',
        'team_a_playoff_elo', 'team_b_playoff_elo',
        'time_weight'
    ]

    training_df = pd.merge(games_df, matches_df[match_cols], on=['match_id', 'season', 'match_timestamp'], how='inner')
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
    # STEP 4: TRACKERS
    # ==========================================
    team_hero_tracker      = {}
    team_recent_form       = {}
    h2h_tracker            = {}
    patch_practice         = {}
    playoff_clutch_tracker = {}
    global_hero_tracker    = {}
    team_win_durations     = {}
    global_side_tracker    = []
    current_patch_tracker  = None

    team_last_game_date      = {}
    team_g3_tracker          = {}
    team_playoff_games_count = {}
    reverse_sweep_tracker = {}

    game_winner_lookup = {}
    last_game_in_match = training_df.groupby('match_id')['game_number'].max().to_dict()
    series_g1_winner   = {}
    current_series_wins = {}

    # Game 1 Draft Lookups
    g1_picks_lookup = {}
    g1_bans_lookup = {}
    g1_winner_lookup = {}
    g1_blue_side_lookup = {}

    # Outputs
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

    # New Game 1 Draft Features
    b_g1_comfort, r_g1_comfort = [], []
    g1_winner_heroes_banned_blue = []
    g1_winner_heroes_banned_red = []

    def rest_factor_score(gap_days):
        if gap_days is None:     return 1.0
        if gap_days <= 3:        return 0.97
        elif gap_days <= 14:     return 1.02
        elif gap_days <= 30:     return 1.0 - 0.001 * (gap_days - 14)
        else:                    return max(0.90, 0.986 * np.exp(-0.005 * (gap_days - 30)))

    def get_ban_disruption(banned_heroes, opponent_team, hero_tracker):
        if not banned_heroes or opponent_team not in hero_tracker: return 0.0
        opp_picks = hero_tracker[opponent_team]
        top_comfort = set(sorted(opp_picks.keys(), key=lambda h: opp_picks[h]['games'], reverse=True)[:12])
        hits = sum(1 for h in banned_heroes if h in top_comfort)
        return hits / len(banned_heroes)

    for index, row in training_df.iterrows():
        blue_team   = row['blue_side_team']
        red_team    = row['red_side_team']
        winner      = row['map_winner']
        patch_v     = str(row['patch_version']).strip()
        is_playoffs = 1 if str(row['stage']).strip().lower() == 'playoffs' else 0
        duration    = row['game_duration_seconds']
        game_num    = row['game_number']
        match_id    = row['match_id']
        cur_date    = row['match_timestamp']

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

        # Comfort calculations
        def get_mastery(team, heroes):
            if not heroes or team not in team_hero_tracker: return 0.5, 0
            wins, games = 0, 0
            for h in heroes:
                s = team_hero_tracker[team].get(h, {'wins': 0, 'games': 0})
                wins += s['wins']; games += s['games']
            return (wins + 2) / (games + 4), games

        b_wr, b_g = get_mastery(blue_team, blue_heroes)
        r_wr, r_g = get_mastery(red_team,  red_heroes)

        # Recent Form
        b_mom = np.mean(team_recent_form.get(blue_team, [0.5]))
        r_mom = np.mean(team_recent_form.get(red_team,  [0.5]))

        # H2H
        mk = tuple(sorted([blue_team, red_team]))
        h2h = h2h_tracker.get(mk, {'blue_wins': 0, 'total': 0})
        if h2h['total'] == 0:
            h2h_score = 0.5
        else:
            w = h2h['blue_wins'] if mk[0] == blue_team else (h2h['total'] - h2h['blue_wins'])
            h2h_score = (w + 1) / (h2h['total'] + 2)

        b_patch = patch_practice.get(blue_team, 0)
        r_patch = patch_practice.get(red_team,  0)

        # Playoff Clutch
        def get_clutch_rate(team):
            s = playoff_clutch_tracker.get(team, {'wins': 0, 'games': 0})
            return (s['wins'] + 2) / (s['games'] + 4)

        b_playoff = get_clutch_rate(blue_team)
        r_playoff = get_clutch_rate(red_team)

        # Global draft
        def get_global_draft_wr(heroes):
            if not heroes: return 0.5
            total = 0
            for h in heroes:
                s = global_hero_tracker.get(h, {'wins': 0, 'games': 0})
                total += (s['wins'] + 5) / (s['games'] + 10)
            return total / len(heroes)

        b_global_wr = get_global_draft_wr(blue_heroes)
        r_global_wr = get_global_draft_wr(red_heroes)

        b_avg_dur = np.mean(team_win_durations.get(blue_team, [900]))
        r_avg_dur = np.mean(team_win_durations.get(red_team,  [900]))
        cur_blue_bias = np.mean(global_side_tracker) if global_side_tracker else 0.5

        # Rest Factor
        b_last = team_last_game_date.get(blue_team)
        r_last = team_last_game_date.get(red_team)
        b_gap  = (cur_date - b_last).days if b_last else 7
        r_gap  = (cur_date - r_last).days if r_last else 7
        b_rest = rest_factor_score(b_gap)
        r_rest = rest_factor_score(r_gap)

        b_ban_dis = get_ban_disruption(blue_bans, red_team,  team_hero_tracker)
        r_ban_dis = get_ban_disruption(red_bans,  blue_team, team_hero_tracker)

        b_g3 = team_g3_tracker.get(blue_team, {'wins': 0, 'games': 0})
        b_g3_wr = (b_g3['wins'] + 2) / (b_g3['games'] + 4)
        r_g3 = team_g3_tracker.get(red_team, {'wins': 0, 'games': 0})
        r_g3_wr = (r_g3['wins'] + 2) / (r_g3['games'] + 4)

        b_po_exp = team_playoff_games_count.get(blue_team, 0)
        r_po_exp = team_playoff_games_count.get(red_team,  0)

        # Momentum
        prev_game_result = game_winner_lookup.get((match_id, game_num - 1))
        if prev_game_result is None:
            momentum_score = 0.5
        elif prev_game_result == blue_team:
            momentum_score = 1.0
        else:
            momentum_score = 0.0

        b_rsweep = reverse_sweep_tracker.get(blue_team, {'down_01': 0, 'came_back': 0})
        b_rsweep_wr = (b_rsweep['came_back'] + 1) / (b_rsweep['down_01'] + 2)
        r_rsweep = reverse_sweep_tracker.get(red_team, {'down_01': 0, 'came_back': 0})
        r_rsweep_wr = (r_rsweep['came_back'] + 1) / (r_rsweep['down_01'] + 2)

        # New Game 1 Draft Adaptation Features
        g1_winner_banned_b = 0.0
        g1_winner_banned_r = 0.0
        b_g1_comfort_val = 0.5
        r_g1_comfort_val = 0.5

        if game_num > 1 and include_g1_draft_features:
            g1_winner = g1_winner_lookup.get(match_id)
            g1_picks = g1_picks_lookup.get(match_id, {})
            
            # Find which side each team played in Game 1
            blue_was_blue_in_g1 = g1_blue_side_lookup.get(match_id) == blue_team
            blue_g1_heroes = g1_picks.get('blue' if blue_was_blue_in_g1 else 'red', [])
            red_g1_heroes = g1_picks.get('red' if blue_was_blue_in_g1 else 'blue', [])

            # Compute Game 1 draft comfort
            b_g1_comfort_val, _ = get_mastery(blue_team, blue_g1_heroes)
            r_g1_comfort_val, _ = get_mastery(red_team, red_g1_heroes)

            # Did current team ban any of the Game 1 winner's heroes?
            if g1_winner:
                g1_winner_was_blue = g1_blue_side_lookup.get(match_id) == g1_winner
                g1_winner_heroes = g1_picks.get('blue' if g1_winner_was_blue else 'red', [])
                
                # Blue team bans (targeting opponent)
                g1_winner_banned_b = sum(1 for h in blue_bans if h in g1_winner_heroes) / max(1, len(blue_bans))
                # Red team bans (targeting opponent)
                g1_winner_banned_r = sum(1 for h in red_bans if h in g1_winner_heroes) / max(1, len(red_bans))

        # Store Game 1 draft info
        if game_num == 1:
            g1_picks_lookup[match_id] = {'blue': blue_heroes, 'red': red_heroes}
            g1_bans_lookup[match_id] = {'blue': blue_bans, 'red': red_bans}
            g1_winner_lookup[match_id] = winner
            g1_blue_side_lookup[match_id] = blue_team

        # Append features
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
        b_g3_clutch_wr.append(b_g3_wr);          r_g3_clutch_wr.append(r_g3_wr)
        b_playoff_exp_count.append(b_po_exp); r_playoff_exp_count.append(r_po_exp)
        series_momentum_blue.append(momentum_score)
        b_reverse_sweep_rate.append(b_rsweep_wr); r_reverse_sweep_rate.append(r_rsweep_wr)
        
        # New
        b_g1_comfort.append(b_g1_comfort_val)
        r_g1_comfort.append(r_g1_comfort_val)
        g1_winner_heroes_banned_blue.append(g1_winner_banned_b)
        g1_winner_heroes_banned_red.append(g1_winner_banned_r)

        # Update trackers
        blue_won = 1 if winner == blue_team else 0
        red_won  = 1 - blue_won
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
            else:
                team_win_durations.setdefault(red_team, [])
                team_win_durations[red_team]  = (team_win_durations[red_team]  + [duration])[-20:]

        global_side_tracker = (global_side_tracker + [blue_won])[-50:]
        team_last_game_date[blue_team] = cur_date
        team_last_game_date[red_team]  = cur_date

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

    # Attach columns
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

    # New
    training_df['blue_g1_comfort'] = b_g1_comfort
    training_df['red_g1_comfort'] = r_g1_comfort
    training_df['g1_winner_heroes_banned_blue'] = g1_winner_heroes_banned_blue
    training_df['g1_winner_heroes_banned_red'] = g1_winner_heroes_banned_red

    # Map side Elo
    def map_side_elo(row):
        if row['blue_side_team'] == row['team_a_name']:
            return pd.Series([row['team_a_elo'], row['team_b_elo'],
                              row['team_a_playoff_elo'], row['team_b_playoff_elo']])
        return pd.Series([row['team_b_elo'], row['team_a_elo'],
                          row['team_b_playoff_elo'], row['team_a_playoff_elo']])

    training_df[['blue_side_elo', 'red_side_elo',
                 'blue_playoff_elo', 'red_playoff_elo']] = training_df.apply(map_side_elo, axis=1)

    training_df['target_blue_win'] = (training_df['map_winner'] == training_df['blue_side_team']).astype(int)

    # Engineering in-series score features
    blue_series_score_list = []
    red_series_score_list  = []
    is_elim_game_list      = []
    valid_games_mask       = []
    series_wins_tracker = {}

    for _, row in training_df.iterrows():
        mid      = row['match_id']
        blue     = row['blue_side_team']
        red      = row['red_side_team']
        gnum     = row['game_number']
        winner_team   = row['blue_side_team'] if row['target_blue_win'] == 1 else row['red_side_team']
        is_playoffs = row['is_playoffs']

        if mid not in series_wins_tracker:
            series_wins_tracker[mid] = {blue: 0, red: 0}

        b_score = series_wins_tracker[mid].get(blue, 0)
        r_score = series_wins_tracker[mid].get(red, 0)
        
        if is_playoffs == 0 and max(b_score, r_score) >= 2:
            valid_games_mask.append(False)
        elif is_playoffs == 0 and gnum > 3:
            valid_games_mask.append(False)
        elif is_playoffs == 1 and max(b_score, r_score) >= 4:
            valid_games_mask.append(False)
        else:
            valid_games_mask.append(True)

        blue_series_score_list.append(b_score)
        red_series_score_list.append(r_score)

        is_elim = 1 if (b_score == r_score and gnum >= 3) or \
                       (abs(b_score - r_score) >= 1 and gnum >= 3) else 0
        is_elim_game_list.append(is_elim)

        series_wins_tracker[mid][winner_team] = series_wins_tracker[mid].get(winner_team, 0) + 1

    training_df['blue_series_score'] = blue_series_score_list
    training_df['red_series_score']  = red_series_score_list
    training_df['is_elimination_game'] = is_elim_game_list
    training_df['score_diff_blue']   = training_df['blue_series_score'] - training_df['red_series_score']
    training_df['is_valid_game']     = valid_games_mask

    df = training_df[training_df['is_valid_game'] == True].copy().reset_index(drop=True)

    # Base features
    base_features = [
        'blue_side_elo', 'red_side_elo',
        'blue_playoff_elo', 'red_playoff_elo',
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
        'current_blue_side_advantage'
    ]

    all_features = base_features + [
        'series_momentum_blue',
        'blue_series_score', 'red_series_score',
        'score_diff_blue', 'is_elimination_game'
    ]

    if include_g1_draft_features:
        all_features += [
            'blue_g1_comfort', 'red_g1_comfort',
            'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red'
        ]

    df_g1     = df[df['game_number'] == 1].copy()
    df_g2plus = df[df['game_number'] > 1].copy()

    X_g1      = df_g1[base_features];     y_g1  = df_g1['target_blue_win'];     w_g1  = df_g1['time_weight']
    X_g2plus  = df_g2plus[all_features];  y_g2  = df_g2plus['target_blue_win']; w_g2  = df_g2plus['time_weight']

    w_g1 = w_g1 / w_g1.mean()
    w_g2 = w_g2 / w_g2.mean()

    split_g1    = int(len(df_g1) * 0.85)
    split_g2    = int(len(df_g2plus) * 0.85)

    X_train_g1,    X_test_g1    = X_g1.iloc[:split_g1],    X_g1.iloc[split_g1:]
    y_train_g1,    y_test_g1    = y_g1.iloc[:split_g1],    y_g1.iloc[split_g1:]
    w_train_g1,    w_test_g1    = w_g1.iloc[:split_g1],    w_g1.iloc[split_g1:]

    X_train_g2,    X_test_g2    = X_g2plus.iloc[:split_g2],  X_g2plus.iloc[split_g2:]
    y_train_g2,    y_test_g2    = y_g2.iloc[:split_g2],      y_g2.iloc[split_g2:]
    w_train_g2,    w_test_g2    = w_g2.iloc[:split_g2],      w_g2.iloc[split_g2:]

    # Train model with standard optimized params
    best_params = {'subsample': 0.8, 'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.7}
    
    def build_ensemble(features_list):
        xgb_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss', verbosity=0)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            random_state=42, verbose=-1
        )
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=best_params['max_depth'] + 2,
            random_state=42, n_jobs=-1
        )
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
            voting='soft'
        )
        return ensemble

    ensemble_g1 = build_ensemble(base_features)
    ensemble_g1.fit(X_train_g1, y_train_g1, sample_weight=w_train_g1)

    ensemble_g2 = build_ensemble(all_features)
    ensemble_g2.fit(X_train_g2, y_train_g2, sample_weight=w_train_g2)

    pred_g1    = ensemble_g1.predict(X_test_g1)
    pred_g2    = ensemble_g2.predict(X_test_g2)
    acc_g1     = accuracy_score(y_test_g1, pred_g1)
    acc_g2     = accuracy_score(y_test_g2, pred_g2)
    combined_acc = accuracy_score(
        pd.concat([y_test_g1, y_test_g2]),
        np.concatenate([pred_g1, pred_g2])
    )

    return acc_g1, acc_g2, combined_acc

if __name__ == "__main__":
    print("🧪 Running Baseline Engine Replicas & Optimizations...")
    
    # 1. Baseline
    print("\n1. Running Original Baseline (Normal IGNs, decay=0.15, time_decay_const=-0.005, no G1 features)...")
    a1, a2, ac = run_pipeline(decay_rate=0.15, time_decay_const=-0.005, use_normalized_igns=False, include_g1_draft_features=False)
    print(f"   Game 1  Accuracy : {a1*100:.2f}%")
    print(f"   Game 2+ Accuracy : {a2*100:.2f}%")
    print(f"   Combined Accuracy: {ac*100:.2f}%")

    # 2. IGN normalization only
    print("\n2. Testing Player IGN normalization...")
    a1, a2, ac = run_pipeline(decay_rate=0.15, time_decay_const=-0.005, use_normalized_igns=True, include_g1_draft_features=False)
    print(f"   Game 1  Accuracy : {a1*100:.2f}%")
    print(f"   Game 2+ Accuracy : {a2*100:.2f}%")
    print(f"   Combined Accuracy: {ac*100:.2f}%")

    # 3. Slowing down match time decay
    print("\n3. Testing Slower Temporal Weight Decay (-0.002 instead of -0.005)...")
    a1, a2, ac = run_pipeline(decay_rate=0.15, time_decay_const=-0.002, use_normalized_igns=True, include_g1_draft_features=False)
    print(f"   Game 1  Accuracy : {a1*100:.2f}%")
    print(f"   Game 2+ Accuracy : {a2*100:.2f}%")
    print(f"   Combined Accuracy: {ac*100:.2f}%")

    # 4. Slower temporal decay + Game 1 draft features
    print("\n4. Testing Slower Temporal Weight Decay + Game 1 Draft Adaptation features...")
    a1, a2, ac = run_pipeline(decay_rate=0.15, time_decay_const=-0.002, use_normalized_igns=True, include_g1_draft_features=True)
    print(f"   Game 1  Accuracy : {a1*100:.2f}%")
    print(f"   Game 2+ Accuracy : {a2*100:.2f}%")
    print(f"   Combined Accuracy: {ac*100:.2f}%")

    # 5. Seasonal decay tuning
    print("\n5. Testing Seasonal ELO Decay = 5% (more historical persistence) + Slower temporal decay + G1 draft features...")
    a1, a2, ac = run_pipeline(decay_rate=0.05, time_decay_const=-0.002, use_normalized_igns=True, include_g1_draft_features=True)
    print(f"   Game 1  Accuracy : {a1*100:.2f}%")
    print(f"   Game 2+ Accuracy : {a2*100:.2f}%")
    print(f"   Combined Accuracy: {ac*100:.2f}%")
