import json

# This script rebuilds Prediction_v1.ipynb from the current local pipeline.
# Accuracy should be checked after each data or feature change.

def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

# --- CELL 1: Setup ---
cell1_setup = r"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1. ROBUST DATA LOADING
# ----------------------------------------------------
def resolve_path(filename):
    possible_paths = [
        os.path.join('csv_data', filename),
        os.path.join('..', 'csv_data', filename),
        os.path.join('Predictive analysis practice', 'csv_data', filename)
    ]
    for path in possible_paths:
        if os.path.exists(path): return path
    raise FileNotFoundError(f"Could not find {filename}")

print("Loading datasets...")
df_main = pd.read_csv(resolve_path('ML_Feature_Matrix.csv'))
games_df = pd.read_csv(resolve_path('games.csv'))
rosters_df = pd.read_csv(resolve_path('season_rosters.csv'))
matches_raw = pd.read_csv(resolve_path('matches.csv'))

# Preprocess team names
TEAM_ALIAS_MAP = {
    "Aether Main": "Team Falcons PH", "Aether Valkyrie": "Team Falcons PH",
    "Sunsparks": "Team Liquid PH", "ECHO": "Team Liquid PH",
    "Execration": "Omega Esports", "SGD Omega": "Omega Esports",
    "ONIC Esports PH": "ONIC PH", "Onic Philippines": "ONIC PH",
    "Fnatic ONIC PH": "ONIC PH", "Work-Auster Force": "TNC Pro Team",
}

def resolve_team_name(team_name, season=None):
    t = str(team_name).strip()
    if t == "Aurora" and season is not None:
        try:
            s_int = int(str(season).replace('S', ''))
            if s_int <= 10: return "Team Liquid PH"
        except: pass
    if t == "AP.Bren" and season is not None:
        try:
            s_int = int(str(season).replace('S', ''))
            if s_int < 15: return "Team Falcons PH"
        except: pass
    elif t == "Falcons AP.Bren" or t == "FC AP.Bren": return "Team Falcons PH"
    return TEAM_ALIAS_MAP.get(t, t)

# Resolve names for Elo only
for target in [games_df, rosters_df]:
    for col in ['blue_side_team', 'red_side_team', 'team_name', 'map_winner']:
        if col in target.columns:
            target[col] = target.apply(lambda r: resolve_team_name(r[col], r.get('season')), axis=1)

rosters_df['season'] = rosters_df['season'].astype(str).str.strip()
matches_df = pd.merge(games_df, matches_raw[['match_id', 'stage']], on='match_id', how='inner')
matches_df['season'] = matches_df['season'].astype(str).str.strip()
"""

# --- CELL 2: Elo ---
cell2_elo = r"""
# ----------------------------------------------------
# 2. ON-THE-FLY ELO RE-COMPUTATION
# ----------------------------------------------------
print("Re-computing Elo ratings...")
team_rosters = {}
for _, row in rosters_df.iterrows():
    try: team_rosters.setdefault(row['team_name'], {})[row['season']] = set([p['ign'].strip().lower() for p in json.loads(row['players'])])
    except: pass

k_reg, k_play, decay_rate = 24, 64, 0.15
player_elos, player_p_elos = {}, {}
blue_elos, red_elos, b_p_elos, r_p_elos = [], [], [], []
current_season = None
pending_updates, pending_p_updates = {}, {}

sorted_matches = matches_df.sort_values('match_timestamp').copy()
last_game = sorted_matches.groupby('match_id')['game_number'].max().to_dict()

def get_elo(team, season, elos):
    r = team_rosters.get(team, {}).get(season, set())
    return sum([elos.get(i, 1500) for i in r]) / len(r) if r else 1500

for _, row in sorted_matches.iterrows():
    m_id, s, b_t, r_t, win, g_n, st = row['match_id'], row['season'], row['blue_side_team'], row['red_side_team'], row['map_winner'], row['game_number'], str(row['stage']).lower()
    is_p = any(x in st for x in ['playoff', 'grand final', 'upper', 'lower'])
    if s != current_season:
        if current_season:
            for i in player_elos: player_elos[i] = 1500 + (player_elos[i]-1500)*(1-decay_rate)
            for i in player_p_elos: player_p_elos[i] = 1500 + (player_p_elos[i]-1500)*(1-decay_rate)
        current_season = s
    e_a, e_b = get_elo(b_t, s, player_elos), get_elo(r_t, s, player_elos)
    pe_a, pe_b = get_elo(b_t, s, player_p_elos), get_elo(r_t, s, player_p_elos)
    blue_elos.append(e_a); red_elos.append(e_b); b_p_elos.append(pe_a); r_p_elos.append(pe_b)
    exp_a = 1 / (1 + 10**((e_b-e_a)/400)); act_a = 1 if win == b_t else 0
    k = k_play if is_p else k_reg
    for i in team_rosters.get(b_t, {}).get(s, []): pending_updates[i] = pending_updates.get(i,0) + k*(act_a-exp_a)
    for i in team_rosters.get(r_t, {}).get(s, []): pending_updates[i] = pending_updates.get(i,0) + k*((1-act_a)-(1-exp_a))
    if is_p:
        pexp_a = 1 / (1 + 10**((pe_b-pe_a)/400))
        for i in team_rosters.get(b_t, {}).get(s, []): pending_p_updates[i] = pending_p_updates.get(i,0) + k_play*(act_a-pexp_a)
        for i in team_rosters.get(r_t, {}).get(s, []): pending_p_updates[i] = pending_p_updates.get(i,0) + k_play*((1-act_a)-(1-pexp_a))
    if g_n == last_game.get(m_id):
        for i, d in pending_updates.items(): player_elos[i] = player_elos.get(i, 1500) + d
        for i, d in pending_p_updates.items(): player_p_elos[i] = player_p_elos.get(i, 1500) + d
        pending_updates.clear(); pending_p_updates.clear()

sorted_matches['blue_side_elo_new'] = blue_elos; sorted_matches['red_side_elo_new'] = red_elos
sorted_matches['blue_playoff_elo_new'] = b_p_elos; sorted_matches['red_playoff_elo_new'] = r_p_elos

df_main['match_timestamp'] = pd.to_datetime(df_main['match_timestamp'])
df = df_main.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)
df = df.merge(sorted_matches[['match_id', 'game_number', 'blue_side_elo_new', 'red_side_elo_new', 'blue_playoff_elo_new', 'red_playoff_elo_new']], on=['match_id', 'game_number'], how='left')
df['blue_side_elo'] = df['blue_side_elo_new'].fillna(df['blue_side_elo'])
df['red_side_elo'] = df['red_side_elo_new'].fillna(df['red_side_elo'])
df['blue_playoff_elo'] = df['blue_playoff_elo_new'].fillna(df['blue_playoff_elo'])
df['red_playoff_elo'] = df['red_playoff_elo_new'].fillna(df['red_playoff_elo'])
"""

# --- CELL 3: Features ---
cell3_features = r"""
# ----------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------
print("Engineering features...")
df['momentum_x_side_advantage'] = df['series_momentum_blue'] * df['current_blue_side_advantage']

diff_map = {
    'roster_stability': 'diff_roster_stability', 'patch_practice': 'diff_patch_practice',
    'g3_clutch_wr': 'diff_g3_clutch', 'reverse_sweep_rate': 'diff_reverse_sweep',
    'rest_factor': 'diff_rest', 'playoff_exp': 'diff_playoff_exp',
    'championship_dna': 'diff_championship_dna', 'playoff_winrate': 'diff_playoff_winrate',
    'avg_win_duration': 'diff_avg_win_duration', 'avg_loss_duration': 'diff_avg_loss_duration',
    'lategame_winrate': 'diff_lategame_winrate', 'prev_season_match_wr': 'diff_prev_season_match_wr',
    'prev_season_game_wr': 'diff_prev_season_game_wr', 'prev_season_kda': 'diff_prev_season_kda',
    'draft_exhaustion': 'diff_draft_exhaustion'
}
for base, target in diff_map.items():
    if f'blue_{base}' in df.columns: df[target] = df[f'blue_{base}'] - df[f'red_{base}']
df['diff_rs_rank'] = (df['blue_rs_rank'] - df['red_rs_rank']) * df['is_playoffs']

# Feature lists
base_features_dense = [
    'blue_roster_stability', 'red_roster_stability', 'diff_roster_stability',
    'blue_side_elo', 'red_side_elo', 'blue_playoff_elo', 'red_playoff_elo',
    'blue_championship_dna', 'red_championship_dna', 'blue_playoff_winrate', 'red_playoff_winrate',
    'blue_momentum', 'red_momentum', 'blue_h2h_winrate', 'blue_patch_practice', 'red_patch_practice',
    'diff_patch_practice', 'diff_g3_clutch', 'diff_reverse_sweep', 'diff_rest', 'diff_rs_rank',
    'blue_is_defending_champ', 'red_is_defending_champ', 'is_playoffs', 'blue_g3_clutch_wr', 'red_g3_clutch_wr',
    'blue_reverse_sweep_rate', 'red_reverse_sweep_rate', 'blue_rest_factor', 'red_rest_factor',
    'blue_avg_win_duration', 'red_avg_win_duration', 'current_blue_side_advantage',
    'blue_comfort_patch_score', 'red_comfort_patch_score', 'blue_expected_comfort', 'red_expected_comfort',
    'blue_prev_season_match_wr', 'red_prev_season_match_wr', 'blue_prev_season_game_wr', 'red_prev_season_game_wr',
    'blue_prev_season_kda', 'red_prev_season_kda', 'blue_prev_season_avg_kills', 'red_prev_season_avg_kills',
    'blue_prev_season_avg_deaths', 'red_prev_season_avg_deaths', 'blue_prev_season_avg_assists', 'red_prev_season_avg_assists',
    'diff_prev_season_match_wr', 'diff_prev_season_game_wr', 'diff_prev_season_kda',
]

g1_features = base_features_dense + ['draft_style_sim', 'blue_playoff_exp', 'red_playoff_exp', 'diff_playoff_exp']

series_features = ['series_momentum_blue', 'prev_stomp_margin', 'is_side_swap']
pre_match_series_features = ['blue_draft_exhaustion', 'red_draft_exhaustion', 'diff_draft_exhaustion', 'blue_prev_winner_exhaustion', 'red_prev_winner_exhaustion', 'blue_g1_comfort', 'red_g1_comfort', 'blue_prev_comfort', 'red_prev_comfort']

g2p_features = base_features_dense + series_features + pre_match_series_features + ['draft_style_sim', 'momentum_x_side_advantage', 'blue_playoff_exp', 'red_playoff_exp', 'diff_playoff_exp']

# Split matches chronologically
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)
match_start = df.groupby('match_id')['match_timestamp'].min().sort_values(kind='mergesort')
cutoff = match_start.iloc[int(len(match_start)*0.85)]
train_df = df[df['match_id'].isin(match_start[match_start < cutoff].index)].copy()
test_df = df[df['match_id'].isin(match_start[match_start >= cutoff].index)].copy()
"""

# --- CELL 4: Training ---
cell4_training = r"""
# ----------------------------------------------------
# 4. MODEL TRAINING
# ----------------------------------------------------
print("Training Game 1 XGBoost...")
g1_train = train_df[train_df['game_number'] == 1]; g1_test = test_df[test_df['game_number'] == 1]
g1_model = xgb.XGBClassifier(n_estimators=400, max_depth=2, learning_rate=0.02, subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, eval_metric='logloss', verbosity=0)
g1_model.fit(g1_train[g1_features], g1_train['target_blue_win'])

print("Training Game 2+ Ensemble (CatBoost + RF Blend)...")
g2p_train = train_df[train_df['game_number'] > 1]; g2p_test = test_df[test_df['game_number'] > 1]
c1 = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, subsample=0.9, l2_leaf_reg=3.0, random_seed=42, verbose=0)
rf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=42)
c2 = CatBoostClassifier(iterations=50, depth=4, learning_rate=0.05, subsample=0.9, l2_leaf_reg=3.0, random_seed=42, verbose=0)

for m in [c1, rf, c2]: m.fit(g2p_train[g2p_features], g2p_train['target_blue_win'], sample_weight=g2p_train['time_weight'])

# Final Evaluation
p1 = g1_model.predict_proba(g1_test[g1_features])[:, 1]
p2 = (1*c1.predict_proba(g2p_test[g2p_features])[:,1] + 3*rf.predict_proba(g2p_test[g2p_features])[:,1] + 2*c2.predict_proba(g2p_test[g2p_features])[:,1]) / 6
acc_g1, acc_g2 = accuracy_score(g1_test['target_blue_win'], (p1>=0.5).astype(int)), accuracy_score(g2p_test['target_blue_win'], (p2>=0.5).astype(int))
comb_acc = accuracy_score(pd.concat([g1_test['target_blue_win'], g2p_test['target_blue_win']]), np.concatenate([(p1>=0.5).astype(int), (p2>=0.5).astype(int)]))

print("\n" + "="*40)
print(f"Game 1 Accuracy: {acc_g1:.2%}")
print(f"Game 2+ Accuracy: {acc_g2:.2%}")
print(f"FINAL COMBINED ACCURACY: {comb_acc:.2%}")
print("="*40)
"""

# --- CELL 5: Post-Draft ---
cell5_post_draft = r"""
# ----------------------------------------------------
# 5. POST-DRAFT BENCHMARK (FOR ANALYSIS ONLY)
# ----------------------------------------------------
print("Evaluating Post-Draft Benchmark...")
draft_features = ['blue_synergy', 'red_synergy', 'blue_counter', 'red_counter', 'blue_buffs_in_draft', 'blue_nerfs_in_draft', 'red_buffs_in_draft', 'red_nerfs_in_draft']
X_pd_train = g2p_train[g2p_features + draft_features]
X_pd_test = g2p_test[g2p_features + draft_features]
pd_model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
pd_model.fit(X_pd_train, g2p_train['target_blue_win'], sample_weight=g2p_train['time_weight'])
pd_acc = accuracy_score(g2p_test['target_blue_win'], pd_model.predict(X_pd_test))
print(f"Post-Draft Game 2+ Accuracy: {pd_acc:.2%} (vs {acc_g2:.2%} pre-match)")
"""

# --- CELL 6: Simulator ---
cell6_simulator = r"""
# ----------------------------------------------------
# 6. SERIES SIMULATOR
# ----------------------------------------------------
def get_recent_stats(team_name):
    t = resolve_team_name(team_name)
    team_games = df[(df['blue_side_team']==t)|(df['red_side_team']==t)]
    if team_games.empty: return None
    last = team_games.iloc[-1]
    side = 'blue' if last['blue_side_team'] == t else 'red'
    return {col.replace(f'{side}_', ''): last[col] for col in df.columns if col.startswith(f'{side}_')}

def predict_matchup(blue_team, red_team, game_num=1, b_won_last=None):
    b_stats, r_stats = get_recent_stats(blue_team), get_recent_stats(red_team)
    if not b_stats or not r_stats: return "Team not found"
    row = {}
    for f in g2p_features:
        if f.startswith('diff_'): base = f.replace('diff_', ''); row[f] = b_stats.get(base, 0) - r_stats.get(base, 0)
        elif f.startswith('blue_'): row[f] = b_stats.get(f.replace('blue_', ''), 0)
        elif f.startswith('red_'): row[f] = r_stats.get(f.replace('red_', ''), 0)
        else: row[f] = 0
    row['series_momentum_blue'] = 0.5 if b_won_last is None else (1.0 if b_won_last else 0.0)
    row['momentum_x_side_advantage'] = row['series_momentum_blue'] * df['current_blue_side_advantage'].iloc[-1]
    row['is_playoffs'] = 1
    inp = pd.DataFrame([row])
    if game_num == 1: pr = g1_model.predict_proba(inp[g1_features])[0, 1]
    else: pr = (1*c1.predict_proba(inp[g2p_features])[0,1] + 3*rf.predict_proba(inp[g2p_features])[0,1] + 2*c2.predict_proba(inp[g2p_features])[0,1]) / 6
    print(f"{blue_team} ({pr:.1%}) vs {red_team} ({1-pr:.1%}) -> Predicted: {'Blue' if pr>0.5 else 'Red'}")

print("\nSample Predictions:")
predict_matchup("Team Liquid PH", "Team Falcons PH")
predict_matchup("ONIC PH", "RSG PH")
"""

notebook = {
    "cells": [
        markdown_cell("# MPL PH Prediction Pipeline\nCurrent experimental notebook generated from `create_prediction_v1_tuned.py`."),
        code_cell(cell1_setup),
        code_cell(cell2_elo),
        code_cell(cell3_features),
        code_cell(cell4_training),
        code_cell(cell5_post_draft),
        code_cell(cell6_simulator)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4, "nbformat_minor": 2,
}

with open("1_NoteBook/Prediction_v1.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Prediction_v1.ipynb rebuilt.")
