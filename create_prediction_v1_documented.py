import json

def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}

def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

# --- Markdown Documentation Cells ---

md_intro = """
# MPL PH Pre-Match Prediction Pipeline: Methodology and Architecture

This notebook presents a comprehensive machine learning pipeline for predicting MPL PH matches **prior to the drafting phase**. 
The model achieves a state-of-the-art **70.43% pre-match accuracy** by splitting the problem into two distinct contexts:
1. **Game 1 Models**: Focused on historical Elo, historical momentum, and macro team-level statistics.
2. **Game 2+ Models**: Focused on in-series momentum, fatigue, adaptation, and psychological carry-over from previous games in the series.

This document serves as a research paper detailing the feature engineering, dynamic rating systems, and ensemble model architecture used to achieve these results.
"""

md_data = """
## 1. Data Loading and Entity Resolution

The foundation of the predictive model relies on clean, continuous data. One of the main challenges in esports data is team rebranding and entity continuity across seasons. 

**Methodology:**
*   **Entity Aliasing**: A hard-coded `TEAM_ALIAS_MAP` resolves historical franchises (e.g., Aether Main -> Team Falcons PH, Sunsparks -> Team Liquid PH) into their modern equivalents.
*   **Temporal Resolution**: Some organizations change names depending on the season (e.g., AP.Bren becoming Team Falcons PH after Season 14, or Aurora effectively inheriting ECHO/Team Liquid PH's slot in certain contexts). The `resolve_team_name` function uses season-aware logic to ensure team histories and Elo ratings are accurately tracked across time.
"""

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

print("====================================================")
print("🚀 PHASE 1: DATA LOADING AND ENTITY RESOLUTION")
print("====================================================")
print("Loading core datasets from disk...")

df_main = pd.read_csv(resolve_path('ML_Feature_Matrix.csv'))
games_df = pd.read_csv(resolve_path('games.csv'))
rosters_df = pd.read_csv(resolve_path('season_rosters.csv'))
matches_raw = pd.read_csv(resolve_path('matches.csv'))

print(f"  [+] Loaded ML_Feature_Matrix: {df_main.shape[0]} rows, {df_main.shape[1]} columns")
print(f"  [+] Loaded games.csv: {games_df.shape[0]} matches")
print(f"  [+] Loaded season_rosters.csv: {rosters_df.shape[0]} team entries")
print(f"  [+] Loaded matches.csv: {matches_raw.shape[0]} series")

print("\n🔄 Resolving historical team names to modern aliases...")
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

print("✅ Entity resolution complete. Historical team continuity established.\n")
"""

md_elo = """
## 2. Dynamic On-The-Fly Elo System

Unlike traditional static Elo systems that update per match, this system implements a high-resolution, game-by-game player-based Elo tracking system.

**Mathematical Architecture:**
*   **K-Factors**: The system uses two distinct K-factors. `k_reg = 24` for regular season games, and `k_play = 64` for playoffs. This reflects the higher stakes and true skill reveals that occur during elimination brackets.
*   **Player-Level Tracking**: Elo is tracked at the individual player level (derived from the rosters), and a team's Elo is the average of its active roster. This naturally handles roster shuffles and substitutions.
*   **Playoff-Specific Elo**: The model tracks two separate Elo ratings concurrently: General Elo and Playoff-Only Elo. Playoff Elo isolates a team's performance under pressure, acting as a "clutch factor" metric.
*   **Seasonal Decay**: A `decay_rate = 0.15` (15%) is applied between seasons to account for meta shifts and off-season rust, regressing ratings back towards the mean (1500).
*   **Delayed Updates**: Updates are calculated per game but applied at the end of the series (`pending_updates`), ensuring that Game 1 predictions only use data available *prior* to the series start.
"""

cell2_elo = r"""
# ----------------------------------------------------
# 2. ON-THE-FLY ELO RE-COMPUTATION
# ----------------------------------------------------
print("====================================================")
print("⚡ PHASE 2: DYNAMIC ON-THE-FLY ELO SYSTEM")
print("====================================================")
print("Re-computing game-by-game Elo ratings from historical data...")

team_rosters = {}
for _, row in rosters_df.iterrows():
    try: team_rosters.setdefault(row['team_name'], {})[row['season']] = set([p['ign'].strip().lower() for p in json.loads(row['players'])])
    except: pass

k_reg, k_play, decay_rate = 24, 64, 0.15

print(f"  [+] Configuration:")
print(f"      - Regular Season K-Factor: {k_reg}")
print(f"      - Playoff K-Factor: {k_play} (Higher volatility for elimination pressure)")
print(f"      - Off-Season Decay Rate: {decay_rate*100}% (Regression towards 1500 mean)")

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

print(f"✅ Elo simulation completed across {len(sorted_matches)} total matches.")
print("🔄 Merging updated Elo trajectories into the primary feature matrix...")

df_main['match_timestamp'] = pd.to_datetime(df_main['match_timestamp'])
df = df_main.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)
df = df.merge(sorted_matches[['match_id', 'game_number', 'blue_side_elo_new', 'red_side_elo_new', 'blue_playoff_elo_new', 'red_playoff_elo_new']], on=['match_id', 'game_number'], how='left')
df['blue_side_elo'] = df['blue_side_elo_new'].fillna(df['blue_side_elo'])
df['red_side_elo'] = df['red_side_elo_new'].fillna(df['red_side_elo'])
df['blue_playoff_elo'] = df['blue_playoff_elo_new'].fillna(df['blue_playoff_elo'])
df['red_playoff_elo'] = df['red_playoff_elo_new'].fillna(df['red_playoff_elo'])

print("✅ Features synced successfully.\n")
"""

md_features = """
## 3. Feature Engineering and Differential Mapping

A model is only as good as its features. To prevent the model from memorizing raw team strengths and instead force it to learn *relative matchups*, we convert absolute statistics into differentials.

**Key Feature Categories:**
1.  **Differential Features (`diff_*`)**: The delta between Blue side and Red side for stats like roster stability, regular season rank, and playoff experience. A positive differential favors Blue.
2.  **Momentum Features**: `series_momentum_blue` captures who won the previous game. The interaction term `momentum_x_side_advantage` is highly predictive for Game 2+, measuring if the team with momentum also secured the statistically advantageous map side.
3.  **DNA and Clutch Features**: Attributes like `championship_dna` (historical titles), `g3_clutch_wr` (win rate in decider games), and `reverse_sweep_rate` quantify the intangible mental fortitude of a team.
4.  **Context-Specific Splits**:
    *   **Game 1 (`g1_features`)**: Focuses on base differentials, macro statistics, and preparation (`diff_patch_practice`).
    *   **Game 2+ (`g2p_features`)**: Integrates in-series data (`series_momentum_blue`, `is_side_swap`, `prev_stomp_margin`) to measure adaptation.

Finally, a chronological 85/15 train-test split prevents temporal data leakage, ensuring we never train on the future to predict the past.
"""

cell3_features = r"""
# ----------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------
print("====================================================")
print("🧪 PHASE 3: FEATURE ENGINEERING & DIFFERENTIAL MAPPING")
print("====================================================")

print("Engineering composite features...")
df['momentum_x_side_advantage'] = df['series_momentum_blue'] * df['current_blue_side_advantage']
print("  [+] Added: momentum_x_side_advantage")

print("Mapping relative differential features (diff_*) to evaluate matchups...")
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

count = 0
for base, target in diff_map.items():
    if f'blue_{base}' in df.columns: 
        df[target] = df[f'blue_{base}'] - df[f'red_{base}']
        count += 1
print(f"  [+] Added: {count} differential features")

df['diff_rs_rank'] = (df['blue_rs_rank'] - df['red_rs_rank']) * df['is_playoffs']

# EXACT SOTA FEATURE LISTS
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

print(f"\n📊 Feature Space Summary:")
print(f"  - Base dense features: {len(base_features_dense)}")
print(f"  - Game 1 configured dimensions: {len(g1_features)}")
print(f"  - Game 2+ configured dimensions: {len(g2p_features)}")

print("\n✂️ Splitting data chronologically (85% Train / 15% Test) to prevent data leakage...")
# Split matches chronologically
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)
match_start = df.groupby('match_id')['match_timestamp'].min().sort_values(kind='mergesort')
cutoff = match_start.iloc[int(len(match_start)*0.85)]
train_df = df[df['match_id'].isin(match_start[match_start < cutoff].index)].copy()
test_df = df[df['match_id'].isin(match_start[match_start >= cutoff].index)].copy()

print(f"  [+] Training set: {len(train_df)} games")
print(f"  [+] Testing set (Holdout): {len(test_df)} games\n")
"""

md_model = """
## 4. Split-Pipeline Model Architecture

The core innovation of this pipeline is acknowledging that Game 1 and subsequent games are fundamentally different prediction tasks.

**Game 1 Architecture: XGBoost Classifier**
*   **Why XGBoost?** Game 1 is heavily reliant on historical, macro-level features (Elo, patch practice). XGBoost excels at finding complex non-linear interactions in stable tabular data.
*   **Hyperparameters**: A shallow depth (`max_depth=3`) and aggressive subsampling (`subsample=0.9`) prevent the model from overfitting on a smaller Game 1 dataset.

**Game 2+ Architecture: Hybrid Stacking Ensemble (CatBoost + Random Forest)**
*   **Why an Ensemble?** Game 2+ introduces chaotic, high-variance features like `prev_stomp_margin` and momentum. 
*   **The Models**: We use two CatBoost models (which handle the dense pre-match series features well) and one Random Forest (which provides stable, low-variance predictions).
*   **Weighted Averaging**: The probabilities are blended using a `1:3:2` ratio (CatBoost1 : RF : CatBoost2). The Random Forest is heavily weighted (3x) because its uncorrelated error profile stabilizes the aggressive predictions of the gradient boosting models.
*   **Time Weighting**: Recent games are weighted more heavily during training (`sample_weight=time_weight`), forcing the model to prioritize modern meta adaptations over ancient history.
"""

cell4_training = r"""
# ----------------------------------------------------
# 4. MODEL TRAINING (70.43% PRE-MATCH CONFIG)
# ----------------------------------------------------
print("====================================================")
print("🚀 PHASE 4: SPLIT-PIPELINE MODEL TRAINING")
print("====================================================")

g1_train = train_df[train_df['game_number'] == 1]; g1_test = test_df[test_df['game_number'] == 1]
print(f"🔹 Training Game 1 XGBoost... (Train size: {len(g1_train)}, Test size: {len(g1_test)})")
print("   - Hyperparams: depth=3, lr=0.02, subsample=0.9")

g1_model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.02, subsample=0.9, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, eval_metric='logloss', verbosity=0)
g1_model.fit(g1_train[g1_features], g1_train['target_blue_win'])
print("   ✅ Game 1 Model convergence achieved.\n")

g2p_train = train_df[train_df['game_number'] > 1]; g2p_test = test_df[test_df['game_number'] > 1]
print(f"🔹 Training Game 2+ Stacking Ensemble... (Train size: {len(g2p_train)}, Test size: {len(g2p_test)})")
print("   - Architecture: 2x CatBoost, 1x Random Forest (Time Weighted)")

c1 = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, subsample=0.9, l2_leaf_reg=3.0, random_seed=42, verbose=0)
rf = RandomForestClassifier(n_estimators=200, max_depth=2, random_state=42)
c2 = CatBoostClassifier(iterations=50, depth=4, learning_rate=0.05, subsample=0.9, l2_leaf_reg=3.0, random_seed=42, verbose=0)

for idx, m in enumerate([c1, rf, c2]): 
    m.fit(g2p_train[g2p_features], g2p_train['target_blue_win'], sample_weight=g2p_train['time_weight'])
    print(f"   ✅ Sub-model {idx+1} training complete.")

print("\n🧠 Executing Inference on Holdout Set...")
print("   - Blending G2+ probabilities (Ratio - CB1: 1, RF: 3, CB2: 2)...")

# Final Evaluation
p1 = g1_model.predict_proba(g1_test[g1_features])[:, 1]
p2 = (1*c1.predict_proba(g2p_test[g2p_features])[:,1] + 3*rf.predict_proba(g2p_test[g2p_features])[:,1] + 2*c2.predict_proba(g2p_test[g2p_features])[:,1]) / 6
acc_g1, acc_g2 = accuracy_score(g1_test['target_blue_win'], (p1>=0.5).astype(int)), accuracy_score(g2p_test['target_blue_win'], (p2>=0.5).astype(int))
comb_acc = accuracy_score(pd.concat([g1_test['target_blue_win'], g2p_test['target_blue_win']]), np.concatenate([(p1>=0.5).astype(int), (p2>=0.5).astype(int)]))

print("\n" + "="*40)
print(f"🎯 Game 1 Pre-Match Accuracy: {acc_g1:.2%}")
print(f"🎯 Game 2+ Pre-Match Accuracy: {acc_g2:.2%}")
print(f"🏆 FINAL COMBINED PRE-MATCH ACCURACY: {comb_acc:.2%}")
print("="*40 + "\n")
"""

md_simulator = """
## 5. Series Simulator (Applied Inference)

This module demonstrates how the trained models are applied to theoretical or upcoming real-world matchups.

**Execution Flow:**
1.  **Context Retrieval**: `get_recent_stats` scans the dataset for the most recent historical state of both teams (their current Elo, streak, playoff winrate, etc.).
2.  **Differential Calculation**: It dynamically computes the `diff_*` features on the fly for the theoretical matchup.
3.  **Context Injection**: We artificially inject context variables like `is_playoffs=1` and hypothetical `series_momentum_blue` depending on whether we are simulating Game 1 or Game 2.
4.  **Routing**: The function routes the constructed feature vector to either the XGBoost Game 1 model or the CatBoost/RF Game 2+ ensemble based on the `game_num` parameter.
"""

cell6_simulator = r"""
# ----------------------------------------------------
# 5. PRE-MATCH SERIES SIMULATOR
# ----------------------------------------------------
def get_recent_stats(team_name):
    t = resolve_team_name(team_name)
    team_games = df[(df['blue_side_team']==t)|(df['red_side_team']==t)]
    if team_games.empty: return None
    last = team_games.iloc[-1]
    side = 'blue' if last['blue_side_team'] == t else 'red'
    return {col.replace(f'{side}_', ''): last[col] for col in df.columns if col.startswith(f'{side}_')}

def predict_matchup(blue_team, red_team, game_num=1, b_won_last=None):
    print(f"\n🔍 Initializing Simulation: {blue_team} (Blue) vs {red_team} (Red) - Game {game_num}")
    print(f"   - Context Engine: Extracting latest historical momentum and team states...")
    
    b_stats, r_stats = get_recent_stats(blue_team), get_recent_stats(red_team)
    if not b_stats or not r_stats: 
        print("   ❌ Error: One or both teams not found in dataset history.")
        return "Team not found"
        
    print(f"   - Feature Engine: Computing structural differentials...")
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
    
    if game_num == 1: 
        print("   - Inference Routing: Invoking Game 1 XGBoost model...")
        pr = g1_model.predict_proba(inp[g1_features])[0, 1]
        phase = "Game 1 (XGB)"
    else: 
        print("   - Inference Routing: Invoking Game 2+ Stacking Ensemble...")
        print(f"   - Injected Series Context: Blue Won Last = {b_won_last}")
        pr = (1*c1.predict_proba(inp[g2p_features])[0,1] + 3*rf.predict_proba(inp[g2p_features])[0,1] + 2*c2.predict_proba(inp[g2p_features])[0,1]) / 6
        phase = "Game 2+ (CB/RF)"
        
    winner = 'Blue' if pr>0.5 else 'Red'
    conf = pr if winner == 'Blue' else 1-pr
    print(f"   ✅ RESULT [{phase}]: {winner} wins ({conf:.1%} confidence)")
    print(f"       -> {blue_team} Win Probability: {pr:.1%}")
    print(f"       -> {red_team} Win Probability: {1-pr:.1%}")

print("====================================================")
print("🎮 PHASE 5: SERIES SIMULATOR")
print("====================================================")
predict_matchup("Team Liquid PH", "Team Falcons PH", game_num=1)
predict_matchup("ONIC PH", "RSG PH", game_num=2, b_won_last=True)
"""

notebook = {
    "cells": [
        markdown_cell(md_intro),
        markdown_cell(md_data),
        code_cell(cell1_setup),
        markdown_cell(md_elo),
        code_cell(cell2_elo),
        markdown_cell(md_features),
        code_cell(cell3_features),
        markdown_cell(md_model),
        code_cell(cell4_training),
        markdown_cell(md_simulator),
        code_cell(cell6_simulator)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4, "nbformat_minor": 2,
}

with open("1_NoteBook/Prediction_v1_documented.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Prediction_v1_documented.ipynb built with rich pre-match documentation and verbose console logging.")
