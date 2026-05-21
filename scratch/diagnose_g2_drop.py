"""
Diagnostic: Why did G2+ accuracy drop?

Test three hypotheses:
1. The new V8 features are hurting G2+ (feature noise)
2. The shared hyperparameters are suboptimal for G2+
3. The V8 features need DIFFERENT params specifically for G2+
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load feature matrix
df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')
df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)

# Replicate series dynamics preprocessing
series_wins_tracker = {}
blue_series_score_list, red_series_score_list, is_elim_game_list, valid_games_mask = [], [], [], []
for _, row in df.iterrows():
    mid   = row['match_id']
    blue  = row['blue_side_team']
    red   = row['red_side_team']
    gnum  = row['game_number']
    winner= row['blue_side_team'] if row['target_blue_win'] == 1 else row['red_side_team']
    is_po = row['is_playoffs']
    if mid not in series_wins_tracker:
        series_wins_tracker[mid] = {blue: 0, red: 0}
    b_score = series_wins_tracker[mid].get(blue, 0)
    r_score = series_wins_tracker[mid].get(red, 0)
    if is_po == 0 and max(b_score, r_score) >= 2:
        valid_games_mask.append(False)
    elif is_po == 0 and gnum > 3:
        valid_games_mask.append(False)
    elif is_po == 1 and max(b_score, r_score) >= 4:
        valid_games_mask.append(False)
    else:
        valid_games_mask.append(True)
    blue_series_score_list.append(b_score)
    red_series_score_list.append(r_score)
    is_elim = 1 if (b_score == r_score and gnum >= 3) or (abs(b_score - r_score) >= 1 and gnum >= 3) else 0
    is_elim_game_list.append(is_elim)
    series_wins_tracker[mid][winner] = series_wins_tracker[mid].get(winner, 0) + 1

df['blue_series_score']   = blue_series_score_list
df['red_series_score']    = red_series_score_list
df['is_elimination_game'] = is_elim_game_list
df['score_diff_blue']     = df['blue_series_score'] - df['red_series_score']
df['is_valid_game']       = valid_games_mask
df = df[df['is_valid_game'] == True].copy()
print(f"✅ {len(df)} valid games loaded.\n")

# Feature definitions (matching copy_code.py exactly)
base_v7_only = [
    'blue_side_elo','red_side_elo','blue_playoff_elo','red_playoff_elo',
    'blue_comfort_wr','red_comfort_wr','blue_draft_experience','red_draft_experience',
    'blue_global_draft_wr','red_global_draft_wr','blue_ban_disruption','red_ban_disruption',
    'blue_buffs_in_draft','blue_nerfs_in_draft','red_buffs_in_draft','red_nerfs_in_draft',
    'blue_momentum','red_momentum','blue_h2h_winrate',
    'blue_patch_practice','red_patch_practice','is_playoffs',
    'blue_playoff_clutch','red_playoff_clutch','blue_playoff_exp','red_playoff_exp',
    'blue_g3_clutch_wr','red_g3_clutch_wr','blue_reverse_sweep_rate','red_reverse_sweep_rate',
    'blue_rest_factor','red_rest_factor','blue_avg_win_duration','red_avg_win_duration',
    'current_blue_side_advantage','blue_comfort_patch_score','red_comfort_patch_score'
]
v8_features = [
    'blue_avg_loss_duration','red_avg_loss_duration',
    'blue_execution_margin','red_execution_margin',
    'blue_execution_punish_score','red_execution_punish_score',
    'blue_lategame_winrate','red_lategame_winrate'
]
g2_adapt = [
    'series_momentum_blue','blue_series_score','red_series_score',
    'score_diff_blue','is_elimination_game',
    'blue_g1_comfort','red_g1_comfort',
    'g1_winner_heroes_banned_blue','g1_winner_heroes_banned_red',
    'blue_prev_comfort','red_prev_comfort',
    'prev_winner_heroes_banned_blue','prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue','prev_played_comfort_banned_red'
]

df_g2plus = df[df['game_number'] > 1].copy()
y_g2 = df_g2plus['target_blue_win']
w_g2 = df_g2plus['time_weight'] / df_g2plus['time_weight'].mean()
split_g2 = int(len(df_g2plus) * 0.85)
y_train_g2, y_test_g2 = y_g2.iloc[:split_g2], y_g2.iloc[split_g2:]
w_train_g2 = w_g2.iloc[:split_g2]

def build_ensemble(params):
    xm = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)
    lm = lgb.LGBMClassifier(
        n_estimators=params.get('n_estimators',400),
        learning_rate=params.get('learning_rate',0.02),
        max_depth=params.get('max_depth',3),
        subsample=params.get('subsample',0.9),
        colsample_bytree=params.get('colsample_bytree',0.8),
        random_state=42, verbose=-1
    )
    rm = RandomForestClassifier(
        n_estimators=300, max_depth=params.get('max_depth',3)+2,
        random_state=42, n_jobs=-1
    )
    return VotingClassifier(estimators=[('xgb',xm),('lgb',lm),('rf',rm)], voting='soft')

def test_g2(features, params, label):
    X_train = df_g2plus[features].iloc[:split_g2]
    X_test  = df_g2plus[features].iloc[split_g2:]
    ens = build_ensemble(params)
    ens.fit(X_train, y_train_g2, sample_weight=w_train_g2)
    preds = ens.predict(X_test)
    acc = accuracy_score(y_test_g2, preds)
    print(f"  {label}: {acc*100:.2f}%")
    return acc

print("=" * 60)
print("🔬 G2+ DIAGNOSTIC — Testing different feature/param combos")
print("=" * 60)

# Params
v7_params    = {'subsample':0.8,'n_estimators':600,'max_depth':3,'learning_rate':0.02,'colsample_bytree':0.7}
v8_params    = {'subsample':0.9,'n_estimators':400,'max_depth':4,'learning_rate':0.02,'colsample_bytree':0.8}
# Try separate optimized G2+ params (deeper tree, more estimators)
g2_params_a  = {'subsample':0.9,'n_estimators':600,'max_depth':5,'learning_rate':0.01,'colsample_bytree':0.8}
g2_params_b  = {'subsample':0.8,'n_estimators':500,'max_depth':4,'learning_rate':0.02,'colsample_bytree':0.9}
g2_params_c  = {'subsample':1.0,'n_estimators':400,'max_depth':4,'learning_rate':0.02,'colsample_bytree':0.8}

print("\n📋 Hypothesis 1: V7 features only (no V8 noise) — different params")
test_g2(base_v7_only + g2_adapt, v7_params, "V7 feats + V7 params (reference)")
test_g2(base_v7_only + g2_adapt, v8_params, "V7 feats + V8 params")

print("\n📋 Hypothesis 2: All V8 features — different params for G2+")
test_g2(base_v7_only + v8_features + g2_adapt, v8_params,   "V8 feats + shared V8 params")
test_g2(base_v7_only + v8_features + g2_adapt, g2_params_a, "V8 feats + G2-specific params A (deeper/slower)")
test_g2(base_v7_only + v8_features + g2_adapt, g2_params_b, "V8 feats + G2-specific params B")
test_g2(base_v7_only + v8_features + g2_adapt, g2_params_c, "V8 feats + G2-specific params C (no subsample)")

print("\n📋 Hypothesis 3: Cherry-pick best V8 features for G2+")
# Only lategame_winrate (strongest G2+ predictor from ablation)
test_g2(base_v7_only + ['blue_lategame_winrate','red_lategame_winrate'] + g2_adapt,
        v8_params, "V7 + lategame_winrate only + V8 params")
test_g2(base_v7_only + ['blue_execution_punish_score','red_execution_punish_score'] + g2_adapt,
        v8_params, "V7 + execution_punish only + V8 params")

print("\n✅ Diagnostic complete.")
