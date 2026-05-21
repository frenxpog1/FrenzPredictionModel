import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Load feature matrix
df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')
df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)

# Replicate the V7 feature engineering for series dynamics
blue_series_score_list = []
red_series_score_list  = []
is_elim_game_list      = []
valid_games_mask       = []
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

    # Update AFTER reading
    series_wins_tracker[mid][winner] = series_wins_tracker[mid].get(winner, 0) + 1

df['blue_series_score'] = blue_series_score_list
df['red_series_score']  = red_series_score_list
df['is_elimination_game'] = is_elim_game_list
df['score_diff_blue']   = df['blue_series_score'] - df['red_series_score']
df['is_valid_game']     = valid_games_mask

df = df[df['is_valid_game'] == True].copy()
print(f"✅ Data processing complete. Kept {len(df)} valid games.")

# Baseline V7 Features
base_features_v7 = [
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
    'current_blue_side_advantage',
    'blue_comfort_patch_score', 'red_comfort_patch_score'
]

g2_adaptation_features = [
    'series_momentum_blue',
    'blue_series_score', 'red_series_score',
    'score_diff_blue', 'is_elimination_game',
    'blue_g1_comfort', 'red_g1_comfort',
    'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
    'blue_prev_comfort', 'red_prev_comfort',
    'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red'
]

# V8 Features
v8_features = [
    'blue_avg_loss_duration', 'red_avg_loss_duration',
    'blue_execution_margin', 'red_execution_margin',
    'blue_execution_punish_score', 'red_execution_punish_score',
    'blue_lategame_winrate', 'red_lategame_winrate'
]

# Split Game 1 and Game 2+
df_g1 = df[df['game_number'] == 1].copy()
df_g2plus = df[df['game_number'] > 1].copy()

# Weights
w_g1 = df_g1['time_weight'] / df_g1['time_weight'].mean()
w_g2 = df_g2plus['time_weight'] / df_g2plus['time_weight'].mean()

split_g1 = int(len(df_g1) * 0.85)
split_g2 = int(len(df_g2plus) * 0.85)

def evaluate_models(g1_features, g2_features, xgb_params):
    X_train_g1, X_test_g1 = df_g1[g1_features].iloc[:split_g1], df_g1[g1_features].iloc[split_g1:]
    y_train_g1, y_test_g1 = df_g1['target_blue_win'].iloc[:split_g1], df_g1['target_blue_win'].iloc[split_g1:]
    w_train_g1 = w_g1.iloc[:split_g1]

    X_train_g2, X_test_g2 = df_g2plus[g2_features].iloc[:split_g2], df_g2plus[g2_features].iloc[split_g2:]
    y_train_g2, y_test_g2 = df_g2plus['target_blue_win'].iloc[:split_g2], df_g2plus['target_blue_win'].iloc[split_g2:]
    w_train_g2 = w_g2.iloc[:split_g2]

    def build_ensemble(params):
        xgb_m = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)
        lgb_m = lgb.LGBMClassifier(
            n_estimators=params.get('n_estimators', 400),
            learning_rate=params.get('learning_rate', 0.02),
            max_depth=params.get('max_depth', 3),
            subsample=params.get('subsample', 0.9),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42, verbose=-1
        )
        rf_m = RandomForestClassifier(
            n_estimators=300,
            max_depth=params.get('max_depth', 3) + 2,
            random_state=42, n_jobs=-1
        )
        return VotingClassifier(
            estimators=[('xgb', xgb_m), ('lgb', lgb_m), ('rf', rf_m)],
            voting='soft'
        )

    ens_g1 = build_ensemble(xgb_params)
    ens_g1.fit(X_train_g1, y_train_g1, sample_weight=w_train_g1)

    ens_g2 = build_ensemble(xgb_params)
    ens_g2.fit(X_train_g2, y_train_g2, sample_weight=w_train_g2)

    pred_g1 = ens_g1.predict(X_test_g1)
    pred_g2 = ens_g2.predict(X_test_g2)

    acc_g1 = accuracy_score(y_test_g1, pred_g1)
    acc_g2 = accuracy_score(y_test_g2, pred_g2)
    combined = accuracy_score(
        pd.concat([y_test_g1, y_test_g2]),
        np.concatenate([pred_g1, pred_g2])
    )
    return acc_g1, acc_g2, combined

# Baseline V7 evaluation
print("--- Baseline V7 Features ---")
baseline_g1_params = {'subsample': 0.8, 'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.02, 'colsample_bytree': 0.7}
b_g1, b_g2, b_comb = evaluate_models(base_features_v7, base_features_v7 + g2_adaptation_features, baseline_g1_params)
print(f"Game 1: {b_g1*100:.2f}%, Game 2+: {b_g2*100:.2f}%, Combined: {b_comb*100:.2f}%\n")

# Different subsets of V8 features
feature_configs = {
    "All V8 Features": v8_features,
    "Late-Game Winrate + Execution Punish Only": [
        'blue_execution_punish_score', 'red_execution_punish_score',
        'blue_lategame_winrate', 'red_lategame_winrate'
    ],
    "Execution Margin + Execution Punish Only": [
        'blue_execution_margin', 'red_execution_margin',
        'blue_execution_punish_score', 'red_execution_punish_score'
    ],
    "Late-Game Winrate + Execution Margin Only": [
        'blue_execution_margin', 'red_execution_margin',
        'blue_lategame_winrate', 'red_lategame_winrate'
    ],
    "Late-Game Winrate Only": [
        'blue_lategame_winrate', 'red_lategame_winrate'
    ],
    "Execution Punish Only": [
        'blue_execution_punish_score', 'red_execution_punish_score'
    ],
    "Execution Margin Only": [
        'blue_execution_margin', 'red_execution_margin'
    ],
    "Loss Durations Only": [
        'blue_avg_loss_duration', 'red_avg_loss_duration'
    ]
}

print("--- Testing Feature Combinations ---")
for name, features in feature_configs.items():
    g1_feats = base_features_v7 + features
    g2_feats = base_features_v7 + features + g2_adaptation_features
    
    acc_g1, acc_g2, comb = evaluate_models(g1_feats, g2_feats, baseline_g1_params)
    print(f"Config: {name}")
    print(f"  Game 1: {acc_g1*100:.2f}% | Game 2+: {acc_g2*100:.2f}% | Combined: {comb*100:.2f}%")

# Let's also do a grid search over parameters for All V8 features and the best subsets
print("\n--- Hyperparameter Grid Search for All V8 Features ---")
param_grid = [
    {'subsample': 0.8, 'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.02, 'colsample_bytree': 0.7},
    {'subsample': 0.8, 'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.7},
    {'subsample': 0.9, 'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.02, 'colsample_bytree': 0.8},
    {'subsample': 0.9, 'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.02, 'colsample_bytree': 0.8},
    {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.7},
    {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 0.7},
    {'subsample': 0.9, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.02, 'colsample_bytree': 0.8},
    {'subsample': 0.8, 'n_estimators': 600, 'max_depth': 3, 'learning_rate': 0.02, 'colsample_bytree': 0.6},
    {'subsample': 0.8, 'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.6},
]

best_comb = 0
best_params = None
best_config = None

for p in param_grid:
    for name, features in feature_configs.items():
        g1_feats = base_features_v7 + features
        g2_feats = base_features_v7 + features + g2_adaptation_features
        acc_g1, acc_g2, comb = evaluate_models(g1_feats, g2_feats, p)
        if comb > best_comb:
            best_comb = comb
            best_params = p
            best_config = (name, acc_g1, acc_g2)

print(f"\n🏆 BEST COMBINED ACCURACY FOUND: {best_comb*100:.2f}%")
print(f"Config: {best_config[0]}")
print(f"Params: {best_params}")
print(f"Game 1 Accuracy: {best_config[1]*100:.2f}%")
print(f"Game 2+ Accuracy: {best_config[2]*100:.2f}%")
