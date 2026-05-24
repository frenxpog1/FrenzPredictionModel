import json


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


notebook = {
    "cells": [
        markdown_cell(
            """
# MPL PH Leak-Safe Prediction Pipeline

This notebook now separates the question into three honest buckets:

- Game 1 pre-match model
- Game 2 pre-match candidate model
- Game 3+ pre-match candidate model

It also keeps the pooled Game 2+ model as the production champion unless the split candidate proves better. This matters because Game 3+ has far fewer samples, so splitting can look logical while making predictions weaker.

Current-game draft features are still blocked from the pre-match model. A separate post-draft benchmark appears later and is clearly labeled.
"""
        ),
        code_cell(
            """
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------
# 1. DATA IMPORT
# ----------------------------------------------------
csv_dir = 'csv_data' if os.path.exists('csv_data') else '../csv_data'
print(f"Resolved CSV Directory: '{csv_dir}'")

df = pd.read_csv(f'{csv_dir}/ML_Feature_Matrix.csv')
df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)
print(f"Loaded Feature Matrix: {len(df)} games")


def add_safe_series_state(dataframe):
    out = dataframe.copy()
    series_wins = {}
    prev_game = {}

    blue_scores = []
    red_scores = []
    score_diffs = []
    blue_leads = []
    red_leads = []
    abs_diffs = []
    games_played = []
    late_series = []
    prev_blue_side_won = []

    for _, row in out.iterrows():
        match_id = row['match_id']
        blue_team = row['blue_side_team']
        red_team = row['red_side_team']
        game_number = int(row['game_number'])

        series_wins.setdefault(match_id, {})
        blue_score = series_wins[match_id].get(blue_team, 0)
        red_score = series_wins[match_id].get(red_team, 0)

        blue_scores.append(blue_score)
        red_scores.append(red_score)
        score_diffs.append(blue_score - red_score)
        blue_leads.append(int(blue_score > red_score))
        red_leads.append(int(red_score > blue_score))
        abs_diffs.append(abs(blue_score - red_score))
        games_played.append(max(0, game_number - 1))
        late_series.append(int(game_number >= 3))

        previous = prev_game.get(match_id)
        prev_blue_side_won.append(0.5 if previous is None else previous['target_blue_win'])

        winner = blue_team if row['target_blue_win'] == 1 else red_team
        series_wins[match_id][winner] = series_wins[match_id].get(winner, 0) + 1
        prev_game[match_id] = {'target_blue_win': row['target_blue_win']}

    out['blue_series_score_safe'] = blue_scores
    out['red_series_score_safe'] = red_scores
    out['score_diff_blue_safe'] = score_diffs
    out['blue_leads_series'] = blue_leads
    out['red_leads_series'] = red_leads
    out['series_score_abs_diff'] = abs_diffs
    out['series_games_played'] = games_played
    out['is_late_series_game'] = late_series
    out['prev_blue_side_won'] = prev_blue_side_won
    return out


df = add_safe_series_state(df)
df['momentum_x_side_advantage'] = df['series_momentum_blue'] * df['current_blue_side_advantage']

df['diff_draft_experience'] = df['blue_draft_experience'] - df['red_draft_experience']
df['diff_patch_practice'] = df['blue_patch_practice'] - df['red_patch_practice']
df['diff_roster_stability'] = df['blue_roster_stability'] - df['red_roster_stability']
df['diff_patch_wr'] = df['blue_patch_winrate'] - df['red_patch_winrate']
df['diff_patch_adapt'] = df['blue_patch_adaptation'] - df['red_patch_adaptation']
df['diff_playoff_clutch'] = df['blue_playoff_clutch'] - df['red_playoff_clutch']
df['diff_playoff_exp'] = df['blue_playoff_exp'] - df['red_playoff_exp']
df['diff_g3_clutch'] = df['blue_g3_clutch_wr'] - df['red_g3_clutch_wr']
df['diff_reverse_sweep'] = df['blue_reverse_sweep_rate'] - df['red_reverse_sweep_rate']
df['diff_rest'] = df['blue_rest_factor'] - df['red_rest_factor']

# Only apply RS Rank diff if it's playoffs! Otherwise it's noise.
df['diff_rs_rank'] = (df['blue_rs_rank'] - df['red_rs_rank']) * df['is_playoffs']

# ----------------------------------------------------
# 1.5 SOTA Game 2+ Feature Engineering
# ----------------------------------------------------
print("🧪 Engineering SOTA Game 2+ Features from Game 1 state...")
try:
    games_df = pd.read_csv('csv_data/games.csv')
    g1_games = games_df[games_df['game_number'] == 1][['match_id', 'game_duration_seconds']].copy()
    g1_games.rename(columns={'game_duration_seconds': 'g1_duration'}, inplace=True)
    df = df.merge(g1_games, on='match_id', how='left')
    df['G1_Duration_Deviation'] = df['g1_duration'].fillna(900) - 900
except Exception as e:
    print(f"Warning: Could not load games.csv for duration: {e}")
    df['G1_Duration_Deviation'] = 0

df['G1_Upset_Magnitude'] = np.where(
    df['series_momentum_blue'] == 1,
    df['red_side_elo'] - df['blue_side_elo'],
    np.where(df['series_momentum_blue'] == 0, df['blue_side_elo'] - df['red_side_elo'], 0)
)

df['G1_Winner_Comfort_Advantage'] = np.where(
    df['series_momentum_blue'] == 1,
    df['blue_g1_comfort'] - df['red_g1_comfort'],
    np.where(df['series_momentum_blue'] == 0, df['red_g1_comfort'] - df['blue_g1_comfort'], 0)
)

df['diff_draft_exhaustion'] = df['blue_draft_exhaustion'] - df['red_draft_exhaustion']

df = df.copy() # De-fragment dataframe

# ----------------------------------------------------
# 2. FEATURE POLICY
# ----------------------------------------------------
forbidden_pre_match_features = {
    'blue_synergy', 'red_synergy',
    'blue_counter', 'red_counter',
    'blue_comfort_wr', 'red_comfort_wr',
    'blue_draft_experience', 'red_draft_experience',
    'blue_global_draft_wr', 'red_global_draft_wr',
    'blue_ban_disruption', 'red_ban_disruption',
    'blue_buffs_in_draft', 'blue_nerfs_in_draft',
    'red_buffs_in_draft', 'red_nerfs_in_draft',
    'blue_heroes_stolen', 'red_heroes_stolen',
    'blue_synergy_delta', 'red_synergy_delta',
    'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
    'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red',
    'blue_turtles', 'red_turtles', 'blue_lords', 'red_lords',
    'blue_draft_overlap', 'red_draft_overlap', 'diff_draft_overlap',
}

base_features = [
    'blue_roster_stability', 'red_roster_stability', 'diff_roster_stability',
    'blue_side_elo', 'red_side_elo',
    'blue_playoff_elo', 'red_playoff_elo',
    'blue_momentum', 'red_momentum',
    'blue_h2h_winrate',
    'blue_patch_practice', 'red_patch_practice',
    'diff_patch_practice',
    'diff_playoff_clutch', 'diff_playoff_exp',
    'diff_g3_clutch', 'diff_reverse_sweep', 'diff_rest',
    'diff_rs_rank', 'blue_is_defending_champ', 'red_is_defending_champ',
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
    # Official Seasonal Statistics (Prior Season)
    'blue_prev_season_match_wr', 'red_prev_season_match_wr',
    'blue_prev_season_game_wr', 'red_prev_season_game_wr',
    'blue_prev_season_kda', 'red_prev_season_kda',
    'blue_prev_season_avg_kills', 'red_prev_season_avg_kills',
    'blue_prev_season_avg_deaths', 'red_prev_season_avg_deaths',
    'blue_prev_season_avg_assists', 'red_prev_season_avg_assists',
    'diff_prev_season_match_wr', 'diff_prev_season_game_wr',
    'diff_prev_season_kda',
]

pace_features = [
    'blue_avg_loss_duration', 'red_avg_loss_duration',
    'blue_execution_margin', 'red_execution_margin',
    'blue_execution_punish_score', 'red_execution_punish_score',
    'blue_lategame_winrate', 'red_lategame_winrate',
]

series_features = [
    'series_momentum_blue',
    'prev_stomp_margin',
    'is_side_swap',
]

series_score_features = [
    'prev_blue_side_won',
    'blue_series_score_safe',
    'red_series_score_safe',
    'score_diff_blue_safe',
    'blue_leads_series',
    'red_leads_series',
    'G1_Duration_Deviation',
    'G1_Upset_Magnitude',
    'G1_Winner_Comfort_Advantage',
    'is_late_series_game'
]

late_series_features = [
    'series_score_abs_diff',
    'series_games_played',
    'is_late_series_game',
]

patch_adaptation_features = ['diff_patch_wr', 'diff_patch_adapt']

pre_match_series_features = [
    'blue_draft_exhaustion', 'red_draft_exhaustion', 'diff_draft_exhaustion',
    'blue_prev_winner_exhaustion', 'red_prev_winner_exhaustion',
    'blue_g1_comfort', 'red_g1_comfort',
    'blue_prev_comfort', 'red_prev_comfort',
]
g1_features = base_features + ['draft_style_sim']
g2plus_features = base_features + pace_features + series_features + pre_match_series_features + patch_adaptation_features + ['draft_style_sim', 'momentum_x_side_advantage']

# Candidate split models. These are evaluated against the pooled champion.
g2_features = [
    'blue_side_elo', 'red_side_elo',
    'series_momentum_blue',
    'current_blue_side_advantage',
    'G1_Duration_Deviation',
    'G1_Upset_Magnitude',
    'G1_Winner_Comfort_Advantage',
    'momentum_x_side_advantage',
    'blue_g1_comfort', 'red_g1_comfort',
    'blue_playoff_elo', 'red_playoff_elo',
    'is_side_swap'
]
g3plus_features = base_features + pace_features + series_features + pre_match_series_features + patch_adaptation_features + ['draft_style_sim', 'momentum_x_side_advantage']

pre_match_feature_sets = {
    'g1': g1_features,
    'g2plus': g2plus_features,
    'g2': g2_features,
    'g3plus': g3plus_features,
}

used_pre_match_features = set()
for feature_list in pre_match_feature_sets.values():
    used_pre_match_features.update(feature_list)

leaky_features_used = sorted(used_pre_match_features & forbidden_pre_match_features)
if leaky_features_used:
    raise ValueError(f"Pre-match leakage guard failed: {leaky_features_used}")

required_columns = used_pre_match_features | {
    'match_id', 'match_timestamp', 'game_number', 'target_blue_win', 'time_weight',
}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Pre-match feature counts:")
print(f"  Game 1       : {len(g1_features)}")
print(f"  Pooled Game2+: {len(g2plus_features)}")
print(f"  Game 2 split : {len(g2_features)}")
print(f"  Game 3+ split: {len(g3plus_features)}")

# ----------------------------------------------------
# 3. CHRONOLOGICAL MATCH SPLIT
# ----------------------------------------------------
def chronological_match_split(dataframe, train_fraction=0.85):
    match_start = dataframe.groupby('match_id')['match_timestamp'].min().sort_values(kind='mergesort')
    split_pos = int(len(match_start) * train_fraction)
    cutoff_ts = match_start.iloc[split_pos]

    train_ids = match_start[match_start < cutoff_ts].index
    test_ids = match_start[match_start >= cutoff_ts].index

    train = dataframe[dataframe['match_id'].isin(train_ids)].copy()
    test = dataframe[dataframe['match_id'].isin(test_ids)].copy()
    return train, test, cutoff_ts


train_df, test_df, cutoff_ts = chronological_match_split(df)

print("Chronological holdout:")
print(f"  Cutoff date      : {cutoff_ts.date()}")
print(f"  Game 1 split     : {(train_df['game_number'] == 1).sum()} train / {(test_df['game_number'] == 1).sum()} test")
print(f"  Game 2 split     : {(train_df['game_number'] == 2).sum()} train / {(test_df['game_number'] == 2).sum()} test")
print(f"  Game 3+ split    : {(train_df['game_number'] >= 3).sum()} train / {(test_df['game_number'] >= 3).sum()} test")
"""
        ),
        code_cell(
            """
# ----------------------------------------------------
# 4. PRE-MATCH TRAINING AND EVALUATION
# ----------------------------------------------------
g1_params = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'tree_method': 'hist',
}

g2plus_params = {
    'n_estimators': 10,
    'max_depth': 2,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'tree_method': 'hist',
}

split_params = {
    'n_estimators': 75,
    'max_depth': 3,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'tree_method': 'hist',
}


def build_ensemble(params, is_g2plus=False):
    if is_g2plus:
        xgb_model = xgb.XGBClassifier(
            n_estimators=10,
            learning_rate=0.02,
            max_depth=2,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            tree_method='hist'
        )
        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=2,
            random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model)
            ],
            voting='soft', weights=None
        )
        return ensemble

    xgb_model = xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 200),
        learning_rate=params.get('learning_rate', 0.02),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 0.9),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        reg_lambda=params.get('reg_lambda', 2.0),
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        tree_method='hist'
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=params.get('n_estimators', 200),
        learning_rate=params.get('learning_rate', 0.02),
        max_depth=params.get('max_depth', 3),
        subsample=params.get('subsample', 0.9),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        random_state=42, verbose=-1
    )
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=params.get('max_depth', 3) + 2,
        random_state=42
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
        voting='soft', weights=None
    )
    return ensemble


def fit_bucket(train, condition, features, params, is_g2plus=False):
    bucket = train[condition(train)].copy()
    model = build_ensemble(params, is_g2plus=is_g2plus).fit(
        bucket[features],
        bucket['target_blue_win'],
        sample_weight=bucket['time_weight'],
    )
    return model


def score_bucket(model, data, condition, features, label):
    bucket = data[condition(data)].copy()
    pred = model.predict(bucket[features])
    return {
        'label': label,
        'accuracy': accuracy_score(bucket['target_blue_win'], pred),
        'count': len(bucket),
        'y_true': bucket['target_blue_win'],
        'y_pred': pred,
    }


def train_pooled_architecture(train):
    return {
        'g1': fit_bucket(train, lambda d: d['game_number'] == 1, g1_features, g1_params, is_g2plus=False),
        'g2plus': fit_bucket(train, lambda d: d['game_number'] > 1, g2plus_features, g2plus_params, is_g2plus=True),
    }


def evaluate_pooled_architecture(models, data):
    rows = [
        score_bucket(models['g1'], data, lambda d: d['game_number'] == 1, g1_features, 'Game 1'),
        score_bucket(models['g2plus'], data, lambda d: d['game_number'] > 1, g2plus_features, 'Game 2+'),
    ]
    y_true = pd.concat([row['y_true'] for row in rows])
    y_pred = np.concatenate([row['y_pred'] for row in rows])
    return rows, accuracy_score(y_true, y_pred)


def train_split_architecture(train):
    # Game 1 model (ensemble of XGB+LGB+RF+Cat)
    g1_model = fit_bucket(train, lambda d: d['game_number'] == 1, g1_features, g1_params, is_g2plus=False)
    
    # Game 2 model (Logistic Regression with SOTA features)
    g2_bucket = train[train['game_number'] == 2].copy()
    g2_model = LogisticRegression(C=0.01, random_state=42, max_iter=1000)
    g2_model.fit(
        g2_bucket[g2_features],
        g2_bucket['target_blue_win'],
        sample_weight=g2_bucket['time_weight']
    )
    
    # Game 3+ model: Random Forest (depth=2, n_est=50)
    g2plus_bucket = train[train['game_number'] > 1].copy()
    g3plus_model = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42)
    g3plus_model.fit(
        g2plus_bucket[g3plus_features],
        g2plus_bucket['target_blue_win'],
        sample_weight=g2plus_bucket['time_weight']
    )
    
    return {
        'g1': g1_model,
        'g2': g2_model,
        'g3plus': g3plus_model,
    }


def evaluate_split_architecture(models, data):
    rows = [
        score_bucket(models['g1'], data, lambda d: d['game_number'] == 1, g1_features, 'Game 1'),
        score_bucket(models['g2'], data, lambda d: d['game_number'] == 2, g2_features, 'Game 2'),
        score_bucket(models['g3plus'], data, lambda d: d['game_number'] >= 3, g3plus_features, 'Game 3+'),
    ]
    y_true = pd.concat([row['y_true'] for row in rows])
    y_pred = np.concatenate([row['y_pred'] for row in rows])
    return rows, accuracy_score(y_true, y_pred)


def summarize_architecture(name, test_rows, test_combined, train_rows, train_combined):
    print("\\n" + "=" * 72)
    print(name)
    print("=" * 72)
    for row in test_rows:
        train_row = next(item for item in train_rows if item['label'] == row['label'])
        print(
            f"{row['label']:<8} Test: {row['accuracy']:.2%} ({row['count']} games) | "
            f"Train: {train_row['accuracy']:.2%} ({train_row['count']} games)"
        )
    print("-" * 72)
    print(f"Combined Test : {test_combined:.2%}")
    print(f"Combined Train: {train_combined:.2%}")


pooled_models = train_pooled_architecture(train_df)
pooled_test_rows, pooled_test_combined = evaluate_pooled_architecture(pooled_models, test_df)
pooled_train_rows, pooled_train_combined = evaluate_pooled_architecture(pooled_models, train_df)

split_models = train_split_architecture(train_df)
split_test_rows, split_test_combined = evaluate_split_architecture(split_models, test_df)
split_train_rows, split_train_combined = evaluate_split_architecture(split_models, train_df)

summarize_architecture("POOLED PRE-MATCH CHAMPION", pooled_test_rows, pooled_test_combined, pooled_train_rows, pooled_train_combined)
summarize_architecture("SPLIT PRE-MATCH CHALLENGER", split_test_rows, split_test_combined, split_train_rows, split_train_combined)

if split_test_combined > pooled_test_combined:
    champion_architecture = 'split'
    print("\\nChampion for simulator: split Game 1 / Game 2 / Game 3+")
else:
    champion_architecture = 'pooled'
    print("\\nChampion for simulator: pooled Game 2+ remains better on this holdout")

print("\\nGame 2+ report for pooled champion:")
pooled_g2plus = test_df[test_df['game_number'] > 1].copy()
pooled_g2plus_pred = pooled_models['g2plus'].predict(pooled_g2plus[g2plus_features])
print(classification_report(pooled_g2plus['target_blue_win'], pooled_g2plus_pred, zero_division=0))
"""
        ),
        code_cell(
            """
# ----------------------------------------------------
# 5. ROLLING VALIDATION: POOLED VS SPLIT
# ----------------------------------------------------
def rolling_validation(dataframe, folds=5):
    match_start = dataframe.groupby('match_id')['match_timestamp'].min().sort_values(kind='mergesort')
    match_ids = list(match_start.index)
    fold_size = len(match_ids) // (folds + 1)
    rows = []

    for fold in range(1, folds + 1):
        train_ids = match_ids[:fold_size * fold]
        if fold < folds:
            test_ids = match_ids[fold_size * fold:fold_size * (fold + 1)]
        else:
            test_ids = match_ids[fold_size * fold:]

        fold_train = dataframe[dataframe['match_id'].isin(train_ids)].copy()
        fold_test = dataframe[dataframe['match_id'].isin(test_ids)].copy()

        pooled = train_pooled_architecture(fold_train)
        pooled_rows, pooled_combined = evaluate_pooled_architecture(pooled, fold_test)

        split = train_split_architecture(fold_train)
        split_rows, split_combined = evaluate_split_architecture(split, fold_test)

        rows.append({
            'fold': fold,
            'train_matches': len(train_ids),
            'test_matches': len(test_ids),
            'pooled_combined': pooled_combined,
            'split_combined': split_combined,
            'pooled_g2plus': pooled_rows[1]['accuracy'],
            'split_g2': split_rows[1]['accuracy'],
            'split_g3plus': split_rows[2]['accuracy'],
        })

    return pd.DataFrame(rows)


rolling_results = rolling_validation(df)
print("Rolling validation by chronological match block:")
print(rolling_results.to_string(index=False, formatters={
    'pooled_combined': '{:.2%}'.format,
    'split_combined': '{:.2%}'.format,
    'pooled_g2plus': '{:.2%}'.format,
    'split_g2': '{:.2%}'.format,
    'split_g3plus': '{:.2%}'.format,
}))

print("\\nAverage combined accuracy:")
print(f"  Pooled Game2+ : {rolling_results['pooled_combined'].mean():.2%}")
print(f"  Split G2/G3+  : {rolling_results['split_combined'].mean():.2%}")
"""
        ),
        markdown_cell(
            """
### Post-Draft Benchmark

The next cell is not part of the leak-safe pre-match score. It answers a different question: what happens if picks and bans are already known?

Use this only after draft data exists for the match you are predicting.
"""
        ),
        code_cell(
            """
# ----------------------------------------------------
# 6. OPTIONAL POST-DRAFT BENCHMARK
# ----------------------------------------------------
current_draft_features = [
    'blue_synergy', 'red_synergy',
    'blue_counter', 'red_counter',
    'blue_comfort_wr', 'red_comfort_wr',
    'blue_draft_experience', 'red_draft_experience',
    'blue_global_draft_wr', 'red_global_draft_wr',
    'blue_ban_disruption', 'red_ban_disruption',
    'blue_buffs_in_draft', 'blue_nerfs_in_draft',
    'red_buffs_in_draft', 'red_nerfs_in_draft',
]

post_draft_adaptation_features = [
    'blue_heroes_stolen', 'red_heroes_stolen',
    'blue_synergy_delta', 'red_synergy_delta',
    'g1_winner_heroes_banned_blue', 'g1_winner_heroes_banned_red',
    'prev_winner_heroes_banned_blue', 'prev_winner_heroes_banned_red',
    'prev_played_comfort_banned_blue', 'prev_played_comfort_banned_red',
    'blue_draft_overlap', 'red_draft_overlap', 'diff_draft_overlap',
]

post_g1_features = g1_features + current_draft_features
post_g2plus_features = g2plus_features + current_draft_features + post_draft_adaptation_features

post_missing = sorted(set(post_g1_features + post_g2plus_features) - set(df.columns))
if post_missing:
    print(f"Post-draft benchmark skipped. Missing columns: {post_missing}")
else:
    post_params = {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.02,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_lambda': 2.0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
        'tree_method': 'hist',
    }

    post_g1_model = fit_bucket(train_df, lambda d: d['game_number'] == 1, post_g1_features, post_params)
    post_g2plus_model = fit_bucket(train_df, lambda d: d['game_number'] > 1, post_g2plus_features, post_params)

    post_rows = [
        score_bucket(post_g1_model, test_df, lambda d: d['game_number'] == 1, post_g1_features, 'Game 1 post-draft'),
        score_bucket(post_g2plus_model, test_df, lambda d: d['game_number'] > 1, post_g2plus_features, 'Game 2+ post-draft'),
    ]
    post_combined = accuracy_score(
        pd.concat([row['y_true'] for row in post_rows]),
        np.concatenate([row['y_pred'] for row in post_rows]),
    )

    print("POST-DRAFT BENCHMARK ONLY")
    for row in post_rows:
        print(f"{row['label']:<18}: {row['accuracy']:.2%} ({row['count']} games)")
    print(f"Combined post-draft : {post_combined:.2%}")
"""
        ),
        code_cell(
            """
# ----------------------------------------------------
# 7. PLAYOFF SERIES SIMULATION HARNESS
# ----------------------------------------------------
def get_recent_stats(team_name, dataframe):
    team_games = dataframe[
        (dataframe['blue_side_team'] == team_name) |
        (dataframe['red_side_team'] == team_name)
    ]
    if team_games.empty:
        return None

    last_game = team_games.iloc[-1]
    on_blue = last_game['blue_side_team'] == team_name
    side = 'blue' if on_blue else 'red'

    stats_dict = {
        'elo': last_game[f'{side}_side_elo'],
        'playoff_elo': last_game[f'{side}_playoff_elo'],
        'roster_stability': last_game[f'{side}_roster_stability'],
        'draft_overlap': last_game[f'{side}_draft_overlap'],
        'prev_winner_exhaustion': last_game[f'{side}_prev_winner_exhaustion'],
        'ban_disruption': last_game[f'{side}_ban_disruption'],
        'playoff_clutch': last_game[f'{side}_playoff_clutch'],
        'playoff_exp': last_game[f'{side}_playoff_exp'],
        'g3_clutch_wr': last_game[f'{side}_g3_clutch_wr'],
        'reverse_sweep_rate': last_game[f'{side}_reverse_sweep_rate'],
        'rest_factor': last_game[f'{side}_rest_factor'],
        'momentum': last_game[f'{side}_momentum'],
        'patch_practice': last_game[f'{side}_patch_practice'],
        'patch_winrate': last_game[f'{side}_patch_winrate'],
        'patch_adaptation': last_game[f'{side}_patch_adaptation'],
        'avg_win_duration': last_game[f'{side}_avg_win_duration'],
        'avg_loss_duration': last_game[f'{side}_avg_loss_duration'],
        'execution_margin': last_game[f'{side}_execution_margin'],
        'execution_punish_score': last_game[f'{side}_execution_punish_score'],
        'lategame_winrate': last_game[f'{side}_lategame_winrate'],
        'h2h_winrate': last_game['blue_h2h_winrate'] if on_blue else 1.0 - last_game['blue_h2h_winrate'],
        'comfort_patch_score': last_game[f'{side}_comfort_patch_score'],
        'expected_comfort': last_game[f'{side}_expected_comfort'],
        'prev_season_match_wr': last_game[f'{side}_prev_season_match_wr'],
        'prev_season_game_wr': last_game[f'{side}_prev_season_game_wr'],
        'prev_season_kda': last_game[f'{side}_prev_season_kda'],
        'prev_season_avg_kills': last_game[f'{side}_prev_season_avg_kills'],
        'prev_season_avg_deaths': last_game[f'{side}_prev_season_avg_deaths'],
        'prev_season_avg_assists': last_game[f'{side}_prev_season_avg_assists'],
        'rs_rank': last_game[f'{side}_rs_rank'],
        'is_defending_champ': last_game[f'{side}_is_defending_champ'],
    }
    for i in range(16):
        stats_dict[f'draft_emb_{i}'] = last_game[f'{side}_draft_emb_{i}']
    return stats_dict


def simulate_matchup(
    blue_team,
    red_team,
    is_playoffs=True,
    game_number=1,
    blue_won_last_game=None,
    prev_stomp_margin=0.0,
    is_side_swap=0,
    blue_draft_exhaustion=0.0,
    red_draft_exhaustion=0.0,
    blue_prev_winner_exhaustion=0.0,
    red_prev_winner_exhaustion=0.0,
    blue_g1_comfort=0.53,
    red_g1_comfort=0.53,
    blue_prev_comfort=0.53,
    red_prev_comfort=0.53,
    architecture=None,
):
    architecture = architecture or champion_architecture
    blue_stats = get_recent_stats(blue_team, df)
    red_stats = get_recent_stats(red_team, df)

    if blue_stats is None:
        print(f"ERROR: Team '{blue_team}' not found in dataset.")
        return None
    if red_stats is None:
        print(f"ERROR: Team '{red_team}' not found in dataset.")
        return None

    if blue_won_last_game is None:
        series_momentum = 0.5
    elif blue_won_last_game:
        series_momentum = 1.0
    else:
        series_momentum = 0.0

    base_row = {
        'blue_roster_stability': blue_stats['roster_stability'],
        'red_roster_stability': red_stats['roster_stability'],
        'diff_roster_stability': blue_stats['roster_stability'] - red_stats['roster_stability'],
        'blue_side_elo': blue_stats['elo'],
        'red_side_elo': red_stats['elo'],
        'blue_playoff_elo': blue_stats['playoff_elo'],
        'red_playoff_elo': red_stats['playoff_elo'],
        'blue_momentum': blue_stats['momentum'],
        'red_momentum': red_stats['momentum'],
        'blue_h2h_winrate': blue_stats['h2h_winrate'],
        'blue_patch_practice': blue_stats['patch_practice'],
        'red_patch_practice': red_stats['patch_practice'],
        'diff_patch_practice': blue_stats['patch_practice'] - red_stats['patch_practice'],
        'diff_patch_wr': blue_stats['patch_winrate'] - red_stats['patch_winrate'],
        'diff_patch_adapt': blue_stats['patch_adaptation'] - red_stats['patch_adaptation'],
        'diff_playoff_clutch': blue_stats['playoff_clutch'] - red_stats['playoff_clutch'],
        'diff_playoff_exp': blue_stats['playoff_exp'] - red_stats['playoff_exp'],
        'diff_g3_clutch': blue_stats['g3_clutch_wr'] - red_stats['g3_clutch_wr'],
        'diff_reverse_sweep': blue_stats['reverse_sweep_rate'] - red_stats['reverse_sweep_rate'],
        'diff_rest': blue_stats['rest_factor'] - red_stats['rest_factor'],
        'blue_rs_rank': blue_stats['rs_rank'],
        'red_rs_rank': red_stats['rs_rank'],
        'diff_rs_rank': (blue_stats['rs_rank'] - red_stats['rs_rank']) if is_playoffs else 0,
        'blue_is_defending_champ': blue_stats['is_defending_champ'],
        'red_is_defending_champ': red_stats['is_defending_champ'],
        'is_playoffs': 1 if is_playoffs else 0,
        'blue_playoff_clutch': blue_stats['playoff_clutch'],
        'red_playoff_clutch': red_stats['playoff_clutch'],
        'blue_playoff_exp': blue_stats['playoff_exp'],
        'red_playoff_exp': red_stats['playoff_exp'],
        'blue_g3_clutch_wr': blue_stats['g3_clutch_wr'],
        'red_g3_clutch_wr': red_stats['g3_clutch_wr'],
        'blue_reverse_sweep_rate': blue_stats['reverse_sweep_rate'],
        'red_reverse_sweep_rate': red_stats['reverse_sweep_rate'],
        'blue_rest_factor': 1.02,
        'red_rest_factor': 1.02,
        'blue_avg_win_duration': blue_stats['avg_win_duration'],
        'red_avg_win_duration': red_stats['avg_win_duration'],
        'current_blue_side_advantage': df['current_blue_side_advantage'].iloc[-1],
        'momentum_x_side_advantage': series_momentum * df['current_blue_side_advantage'].iloc[-1],
        'blue_comfort_patch_score': blue_stats['comfort_patch_score'],
        'red_comfort_patch_score': red_stats['comfort_patch_score'],
        'blue_expected_comfort': blue_stats['expected_comfort'],
        'red_expected_comfort': red_stats['expected_comfort'],
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

    if game_number == 1:
        matchup = pd.DataFrame([base_row])[g1_features]
        model = pooled_models['g1']
        model_label = 'Game 1'
    elif architecture == 'split' and game_number == 2:
        full_row = {
            **base_row,
            'blue_avg_loss_duration': blue_stats['avg_loss_duration'],
            'red_avg_loss_duration': red_stats['avg_loss_duration'],
            'blue_execution_margin': blue_stats['execution_margin'],
            'red_execution_margin': red_stats['execution_margin'],
            'blue_execution_punish_score': blue_stats['execution_punish_score'],
            'red_execution_punish_score': red_stats['execution_punish_score'],
            'blue_lategame_winrate': blue_stats['lategame_winrate'],
            'red_lategame_winrate': red_stats['lategame_winrate'],
            'series_momentum_blue': series_momentum,
            'prev_stomp_margin': prev_stomp_margin,
            'is_side_swap': is_side_swap,
            'blue_draft_exhaustion': blue_draft_exhaustion,
            'red_draft_exhaustion': red_draft_exhaustion,
            'blue_prev_winner_exhaustion': blue_prev_winner_exhaustion,
            'red_prev_winner_exhaustion': red_prev_winner_exhaustion,
            'blue_g1_comfort': blue_g1_comfort,
            'red_g1_comfort': red_g1_comfort,
            'blue_prev_comfort': blue_prev_comfort,
            'red_prev_comfort': red_prev_comfort,
        }
        matchup = pd.DataFrame([full_row])[g2_features]
        model = split_models['g2']
        model_label = 'Game 2 split'
    elif architecture == 'split' and game_number >= 3:
        full_row = {
            **base_row,
            'blue_avg_loss_duration': blue_stats['avg_loss_duration'],
            'red_avg_loss_duration': red_stats['red_loss_duration'] if 'red_loss_duration' in red_stats else red_stats['avg_loss_duration'],
            'blue_execution_margin': blue_stats['execution_margin'],
            'red_execution_margin': red_stats['execution_margin'],
            'blue_execution_punish_score': blue_stats['execution_punish_score'],
            'red_execution_punish_score': red_stats['execution_punish_score'],
            'blue_lategame_winrate': blue_stats['lategame_winrate'],
            'red_lategame_winrate': red_stats['lategame_winrate'],
            'series_momentum_blue': series_momentum,
            'prev_stomp_margin': prev_stomp_margin,
            'is_side_swap': is_side_swap,
            'blue_draft_exhaustion': blue_draft_exhaustion,
            'red_draft_exhaustion': red_draft_exhaustion,
            'blue_prev_winner_exhaustion': blue_prev_winner_exhaustion,
            'red_prev_winner_exhaustion': red_prev_winner_exhaustion,
            'blue_g1_comfort': blue_g1_comfort,
            'red_g1_comfort': red_g1_comfort,
            'blue_prev_comfort': blue_prev_comfort,
            'red_prev_comfort': red_prev_comfort,
        }
        matchup = pd.DataFrame([full_row])[g3plus_features]
        model = split_models['g3plus']
        model_label = 'Game 3+ split'
    else:
        full_row = {
            **base_row,
            'blue_avg_loss_duration': blue_stats['avg_loss_duration'],
            'red_avg_loss_duration': red_stats['avg_loss_duration'],
            'blue_execution_margin': blue_stats['execution_margin'],
            'red_execution_margin': red_stats['execution_margin'],
            'blue_execution_punish_score': blue_stats['execution_punish_score'],
            'red_execution_punish_score': red_stats['execution_punish_score'],
            'blue_lategame_winrate': blue_stats['lategame_winrate'],
            'red_lategame_winrate': red_stats['lategame_winrate'],
            'series_momentum_blue': series_momentum,
            'prev_stomp_margin': prev_stomp_margin,
            'is_side_swap': is_side_swap,
            'blue_draft_exhaustion': blue_draft_exhaustion,
            'red_draft_exhaustion': red_draft_exhaustion,
            'blue_prev_winner_exhaustion': blue_prev_winner_exhaustion,
            'red_prev_winner_exhaustion': red_prev_winner_exhaustion,
            'blue_g1_comfort': blue_g1_comfort,
            'red_g1_comfort': red_g1_comfort,
            'blue_prev_comfort': blue_prev_comfort,
            'red_prev_comfort': red_prev_comfort,
        }
        matchup = pd.DataFrame([full_row])[g2plus_features]
        model = pooled_models['g2plus']
        model_label = 'pooled Game 2+'

    probs = model.predict_proba(matchup)[0]
    red_prob = probs[0] * 100
    blue_prob = probs[1] * 100
    winner = blue_team if blue_prob > red_prob else red_team

    print("\\n" + "=" * 60)
    print(f"{blue_team} vs {red_team}")
    print(f"Game {game_number} | {'Playoffs' if is_playoffs else 'Regular Season'} | {model_label}")
    print("=" * 60)
    print(f"{blue_team}: {blue_prob:.1f}%")
    print(f"{red_team}: {red_prob:.1f}%")
    print(f"Predicted winner: {winner}")
    print("=" * 60)

    return {
        'blue_team': blue_team,
        'red_team': red_team,
        'blue_probability': blue_prob / 100,
        'red_probability': red_prob / 100,
        'predicted_winner': winner,
        'architecture': architecture,
        'model_label': model_label,
    }


print("Sample playoff matchups:")
simulate_matchup("Team Liquid PH", "AP.Bren", is_playoffs=True, game_number=1)
simulate_matchup("ONIC PH", "RSG PH", is_playoffs=True, game_number=1)
"""
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

with open("1_NoteBook/Prediction_v1.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Prediction_v1.ipynb rebuilt with pooled, split, and post-draft comparisons.")
