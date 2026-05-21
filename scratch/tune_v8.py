import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv('csv_data/ML_Feature_Matrix.csv')
df = df.sort_values(by=['season', 'match_id', 'game_number']).reset_index(drop=True)

df_g1 = df[df['game_number'] == 1].copy()

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
    'current_blue_side_advantage',
    'blue_comfort_patch_score', 'red_comfort_patch_score',
    'blue_lategame_winrate', 'red_lategame_winrate',
    'blue_avg_loss_duration', 'red_avg_loss_duration',
    'blue_execution_margin', 'red_execution_margin',
    'blue_execution_punish_score', 'red_execution_punish_score',
    # 'blue_draft_mastery', 'red_draft_mastery',
    # 'blue_execution_mastery', 'red_execution_mastery',
    # 'blue_draft_reliance', 'red_draft_reliance'
]

X = df_g1[base_features]
y = df_g1['target_blue_win']
weights = df_g1['time_weight']

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, step=0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    losses = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        w_train = weights.iloc[train_index]
        
        model = xgb.XGBClassifier(**param, verbosity=0)
        model.fit(X_train, y_train, sample_weight=w_train)
        preds = model.predict_proba(X_test)[:, 1]
        loss = log_loss(y_test, preds)
        losses.append(loss)
        
    return np.mean(losses)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300, n_jobs=-1)

print('Best parameters for Game 1:')
print(study.best_params)

# Test the best params to show accuracy
model = xgb.XGBClassifier(**study.best_params, random_state=42)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
w_train = weights.iloc[:split_idx]

model.fit(X_train, y_train, sample_weight=w_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f'Retuned Test Accuracy: {acc * 100:.2f}%')
