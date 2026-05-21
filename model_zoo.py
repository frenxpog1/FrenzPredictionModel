import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import math

class ModelZoo:
    def __init__(self, lambda_decay=0.18):
        self.lambda_decay = lambda_decay
        self.model_a = None # LightGBM
        
    def calculate_sample_weights(self, df):
        # Spec 5.2: W(match) = e^(-lambda * (Current_Season - Match_Season))
        current_season = df['season'].max()
        weights = df['season'].apply(lambda s: math.exp(-self.lambda_decay * (current_season - s)))
        return weights

    def temporal_train(self, df):
        """Spec 5.1: Chronological Walk-Forward Validation"""
        seasons = sorted(df['season'].unique())
        # Example: Train on seasons 1-16, validate on 17
        train_df = df[df['season'] < seasons[-1]]
        val_df = df[df['season'] == seasons[-1]]
        
        weights = self.calculate_sample_weights(train_df)
        
        # Spec 4.1: LightGBM Configuration
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'max_depth': 5, # Spec 4.1: Depth limit
            }
            
            dtrain = lgb.Dataset(train_df.drop(columns=['game_winner', 'match_id']), 
                                label=train_df['game_winner'], weight=weights)
            dval = lgb.Dataset(val_df.drop(columns=['game_winner', 'match_id']), 
                              label=val_df['game_winner'])
            
            gbm = lgb.train(params, dtrain, valid_sets=[dval])
            preds = gbm.predict(val_df.drop(columns=['game_winner', 'match_id']))
            return log_loss(val_df['game_winner'], preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        print(f"Optimal Params: {study.best_params}")
        self.model_a = lgb.LGBMClassifier(**study.best_params, max_depth=5)
        self.model_a.fit(train_df.drop(columns=['game_winner', 'match_id']), 
                         train_df['game_winner'], sample_weight=weights)

    def predict_win_prob(self, feature_vector):
        if not self.model_a: return 0.5
        return self.model_a.predict_proba(feature_vector)[0][1]
