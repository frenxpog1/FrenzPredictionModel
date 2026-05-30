import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import math

class ModelZoo:
    """
    ModelZoo contains experimental LightGBM and Optuna walk-forward hyperparameter optimization code.
    It is compatible with 'ML_Feature_Matrix.csv' and uses 'target_blue_win' as the label.
    """
    def __init__(self, lambda_decay=0.18):
        self.lambda_decay = lambda_decay
        self.model_a = None  # LightGBM Classifier
        
    def calculate_sample_weights(self, df):
        # W(match) = e^(-lambda * (Current_Season - Match_Season))
        current_season = df['season'].max()
        weights = df['season'].apply(lambda s: math.exp(-self.lambda_decay * (current_season - s)))
        return weights

    def temporal_train(self, df):
        """Chronological Walk-Forward Validation using LightGBM and Optuna"""
        seasons = sorted(df['season'].unique())
        # Train on all prior seasons, validate on the most recent season
        train_df = df[df['season'] < seasons[-1]].copy()
        val_df = df[df['season'] == seasons[-1]].copy()
        
        weights = self.calculate_sample_weights(train_df)
        
        # Filter down to numeric features and align targets
        def prepare_data(data_frame):
            X = data_frame.select_dtypes(include=[np.number]).copy()
            y = X['target_blue_win'] if 'target_blue_win' in X.columns else data_frame['target_blue_win']
            
            # Drop unnecessary columns if they exist
            cols_to_drop = ['target_blue_win', 'match_id', 'game_number', 'season']
            for col in cols_to_drop:
                if col in X.columns:
                    X = X.drop(columns=[col])
            return X, y

        X_train, y_train = prepare_data(train_df)
        X_val, y_val = prepare_data(val_df)
        
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
                'max_depth': 5,
            }
            
            dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
            dval = lgb.Dataset(X_val, label=y_val)
            
            gbm = lgb.train(params, dtrain, valid_sets=[dval])
            preds = gbm.predict(X_val)
            return log_loss(y_val, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)
        
        print(f"Optimal LightGBM Params: {study.best_params}")
        self.model_a = lgb.LGBMClassifier(**study.best_params, max_depth=5, verbosity=-1)
        self.model_a.fit(X_train, y_train, sample_weight=weights)

    def predict_win_prob(self, feature_vector):
        if not self.model_a:
            return 0.5
        # Filter down feature vector to align with model inputs
        if isinstance(feature_vector, pd.DataFrame):
            cols_to_drop = ['target_blue_win', 'match_id', 'game_number', 'season']
            feature_vector = feature_vector.select_dtypes(include=[np.number]).copy()
            for col in cols_to_drop:
                if col in feature_vector.columns:
                    feature_vector = feature_vector.drop(columns=[col])
        return self.model_a.predict_proba(feature_vector)[0][1]
