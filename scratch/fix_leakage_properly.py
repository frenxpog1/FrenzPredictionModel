import nbformat
import os

def fix_leakage_and_accuracy():
    nb_path = '1_NoteBook/Prediction.ipynb'
    with open(nb_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Modeling Source with STRICT split and proper evaluation
    modeling_source = [
        "import xgboost as xgb\n",
        "from sklearn.metrics import accuracy_score, log_loss\n",
        "\n",
        "print(\"\\n📊 Loading Feature Matrix for V7 Engine...\")\n",
        "df = pd.read_csv(f'{csv_dir}/ML_Feature_Matrix.csv')\n",
        "\n",
        "df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])\n",
        "df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)\n",
        "\n",
        "base_features = [\n",
        "    'blue_side_elo', 'red_side_elo',\n",
        "    'blue_playoff_elo', 'red_playoff_elo',\n",
        "    'blue_comfort_wr', 'red_comfort_wr',\n",
        "    'blue_draft_experience', 'red_draft_experience',\n",
        "    'blue_global_draft_wr', 'red_global_draft_wr',\n",
        "    'blue_ban_disruption', 'red_ban_disruption',\n",
        "    'blue_buffs_in_draft', 'blue_nerfs_in_draft',\n",
        "    'red_buffs_in_draft',  'red_nerfs_in_draft',\n",
        "    'blue_momentum', 'red_momentum',\n",
        "    'blue_h2h_winrate',\n",
        "    'blue_patch_practice', 'red_patch_practice',\n",
        "    'is_playoffs',\n",
        "    'blue_playoff_clutch', 'red_playoff_clutch',\n",
        "    'blue_playoff_exp', 'red_playoff_exp',\n",
        "    'blue_g3_clutch_wr', 'red_g3_clutch_wr',\n",
        "    'blue_reverse_sweep_rate', 'red_reverse_sweep_rate',\n",
        "    'blue_rest_factor', 'red_rest_factor',\n",
        "    'blue_avg_win_duration', 'red_avg_win_duration',\n",
        "    'current_blue_side_advantage'\n",
        "]\n",
        "\n",
        "# Split into G1 and G2+\n",
        "df_g1 = df[df['game_number'] == 1].copy()\n",
        "df_g2plus = df[df['game_number'] > 1].copy()\n",
        "\n",
        "# 85% Chronological Split (No future-data leakage)\n",
        "split_g1 = int(len(df_g1) * 0.85)\n",
        "split_g2 = int(len(df_g2plus) * 0.85)\n",
        "\n",
        "train_g1, test_g1 = df_g1.iloc[:split_g1], df_g1.iloc[split_g1:]\n",
        "train_g2, test_g2 = df_g2plus.iloc[:split_g2], df_g2plus.iloc[split_g2:]\n",
        "\n",
        "print(f\"🤖 Training on {len(train_g1) + len(train_g2)} games...\")\n",
        "print(f\"🧪 Testing on {len(test_g1) + len(test_g2)} unseen games...\")\n",
        "\n",
        "# Optimized Params for real-world accuracy (Regularization)\n",
        "params = {\n",
        "    'n_estimators': 150,\n",
        "    'max_depth': 4,\n",
        "    'learning_rate': 0.05,\n",
        "    'subsample': 0.8,\n",
        "    'colsample_bytree': 0.8,\n",
        "    'random_state': 42\n",
        "}\n",
        "\n",
        "model_g1 = xgb.XGBClassifier(**params).fit(train_g1[base_features], train_g1['target_blue_win'], sample_weight=train_g1['time_weight'])\n",
        "model_g2 = xgb.XGBClassifier(**params).fit(train_g2[base_features], train_g2['target_blue_win'], sample_weight=train_g2['time_weight'])\n",
        "\n",
        "print(\"\\n🏆 REAL-WORLD VALIDATION RESULTS (Unseen Games):\")\n",
        "test_acc_g1 = accuracy_score(test_g1['target_blue_win'], model_g1.predict(test_g1[base_features]))\n",
        "test_acc_g2 = accuracy_score(test_g2['target_blue_win'], model_g2.predict(test_g2[base_features]))\n",
        "print(f\"   Game 1  Test Accuracy : {test_acc_g1:.2%}\")\n",
        "print(f\"   Game 2+ Test Accuracy : {test_acc_g2:.2%}\")\n",
        "\n",
        "print(\"\\n📈 Training (Memorization) Accuracy (for reference):\")\n",
        "train_acc_g1 = accuracy_score(train_g1['target_blue_win'], model_g1.predict(train_g1[base_features]))\n",
        "train_acc_g2 = accuracy_score(train_g2['target_blue_win'], model_g2.predict(train_g2[base_features]))\n",
        "print(f\"   Game 1  Train Accuracy : {train_acc_g1:.2%}\")\n",
        "print(f\"   Game 2+ Train Accuracy : {train_acc_g2:.2%}\")\n"
    ]

    # Update the modeling cell (index 2)
    nb.cells[2].source = "".join(modeling_source)
    
    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)

fix_leakage_and_accuracy()
print('Notebook modeling cell fixed to prevent leakage.')
