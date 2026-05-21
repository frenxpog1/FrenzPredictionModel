import json
import os

notebook_path = "1_NoteBook/Prediction.ipynb"

if not os.path.exists(notebook_path):
    print(f"❌ Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded notebook: {notebook_path}")
print(f"Total cells: {len(nb['cells'])}")

modified_part1 = False
modified_part2 = False

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    
    source_text = "".join(cell["source"])
    
    # 1. Look for final_features in Part 1
    if "final_features = [" in source_text and "target_blue_win" in source_text and not modified_part1:
        print(f"Found Part 1 features cell at index {idx}.")
        
        # We will replace the final_features definition block
        new_source = []
        in_features_block = False
        
        for line in cell["source"]:
            if "final_features = [" in line:
                new_source.append(line)
                new_source.append("    'match_timestamp', 'match_id', 'season', 'game_number', 'patch_version',\n")
                new_source.append("    'blue_side_team', 'red_side_team',\n")
                in_features_block = True
                continue
            
            if in_features_block:
                # Skip the original ID/side columns as they are added above
                if any(x in line for x in ["'match_id'", "'season'", "'game_number'", "'patch_version'", "'blue_side_team'", "'red_side_team'"]):
                    continue
                # Add stage to target
                if "'target_blue_win'" in line:
                    new_source.append("    'time_weight', 'target_blue_win', 'stage'\n")
                    continue
                if "'time_weight'" in line:
                    continue
                if "]" in line and len(line.strip()) == 1:
                    new_source.append(line)
                    in_features_block = False
                    continue
            
            new_source.append(line)
            
        cell["source"] = new_source
        modified_part1 = True
        print(f"✅ Part 1 features cell updated successfully.")

    # 2. Look for the training loop in Part 2
    if "df = pd.read_csv" in source_text and "blue_series_score_list = []" in source_text and not modified_part2:
        print(f"Found Part 2 modeling cell at index {idx}.")
        
        # We will replace this entire cell with the clean training code
        # matching what copy_code.py successfully ran.
        new_source = [
            "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
            "from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV\n",
            "from sklearn.metrics import accuracy_score\n",
            "import lightgbm as lgb\n",
            "\n",
            "print(\"\\n📊 Loading Feature Matrix for V7 Engine...\")\n",
            "df = pd.read_csv(f'{csv_dir}/ML_Feature_Matrix.csv')\n",
            "\n",
            "# ==========================================\n",
            "# NEW V7 FEATURE: ELIMINATION PRESSURE\n",
            "# Computes series score for each team going INTO each game.\n",
            "# ==========================================\n",
            "print(\"🔧 Engineering Elimination Pressure features...\")\n",
            "\n",
            "df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])\n",
            "df = df.sort_values(['match_timestamp', 'game_number']).reset_index(drop=True)\n",
            "\n",
            "blue_series_score_list = []\n",
            "red_series_score_list  = []\n",
            "is_elim_game_list      = []\n",
            "valid_games_mask       = []\n",
            "\n",
            "# Track running wins per match\n",
            "series_wins_tracker = {}   # match_id -> {blue_team: wins, red_team: wins}\n",
            "\n",
            "for _, row in df.iterrows():\n",
            "    mid      = row['match_id']\n",
            "    blue     = row['blue_side_team']\n",
            "    red      = row['red_side_team']\n",
            "    gnum     = row['game_number']\n",
            "    winner   = row['blue_side_team'] if row['target_blue_win'] == 1 else row['red_side_team']\n",
            "    is_playoffs = row['is_playoffs']\n",
            "\n",
            "    if mid not in series_wins_tracker:\n",
            "        series_wins_tracker[mid] = {blue: 0, red: 0}\n",
            "\n",
            "    b_score = series_wins_tracker[mid].get(blue, 0)\n",
            "    r_score = series_wins_tracker[mid].get(red, 0)\n",
            "    \n",
            "    # Dataset bug mitigation: if team already won 2 games in BO3 (Regular Season), match is over.\n",
            "    # If the scraper logged a Game 3 anyway, it's invalid.\n",
            "    if is_playoffs == 0 and max(b_score, r_score) >= 2:\n",
            "        valid_games_mask.append(False)\n",
            "    # Filter Game 4/5 from BO3 just in case, though max score handles it\n",
            "    elif is_playoffs == 0 and gnum > 3:\n",
            "        valid_games_mask.append(False)\n",
            "    # Playoff BO5/BO7 filter (max wins 4)\n",
            "    elif is_playoffs == 1 and max(b_score, r_score) >= 4:\n",
            "        valid_games_mask.append(False)\n",
            "    else:\n",
            "        valid_games_mask.append(True)\n",
            "\n",
            "    blue_series_score_list.append(b_score)\n",
            "    red_series_score_list.append(r_score)\n",
            "\n",
            "    # Is this an elimination game?\n",
            "    is_elim = 1 if (b_score == r_score and gnum >= 3) or \\\n",
            "                   (abs(b_score - r_score) >= 1 and gnum >= 3) else 0\n",
            "    is_elim_game_list.append(is_elim)\n",
            "\n",
            "    # Update AFTER reading\n",
            "    series_wins_tracker[mid][winner] = series_wins_tracker[mid].get(winner, 0) + 1\n",
            "\n",
            "df['blue_series_score'] = blue_series_score_list\n",
            "df['red_series_score']  = red_series_score_list\n",
            "df['is_elimination_game'] = is_elim_game_list\n",
            "df['score_diff_blue']   = df['blue_series_score'] - df['red_series_score']\n",
            "df['is_valid_game']     = valid_games_mask\n",
            "\n",
            "df = df[df['is_valid_game'] == True].copy()\n",
            "print(f\"   Cleaned invalid dataset entries: kept {len(df)} valid games.\")\n",
            "\n",
            "# ==========================================\n",
            "# DEFINE FULL FEATURE SET\n",
            "# ==========================================\n",
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
            "# REMOVED MOMENTUM FEATURES DUE TO SCRAPER BUG\n",
            "# The scraper sorted games within a match by winner. So game_number sequence is entirely corrupted\n",
            "# and using momentum features (e.g. who won game 2, or score diffs) guarantees 100% leakage.\n",
            "all_features = base_features\n",
            "\n",
            "# ==========================================\n",
            "# DUAL MODEL SPLIT (Game 1 vs Game 2+)\n",
            "# ==========================================\n",
            "df_g1   = df[df['game_number'] == 1].copy()\n",
            "df_g2plus = df[df['game_number'] > 1].copy()\n",
            "\n",
            "X_g1      = df_g1[base_features];     y_g1  = df_g1['target_blue_win'];     w_g1  = df_g1['time_weight']\n",
            "X_g2plus  = df_g2plus[all_features];  y_g2  = df_g2plus['target_blue_win']; w_g2  = df_g2plus['time_weight']\n",
            "\n",
            "# Normalize weights so XGBoost doesn't produce empty trees\n",
            "w_g1 = w_g1 / w_g1.mean()\n",
            "w_g2 = w_g2 / w_g2.mean()\n",
            "\n",
            "# Chronological train/test split (15% holdout)\n",
            "split_g1    = int(len(df_g1) * 0.85)\n",
            "split_g2    = int(len(df_g2plus) * 0.85)\n",
            "\n",
            "X_train_g1,    X_test_g1    = X_g1.iloc[:split_g1],    X_g1.iloc[split_g1:]\n",
            "y_train_g1,    y_test_g1    = y_g1.iloc[:split_g1],    y_g1.iloc[split_g1:]\n",
            "w_train_g1,    w_test_g1    = w_g1.iloc[:split_g1],    w_g1.iloc[split_g1:]\n",
            "\n",
            "X_train_g2,    X_test_g2    = X_g2plus.iloc[:split_g2],  X_g2plus.iloc[split_g2:]\n",
            "y_train_g2,    y_test_g2    = y_g2.iloc[:split_g2],      y_g2.iloc[split_g2:]\n",
            "w_train_g2,    w_test_g2    = w_g2.iloc[:split_g2],      w_g2.iloc[split_g2:]\n",
            "\n",
            "print(f\"   Game 1 model  → {len(X_train_g1)} train / {len(X_test_g1)} test\")\n",
            "print(f\"   Game 2+ model → {len(X_train_g2)} train / {len(X_test_g2)} test\")\n"
        ]
        cell["source"] = new_source
        modified_part2 = True
        print(f"✅ Part 2 modeling cell updated successfully.")

# Save modified notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n🎉 Notebook saved successfully!")
