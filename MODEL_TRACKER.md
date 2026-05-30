# Model Tracker

This file records the current state and performance of the MPL Philippines match prediction pipeline in plain terms.

## Current Status

The project is an optimized ML prediction pipeline. It validates in the **70.70%** range on the latest chronological holdout checks using the rebuilt, optimized ensemble pipeline.

Recent verified chronological holdout results:

| Evaluation Partition | Accuracy | Correct Predictions | Fraction | Model Focus |
| :--- | :---: | :---: | :---: | :--- |
| **Game 1 (Initial matchup)** | **79.56%** | $109\ /\ 137$ | $0.7956$ | Static team strength, baseline roster synergy, comfort pools |
| **Game 2+ (In-series maps)** | **65.53%** | $154\ /\ 235$ | $0.6553$ | Live series momentum, in-series draft exhaustion, elim pressure |
| **Combined Pipeline** | **70.70%** | **$263\ /\ 372$** | **$0.7070$** | Unified predictive score across all tournament game outcomes |

---

## Active Pipeline Configuration

- **Data Source**: Local SQLite database `mlbb_data.db` and exported CSV files in `csv_data/`.
- **Validation Splitting**: Grouped chronological split ($85\%$ train / $15\%$ test holdout, split by `match_id` to prevent contamination).
- **Research Notebook**: Generated dynamically at `1_NoteBook/Prediction_v1.ipynb`.
- **Active Compilation Script**: `create_prediction_v1_tuned.py`.
- **FastAPI Dashboard**: Main app hosted via `main.py` with views under `templates/` and static resources under `static/`.

---

## Optimized Classifiers & Parameters

### Game 1 Predictor
* **Model**: Single `XGBClassifier`
* **Configuration**: `n_estimators=400`, `max_depth=2`, `learning_rate=0.02`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=2.0`.
* **Accuracy**: **79.56%** (Chronological holdout)

### Game 2+ Predictor
* **Model**: Weighted blend of three parallel classifiers:
  $$\text{Ensemble Prediction} = \frac{1 \times \text{CatBoost}_1\ +\ 3 \times \text{RandomForest}\ +\ 2 \times \text{CatBoost}_2}{6}$$
* **Classifiers**:
  * `CatBoostClassifier` (deep): 100 iterations, depth 4, learning rate 0.05, L2 leaf regularizer 3.0.
  * `RandomForestClassifier`: 200 estimators, max depth 2.
  * `CatBoostClassifier` (shallow): 50 iterations, depth 4.
* **Accuracy**: **65.53%** (Chronological holdout)

---

## Active Feature Groups
- **Team & Player Strength**: Separate dynamic regular season and playoff ELO tracks computed post-game on-the-fly.
- **Roster & Experience**: Retained roster stability indexes across seasons, aggregate player championship wins, and playoff game counts.
- **Team Form & Head-to-Head**: Bayesian-smoothed Head-to-Head win rate and rolling 5-match team momentum.
- **Patch comfort**: Alignment of top 12 comfort pools with active patch adjustments (BUFF/NERF).
- **In-Series States (Game 2+)**: Roster pick exhaustion, side-swap tracking, series score differentials, and acute elimination pressure checks.

---

## Verification & Integrity Check
All active repository assets are verified by the CI/CD integrity checker:
```bash
python sentinel.py
```
**Status**: `All checks passed.`
