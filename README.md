# Frenz Prediction Model: MPL PH Match Outcome Pipeline

This repository contains a research-grade machine learning pipeline for predicting match outcomes in the **Mobile Legends: Bang Bang Professional League Philippines (MPL PH)** across Seasons 5 to 17. 

By applying strict chronological holdout validation, sequentially-updated ELO ratings, comfort pool analysis, and robust ensemble classifiers, this model delivers state-of-the-art predictive accuracy while maintaining complete resistance to data leakage.

---

## 📊 Empirical Results

The pipeline's performance is rigorously audited on a **chronological test set** representing the final 15% of matches (chronological future). 

Following hyperparameter optimization, the models achieve the following classification accuracies:

| Evaluation Subset | Accuracy | Exact Match Counts | Predictive Focus |
| :--- | :---: | :---: | :--- |
| **Game 1 (Initial Matchup)** | **79.56%** | $109\ /\ 137$ | Historical franchise strength, baseline roster synergy, comfort picks |
| **Game 2+ (In-Series Maps)** | **65.53%** | $154\ /\ 235$ | Dynamic series momentum, in-series draft exhaustion, elimination pressure |
| **Combined Pipeline** | **70.70%** | **$263\ /\ 372$** | Overall predictive performance across all map outcomes |

### Key Research Findings
1. **Initial prediction advantage**: Game 1 matchups display exceptionally high predictability ($79.56\%$), indicating that base team strength, patch alignment, and historical franchise head-to-heads carry high predictive power when teams are fresh.
2. **In-series volatility**: Standalone Game 2+ accuracy ($65.53\%$) reflects the high-variance nature of live professional esports, driven by adaptive coaching, tactical adjustments, side swaps, and acute psychological pressure during elimination maps.

---

## 🔬 Methodology & Core Architecture

### 1. Zero-Leakage Chronological Group Validation
Predicting sports outcomes requires strict temporal ordering to prevent lookahead bias. 
* **Splitting Strategy**: Matches are sorted chronologically. The first 85% of matches constitute the training subset; the remaining 15% are held out exclusively for testing.
* **Match-ID Grouping**: Splits are grouped strictly by `match_id`. Games belonging to the same series are never fragmented or leaked across the training and testing partition boundary.

```
[--- Train Subset: Seasons 5 to 15 (85%) ---] [--- Test Holdout: Seasons 15 to 17 (15%) ---]
◀───────────────────────────────── Time Scale ─────────────────────────────────▶
```

### 2. Sequential Dynamic ELO Ratings
Rather than relying on static season-end rankings, the pipeline computes team and player strengths dynamically:
* **Franchise Mapping**: Successfully resolves historic rebrands contextually across seasons (e.g. mapping *Sunsparks* $\rightarrow$ *ECHO* $\rightarrow$ *Team Liquid PH*, and *AP.Bren* $\rightarrow$ *Team Falcons PH*) to preserve historical Elo memory.
* **Leak-Safe Updates**: Player Elo ratings are updated **strictly post-match**. For any given map prediction, ELO ratings reflect the history *prior* to that series, preventing in-series result leakage.
* **Decay Mechanics**: Applies a $15\%$ decay rate during season transitions to account for roster moves, patch disruption, and offseason meta-shifts.

### 3. Integrated Feature Groups
The pipeline synthesizes over 50 predictive signals across five operational pillars:
1. **Franchise & Player Strength**: Dynamic regular and playoff ELO tracks, championship titles (DNA), and playoff experience counts.
2. **Team Form & Head-to-Head (H2H)**: Bayesian-smoothed head-to-head records and rolling 5-match team momentum.
3. **Patch Comfort & Draft Synergy**: Bayesian-smoothed win rates of players' top 5 most-played comfort heroes, draft synergy, and patch-adaptation comfort alignment scores.
4. **Roster Stability**: Retained player rosters across season shifts.
5. **Series-State Variables (Game 2+)**: Roster exhaustion, cumulative draft overlap, side-swap advantages, series momentum, and acute elimination game pressure.

---

## 🛠️ Model Configurations

To handle different data densities, the pipeline uses a dual-model architecture:

### Game 1 Model: Robust Shallow XGBoost
Optimized for high bias resistance and generalization under high feature dimensionalities.
* **Classifier**: `xgb.XGBClassifier`
* **Parameters**: `n_estimators=400`, `max_depth=2`, `learning_rate=0.02`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=2.0`.

### Game 2+ Model: Custom Weighted Ensemble
Designed to capture complex interaction effects of in-series dynamics.
* **Blending Ratio**: $1 \times \text{CatBoost}_1\ +\ 3 \times \text{RandomForest}\ +\ 2 \times \text{CatBoost}_2$ (Normalizing factor of $6$).
* **Classifiers**:
  * `CatBoostClassifier`: 100 iterations, depth 4, learning rate 0.05, L2 leaf regularizer 3.0.
  * `RandomForestClassifier`: 200 estimators, max depth 2.
  * `CatBoostClassifier` (shallow): 50 iterations, depth 4.

---

## 📂 Repository Layout

| Path | Purpose |
| :--- | :--- |
| **`1_NoteBook/Prediction_v1.ipynb`** | Main research notebook containing the full, reproducible prediction pipeline. |
| **`create_prediction_v1_tuned.py`** | Tuned python pipeline script that automatically rebuilds the research notebook. |
| **`database.py` / `models.py`** | SQLite schema and SQLAlchemy models representing the local data warehouse. |
| **`scraper.py`** | Dynamic Liquipedia scraper utilizing Headless Fetching for match history. |
| **`export_to_csv.py`** | Utility to extract data tables from the SQLite database to clean CSV directories. |
| **`export_data.py`** | Pre-compiles database stats to a static asset `static/data.js` for zero-latency UI load. |
| **`main.py`** | FastAPI server hosting the analytics dashboard. |
| **`sentinel.py`** | CI/CD style integrity checker to verify syntax and pipeline validity. |
| **`csv_data/`** | Repository containing all exported tables and the generated `ML_Feature_Matrix.csv`. |
| **`templates/` / `static/`** | Frontend views and CSS stylings for the HTML dashboard. |

---

## 🚀 Reproduction & Usage

### 1. Environment Setup
Clone the repository, create a virtual environment, and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rebuilding the Research Pipeline
Re-run the optimized hyperparameter search, generate the latest dataset splits, train the models, and compile the research notebook:
```bash
python create_prediction_v1_tuned.py
```

### 3. Pipeline Integrity Check
Ensure all Python files and the generated research notebook are syntactically and logically clean:
```bash
python sentinel.py
```

### 4. Running the Dashboard
Launch the FastAPI analytical server to inspect historical tournament trends, rosters, and patch adjustments:
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000` in your web browser.
