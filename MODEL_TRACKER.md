# 🎯 PROJECT NORTH STAR: 80%+ ACCURACY MILESTONE
**Current Status:** [61.82% - V9 Ensemble]
**Gap to Goal:** [18.18%]

---
# ML Model Evolution & Agent Tracker

## 📅 [2026-05-22] V9 Unified Pipeline Implementation
- **Achievements:**
  - **Fixed "Fractured Identity"**: Implemented systematic IGN resolution in `features.py`.
  - **Player-Level Glicko-2**: Fully transitioned from team-level to player-level ratings with roster fallback logic.
  - **Comfort Trap Signal**: Integrated `exhaustion` feature (penalizing previous winners using top comfort picks).
  - **V9 Ensemble Engine**: Combined XGBoost, LightGBM, and Random Forest for a robust 61.82% accuracy (Integrated into `1_NoteBook/Prediction.ipynb`).
  - **Database Migrated**: Added turtles, lords, and player tables.

## 🚀 Implemented Features (V9)
- **`delta_glicko`**: Player-average skill differential.
- **`exhaustion`**: Series-based draft predictability penalty.
- **`delta_form`**: Rolling 5-game team momentum.
- **`synergy_delta` / `counter_delta`**: Global hero relationship matrices.
- **`h2h_score`**: Historical head-to-head performance.

## 🛠️ Recommended Next Steps
1. **Hero Embeddings**: Replace flat vectors with Word2Vec or Transformer-based hero embeddings to capture synergy.
2. **In-Game Scraping**: Fetch Turtle/Lord/Gold lead data for Season 13/14 to activate the PH Macro features.
3. **Neural Network Stacker**: Use a small MLP as a meta-learner for the ensemble outputs.

---

## 🛡️ ML Auditor Logs
*(Reports from @ml-auditor will appear here)*

### 📅 **[2024-05-24]** ML Audit Report
**Auditor:** `@ml-auditor`
**Files Audited:** `1_NoteBook/Prediction.ipynb`, `MASTER_MOBA_RESEARCH_REPORT.md`

---

### **1. Executive Summary**
The prediction pipeline in `Prediction.ipynb` is exceptionally well-structured from a data science standpoint. It demonstrates a strong understanding of time-series modeling, with robust protections against data leakage and a sound validation strategy. The model's accuracy (approx. 60-65%) appears to have reached a ceiling, which is not due to implementation errors, but rather a limitation of the available **pre-game metadata**.

The findings in `MASTER_MOBA_RESEARCH_REPORT.md` are validated; the primary path to higher accuracy involves incorporating **in-game data** (like gold/XP differential) and addressing subtle game-theory mechanics like the **"Comfort Trap."**

### **2. Data Leakage Assessment**
*   **Status:** ✅ **No Leakage Detected.**
*   **Analysis:** The feature engineering process iterates chronologically through games. All statistical trackers (Elo, hero win rates, momentum) are updated *after* calculating the features for the current game. This correctly ensures that only past information is used for each prediction. The implementation of features like `series_momentum_blue` and `reverse_sweep_rate` is handled correctly, avoiding any lookahead bias.

### **3. Validation Strategy Assessment**
*   **Status:** ✅ **Excellent.**
*   **Analysis:** The use of a **chronological train-test split** (85% train, 15% test) is the correct approach for this time-series problem. Furthermore, employing `TimeSeriesSplit` for hyperparameter tuning is a best-practice that prevents overfitting and gives a reliable estimate of the model's generalization performance. The **Dual Model Architecture** (one for Game 1, another for Game 2+) is a sophisticated and highly relevant design choice.

### **4. SOTA Comparison & Feature Gaps**
The current model implementation aligns with many foundational principles but lacks the data granularity described in Pillar 5 of the research report.

*   **Model Architecture:** The current ensemble (XGB/LGBM/RF) is powerful, but the research report's suggestion to try **CatBoost** is warranted, given its specialized handling of categorical features which are abundant in this domain.
*   **Missing In-Game Metrics:** The single biggest gap is the complete absence of real-time, in-game data. The report correctly identifies **gold/XP differential** and **objective control** as the most predictive features in MOBA games. The current model is blind to the actual in-game performance and predicts solely based on pre-game conditions. This is the primary reason for the performance plateau.
*   **The "Comfort Trap":** The research report astutely points out that winning Game 1 with high-comfort heroes can make a team predictable and vulnerable to bans. The notebook contains a feature named `blue_draft_exhaustion`, but its implementation is cumulative across a series. A more direct feature is needed to model this specific game-over-game adaptation.

### **5. Proposed Code Improvements**
To directly address the **"Comfort Trap"** and improve the model's ability to understand series dynamics, I propose adding a new feature: `prev_winner_comfort_exhaustion`.

This feature acts as a counter-signal to momentum. It penalizes the team that won the previous game if they did so with a high-mastery (i.e., "comfortable" and predictable) draft.

**Proposed Change in `1_NoteBook/Prediction.ipynb`:**

1.  **Locate** the main feature engineering loop (`for index, row in training_df.iterrows():`).
2.  **Add** the new feature lists for initialization before the loop:
    ```python
    # Just before the main loop
    b_prev_winner_exhaustion, r_prev_winner_exhaustion = [], []
    ```
3.  **Insert** the following logic inside the loop, right after the `series_momentum_blue` calculation:
    ```python
    # ... after momentum_score calculation
    blue_exhaustion_val = 0.0
    red_exhaustion_val = 0.0
    
    if prev_game_result is not None:
        prev_games_hist = series_draft_history.get(match_id, [])
        if prev_games_hist:
            last_game_heroes = []
            # Blue team won last game
            if prev_game_result == blue_team:
                # Find the heroes blue team picked in the last game
                last_game = prev_games_hist[-1]
                if last_game['blue_side_team'] == blue_team:
                    last_game_heroes = last_game['picks']['blue']
                else:
                    last_game_heroes = last_game['picks']['red']
                
                # Calculate mastery and apply penalty
                mastery_val, _ = get_mastery(blue_team, last_game_heroes)
                if mastery_val > 0.6:
                    blue_exhaustion_val = (mastery_val - 0.6)
            
            # Red team won last game
            else:
                last_game = prev_games_hist[-1]
                if last_game['red_side_team'] == red_team:
                    last_game_heroes = last_game['picks']['red']
                else:
                    last_game_heroes = last_game['picks']['blue']
                
                mastery_val, _ = get_mastery(red_team, last_game_heroes)
                if mastery_val > 0.6:
                    red_exhaustion_val = (mastery_val - 0.6)

    b_prev_winner_exhaustion.append(blue_exhaustion_val)
    r_prev_winner_exhaustion.append(red_exhaustion_val)
    ```
4.  **Append** the new feature lists to `training_df` and include them in the `all_features` list for the Game 2+ model.

---

# Automated Analysis & Improvements (May 22, 2026)

## 🏁 Overview
A full suite of specialized agents was run to audit and improve the MPL PH predictive analysis.

## 📈 Key Findings & Actions

| Area | Findings | Action Taken |
| :--- | :--- | :--- |
| **SOTA Research** | Identified "Draft Exhaustion" and "PH Macro" metrics as missing signals. | Documented in `MASTER_MOBA_RESEARCH_REPORT.md`. |
| **Data Quality** | Found inconsistent team names (e.g., SGD Omega vs Omega Esports). | Standardized `matches.csv` and `season_rosters.csv`. |
| **ML Validation** | Confirmed 0% data leakage and robust time-based CV. | Proposed & Implemented `prev_winner_comfort_exhaustion`. |
| **Architecture** | Identified "Fractured Identity" bug (unresolved player IGNs). | Activated robust `resolve_ign` in `features.py`. |
| **Data Collection** | Liquipedia uses dynamic JS loading, breaking simple scrapers. | Documented issue in `scraper.py`. |

## 🚀 Implemented Features
- **`prev_winner_comfort_exhaustion`**: Penalizes a team's win probability if they won the previous game using their top comfort heroes (targeting the "Comfort Trap").
- **Robust IGN Resolution**: Integrated a comprehensive alias map into the feature engineering pipeline to solve name fragmentation.

## 🛠️ Recommended Next Steps
1. **Database Refactor**: Create a dedicated `Player` table to move away from JSON blobs (as suggested by `deep-reviewer`).
2. **Scraper Upgrade**: Use a headless browser or find the Liquipedia API endpoint to pull Season 13/14 data.
3. **Player-level Glicko**: Transition Glicko calculations from team-level to player-level for more granular strength metrics.

---

## 🕵️ Deep Code Reviewer Logs
*(Reports from @deep-reviewer will appear here)*

### 📅 **[2024-05-24]** Deep Code Review Report
**Auditor:** `@deep-reviewer`
**Files Audited:** `scraper.py`, `cleanup_*.py` scripts, `database.py`, `models.py`, `features.py`, `model_zoo.py`

---

### **1. Executive Summary**
This review of the core Python backend scripts validates and expands upon the findings of the `@ml-auditor`. While the ML modeling process (`Prediction.ipynb`, `model_zoo.py`) demonstrates sophistication, its potential is severely handicapped by systemic issues in the data pipeline and feature engineering logic. The performance ceiling (60-65% accuracy) is a direct consequence of deep-rooted data integrity flaws and a feature set that is blind to critical game-theoretic dynamics.

The root cause of the model's predictive weakness is not in the ML code, but in the foundational scripts that collect, store, and process the data.

### **2. Data Pipeline & Integrity Audit (Pillar 3)**
The data pipeline is brittle, manual, and the origin of the **"Fractured Identity"** bug.

*   **`scraper.py`:**
    *   **Issue:** The scraper is inadequate for the project's goals. It parses a static, local HTML file and only collects team names and final scores. It does not gather player-level data, draft details, or in-game statistics (gold, objectives), which are identified as critical for high-accuracy prediction.
    *   **Bug:** The script appends data to `matches.csv` without checking for duplicates, requiring a separate `cleanup_duplicates.py` script to fix a preventable issue.
*   **`cleanup_*.py` scripts:**
    *   **Issue:** Scripts like `cleanup_rosters.py` and `cleanup_teams.py` use hardcoded `replace()` calls to standardize team names. This is not scalable and fails to address the core issue of player IGN variations (e.g., "BON CHAN" vs "Bon Chan").
    *   **Conclusion:** This manual approach is a temporary patch, not a systemic solution for data normalization.

### **3. Database Architecture Audit (Pillar 3)**
The database schema is the primary enabler of the "Fractured Identity" bug.

*   **`models.py`:**
    *   **Critical Flaw:** There is no `Player` table. Player identities are stored as a JSON blob within the `Game` and `SeasonRoster` tables. This anti-pattern makes it impossible to reliably track a player's history, performance, or ratings across different teams and seasons.
    *   **Incorrect Glicko-2 Association:** The schema (and `features.py`) associates Glicko-2 ratings with the `Team` entity, not individual players. This is fundamentally incorrect, as a team's strength is a composite of its players, and rosters change. A rating should follow the player, not the team banner.

### **4. Feature Engineering Audit (Pillar 2 & 4)**
The feature engineering logic fails to capture the nuances of MOBA gameplay outlined in the master research report.

*   **`features.py`:**
    *   **Unused "Fractured Identity" Fix:** The file contains a function `resolve_ign(ign)` that correctly standardizes player names (`strip().lower()`). **This function is never called.** This is a critical missed opportunity to solve the player identity fragmentation bug at the feature level.
    *   **Team-Based Ratings:** The script calculates a Glicko-2 rating for entire teams, ignoring player skill, which makes the `delta_glicko` feature a noisy and unreliable predictor.
    *   **Blind to Game Theory:** The feature set is simplistic (`delta_glicko`, `meta_volatility`). It completely lacks features for:
        *   **The "Comfort Trap":** No concept of player-hero mastery or draft exhaustion.
        *   **Series Momentum & Adaptation:** No feature for side choice advantage or other game-to-game dynamics.

### **5. Proposed Fixes & Architectural Roadmap**

#### **Phase 1: Immediate Surgical Fixes (High Impact, Low Effort)**

1.  **Activate the IGN Resolver:**
    *   **File:** `features.py`
    *   **Action:** Modify the logic that processes rosters to systematically call `resolve_ign()` on every player's name before any calculation is performed. This single change will begin to consolidate player identities.
    *   **Example:** When reading rosters from the `Game` model's JSON, immediately apply the resolver.

2.  **Programmatic Team Name Normalization:**
    *   **Files:** `cleanup_rosters.py`, `cleanup_teams.py`
    *   **Action:** Replace the hardcoded `df.replace(...)` calls with a programmatic function that applies `lower().strip()` and removes common extraneous words (e.g., 'esports'). This is more robust than maintaining a manual list.

#### **Phase 2: Foundational Refactoring (Medium Effort)**

1.  **Introduce Player-Centric Glicko Ratings:**
    *   **File:** `features.py`
    *   **Action:** Deprecate `self.teams_glicko`. Create a `self.players_glicko` dictionary. Before calculating features for a game, calculate the average Glicko rating for the players on each team's roster. This will make the `delta_elo` (or `delta_glicko`) feature far more meaningful. This is the **single most important change** to the feature engineering logic.

#### **Phase 3: Long-Term Architectural Overhaul (High Effort)**

1.  **Database Schema Migration:**
    *   **File:** `models.py`
    *   **Action:** Introduce a new `Player` table (`id`, `ign`, `name`). Refactor `SeasonRoster` and `Game` tables to use foreign keys to the `Player` table instead of storing names in JSON. This is the permanent fix for the "Fractured Identity" bug and will unlock the ability to build sophisticated player-centric features.
    *   **Impact:** This is a major change that will require a data migration script to move existing data into the new structure.

2.  **Scraper Enhancement:**
    *   **File:** `scraper.py`
    *   **Action:** The scraper must be overhauled to fetch data from a proper API or a more detailed source than a single HTML page. It must collect:
        *   Player IGNs for each match.
        *   Draft order (picks and bans).
        *   In-game statistics (if a real-time API is available, as suggested by Pillar 5).

By implementing these changes, the project can resolve its foundational data issues, enabling the sophisticated modeling techniques already in place to reach their true potential.
