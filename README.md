# MOBA Win Prediction Engine 🏆

A high-performance machine learning pipeline designed to predict professional MOBA match outcomes (MLBB MPL). By leveraging player-centric metrics, dynamic Glicko-2 tracking, and advanced game theory adaptation, this engine systematically captures the complexities of professional esports series.

## 🚀 Overview

The predictive landscape of competitive MOBAs is highly volatile. Our engine tackles this by distinguishing between the "blank slate" of Game 1 and the complex psychological and strategic shifts of Game 2 and beyond. The latest **V9 Unified Pipeline** represents a major shift toward granular, player-centric analysis and rigorous leak-safe validation.

### 🏁 V9 Unified Pipeline (Production Champion)

The V9 architecture moves beyond simple team-based Elo, treating each match as a composite of individual player skill and series-specific momentum.

**Key Analytical Pillars:**
*   **Tri-State Partitioning:** Breaking the series down into three distinct evaluative states:
    *   **Game 1 (Pre-Match):** Pure structural evaluation using historical H2H, team chemistry, and base skill.
    *   **Game 2 (Adaptation):** Introduces immediate series momentum and Game 1 "Comfort Trap" signals.
    *   **Game 3+ (Late-Series Clutch):** Accounts for high-pressure variables, deciding-game resilience, and fatigue.
*   **Player-Centric Glicko-2:** Skill ratings are now tracked at the individual player level, resolving the "Fractured Identity" bug and allowing for accurate predictions even after roster shifts.
*   **Leak-Safe Feature Engineering:** Strict isolation ensures current-game draft features are entirely blocked from pre-match models, ensuring zero data leakage and rigorous real-world validation.
*   **Unified Ensemble Meta-Learner:** Integrates **CatBoost**, XGBoost, LightGBM, and Random Forest via a Soft Voting Classifier to maximize generalization across different game states.

---

## ✨ SOTA Engine Features

*   **Dual-Track Elo/Glicko System:** Tracks separate ratings for Regular Season and Playoffs to capture "clutch" performances and structural team scaling under pressure.
*   **Match-Level Updates:** Ratings are updated at the end of a match rather than per-game, eliminating extreme recency bias and "volatility traps".
*   **Franchise & IGN Resolver:** Consolidates fragmented player identities and dynamically maps organizational rebrands (e.g., AP.Bren -> Team Falcons PH, Blacklist -> Aurora) right at the ingestion layer.
*   **The Comfort Exhaustion Flag:** Penalizes a team's win probability if they won the previous game using their top comfort heroes (targeting the "Comfort Trap").
*   **PH Macro Metrics:** Incorporates Turtle and Lord control rates as high-signal predictors for macro-dominance.

## 🛠 Tech Stack

*   **Core:** Python 3.10+
*   **Machine Learning:** XGBoost, LightGBM, CatBoost, Scikit-Learn
*   **Data Processing:** Pandas, NumPy, SQLAlchemy (SQLite)
*   **Scraping & Data Acquisition:** Playwright, Scrapling (Liquipedia source integration)
*   **Experimentation:** Jupyter Notebooks

## 📂 Project Structure

*   `create_prediction_v1_tuned.py`: The production V9 pipeline implementing the tri-state ensemble.
*   `copy_code.py`: Feature engineering pipeline and Master Matrix generator.
*   `features.py`: The core feature engineering engine (Player Glicko, Draft Vectors, Temporal features).
*   `models.py`: SQLAlchemy database schema definitions with Player and Performance tracking.
*   `scraper.py`: Automated, robust scraper for MPL Seasons 1-17 with IGN Normalization.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Comprehensive audit detailing MOBA prediction logic and SOTA standards.
*   `MODEL_TRACKER.md`: Live log of model evolution, audit reports, and roadmap.

## 📈 Accuracy Benchmarks (V9 Ensemble)

Recent testing on our strict, purely chronological walk-forward holdout set achieved the following metrics for the V9 Unified Pipeline:

| Metric | Status |
| :--- | :--- |
| **Combined Accuracy (V9 Ensemble)** | **61.82%** |
| **Game 1 Precision** | ~69% |
| **Gap to 80% Goal** | 18.18% |

*(Note: Pre-game MOBA predictions face a natural "accuracy ceiling" of ~75% due to volatility and execution gaps. The V9 pipeline focuses on robustness and leak-prevention to ensure these metrics translate to real-world performance.)*

## 🚦 Getting Started

1.  **Setup Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Acquire Data:**
    ```bash
    python3 scraper.py
    ```
3.  **Run Pipeline:**
    ```bash
    python3 create_prediction_v1_tuned.py
    ```

