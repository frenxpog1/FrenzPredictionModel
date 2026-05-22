# MOBA Win Prediction Engine 🏆

A high-performance machine learning pipeline designed to predict professional MOBA match outcomes (MLBB MPL). By leveraging player-centric metrics, dynamic Elo tracking, and advanced game theory adaptation, this engine sets out to systematically capture the complexities of professional esports series.

## 🚀 Overview

The predictive landscape of competitive MOBAs is highly volatile. Unlike traditional sports or singular matches, professional MOBA series involve progressive adaptation. Our engine tackles this by distinguishing between the "blank slate" of Game 1 and the complex psychological and strategic shifts of Game 2 and beyond.

### 🚧 Current Work In Progress: `Prediction_v1.ipynb`

We are currently developing a next-generation predictive architecture in `1_NoteBook/Prediction_v1.ipynb`. This experimental branch transitions from a dual-model approach to a **highly granular Game-State Partitioned Engine**. 

**Key Analytical Updates in V1 (WIP):**
*   **Tri-State Partitioning:** Breaking the series down into three distinct evaluative states:
    *   **Game 1 (Pre-Match):** Pure structural and historical evaluation (Draft isolation, base Elo, historical H2H).
    *   **Game 2 (Adaptation):** Introduces immediate series momentum and first-game draft exhaustion factors.
    *   **Game 3+ (Late-Series Clutch):** Accounts for high-pressure variables, late-series score differences, and potential fatigue.
*   **Leak-Safe Feature Engineering:** Strict isolation to ensure current-game draft features are entirely blocked from pre-match models, ensuring zero data leakage and rigorous real-world validation.
*   **Candidate vs. Pooled Evaluation:** We dynamically evaluate whether splitting the Game 2+ models into distinct Game 2 and Game 3+ models yields statistically significant improvements over a pooled Game 2+ model, acknowledging the risk of overfitting on the smaller sample size of Game 3+.
*   **Expanded Ensemble:** The experimental pipeline now integrates **CatBoost** alongside XGBoost, LightGBM, and Random Forest via a Soft Voting Classifier to maximize generalization.

---

## ✨ Stable Engine Features (V7 Architecture)

While V1 is under active development, our stable V7 engine continues to provide a robust baseline:

*   **V7 Ensemble Engine:** Combines XGBoost, LightGBM, and Random Forest for regularized, high-confidence predictions.
*   **Dual-Track Elo System:** Tracks separate Elo ratings for Regular Season and Playoffs to accurately capture "clutch" performances and structural team scaling under pressure.
*   **Pillar 3 Data Integrity:** 
    *   **IGN Normalization:** Consolidates fragmented player identities (e.g., merging "BON CHAN" and "Bon Chan") to maintain continuous, accurate skill histories.
    *   **Glicko-2 Stability:** Employs match-based rating updates to prevent "volatility traps" caused by single-map statistical anomalies.
*   **Draft & Meta Analysis:** Engineered features include Hero Comfort Mastery, Patch Practice, Side Advantage, and Series Momentum.

## 🛠 Tech Stack

*   **Core:** Python 3.10+
*   **Machine Learning:** XGBoost, LightGBM, CatBoost, Scikit-Learn
*   **Data Processing:** Pandas, NumPy, SQLAlchemy (SQLite)
*   **Scraping & Data Acquisition:** Playwright, Scrapling (Liquipedia source integration)
*   **Experimentation:** Jupyter Notebooks

## 📂 Project Structure

*   `1_NoteBook/Prediction_v1.ipynb` **(WIP)**: The experimental tri-state prediction pipeline.
*   `1_NoteBook/Prediction.ipynb`: The stable V7 analysis and training hub.
*   `create_prediction_v1_tuned.py`: Script form of the V1 pipeline for rapid iteration and tuning.
*   `features.py`: The core feature engineering engine (Glicko-2, Draft Vectors, Temporal features).
*   `scraper.py`: Automated, robust scraper for MPL Seasons 1-17.
*   `models.py`: SQLAlchemy database schema definitions for Teams, Matches, Games, and Heroes.
*   `csv_data/`: Centralized repository for processed datasets, including the ML Feature Matrices.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Comprehensive audit detailing MOBA prediction logic, SOTA standards, and pipeline integrity.

## 📈 Stable Accuracy Benchmarks (V7)

Our walk-forward validation framework mimics real-world deployment, achieving industry-standard predictive accuracy:

| Stage | Accuracy |
| :--- | :--- |
| **Game 1 (Draft / Baseline)** | ~67.5% |
| **Game 2+ (Adaptation Phase)** | ~59.7% |
| **Combined (Unseen Matches)** | **~66.2%** |

*Note: Pre-game MOBA predictions face a natural "accuracy ceiling" of ~75% due to the inherent volatility, human error, and meta-shifts present in elite competition.*

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
3.  **Run Experimentation:**
    Open `1_NoteBook/Prediction_v1.ipynb` to explore the latest architectural changes, or `1_NoteBook/Prediction.ipynb` to train the stable V7 ensemble.
