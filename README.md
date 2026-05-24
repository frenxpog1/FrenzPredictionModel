# MOBA Win Prediction Engine 🏆

A high-performance machine learning pipeline designed to predict professional MOBA match outcomes (MLBB MPL). By leveraging player-centric metrics, dynamic Elo tracking, and advanced game theory adaptation, this engine systematically captures the complexities of professional esports series.

## 🚀 Overview

The predictive landscape of competitive MOBAs is highly volatile. Unlike traditional sports or singular matches, professional MOBA series involve progressive adaptation. Our engine tackles this by distinguishing between the "blank slate" of Game 1 and the complex psychological and strategic shifts of Game 2 and beyond.

### 🚧 Current Work In Progress: `Prediction_v1.ipynb`

We are developing a next-generation predictive architecture in `1_NoteBook/Prediction_v1.ipynb`. This experimental branch transitions from a dual-model approach to a **highly granular Game-State Partitioned Engine**. 

**Key Analytical Updates in V1:**
*   **Tri-State Partitioning:** Breaking the series down into three distinct evaluative states:
    *   **Game 1 (Pre-Match):** Pure structural and historical evaluation (Draft isolation, base Elo, historical H2H).
    *   **Game 2 (Adaptation):** Introduces immediate series momentum and first-game draft exhaustion factors.
    *   **Game 3+ (Late-Series Clutch):** Accounts for high-pressure variables, late-series score differences, and potential fatigue.
*   **Leak-Safe Feature Engineering:** Strict isolation to ensure current-game draft features are entirely blocked from pre-match models, ensuring zero data leakage and rigorous real-world validation.
*   **Candidate vs. Pooled Evaluation:** We dynamically evaluate whether splitting the Game 2+ models into distinct Game 2 and Game 3+ models yields statistically significant improvements over a pooled Game 2+ model.
*   **Expanded Ensemble:** The pipeline integrates **CatBoost** alongside XGBoost, LightGBM, and Random Forest via a Soft Voting Classifier to maximize generalization.

---

## ✨ SOTA Engine Features (V7 Architecture)

Our stable V7 engine incorporates cutting-edge feature engineering and structural data integrity:

*   **Dual-Track Elo System:** Tracks separate Elo ratings for Regular Season and Playoffs to accurately capture "clutch" performances and structural team scaling under pressure.
*   **Match-Level Elo Updates (The Volatility Fix):** Elo updates are strictly bounded to the end of a match rather than per-game, eliminating extreme recency bias and "volatility traps".
*   **Pillar 3 Data Integrity (IGN Normalization):** Consolidates fragmented player identities directly at the ingestion layer (`scraper.py`) to maintain continuous, mathematically flawless skill histories.
*   **The Comfort Exhaustion Flag:** Implements the "Comfort Trap" discovery from our research—teams winning Game 1 with top comfort picks face predictable bans and draft exhaustion in Game 2.
*   **Side Choice × Series State Interaction:** Amplifies the psychological momentum of Game 2 recoveries for teams that draft well on their mechanically favored side.

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
*   `copy_code.py`: Feature engineering pipeline and Master Matrix generator.
*   `features.py`: The core feature engineering engine (Elo, Draft Vectors, Temporal features).
*   `scraper.py`: Automated, robust scraper for MPL Seasons 1-17 with IGN Normalization.
*   `models.py`: SQLAlchemy database schema definitions for Teams, Matches, Games, and Heroes.
*   `csv_data/`: Centralized repository for processed datasets, including the ML Feature Matrices.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Comprehensive audit detailing MOBA prediction logic, SOTA standards, and pipeline integrity.

## 📈 Accuracy Benchmarks (V7 Walk-Forward Pipeline)

Recent testing on our strict, purely chronological walk-forward holdout set (predicting S16 and S17 based exclusively on prior seasons) achieved new SOTA metrics:

| Stage | Test Accuracy |
| :--- | :--- |
| **Game 1 (Draft / Baseline)** | 68.61% |
| **Game 2+ (Pooled Adaptation)** | 63.83% |
| **Combined (Unseen Matches)** | **65.59%** |

*(Note: Pre-game MOBA predictions face a natural "accuracy ceiling" of ~75% due to the inherent volatility, human error, execution gaps, and meta-shifts present in elite competition. A 65.59% accuracy benchmark without any live-game telemetry places this engine in the upper echelon of pre-match forecasting systems.)*

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
3.  **Run Feature Matrix Generation:**
    ```bash
    python3 copy_code.py
    ```
4.  **Run Experimentation:**
    ```bash
    python3 create_prediction_v1_tuned.py
    ```
    Open `1_NoteBook/Prediction_v1.ipynb` to explore the latest architectural evaluations.
