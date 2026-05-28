# MOBA Win Prediction Engine 🏆

A high-performance machine learning pipeline designed to predict professional MOBA match outcomes (MLBB MPL PH). By leveraging player-centric metrics, dynamic Elo tracking, and advanced game theory adaptation, this engine systematically captures the complexities of professional esports series.

## 🚀 Overview

The predictive landscape of competitive MOBAs is highly volatile. Our engine tackles this by distinguishing between the "blank slate" of Game 1 and the complex psychological and strategic shifts of Game 2 and beyond. The latest **70.43% SOTA Architecture** represents a major breakthrough in modeling in-series momentum and adaptation.

### 🏁 The 70.43% SOTA Pre-Match Pipeline (Production Champion)

The pipeline explicitly rejects "one size fits all" modeling, instead treating the series context as the ultimate feature.

**Key Analytical Pillars:**
*   **Dual-State Partitioning (Split-Pipeline Architecture):** 
    *   **Game 1 (XGBoost):** Focuses heavily on historical Elo, macro team-level statistics, patch practice, and baseline structural strengths.
    *   **Game 2+ (Hybrid Stacking Ensemble):** Blends CatBoost and Random Forest to ingest in-series momentum, adaptation, and psychological carry-over (e.g., `prev_stomp_margin`).
*   **Dynamic, Player-Based Elo System:** Ratings are tracked at the individual player level and averaged for the active roster. It uniquely features Dual-Track ratings (General Elo vs Playoff-Only "Clutch" Elo) and responsive K-factors based on tournament stakes.
*   **Differential Feature Mapping:** Converts raw strengths into relative matchup advantages (`diff_roster_stability`, `diff_playoff_exp`) to force the model to evaluate the specific head-to-head dynamic.

---

## ✨ Core Features & Signals

*   **Momentum & Map Advantage Intersections:** Tracks advanced interactions like `momentum_x_side_advantage` to determine if a team with momentum also secured the statistically advantageous map side.
*   **DNA & Clutch Metrics:** Uses `championship_dna`, `g3_clutch_wr`, and `reverse_sweep_rate` to quantify intangible mental fortitude.
*   **Time-Weighted Training:** The Game 2+ models heavily weight recent games (`sample_weight=time_weight`), forcing the algorithms to prioritize modern meta adaptations over ancient history.
*   **Strict Chronological Validation:** Operates on a pure 85/15 chronological train-test split to completely eliminate temporal data leakage.

## 🛠 Tech Stack

*   **Core:** Python 3.10+
*   **Machine Learning:** XGBoost, CatBoost, Scikit-Learn (Random Forest)
*   **Data Processing:** Pandas, NumPy
*   **Experimentation:** Jupyter Notebooks (Highly documented, research-grade notebooks)

## 📂 Project Structure

*   `create_prediction_v1_documented.py`: Generates the heavily documented, research-grade Jupyter Notebook containing the SOTA pre-match architecture.
*   `1_NoteBook/Prediction_v1_documented.ipynb`: The primary executable environment for the 70.43% pipeline.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Comprehensive audit detailing MOBA prediction logic and game-theoretic concepts.
*   `MODEL_TRACKER.md`: Live log of model evolution, hyperparameter sweeps, and architectural breakthroughs.

## 📈 Accuracy Benchmarks (SOTA Pre-Match Pipeline)

Recent testing on our strict, purely chronological walk-forward holdout set (Season 16/17 context) achieved the following target-breaching metrics:

| Metric | Accuracy | Correct Predictions | Notes |
| :--- | :--- | :--- | :--- |
| **Combined SOTA Accuracy** | **70.43%** | 262 / 372 | Target Breached! |
| **Game 1 Precision (XGBoost)** | **78.83%** | 108 / 137 | Structural/Historical dominance. |
| **Game 2+ Precision (CB/RF Blend)** | **65.53%** | 154 / 235 | High-variance momentum/adaptation phase. |

*(Note: Pre-game MOBA predictions face a natural "accuracy ceiling" of ~75% due to in-game volatility, human error, and execution gaps. Breaking the 70% threshold pre-draft represents an elite understanding of team structure and momentum.)*

## 🚦 Getting Started

1.  **Generate the Documented Pipeline:**
    ```bash
    python3 create_prediction_v1_documented.py
    ```
2.  **Run the Research Notebook:**
    Navigate to `1_NoteBook/Prediction_v1_documented.ipynb` and execute the cells to see the dynamic Elo system, feature generation, and the Series Simulator in action!
