# MOBA Win Prediction: V7 Ensemble Engine 🏆

A high-performance machine learning pipeline designed to predict professional MOBA match outcomes (MLBB MPL) using player-centric metrics and game theory adaptation.

## 🚀 Overview
This project predicts the winner of professional MLBB matches with a focus on "Series Adaptation." Unlike traditional models that treat every game the same, this engine uses a **Dual-Model Architecture** to account for how teams change their strategy after a win or loss.

## ✨ Key Features
*   **V7 Ensemble Engine:** Combines XGBoost, LightGBM, and Random Forest for robust, regularized predictions.
*   **Dual-Track Elo System:** Tracks separate Elo ratings for Regular Season and Playoffs to capture "clutch" performance.
*   **Pillar 3 Data Integrity:** 
    *   **IGN Normalization:** Consolidates fragmented player identities (e.g., merging "RTZY" and "karltzy") to maintain accurate skill history.
    *   **Glicko-2 Stability:** Match-based rating updates to prevent "volatility traps" caused by single-map losses.
*   **Draft & Meta Analysis:** Features include Hero Comfort Mastery, Patch Practice, Side Advantage, and Series Momentum.
*   **Master Pipeline:** A clean, walk-forward validation system that achieves a **realistic ~66-67% accuracy** on unseen games (Industry Standard).

## 🛠 Tech Stack
*   **Core:** Python 3.14
*   **ML:** XGBoost, LightGBM, Scikit-Learn
*   **Data:** Pandas, NumPy, SQLAlchemy (SQLite)
*   **Scraping:** Playwright, Scrapling (Liquipedia source)
*   **Environment:** Jupyter Notebooks

## 📂 Project Structure
*   `1_NoteBook/Prediction.ipynb`: The primary analysis and training hub.
*   `features.py`: The core feature engineering engine (Glicko-2, Draft Vectors).
*   `scraper.py`: Automated scraper for MPL Seasons 1-17.
*   `models.py`: Database schema for Teams, Matches, Games, and Heroes.
*   `csv_data/`: Processed datasets including the ML Feature Matrix.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Detailed audit on MOBA prediction logic and SOTA standards.

## 📈 Accuracy Benchmarks
| Stage | Accuracy |
| :--- | :--- |
| **Game 1 (Draft)** | ~67.5% |
| **Game 2+ (Adaptation)** | ~59.7% |
| **Combined (Unseen)** | **~66.2%** |

## 🚦 Getting Started
1.  **Setup Environment:**
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Scrape Data:**
    ```bash
    python3 scraper.py
    ```
3.  **Train Model:**
    Open `1_NoteBook/Prediction.ipynb` and run all cells to generate the V7 Ensemble.

---
**Note:** This model is designed for predictive analysis and educational research. Pre-game MOBA prediction has a natural "accuracy ceiling" of ~75% due to the high volatility of human competition.
