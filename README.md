# Predicting Professional MOBA Match Outcomes: A Split-Pipeline Approach to In-Series Adaptation

**Project Status:** Active Research & Implementation  
**Current State-of-the-Art (SOTA) Accuracy:** 70.43% (Chronological Holdout Validation)  
**Domain:** Mobile Legends: Bang Bang (MPL Philippines)

---

## 1. Abstract

Predicting the outcome of professional Multiplayer Online Battle Arena (MOBA) matches is a notoriously complex challenge. Unlike traditional sports, MOBAs feature a highly dynamic drafting phase, rapid in-game snowball mechanics, and continuous meta-shifts via software patches. 

This repository presents a **70.43% accurate** machine learning pipeline designed specifically for the MPL Philippines circuit. The core innovation of this research is the rejection of a singular modeling approach. Instead, we propose a **Split-Pipeline Architecture** that treats Game 1 as a structural, historical prediction problem (solved via XGBoost), and Game 2+ as a high-variance, psychological adaptation problem (solved via a time-weighted Stacking Ensemble of CatBoost and Random Forest).

This document outlines the methodology, feature engineering, and architectural decisions that led to breaking the theoretical 70% accuracy ceiling for pre-match predictions.

---

## 2. Methodology: The Split-Pipeline Architecture

Initial experiments demonstrated that a single model struggles to predict an entire Best-of-3 or Best-of-5 series. Game 1 operates in a vacuum; Game 2 is heavily influenced by the trauma, momentum, and strategic reveals of Game 1.

Our pipeline explicitly partitions the problem space:

### 2.1 Game 1: The Structural Baseline (XGBoost)
Game 1 predictions rely purely on pre-series metadata.
*   **Model Choice:** `XGBClassifier` (depth=3, lr=0.02, subsample=0.9). XGBoost excels at finding complex non-linear interactions in stable, macro-level tabular data without overfitting.
*   **Primary Signals:** Player-averaged historical Elo, regular season rank differentials, patch practice volume, and historical head-to-head win rates.
*   **Performance:** **78.83% Accuracy** (108/137 correct on holdout).

### 2.2 Game 2+: The Adaptation Phase (Hybrid Stacking Ensemble)
Once Game 1 concludes, the feature space expands to include in-series variables (momentum, previous game stomp margins, side swaps).
*   **Model Choice:** A heavily weighted `VotingClassifier` blending Gradient Boosting and Bagging techniques.
    *   *CatBoost (x2)*: Exceptionally adept at handling the dense, rapidly shifting categorical and numerical signals of in-series momentum.
    *   *Random Forest (x3)*: Heavily weighted to act as an anchor. Its uncorrelated, low-variance error profile stabilizes the aggressive, high-variance predictions of the boosting models.
*   **Primary Signals:** `series_momentum_blue`, `momentum_x_side_advantage`, and psychological carry-over metrics.
*   **Performance:** **65.53% Accuracy** (154/235 correct on holdout).

---

## 3. Core Feature Engineering & Game Theory

A model is only as intelligent as its features. To prevent the model from memorizing raw team strengths, we enforce relative matchup evaluation.

### 3.1 Differential Mapping (`diff_*`)
Instead of providing the model with absolute values (e.g., Blue Team Roster Stability = 0.8, Red Team Roster Stability = 0.4), the pipeline computes the delta (`diff_roster_stability = +0.4`). This forces the algorithms to evaluate the specific structural advantage of the head-to-head matchup.

### 3.2 Dynamic, Player-Based Elo System
Standard Elo systems track organizations. In esports, rosters change constantly.
*   **Player-Level Granularity:** Elo is calculated for individual players (`resolve_ign` ensures continuity) and a team's true strength is the dynamic average of its active roster on match day.
*   **Dual-Track Tracking:** The system calculates a General Elo and a localized **Playoff-Only Elo**. This isolates a team's "clutch factor" under elimination pressure.
*   **Responsive K-Factors:** We utilize $K = 24$ for regular-season matches and a highly volatile $K = 64$ for playoffs, recognizing that elimination brackets reveal true skill ceilings.
*   **Off-Season Decay:** A 15% regression-to-the-mean is applied between seasons to account for meta-shifts and off-season rust.

### 3.3 Modeling Intangibles: DNA and Momentum
*   `championship_dna`: Quantifies historical organizational resilience.
*   `momentum_x_side_advantage`: An interaction term measuring if the team that just gained momentum also rotated to the statistically favorable map side.
*   `g3_clutch_wr` & `reverse_sweep_rate`: Identifies teams that structurally scale in high-pressure deciding games.

---

## 4. Addressing the Accuracy Ceiling & Future Work

Pre-match MOBA prediction features a theoretical accuracy ceiling of ~75%. This is due to unpredictable in-game execution, random human error (e.g., a missed Smite objective), and draft-phase outplays.

**The "Comfort Trap" Discovery**
Our research identifies a critical vulnerability in Game 2 predictions: The Comfort Trap. When a team wins Game 1 using high-mastery (>60%) comfort heroes, they become predictable. In the MPL PH meta, teams falling for the Comfort Trap see their Game 2 win rate drop to **35.09%**. Future pipeline iterations aim to inject a `prev_winner_comfort_exhaustion` penalty feature to adjust momentum predictions dynamically based on draft reveals.

---

## 5. Validation Strategy: Zero Data Leakage

Time-series forecasting is highly susceptible to data leakage (using future information to predict the past). 
*   **Strict Chronological Split:** The dataset is split 85% Train / 15% Test based on exact `match_timestamp`. 
*   **Delayed Elo Updates:** Elo rating updates are calculated per game but applied at the *end* of the series (`pending_updates`). This guarantees that Game 1 predictions only evaluate Elo as it existed before the series began.

---

## 6. Repository Architecture

*   `create_prediction_v1_documented.py`: The build script. Generates the highly detailed, interactive Jupyter Notebook that serves as the executable research paper.
*   `1_NoteBook/Prediction_v1_documented.ipynb`: The primary pre-match inference engine and simulator.
*   `MASTER_MOBA_RESEARCH_REPORT.md`: Exhaustive 5-pillar audit detailing foundational MOBA prediction logic, the "Comfort Trap," and SOTA standards.
*   `MODEL_TRACKER.md`: Live engineering log tracking hyperparameter sweeps, architecture shifts, and accuracy milestones.

## 7. Execution Guide

To reproduce the findings or run the Series Simulator:

1.  **Generate the Documented Pipeline:**
    ```bash
    python3 create_prediction_v1_documented.py
    ```
2.  **Run the Research Notebook:**
    Navigate to `1_NoteBook/Prediction_v1_documented.ipynb` and execute the cells sequentially to observe data loading, on-the-fly Elo generation, feature engineering, model training, and simulated inference.
