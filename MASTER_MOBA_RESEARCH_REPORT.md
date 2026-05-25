# MOBA Win Prediction: The Master Audit & Research Report

This document consolidates all research, technical audits, and data science findings for the Predictive Analysis project. It is divided into three main pillars: Industry Standards, Game Theory Mechanics, and Systemic Code Integrity.

---

## PILLAR 1: Industry Standards & SOTA Research
State-of-the-Art (SOTA) models for MOBA prediction (Dota 2, LoL, MLBB) typically achieve **75-85% accuracy** for pre-game and **90%+** for real-time.

### Key Gaps in Current Project:
1.  **Draft Embeddings:** SOTA models use **Hero Embeddings** (Word2Vec/Transformers) to understand hero relationships. Your model uses weighted counts, which cannot "see" complex synergies.
2.  **In-Game Granularity:** You are missing gold leads, tower counts, and objective timings. Without these, you are predicting based on "who they are" rather than "how they played."
3.  **Role-Sensitive Metrics:** Professional models weight player impact by role (Jungler vs. Roamer). Your model treats all 5 players as an average Elo.

---

## PILLAR 2: Technical Deep Dive & Game Theory
MOBA series (Bo3/Bo5) are not independent events; they are a sequence of **Action and Reaction**.

### The "Comfort Trap" Discovery:
*   **Empirical Fact:** When a team wins Game 1 with high comfort heroes (>60% mastery), their win rate in Game 2 drops to **35.09%**.
*   **The Logic:** Winning with comfort heroes in Game 1 makes a team **predictable** and **bannable**. Your model currently treats comfort as a permanent positive buff, leading to the accuracy drop in Game 2 (54% vs 67% in Game 1).

### The SOTA Path (Graph Neural Networks):
Professional analysts use **Heterogeneous Information Networks (HIN)**. Instead of a flat list of numbers, the game is modeled as a Graph of Teams, Players, and Heroes, where edges represent Synergy and Counter-pick relationships.

---

## PILLAR 3: Systemic Code & Data Integrity Audit
Technical debt and "silent" data bugs are currently acting as a ceiling for your model's intelligence.

### 1. The "Fractured Identity" Bug (IGN Fragmentation)
*   **The Issue:** Your database has **382 unique IGNs**, but many are duplicates with different casing/spacing (e.g., `"BON CHAN"` vs `"Bon Chan"`).
*   **The Impact:** Every time a name is typed differently, the Elo system treats them as a "New Player" (1500 rating), deleting their historical skill and making `team_avg_elo` inaccurate.

### 2. Glicko-2 Volatility Trap
*   **The Issue:** Ratings are updated map-by-map rather than in periods.
*   **The Impact:** High "Recency Bias" causes ratings to fluctuate wildly after a single map loss, losing the stability needed for long-term prediction.

### 3. The "Momentum vs Adaptation" Blind Spot
*   **The Issue:** Series Momentum only grants a **56% win rate** in Game 2+. 
*   **The Logic:** The loser of Game 1 gets **Side Choice** in Game 2. This mechanical advantage in MLBB often cancels out the "Momentum" psychological advantage, but your features don't weight "Side Choice" heavily enough.

---

## PILLAR 4: Actionable Roadmap
To break the 65-70% accuracy barrier, follow this priority list:

1.  **Data Cleanliness (High Priority):** Normalize all IGNs to lowercase/stripped strings before updating Elo. Consolidate player history.
2.  **Temporal Patch Filling:** Stop allowing `"Unknown"` patches. Use match dates to map matches to the active patch.
3.  **Invert the Comfort Signal:** Add a feature that penalizes the winner of Game 1 if they "exhausted" their top-tier comfort picks.
4.  **Scrape In-Game Logs:** Transition from Liquipedia to sources that provide Gold Leads and Tower Damage.

---

## PILLAR 5: SOTA Research Findings
**Date:** 2024-05-23
**Researcher:** @sota-researcher

### **Executive Summary**

Recent research in MOBA match outcome prediction has shown a clear trend towards gradient boosting and deep learning models, with **CatBoost** emerging as a state-of-the-art (SOTA) model, achieving up to **96.15% accuracy** on professional MPL (Mobile Legends Professional League) datasets. This high accuracy is attributed to its superior handling of categorical features, which are prevalent in MOBA games (e.g., hero names, player roles). For real-time predictions, **LSTMs (Long Short-Term Memory networks)** are favored for their ability to model the time-series nature of matches and predict outcomes based on evolving game states. Key predictive features include pre-game draft information (hero synergies, player comfort picks) and in-game metrics, with the **gold and experience differential** being the most critical real-time predictor. For MPL PH, it's crucial to consider regional meta factors like the "PH Macro" (disciplined, objective-focused play) and unique drafting strategies.

### **SOTA Models for MOBA Outcome Prediction**

1.  **CatBoost & XGBoost (Gradient Boosting)**
    *   **CatBoost:** Consistently ranks as the top-performing model for pre-match prediction, with a reported accuracy of **96.15%** on MPL data. Its primary advantage is its built-in, sophisticated handling of categorical features, which eliminates the need for extensive pre-processing like one-hot encoding.
    *   **XGBoost:** A close competitor to CatBoost, achieving around **94.75%** accuracy. It is significantly faster than CatBoost, making it a viable alternative, especially for applications requiring rapid model retraining.
    *   **Use Case:** Best suited for pre-match prediction using a comprehensive set of features, including historical player data, hero matchups, and team statistics.

2.  **LSTMs & Transformers (Deep Learning)**
    *   **LSTMs:** The preferred architecture for real-time win probability prediction. Research shows that LSTMs can effectively model the sequential nature of a MOBA match, using time-series data like gold/XP differentials to make accurate predictions as the game unfolds. They have been shown to predict winners with high confidence from as early as the 10-minute mark.
    *   **Transformers:** While powerful, some studies indicate that Transformers can be slower than LSTMs for real-time inference in this context, making LSTMs a more practical choice for live applications.
    *   **Use Case:** Ideal for in-game win probability models that provide live updates to viewers or for real-time analysis tools.

### **Key Features for High-Accuracy Prediction**

**A. Pre-Game (Draft Phase)**
*   **Hero Synergy & Counters:** Analyzing the composition of a team to evaluate hero synergies and counter-picks. This can be enhanced by using hero embeddings.
*   **Player-Hero Comfort:** A player's historical performance (win rate, KDA) on a specific hero is a strong predictor.
*   **Regional Meta Trends:** The pick/ban priority of heroes in the current patch for a specific region (e.g., MPL PH).
*   **Team Cohesion:** Metrics that quantify how long a team roster has been playing together.

**B. In-Game (Real-Time)**
*   **Gold & XP Differential:** The most significant real-time predictor. A lead of 3,000 gold around the 8-minute mark in the MPL PH meta often correlates with a >80% win probability.
*   **Objective Control:** Number of Turtles and Lords secured. In the PH meta, Lord control is often a more decisive factor than the total number of kills.
*   **Turret Pressure & Map Control:** The number of remaining turrets is a strong indicator of map control and game advantage.
*   **KDA & Kill Participation:** Individual player performance, especially that of the Jungler and Gold Laner.

### **MPL PH Specific Considerations**

To achieve "near-perfect" prediction accuracy for MPL PH, models must be tuned to the specific characteristics of the region:

*   **The "PH Macro" Factor:** Filipino teams are renowned for their disciplined, objective-focused, and macro-heavy playstyle (often referred to as the "Ube" strategy). They are known for their patient high-ground defense and calculated comebacks. Models trained on more aggressive regions may not perform as well.
*   **Drafting Mind Games:** MPL PH coaches are known for their sophisticated drafting strategies, including "bait" picks to mislead opponents. Analyzing the draft sequence (the order of picks and bans) can provide additional predictive power.
*   **Patch Impact & Recency Weighting:** The MLBB meta shifts frequently with game patches. Models should apply recency weighting, giving more importance to matches played on the current or recent patches.

### **Recommendations for the Project**

1.  **Prioritize CatBoost for Pre-Match Prediction:** Given the project's existing data, a CatBoost model should be the primary focus for building a high-accuracy pre-match prediction system.
2.  **Engineer MPL PH-Specific Features:**
    *   Create features that capture the "PH Macro" style, such as the timing of the first Turtle, the number of Lords secured, and the average game duration.
    *   Develop "Player-Hero Comfort" features by analyzing the historical performance of MPL PH players.
3.  **Develop a Real-Time Prediction Model using LSTMs:** If real-time data can be acquired, an LSTM model should be developed to predict win probability based on live in-game data (gold/XP lead, objectives).
4.  **Implement Time-Series Cross-Validation:** To build a robust model that accounts for meta shifts, use time-series cross-validation (e.g., train on past seasons to predict the current season).
5.  **Data-centric Approach:** The high accuracy of the SOTA models is heavily dependent on the quality and granularity of the data. The focus should be on collecting detailed match data, including draft information and time-series data for in-game events.

---

## Final Verdict
Your project has a **world-class foundation** for data handling. You aren't "wrong"—you've simply reached the limit of what **Pre-Game metadata** can predict. By fixing the data fragmentation and accounting for "Draft Exhaustion," you can unlock the next 5-10% of accuracy.
