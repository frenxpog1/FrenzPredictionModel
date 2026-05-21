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

## Final Verdict
Your project has a **world-class foundation** for data handling. You aren't "wrong"—you've simply reached the limit of what **Pre-Game metadata** can predict. By fixing the data fragmentation and accounting for "Draft Exhaustion," you can unlock the next 5-10% of accuracy.
