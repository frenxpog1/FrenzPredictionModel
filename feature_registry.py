# Feature Registry and Lifecycle Classification Metadata
# Classifies every column in ML_Feature_Matrix.csv into its strictly audited lifecycle stage:
# 1. "pre_match": Available strictly prior to the series starting. Fully chronological and historical.
# 2. "post_draft": Requires hero draft selections for the current map, computed before the map begins.
# 3. "in_series": Available during live series maps (Game 2+), dynamically computed based on prior map outcomes in the same series.
# 4. "forbidden": Current-game labels or future indicators which MUST NEVER be used as features during prediction.

import numpy as np

FEATURE_REGISTRY = {
    # Match identifiers & metadata
    "match_id": "forbidden",
    "season": "forbidden",
    "match_timestamp": "forbidden",
    "patch_version": "forbidden",
    "stage": "forbidden",
    "blue_side_team": "forbidden",
    "red_side_team": "forbidden",
    "game_number": "forbidden",
    "picks": "forbidden",
    "bans": "forbidden",
    
    # ELO & Franchise Ratings (Historical)
    "blue_side_elo": "pre_match",
    "red_side_elo": "pre_match",
    "blue_playoff_elo": "pre_match",
    "red_playoff_elo": "pre_match",
    "diff_side_elo": "pre_match",
    "diff_playoff_elo": "pre_match",
    
    # Team History & Forms
    "blue_championship_dna": "pre_match",
    "red_championship_dna": "pre_match",
    "diff_championship_dna": "pre_match",
    "blue_playoff_winrate": "pre_match",
    "red_playoff_winrate": "pre_match",
    "diff_playoff_winrate": "pre_match",
    "blue_momentum": "pre_match",
    "red_momentum": "pre_match",
    "blue_h2h_winrate": "pre_match",
    
    # Roster & Practice Signals
    "blue_roster_stability": "pre_match",
    "red_roster_stability": "pre_match",
    "diff_roster_stability": "pre_match",
    "blue_patch_practice": "pre_match",
    "red_patch_practice": "pre_match",
    "diff_patch_practice": "pre_match",
    
    # Playstyle Clashes (Chronological embeddings fit on prior seasons)
    "draft_style_sim": "post_draft",
    
    # In-Series Live Status (Game 2+ live features)
    "series_momentum_blue": "in_series",
    "prev_stomp_margin": "in_series",
    "is_side_swap": "in_series",
    "blue_draft_exhaustion": "in_series",
    "red_draft_exhaustion": "in_series",
    "diff_draft_exhaustion": "in_series",
    "momentum_x_side_advantage": "in_series",
    
    # Target Labels & Forbidden indicators
    "map_winner": "forbidden",
    "target_blue_win": "forbidden",
}

# Auto-add embedding coordinate features as 'post_draft'
for i in range(16):
    FEATURE_REGISTRY[f"blue_draft_emb_{i}"] = "post_draft"
    FEATURE_REGISTRY[f"red_draft_emb_{i}"] = "post_draft"
    FEATURE_REGISTRY[f"diff_draft_emb_{i}"] = "post_draft"

def get_feature_class(feature_name: str) -> str:
    """Returns the availability class of a feature, defaulting to 'pre_match' if not registered."""
    return FEATURE_REGISTRY.get(feature_name, "pre_match")

def audit_features(features_list: list, allowed_stages: list) -> list:
    """Audits a list of features to ensure they belong strictly to allowed operational stages."""
    violators = []
    for f in features_list:
        stage = get_feature_class(f)
        if stage not in allowed_stages:
            violators.append((f, stage))
    return violators
