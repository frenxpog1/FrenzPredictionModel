import pandas as pd
import numpy as np
import glicko2
import math
import datetime
from sqlalchemy.orm import Session
import models

class FeatureEngine:
    def __init__(self, db: Session):
        self.db = db
        self.teams_glicko = {} # name: glicko2.Player()
        self.hero_universe = self.get_hero_universe()
        
    def get_hero_universe(self):
        heroes = self.db.query(models.Hero).all()
        return {h.name: i for i, h in enumerate(heroes)}

    def init_team_glicko(self, name):
        if name not in self.teams_glicko:
            # Spec 3.2: Initialize at mu=1500, phi=350, sigma=0.06
            self.teams_glicko[name] = glicko2.Player(rating=1500, rd=350, vol=0.06)
        return self.teams_glicko[name]

    def calculate_meta_volatility(self, match_date, patch_release_date, gamma=0.1):
        # Spec 3.3: V = e^(-gamma * Days_Since_Patch)
        if not patch_release_date: return 1.0
        days_since = (match_date - patch_release_date).days
        return math.exp(-gamma * max(0, days_since))

    def generate_draft_vector(self, blue_picks, red_picks):
        # Spec 3.1: 2.0 for Phase 1 (Power Picks), 1.0 for Phase 2
        # Simplified: First 3 picks = Phase 1, Last 2 = Phase 2
        vector_size = len(self.hero_universe)
        if vector_size == 0: vector_size = 150 # Fallback
        
        blue_vec = np.zeros(vector_size)
        red_vec = np.zeros(vector_size)
        
        for i, hero in enumerate(blue_picks[:5]):
            if hero in self.hero_universe:
                weight = 2.0 if i < 3 else 1.0
                blue_vec[self.hero_universe[hero]] = weight
                
        for i, hero in enumerate(red_picks[:5]):
            if hero in self.hero_universe:
                weight = 2.0 if i < 3 else 1.0
                red_vec[self.hero_universe[hero]] = weight
                
        return blue_vec, red_vec

    def resolve_ign(self, ign):
        return str(ign).strip().lower()

    def process_match_history(self):
        """Match-based processing to update Glicko-2 ratings for stability"""
        matches = self.db.query(models.Match).order_by(models.Match.match_date).all()
        features = []
        
        for match in matches:
            t1 = self.init_team_glicko(match.team_a_name)
            t2 = self.init_team_glicko(match.team_b_name)
            
            # Record state before match
            delta_glicko = t1.rating - t2.rating
            
            # Patch metadata
            patch = self.db.query(models.Patch).filter_by(patch_version=match.patch_version).first()
            volatility = self.calculate_meta_volatility(match.match_date, patch.release_date if patch else None)
            
            # Process games for features, but update ratings only once per match
            match_won_a = 0
            match_won_b = 0
            
            for game in match.games:
                blue_vec, red_vec = self.generate_draft_vector(game.picks.get('blue', []), game.picks.get('red', []))
                
                if game.game_winner == match.team_a_name: match_won_a += 1
                else: match_won_b += 1
                
                features.append({
                    "match_id": match.match_id,
                    "delta_glicko": delta_glicko,
                    "meta_volatility": volatility,
                    "game_winner": 1 if game.game_winner == match.team_a_name else 0,
                })
            
            # Match-level Glicko-2 update (Stability Fix)
            actual_score = 1.0 if match_won_a > match_won_b else 0.0
            t1.update_player([t2.rating], [t2.rd], [actual_score])
            t2.update_player([t1.rating], [t1.rd], [1.0 - actual_score])
        
        return pd.DataFrame(features)
