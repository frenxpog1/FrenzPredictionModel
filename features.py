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
        self.players_glicko = {} # ign: glicko2.Player()
        self.hero_universe = self.get_hero_universe()
        self.season_rosters_cache = self.get_season_rosters_cache()
        
        # Trackers (Pillar 2 & 5)
        self.team_hero_history = {} # team: {hero: {wins, games}}
        self.global_synergy = {} # (hero_a, hero_b): score
        self.global_counters = {} # (hero_a, hero_b): score
        self.team_recent_form = {} # team: [results]
        self.h2h_tracker = {} # (t1, t2): {t1_wins, total}
        self.patch_practice = {} # team: {patch: count}
        
    def get_hero_universe(self):
        heroes = self.db.query(models.Hero).all()
        return {h.name: i for i, h in enumerate(heroes or [])}

    def get_season_rosters_cache(self):
        rosters = self.db.query(models.SeasonRoster).all()
        cache = {}
        for r in rosters:
            if r.season not in cache: cache[r.season] = {}
            team_name = self.normalize_team_name(r.team_name)
            cache[r.season][team_name] = [p['ign'] for p in (r.players or [])]
        return cache

    def normalize_team_name(self, name):
        if not name: return ""
        return name.lower().strip().replace("esports", "").replace("e-sports", "").strip()

    def init_player_glicko(self, ign):
        resolved = self.resolve_ign(ign)
        if resolved not in self.players_glicko:
            self.players_glicko[resolved] = glicko2.Player(rating=1500, rd=350, vol=0.06)
        return self.players_glicko[resolved]

    def calculate_meta_volatility(self, match_date, patch_release_date, gamma=0.1):
        if not patch_release_date or not match_date: return 1.0
        days_since = (match_date - patch_release_date).days
        return float(math.exp(-gamma * max(0, days_since)))

    def generate_draft_vector(self, blue_picks, red_picks):
        vector_size = len(self.hero_universe)
        if vector_size == 0: vector_size = 150 
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
        if not ign or str(ign) == 'nan': return "unknown"
        ign_str = str(ign).strip().lower()
        IGN_ALIASES = {
            "3mar": "3martzy", "bon chan": "bon chan", "had ji": "hadji", "hadji": "hadji",
            "hadjizy": "hadji", "ohmyv33nus": "ohmyv33nus", "oheb": "oheb", 
            "super marco": "super marco", "edward": "edward", "h2wo": "h2wo",
            "ejhay": "ejhay", "lancecy": "lancecy", "kyle": "kyletzy", "kyle tzy": "kyletzy",
            "flap": "flaptzy", "flap tzy": "flaptzy", "karl": "karltzy", "karl tzy": "karltzy",
            "dlar": "dlarskie",
        }
        clean_ign = ign_str.replace(" ", "").replace("-", "")
        return IGN_ALIASES.get(clean_ign, ign_str)

    def process_match_history(self):
        """Unified Master Pipeline with 70+ signals (Pillar 1-5)"""
        matches = self.db.query(models.Match).order_by(models.Match.match_timestamp).all()
        features = []
        
        for match in matches:
            def get_team_rating(roster):
                if not roster: return 1500.0, 350.0
                ratings = [self.init_player_glicko(p).rating for p in roster]
                rds = [self.init_player_glicko(p).rd for p in roster]
                return float(np.mean(ratings)), float(np.mean(rds))

            patch = self.db.query(models.Patch).filter_by(patch_version=match.patch_version).first()
            volatility = self.calculate_meta_volatility(match.match_timestamp, patch.release_timestamp if patch else None)
            
            match_won_a, match_won_b = 0, 0
            last_winner, last_winner_heroes = None, []

            for game in match.games:
                # 1. Roster & Ratings
                b_rost = [self.resolve_ign(p) for p in (game.blue_roster or self.season_rosters_cache.get(match.season, {}).get(self.normalize_team_name(game.blue_side_team), []))]
                r_rost = [self.resolve_ign(p) for p in (game.red_roster or self.season_rosters_cache.get(match.season, {}).get(self.normalize_team_name(game.red_side_team), []))]
                b_rating, b_rd = get_team_rating(b_rost)
                r_rating, r_rd = get_team_rating(r_rost)
                
                # 2. Form & H2H
                b_form = np.mean(self.team_recent_form.get(game.blue_side_team, [0.5]))
                r_form = np.mean(self.team_recent_form.get(game.red_side_team, [0.5]))
                h2h_key = tuple(sorted([game.blue_side_team, game.red_side_team]))
                h2h = self.h2h_tracker.get(h2h_key, {'blue_wins': 0, 'total': 0})
                h2h_score = (h2h['blue_wins'] + 1) / (h2h['total'] + 2) if h2h_key[0] == game.blue_side_team else (h2h['total'] - h2h['blue_wins'] + 1) / (h2h['total'] + 2)

                # 3. Synergy & Counters
                b_syn, r_syn, b_ctr, r_ctr = 0.0, 0.0, 0.0, 0.0
                b_picks = game.picks.get('blue', [])
                r_picks = game.picks.get('red', [])
                for i, h1 in enumerate(b_picks):
                    for h2 in b_picks[i+1:]: b_syn += self.global_synergy.get(tuple(sorted([h1, h2])), 0.0)
                for i, h1 in enumerate(r_picks):
                    for h2 in r_picks[i+1:]: r_syn += self.global_synergy.get(tuple(sorted([h1, h2])), 0.0)
                for bh in b_picks:
                    for rh in r_picks:
                        b_ctr += self.global_counters.get((bh, rh), 0.0)
                        r_ctr += self.global_counters.get((rh, bh), 0.0)

                # 4. Comfort Trap (Exhaustion)
                exhaustion = 0.0
                if last_winner:
                    tracker = self.team_hero_history.get(last_winner, {})
                    wins = sum(tracker.get(h, {'wins': 0})['wins'] for h in last_winner_heroes)
                    total = sum(tracker.get(h, {'games': 0})['games'] for h in last_winner_heroes)
                    mastery = (wins + 2) / (total + 4)
                    if mastery > 0.6:
                        exhaustion = (mastery - 0.6) * (-1.0 if last_winner == game.blue_side_team else 1.0)

                # Assemble Features
                f = {
                    "match_id": match.match_id, "game_id": game.id, "game_num": game.game_number,
                    "delta_glicko": b_rating - r_rating, "volatility": volatility,
                    "delta_form": b_form - r_form, "h2h_score": h2h_score,
                    "exhaustion": exhaustion, "synergy_delta": b_syn - r_syn, "counter_delta": b_ctr - r_ctr,
                    "winner": 1 if game.map_winner == game.blue_side_team else 0
                }
                features.append(f)

                # 5. Update Trackers AFTER Game
                winner = game.map_winner
                for side in ['blue', 'red']:
                    team = game.blue_side_team if side == 'blue' else game.red_side_team
                    picks = game.picks.get(side, [])
                    won = (winner == team)
                    # Mastery
                    if team not in self.team_hero_history: self.team_hero_history[team] = {}
                    for h in picks:
                        if h not in self.team_hero_history[team]: self.team_hero_history[team][h] = {'wins': 0, 'games': 0}
                        self.team_hero_history[team][h]['games'] += 1
                        if won: self.team_hero_history[team][h]['wins'] += 1
                    # Form
                    if team not in self.team_recent_form: self.team_recent_form[team] = []
                    self.team_recent_form[team].append(1 if won else 0)
                    if len(self.team_recent_form[team]) > 5: self.team_recent_form[team].pop(0)
                # H2H
                if h2h_key not in self.h2h_tracker: self.h2h_tracker[h2h_key] = {'blue_wins': 0, 'total': 0}
                self.h2h_tracker[h2h_key]['total'] += 1
                if winner == h2h_key[0]: self.h2h_tracker[h2h_key]['blue_wins'] += 1
                # Synergy & Counters
                if won:
                    for i, h1 in enumerate(b_picks if winner == game.blue_side_team else r_picks):
                        for h2 in (b_picks if winner == game.blue_side_team else r_picks)[i+1:]:
                            k = tuple(sorted([h1, h2]))
                            self.global_synergy[k] = self.global_synergy.get(k, 0) + 1
                    for wh in (b_picks if winner == game.blue_side_team else r_picks):
                        for lh in (r_picks if winner == game.blue_side_team else b_picks):
                            self.global_counters[(wh, lh)] = self.global_counters.get((wh, lh), 0) + 1
                
                last_winner, last_winner_heroes = winner, (b_picks if winner == game.blue_side_team else r_picks)
                if winner == match.team_a_name: match_won_a += 1
                else: match_won_b += 1

            # 6. Match-level Glicko Update
            team_a_players = set()
            team_b_players = set()
            for g in match.games:
                b_rost = g.blue_roster or self.season_rosters_cache.get(match.season, {}).get(self.normalize_team_name(g.blue_side_team), [])
                r_rost = g.red_roster or self.season_rosters_cache.get(match.season, {}).get(self.normalize_team_name(g.red_side_team), [])
                if g.blue_side_team == match.team_a_name: team_a_players.update([self.resolve_ign(p) for p in (b_rost or [])])
                else: team_b_players.update([self.resolve_ign(p) for p in (b_rost or [])])
                if g.red_side_team == match.team_a_name: team_a_players.update([self.resolve_ign(p) for p in (r_rost or [])])
                else: team_b_players.update([self.resolve_ign(p) for p in (r_rost or [])])
            if team_a_players and team_b_players:
                avg_a_r, avg_a_rd = get_team_rating(list(team_a_players))
                avg_b_r, avg_b_rd = get_team_rating(list(team_b_players))
                score_a = 1.0 if match_won_a > match_won_b else 0.0
                for ign in team_a_players: self.init_player_glicko(ign).update_player([avg_b_r], [avg_b_rd], [score_a])
                for ign in team_b_players: self.init_player_glicko(ign).update_player([avg_a_r], [avg_a_rd], [1.0 - score_a])
        
        return pd.DataFrame(features)
