from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from database import Base
import datetime
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    glicko_mu = Column(Float, default=1500.0)
    glicko_phi = Column(Float, default=350.0)
    glicko_sigma = Column(Float, default=0.06)

class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True, index=True)
    ign = Column(String, unique=True, index=True)
    real_name = Column(String, nullable=True)
    current_team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    glicko_mu = Column(Float, default=1500.0)
    glicko_phi = Column(Float, default=350.0)
    glicko_sigma = Column(Float, default=0.06)
    
    performances = relationship("PlayerPerformance", back_populates="player")

class Patch(Base):
    __tablename__ = "patches"
    patch_version = Column(String, primary_key=True, index=True)
    release_timestamp = Column(DateTime)
    hero_adjustments = Column(JSON, nullable=True) # {"HeroName": "BUFF" | "NERF" | "ADJUST" | "REVAMP"}
    item_adjustments = Column(JSON, nullable=True) # {"ItemName": "BUFF" | "NERF" | "ADJUST"}

class Match(Base):
    __tablename__ = "matches"
    match_id = Column(String, primary_key=True, default=generate_uuid)
    season = Column(Integer, index=True)
    stage = Column(String) # 'Regular Season' | 'Playoffs'
    match_timestamp = Column(DateTime, index=True)
    patch_version = Column(String, ForeignKey("patches.patch_version"), nullable=True)
    team_a_name = Column(String)
    team_b_name = Column(String)
    series_score_a = Column(Integer)
    series_score_b = Column(Integer)
    group = Column(String, nullable=True) # "Upper Bracket", "Lower Bracket", "Grand Final", "Group A", etc.
    
    games = relationship("Game", back_populates="match", cascade="all, delete-orphan")

class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(String, ForeignKey("matches.match_id"))
    game_number = Column(Integer)
    game_duration_seconds = Column(Integer, nullable=True)
    blue_side_team = Column(String)
    red_side_team = Column(String)
    map_winner = Column(String) # Team Name
    bans = Column(JSON) # {"blue": [5 heroes], "red": [5 heroes]}
    picks = Column(JSON) # {"blue": [5 heroes], "red": [5 heroes]}
    pick_ban_sequence = Column(JSON) # Chronological sequence
    blue_roster = Column(JSON, nullable=True) # [5 ign-names]
    red_roster = Column(JSON, nullable=True) # [5 ign-names]
    
    # Pillar 5: PH Macro Metrics
    blue_turtles = Column(Integer, default=0)
    red_turtles = Column(Integer, default=0)
    blue_lords = Column(Integer, default=0)
    red_lords = Column(Integer, default=0)
    
    match = relationship("Match", back_populates="games")
    player_performances = relationship("PlayerPerformance", back_populates="game")

class PlayerPerformance(Base):
    __tablename__ = "player_performances"
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    game_id = Column(Integer, ForeignKey("games.id"))
    hero_id = Column(Integer, ForeignKey("heroes.id"), nullable=True)
    team_name = Column(String)
    role = Column(String, nullable=True) # 'Jungler', 'Roamer', etc.
    side = Column(String) # 'Blue' or 'Red'
    
    player = relationship("Player", back_populates="performances")
    game = relationship("Game", back_populates="player_performances")

class Hero(Base):
    __tablename__ = "heroes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    # Base stats can be added here as needed for Feature Engineering

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    stats = Column(JSON, nullable=True)

class SeasonRoster(Base):
    __tablename__ = "season_rosters"
    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, index=True)
    team_name = Column(String, index=True)
    players = Column(JSON) # List of dictionaries: [{"ign": "...", "role": "...", "name": "..."}]
    staff = Column(JSON)   # List of dictionaries: [{"name": "...", "role": "..."}]
