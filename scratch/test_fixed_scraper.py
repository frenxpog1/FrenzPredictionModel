import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper
import models
from database import Base

def main():
    print("Initializing test database...")
    test_db_url = "sqlite:///./scratch/test_mlbb.db"
    
    # If the test database exists, remove it
    if os.path.exists("./scratch/test_mlbb.db"):
        os.remove("./scratch/test_mlbb.db")
        
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Override scraper's database session
    scraper.SessionLocal = SessionLocal
    
    # Initialize the database schema
    Base.metadata.create_all(bind=engine)
    print("Schema initialized.")
    
    # Let's scrape Season 14
    print("Scraping Season 14 for verification...")
    scraper.scrape_season(14)
    
    # Now query the test database and inspect
    db = SessionLocal()
    try:
        games = db.query(models.Game).all()
        print(f"\nSuccessfully scraped {len(games)} games.")
        
        # Let's group games by match_id and print 2-1 matches
        from collections import defaultdict
        match_games = defaultdict(list)
        for g in games:
            match_games[g.match_id].append(g)
            
        print("\nChecking first 5 Best-of-3 (or longer) matches:")
        count = 0
        for match_id, g_list in match_games.items():
            g_list = sorted(g_list, key=lambda x: x.game_number)
            if len(g_list) >= 3:
                match = db.query(models.Match).filter_by(match_id=match_id).first()
                if match:
                    print(f"\nMatch: {match.team_a_name} ({match.series_score_a}) vs {match.team_b_name} ({match.series_score_b})")
                    for g in g_list:
                        print(f"  Game {g.game_number}: Blue={g.blue_side_team}, Red={g.red_side_team}, Winner={g.map_winner}")
                    count += 1
                    if count >= 5:
                        break
    finally:
        db.close()

if __name__ == "__main__":
    main()
