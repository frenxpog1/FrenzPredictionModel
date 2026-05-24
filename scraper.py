import csv
import uuid
from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import time
from datetime import datetime

def generate_uuid():
    return str(uuid.uuid4())

def scrape_mpl_season(season_num):
    url = f"https://liquipedia.net/mobilelegends/MPL/Philippines/Season_{season_num}"
    print(f"Scraping Season {season_num} from {url}...")
    
    fetcher = DynamicFetcher()
    try:
        result = fetcher.fetch(url)
        soup = BeautifulSoup(result.html_content, 'html.parser')
        
        matches = []
        # Liquipedia match structure for MPL
        match_containers = soup.select('.brkts-match-info')
        for match in match_containers:
            teams = [t.get_text(strip=True) for t in match.select('.brkts-opponent-title-wrapper')]
            scores = [s.get_text(strip=True) for s in match.select('.brkts-opponent-score-wrapper')]
            date_el = match.select_one('.brkts-match-line-datetime')
            date_str = date_el.get_text(strip=True) if date_el else ""
            
            # Simple date parsing or placeholder
            try:
                # Example: March 15, 2024 - 18:00
                # Liquipedia dates can be tricky, using a placeholder if parsing fails
                timestamp = date_str 
            except:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if len(teams) == 2:
                matches.append({
                    'match_id': generate_uuid(),
                    'season': season_num,
                    'stage': 'Regular Season', # Default
                    'match_timestamp': timestamp,
                    'patch_version': '', # Can be filled later
                    'team_a_name': teams[0],
                    'team_b_name': teams[1],
                    'series_score_a': scores[0] if len(scores) > 0 and scores[0].isdigit() else "0",
                    'series_score_b': scores[1] if len(scores) > 1 and scores[1].isdigit() else "0",
                    'group': ''
                })
        
        return matches
    except Exception as e:
        print(f"Error scraping Season {season_num}: {e}")
        return []

def save_matches(matches):
    if not matches:
        return
        
    fieldnames = ['match_id', 'season', 'stage', 'match_timestamp', 'patch_version', 'team_a_name', 'team_b_name', 'series_score_a', 'series_score_b', 'group']
    file_path = 'csv_data/matches.csv'

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(matches)

def scrape_rosters(season_num):
    url = f"https://liquipedia.net/mobilelegends/MPL/Philippines/Season_{season_num}"
    print(f"Scraping Rosters for Season {season_num}...")

    fetcher = DynamicFetcher()
    try:
        result = fetcher.fetch(url)
        soup = BeautifulSoup(result.html_content, "html.parser")

        team_cards = soup.select(".teamcard, .participant-card")
        rosters = []

        for card in team_cards:
            name_el = card.select_one("center a, b a, .participant-card-header a")
            if not name_el: name_el = card.find("a")
            team_name = name_el.get_text(strip=True) if name_el else "Unknown"

            players = []
            rows = card.find_all("tr")
            for row in rows:
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    img = th.select_one("img")
                    role = img.get("title") or img.get("alt") if img else th.get_text(strip=True)
                    ign_el = td.select_one("a")
                    ign_raw = ign_el.get_text(strip=True) if ign_el else td.get_text(strip=True)
                    ign = ign_raw.strip().lower()
                    players.append({"ign": ign, "role": role})

            rosters.append({
                "season": season_num,
                "team_name": team_name,
                "players": players
            })
            print(f"Found {len(players)} players for {team_name}")

            return rosters

    except Exception as e:
        print(f"Error scraping rosters for Season {season_num}: {e}")
        return []

def save_rosters_to_db(rosters):
    from database import SessionLocal
    import models

    print(f"Saving {len(rosters)} rosters to DB...")
    db = SessionLocal()
    try:
        for r in rosters:
            team_name = r['team_name']
            season = r['season']
            players = r['players']
            
            # Update SeasonRoster table
            existing = db.query(models.SeasonRoster).filter_by(season=season, team_name=team_name).first()
            if existing:
                existing.players = players
            else:
                new_roster = models.SeasonRoster(season=season, team_name=team_name, players=players)
                db.add(new_roster)

            # Update Player table for Pillar 3 Identity resolution
            for p in players:
                player_ign = p['ign'].strip().lower()
                if not player_ign:
                    continue
                
                player = db.query(models.Player).filter_by(ign=player_ign).first()
                if not player:
                    new_player = models.Player(ign=player_ign)
                    db.add(new_player)
                    print(f"  + Added Player: {player_ign}")
        
        db.commit()
        print("Database commit successful.")
    except Exception as e:
        print(f"Error saving rosters: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == '__main__':
    for season in [13, 14]:
        # Scrape Matches
        season_matches = scrape_mpl_season(season)
        if season_matches:
            save_matches(season_matches)

        # Scrape Rosters
        season_rosters = scrape_rosters(season)
        if season_rosters:
            save_rosters_to_db(season_rosters)
            print(f"Successfully updated rosters and players for Season {season}.")

        time.sleep(2)

