import sqlite3
import csv
import os
import json

def delete_old_csvs():
    old_files = ['matches.csv', 'games.csv', 'heroes.csv', 'patches.csv', 'season_rosters.csv']
    print("Cleaning up old CSV files in the root directory...")
    for filename in old_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"  Deleted: {filename}")
            except Exception as e:
                print(f"  Error deleting {filename}: {e}")

def export_table_to_csv(cursor, query, csv_filepath):
    print(f"Exporting: {os.path.basename(csv_filepath)}...")
    cursor.execute(query)
    
    # Get column headers
    headers = [description[0] for description in cursor.description]
    
    # Fetch all rows
    rows = cursor.fetchall()
    
    # Write to CSV
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"  Saved {len(rows)} rows to {csv_filepath}")
    return len(rows)

def aggregate_hero_stats(cursor, csv_filepath):
    print(f"Aggregating hero stats for {os.path.basename(csv_filepath)}...")
    
    # Query games and matches to count picks, bans, wins per season & stage
    query = """
        SELECT 
            m.season, 
            m.stage, 
            g.blue_side_team, 
            g.red_side_team, 
            g.map_winner, 
            g.picks, 
            g.bans 
        FROM games g
        JOIN matches m ON g.match_id = m.match_id
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # hero_stats[(season, stage, hero_name)] = {picks, bans, wins}
    hero_stats = {}
    
    for season, stage, blue_team, red_team, winner, picks_json, bans_json in rows:
        # Load picks & bans JSON
        try:
            picks = json.loads(picks_json) if picks_json else {}
        except Exception:
            picks = {}
            
        try:
            bans = json.loads(bans_json) if bans_json else {}
        except Exception:
            bans = {}
            
        blue_picks = picks.get('blue', [])
        red_picks = picks.get('red', [])
        blue_bans = bans.get('blue', [])
        red_bans = bans.get('red', [])
        
        # Count picks and wins
        # Blue picks
        for hero in blue_picks:
            if not hero or hero.startswith('Mock'):
                continue
            key = (season, stage, hero)
            if key not in hero_stats:
                hero_stats[key] = {'picks': 0, 'bans': 0, 'wins': 0}
            hero_stats[key]['picks'] += 1
            if blue_team == winner:
                hero_stats[key]['wins'] += 1
                
        # Red picks
        for hero in red_picks:
            if not hero or hero.startswith('Mock'):
                continue
            key = (season, stage, hero)
            if key not in hero_stats:
                hero_stats[key] = {'picks': 0, 'bans': 0, 'wins': 0}
            hero_stats[key]['picks'] += 1
            if red_team == winner:
                hero_stats[key]['wins'] += 1
                
        # Count bans
        for hero in blue_bans + red_bans:
            if not hero or hero.startswith('Mock'):
                continue
            key = (season, stage, hero)
            if key not in hero_stats:
                hero_stats[key] = {'picks': 0, 'bans': 0, 'wins': 0}
            hero_stats[key]['bans'] += 1

    # Convert aggregated dict into a flat list of rows
    output_rows = []
    for (season, stage, hero_name), stats in hero_stats.items():
        picks_count = stats['picks']
        bans_count = stats['bans']
        wins_count = stats['wins']
        
        # Calculate win rate and P+B rate
        win_rate = (wins_count / picks_count * 100) if picks_count > 0 else 0.0
        pb_rate = picks_count + bans_count
        
        # Calculate Tier based on dashboard rules:
        # - picks < 2: Data Limited
        # - win_rate >= 55: S-Tier
        # - win_rate >= 50: A-Tier
        # - win_rate >= 45: B-Tier
        # - else: C-Tier
        if picks_count < 2:
            tier = 'Data Limited'
        elif win_rate >= 55.0:
            tier = 'S-Tier'
        elif win_rate >= 50.0:
            tier = 'A-Tier'
        elif win_rate >= 45.0:
            tier = 'B-Tier'
        else:
            tier = 'C-Tier'
            
        output_rows.append({
            'season': season,
            'stage': stage,
            'hero_name': hero_name,
            'picks': picks_count,
            'bans': bans_count,
            'wins': wins_count,
            'win_rate': f"{win_rate:.1f}%",
            'pb_rate': pb_rate,
            'tier': tier
        })
        
    # Sort output rows:
    # 1. Chronologically by season ASC
    # 2. Stage (Regular Season first, then Playoffs)
    # 3. By picks DESC for a clear ranking
    # 4. Hero name alphabetically for stable tie-breaks
    stage_order = {'Regular Season': 0, 'Playoffs': 1}
    output_rows.sort(key=lambda r: (
        r['season'], 
        stage_order.get(r['stage'], 2), 
        -r['picks'], 
        r['hero_name']
    ))
    
    # Write to CSV
    headers = ['season', 'stage', 'hero_name', 'picks', 'bans', 'wins', 'win_rate', 'pb_rate', 'tier']
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)
            
    print(f"  Aggregated & saved {len(output_rows)} hero-season-stage records to {csv_filepath}")
    return len(output_rows)

def main():
    db_path = 'mlbb_data.db'
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found in the current directory.")
        return
        
    # 1. Clean up old files in the root folder
    delete_old_csvs()
    
    # 2. Create dedicated folder
    output_dir = 'csv_data'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder '{output_dir}' is ready.\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 3. Export matches
    matches_query = """
        SELECT * FROM matches 
        ORDER BY season ASC, match_timestamp ASC
    """
    export_table_to_csv(cursor, matches_query, os.path.join(output_dir, 'matches.csv'))
    
    # 4. Export games with season & match_timestamp included right after match_id!
    games_query = """
        SELECT 
            g.id, 
            g.match_id, 
            m.season, 
            m.match_timestamp, 
            g.game_number, 
            g.game_duration_seconds, 
            g.blue_side_team, 
            g.red_side_team, 
            g.map_winner, 
            g.bans, 
            g.picks, 
            g.pick_ban_sequence, 
            g.blue_roster, 
            g.red_roster 
        FROM games g
        LEFT JOIN matches m ON g.match_id = m.match_id
        ORDER BY m.season ASC, m.match_timestamp ASC, g.game_number ASC
    """
    export_table_to_csv(cursor, games_query, os.path.join(output_dir, 'games.csv'))
    
    # 5. Export season_rosters
    rosters_query = """
        SELECT * FROM season_rosters 
        ORDER BY season ASC, id ASC
    """
    export_table_to_csv(cursor, rosters_query, os.path.join(output_dir, 'season_rosters.csv'))
    
    # 6. Export patches
    patches_query = """
        SELECT * FROM patches 
        ORDER BY release_timestamp ASC
    """
    export_table_to_csv(cursor, patches_query, os.path.join(output_dir, 'patches.csv'))
    
    # 7. Aggregate & export heroes by season & stage
    aggregate_hero_stats(cursor, os.path.join(output_dir, 'heroes.csv'))
    
    # Quick confirmation showing row count per season for the matches table
    print("\n--- Row Count Per Season for Matches Table ---")
    cursor.execute("""
        SELECT season, COUNT(*) as match_count 
        FROM matches 
        GROUP BY season 
        ORDER BY season ASC
    """)
    season_counts = cursor.fetchall()
    total_matches = 0
    for season, count in season_counts:
        print(f"Season {season:2}: {count} matches")
        total_matches += count
    print(f"Total Matches: {total_matches}")
    
    conn.close()
    print("\nCSV export to 'csv_data/' completed successfully!")

if __name__ == '__main__':
    main()
