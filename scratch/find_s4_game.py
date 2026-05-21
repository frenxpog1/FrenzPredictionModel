import sqlite3
import json

conn = sqlite3.connect("mlbb_data.db")
cursor = conn.cursor()

# Find matches between ONIC PH and SGD Omega in Season 4
cursor.execute("""
    SELECT match_id, season, team_a_name, team_b_name 
    FROM matches 
    WHERE season = 4 
      AND ((team_a_name LIKE '%ONIC%' AND team_b_name LIKE '%Omega%') 
           OR (team_a_name LIKE '%Omega%' AND team_b_name LIKE '%ONIC%'))
""")
matches = cursor.fetchall()
print(f"Found {len(matches)} matches in Season 4:")
for m in matches:
    match_id, season, t1, t2 = m
    print(f"Match ID: {match_id} | {t1} vs {t2}")
    
    # Query games for this match
    cursor.execute("""
        SELECT game_number, game_duration_seconds, blue_side_team, red_side_team, map_winner, picks, bans 
        FROM games 
        WHERE match_id = ?
        ORDER BY game_number
    """, (match_id,))
    games = cursor.fetchall()
    for g in games:
        gnum, duration, blue, red, winner, picks, bans = g
        print(f"  Game {gnum}: Duration = {duration} | Blue = {blue} | Red = {red} | Winner = {winner}")

conn.close()
