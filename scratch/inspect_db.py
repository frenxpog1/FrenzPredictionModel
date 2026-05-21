import sqlite3

conn = sqlite3.connect("mlbb_data.db")
cursor = conn.cursor()

# 1. Total matches
cursor.execute("SELECT COUNT(*) FROM matches")
total_matches = cursor.fetchone()[0]

# 2. Matches with 00:00:00 time component
cursor.execute("SELECT COUNT(*) FROM matches WHERE strftime('%H:%M:%S', match_timestamp) = '00:00:00'")
zero_time_matches = cursor.fetchone()[0]

# 3. Total games
cursor.execute("SELECT COUNT(*) FROM games")
total_games = cursor.fetchone()[0]

# 4. Games with None or 0 duration
cursor.execute("SELECT COUNT(*) FROM games WHERE game_duration_seconds IS NULL OR game_duration_seconds = 0")
null_duration_games = cursor.fetchone()[0]

print(f"Total Matches: {total_matches}")
print(f"Matches with 00:00:00 time: {zero_time_matches} ({zero_time_matches/total_matches*100:.1f}%)")
print(f"Total Games: {total_games}")
print(f"Games with missing/zero duration: {null_duration_games} ({null_duration_games/total_games*100:.1f}%)")

# Let's inspect some of the games that do have a duration to see if it's there
cursor.execute("SELECT game_duration_seconds, blue_side_team, red_side_team, map_winner FROM games WHERE game_duration_seconds IS NOT NULL LIMIT 5")
print("\nSample games with duration:")
for row in cursor.fetchall():
    print(row)

conn.close()
