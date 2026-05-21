import sqlite3

def main():
    print("Clearing matches and games tables from mlbb_data.db...")
    conn = sqlite3.connect("mlbb_data.db")
    cursor = conn.cursor()
    
    # Check row counts before clearing
    cursor.execute("SELECT COUNT(*) FROM matches")
    print(f"  Matches before: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM games")
    print(f"  Games before: {cursor.fetchone()[0]}")
    
    # Delete matches and games
    cursor.execute("DELETE FROM games")
    cursor.execute("DELETE FROM matches")
    conn.commit()
    
    # Check row counts after clearing
    cursor.execute("SELECT COUNT(*) FROM matches")
    print(f"  Matches after: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM games")
    print(f"  Games after: {cursor.fetchone()[0]}")
    
    conn.close()
    print("Clear completed successfully.")

if __name__ == "__main__":
    main()
