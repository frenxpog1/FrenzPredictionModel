from database import SessionLocal
import models

db = SessionLocal()
matches = db.query(models.Match).all()

seen = set()
duplicates = []

for m in matches:
    key = (m.season, m.team_a_name, m.team_b_name, m.match_timestamp)
    if key in seen:
        duplicates.append(m)
    else:
        seen.add(key)

print(f"Found {len(duplicates)} duplicate matches.")

for d in duplicates:
    # Games will be deleted via cascade
    db.delete(d)

db.commit()
print("Cleanup complete.")
db.close()
