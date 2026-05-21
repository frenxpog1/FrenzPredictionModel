from database import SessionLocal, init_db
import models
import datetime

def seed_data():
    db = SessionLocal()
    init_db()

    # Seed Patch
    v1_8_44 = models.Patch(version="1.8.44", release_date=datetime.datetime(2023, 12, 1))
    db.merge(v1_8_44)
    db.commit()

    # Seed Items
    demon_shoes = models.Item(
        name="Demon Shoes",
        cost=720,
        stats={"mana_regen": 6, "movement_speed": 40},
        passive="Mysticism: Assists or hero kills restore 10% of Max Mana."
    )
    db.merge(demon_shoes)
    
    # Seed Heroes
    miya = models.Hero(name="Miya", role="Marksman", win_rate=48.5)
    db.merge(miya)
    
    # Seed Teams
    team_a = models.Team(name="Blacklist International", location="Philippines")
    team_b = models.Team(name="AP.Bren", location="Philippines")
    db.merge(team_a)
    db.merge(team_b)
    db.commit()
    
    # Seed Match with Patch
    match = models.Match(
        tournament="MPL PH Season 12",
        patch_id=v1_8_44.id,
        team1_id=team_a.id,
        team2_id=team_b.id,
        score1=2,
        score2=0,
        date=datetime.datetime(2023, 12, 15)
    )
    db.add(match)
    db.commit()
    
    print("Patch-aware sample data seeded!")
    db.close()

if __name__ == "__main__":
    seed_data()
