from database import SessionLocal
import models

TEAM_MAP = {
    "Blacklist International":      "Blacklist Intl.",
    "AP.Bren":                      "AP.Bren",
    "Bren Esports":                 "AP.Bren",
    "Falcons AP.Bren":              "AP.Bren",
    "Team Falcons PH":              "AP.Bren",
    "ONIC Philippines":             "ONIC PH",
    "Fnatic ONIC PH":               "ONIC PH",
    "ECHO":                         "Team Liquid PH",
    "Liquid Echo":                  "Team Liquid PH",
    "Aurora Gaming PH":             "Aurora",
    "Aurora PH":                    "Aurora",
    "Aura PH":                      "Aurora",
    "RSG Philippines":              "RSG PH",
    "RSG Slate PH":                 "RSG PH",
    "Nexplay Solid":                "Nexplay EVOS",
}

db = SessionLocal()

matches = db.query(models.Match).all()
for m in matches:
    if m.team_a_name in TEAM_MAP:
        m.team_a_name = TEAM_MAP[m.team_a_name]
    if m.team_b_name in TEAM_MAP:
        m.team_b_name = TEAM_MAP[m.team_b_name]

games = db.query(models.Game).all()
for g in games:
    if g.blue_side_team in TEAM_MAP:
        g.blue_side_team = TEAM_MAP[g.blue_side_team]
    if g.red_side_team in TEAM_MAP:
        g.red_side_team = TEAM_MAP[g.red_side_team]
    if g.map_winner in TEAM_MAP:
        g.map_winner = TEAM_MAP[g.map_winner]

db.commit()
print("Extended team normalization complete.")
db.close()
