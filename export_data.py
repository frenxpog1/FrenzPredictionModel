"""
export_data.py
--------------
Reads mlbb_data.db and exports all match, game, hero, item data
into static/data.js so the HTML dashboard works without a backend.

Run this AFTER scraping:
    python export_data.py
"""

import json
import sys
from database import SessionLocal
import models

PATCH_TYPES = {"BUFF", "NERF", "ADJUST", "NEW"}


def serialize_datetime(dt):
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d")


def clean_hero_adjustments(adjustments):
    cleaned = {}
    for hero, patch_type in (adjustments or {}).items():
        patch_type = str(patch_type or "").upper()
        if patch_type not in PATCH_TYPES:
            continue
        if not hero or " - " in hero:
            continue
        cleaned[hero] = patch_type
    return cleaned


def export_to_js():
    db = SessionLocal()
    print("Reading database...")

    # ── Heroes ───────────────────────────────────────────────────
    heroes = db.query(models.Hero).order_by(models.Hero.name).all()
    heroes_data = [{"id": h.id, "name": h.name} for h in heroes]

    # ── Items ─────────────────────────────────────────────────────
    items = db.query(models.Item).order_by(models.Item.name).all()
    items_data = [{"id": i.id, "name": i.name, "stats": i.stats or {}} for i in items]

    # ── Matches + Games ───────────────────────────────────────────
    matches = (
        db.query(models.Match)
        .order_by(models.Match.season.desc(), models.Match.match_timestamp.desc())
        .all()
    )

    matches_data = []
    for m in matches:
        games = db.query(models.Game).filter_by(match_id=m.match_id).order_by(models.Game.game_number).all()

        games_data = []
        for g in games:
            games_data.append({
                "game_number":          g.game_number,
                "blue_side_team":       g.blue_side_team,
                "red_side_team":        g.red_side_team,
                "winner":               g.map_winner,
                "duration_seconds":     g.game_duration_seconds,
                "bans":                 g.bans or {"blue": [], "red": []},
                "picks":                g.picks or {"blue": [], "red": []},
                "pick_ban_sequence":    g.pick_ban_sequence or [],
                "blue_roster":          g.blue_roster or [],
                "red_roster":           g.red_roster or [],
            })

        winner = None
        if m.series_score_a is not None and m.series_score_b is not None:
            if m.series_score_a > m.series_score_b:
                winner = m.team_a_name
            elif m.series_score_b > m.series_score_a:
                winner = m.team_b_name

        matches_data.append({
            "match_id":     m.match_id,
            "season":       m.season,
            "stage":        m.stage,
            "date":         serialize_datetime(m.match_timestamp),
            "patch":        m.patch_version,
            "team_a":       m.team_a_name,
            "team_b":       m.team_b_name,
            "score_a":      m.series_score_a,
            "score_b":      m.series_score_b,
            "group":        m.group,
            "winner":       winner,
            "games":        games_data,
        })

    # ── Build hero stats per season ───────────────────────────────
    # Count picks, bans, wins for each hero across all games
    hero_stats = {}  # hero_name -> {season -> {picks, bans, wins, games_played}}

    for m in matches_data:
        season = m["season"]
        for g in m["games"]:
            blue_picks = g["picks"].get("blue", [])
            red_picks  = g["picks"].get("red", [])
            blue_bans  = g["bans"].get("blue", [])
            red_bans   = g["bans"].get("red", [])
            winner     = g["winner"]
            blue_team  = g["blue_side_team"]
            red_team   = g["red_side_team"]

            all_picked = [(h, "blue", blue_team) for h in blue_picks] + [(h, "red", red_team) for h in red_picks]
            all_banned = blue_bans + red_bans

            for hero, side, team in all_picked:
                if not hero or hero.startswith("Mock"):
                    continue
                key = hero
                if key not in hero_stats:
                    hero_stats[key] = {}
                if season not in hero_stats[key]:
                    hero_stats[key][season] = {"picks": 0, "bans": 0, "wins": 0, "games": 0}
                hero_stats[key][season]["picks"]  += 1
                hero_stats[key][season]["games"]  += 1
                if team == winner:
                    hero_stats[key][season]["wins"] += 1

            for hero in all_banned:
                if not hero or hero.startswith("Mock"):
                    continue
                if hero not in hero_stats:
                    hero_stats[hero] = {}
                if season not in hero_stats[hero]:
                    hero_stats[hero][season] = {"picks": 0, "bans": 0, "wins": 0, "games": 0}
                hero_stats[hero][season]["bans"] += 1

    # ── Patches ──────────────────────────────────────────────────
    patches = db.query(models.Patch).order_by(models.Patch.release_timestamp.desc()).all()
    patches_data = []
    for p in patches:
        hero_adjustments = clean_hero_adjustments(p.hero_adjustments)
        if not hero_adjustments:
            continue
        patches_data.append({
            "version": p.patch_version,
            "date": serialize_datetime(p.release_timestamp),
            "hero_adjustments": hero_adjustments
        })

    # ── Rosters ──────────────────────────────────────────────────
    rosters = db.query(models.SeasonRoster).all()
    rosters_data = []
    for r in rosters:
        rosters_data.append({
            "season": r.season,
            "team": r.team_name,
            "players": r.players or [],
            "staff": r.staff or []
        })

    db.close()

    # ── Write data.js ─────────────────────────────────────────────
    js_content = f"""// Auto-generated by export_data.py — DO NOT EDIT MANUALLY
// Run: python export_data.py  to regenerate

const MLBB_DATA = {{
  heroes:  {json.dumps(heroes_data,  indent=2)},
  items:   {json.dumps(items_data,   indent=2)},
  matches: {json.dumps(matches_data, indent=2)},
  hero_stats: {json.dumps(hero_stats, indent=2)},
  patches: {json.dumps(patches_data, indent=2)},
  rosters: {json.dumps(rosters_data, indent=2)},
  seasons: {json.dumps(sorted(list(set(m["season"] for m in matches_data))))}
}};
"""

    out_path = "static/data.js"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    season_counts = {}
    for m in matches_data:
        s = m["season"]
        season_counts[s] = season_counts.get(s, 0) + 1

    print(f"\n✅ Exported to {out_path}")
    print(f"   Heroes  : {len(heroes_data)}")
    print(f"   Items   : {len(items_data)}")
    print(f"   Matches : {len(matches_data)}")
    print(f"   Seasons : {sorted(season_counts.keys())}")
    for s, c in sorted(season_counts.items()):
        print(f"     Season {s:2d}: {c} matches")


if __name__ == "__main__":
    export_to_js()
