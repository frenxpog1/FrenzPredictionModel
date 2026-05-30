import argparse
import csv
import time
import uuid
from pathlib import Path

from bs4 import BeautifulSoup
from scrapling import DynamicFetcher


BASE_URL = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_{season}"
CSV_DIR = Path("csv_data")
MATCH_FIELDS = [
    "match_id",
    "season",
    "stage",
    "match_timestamp",
    "patch_version",
    "team_a_name",
    "team_b_name",
    "series_score_a",
    "series_score_b",
    "group",
]


def generate_uuid():
    return str(uuid.uuid4())


def fetch_season_page(season_num):
    url = BASE_URL.format(season=season_num)
    print(f"Fetching MPL PH Season {season_num}: {url}")
    result = DynamicFetcher().fetch(url)
    return BeautifulSoup(result.html_content, "html.parser")


def parse_score(value):
    text = str(value).strip()
    return int(text) if text.isdigit() else 0


def parse_matches(soup, season_num):
    matches = []

    for match in soup.select(".brkts-match-info"):
        teams = [team.get_text(strip=True) for team in match.select(".brkts-opponent-title-wrapper")]
        scores = [score.get_text(strip=True) for score in match.select(".brkts-opponent-score-wrapper")]
        date_el = match.select_one(".brkts-match-line-datetime")

        if len(teams) != 2:
            continue

        # Traverse up to find if the match is in a playoff/bracket section
        stage = "Regular Season"
        parent = match
        while parent:
            if parent.name == "div" and any(cls in parent.get("class", []) for cls in ["playoffs", "playoff", "bracket"]):
                stage = "Playoffs"
                break
            parent = parent.parent

        matches.append(
            {
                "match_id": generate_uuid(),
                "season": season_num,
                "stage": stage,
                "match_timestamp": date_el.get_text(strip=True) if date_el else "",
                "patch_version": "",
                "team_a_name": teams[0],
                "team_b_name": teams[1],
                "series_score_a": parse_score(scores[0]) if len(scores) > 0 else 0,
                "series_score_b": parse_score(scores[1]) if len(scores) > 1 else 0,
                "group": "",
            }
        )

    return matches


def parse_rosters(soup, season_num):
    rosters = []

    for card in soup.select(".teamcard, .participant-card"):
        name_el = card.select_one("center a, b a, .participant-card-header a") or card.find("a")
        team_name = name_el.get_text(strip=True) if name_el else ""
        if not team_name:
            continue

        players = []
        for row in card.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if not header or not value:
                continue

            role_icon = header.select_one("img")
            role = (role_icon.get("title") or role_icon.get("alt")) if role_icon else header.get_text(strip=True)
            ign_el = value.select_one("a")
            ign = (ign_el.get_text(strip=True) if ign_el else value.get_text(strip=True)).strip().lower()
            if ign:
                players.append({"ign": ign, "role": role})

        if players:
            rosters.append({"season": season_num, "team_name": team_name, "players": players})

    return rosters


def append_matches_csv(matches, path=CSV_DIR / "matches.csv"):
    if not matches:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing matches to deduplicate and preserve original UUIDs
    existing_keys = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # Unique key matching on season and resolved/stripped team names
                key = (
                    str(row["season"]),
                    str(row["team_a_name"]).strip().lower(),
                    str(row["team_b_name"]).strip().lower()
                )
                existing_keys[key] = row["match_id"]

    new_matches = []
    for match in matches:
        key = (
            str(match["season"]),
            str(match["team_a_name"]).strip().lower(),
            str(match["team_b_name"]).strip().lower()
        )
        if key in existing_keys:
            # Duplicate match found. Do not append it again!
            # Optionally update fields if we want, but definitely preserve original UUID
            print(f"Skipping duplicate match or preserving match_id: {match['team_a_name']} vs {match['team_b_name']} (Season {match['season']})")
        else:
            new_matches.append(match)

    if not new_matches:
        print("No new matches found. matches.csv is fully up to date.")
        return

    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_matches)
    print(f"Successfully appended {len(new_matches)} new matches to matches.csv.")


def save_rosters_to_db(rosters):
    if not rosters:
        return

    from database import SessionLocal
    import models

    db = SessionLocal()
    try:
        for roster in rosters:
            existing = (
                db.query(models.SeasonRoster)
                .filter_by(season=roster["season"], team_name=roster["team_name"])
                .first()
            )
            if existing:
                existing.players = roster["players"]
            else:
                db.add(
                    models.SeasonRoster(
                        season=roster["season"],
                        team_name=roster["team_name"],
                        players=roster["players"],
                    )
                )

            for player in roster["players"]:
                ign = player["ign"].strip().lower()
                if ign and not db.query(models.Player).filter_by(ign=ign).first():
                    db.add(models.Player(ign=ign))

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def scrape_season(season_num, save_matches=True, save_rosters=True):
    soup = fetch_season_page(season_num)
    matches = parse_matches(soup, season_num)
    rosters = parse_rosters(soup, season_num)

    if save_matches:
        append_matches_csv(matches)
    if save_rosters:
        save_rosters_to_db(rosters)

    print(f"Season {season_num}: {len(matches)} matches, {len(rosters)} rosters")
    return matches, rosters


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape MPL PH matches and rosters from Liquipedia.")
    parser.add_argument("seasons", nargs="+", type=int, help="Season numbers to scrape, for example: 13 14")
    parser.add_argument("--skip-matches", action="store_true", help="Do not append scraped matches to csv_data/matches.csv")
    parser.add_argument("--skip-rosters", action="store_true", help="Do not write scraped rosters to the local database")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between seasons in seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    for index, season in enumerate(args.seasons):
        scrape_season(
            season,
            save_matches=not args.skip_matches,
            save_rosters=not args.skip_rosters,
        )
        if index < len(args.seasons) - 1:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
