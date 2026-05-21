"""
scraper.py — MPL PH Seasons 1-17 scraper
-----------------------------------------
Scrapes Regular Season + Playoffs for all seasons from Liquipedia.
Uses a fast static fetch first, then DynamicFetcher as a fallback.
Extracts real match dates from timer spans or plain match text.

Usage:
    python3 scraper.py           # scrape all seasons 1-17 (default)
    python3 scraper.py 14        # scrape only season 14
    python3 scraper.py 13 14 15  # scrape specific seasons
"""

import sys
import time
import datetime
import uuid
import random
import traceback
import re
import logging
from bs4 import BeautifulSoup
from scrapling import DynamicFetcher, Fetcher
from database import SessionLocal, init_db
import models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scraper.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

BASE_URL = "https://liquipedia.net/mobilelegends"
static_fetcher = Fetcher()
dynamic_fetcher = DynamicFetcher()

# ─── Team Name Normalization ──────────────────────────────────────────────────
TEAM_MAP = {
    "Blacklist International":      "Blacklist Intl.",
    "AP.Bren":                      "AP.Bren",
    "Bren Esports":                 "AP.Bren",
    "Falcons AP.Bren":              "AP.Bren",
    "Team Falcons PH":              "Team Falcons PH",
    "ONIC Philippines":             "ONIC PH",
    "Fnatic ONIC PH":               "ONIC PH",
    "ECHO":                         "Team Liquid PH",
    "Liquid Echo":                  "Team Liquid PH",
    "Team Liquid PH":               "Team Liquid PH",
    "Aurora Gaming PH":             "Aurora",
    "Aurora PH":                    "Aurora",
    "Aurora":                       "Aurora",
    "Aura PH":                      "Aurora",
    "RSG Philippines":              "RSG PH",
    "RSG Slate PH":                 "RSG PH",
    "RSG PH":                       "RSG PH",
    "Minana EVOS":                  "Minana EVOS",
    "Nexplay EVOS":                 "Nexplay EVOS",
    "Nexplay Solid":                "Nexplay EVOS",
}

def normalize_team(name: str) -> str:
    if not name: return name
    name = name.strip()
    return TEAM_MAP.get(name, name)


# ─── Season date ranges (fallback if scraping fails) ──────────────────────────
SEASON_YEARS = {
    1: 2018, 2: 2018, 3: 2019, 4: 2019, 5: 2020, 6: 2020,
    7: 2021, 8: 2021, 9: 2022, 10: 2022, 11: 2023, 12: 2023,
    13: 2024, 14: 2024, 15: 2025, 16: 2025, 17: 2025,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_soup(url: str, min_d=1.0, max_d=2.0):
    delay = random.uniform(min_d, max_d)
    log.info(f"[{delay:.1f}s] {url}")
    time.sleep(delay)
    try:
        r = static_fetcher.get(url)
        if r.status == 200:
            return BeautifulSoup(r.html_content, "html.parser")
        log.warning(f"HTTP {r.status} for {url}")
    except Exception as e:
        log.warning(f"Static fetch error {url}: {e}")
    try:
        r = dynamic_fetcher.fetch(url)
        if r.status == 200:
            return BeautifulSoup(r.html_content, "html.parser")
        log.warning(f"Dynamic HTTP {r.status} for {url}")
    except Exception as e:
        log.error(f"Dynamic fetch error {url}: {e}")
    return None


def parse_real_date(popup, season_num: int, fallback_month_day: str = None) -> datetime.datetime:
    if popup:
        timer = popup.select_one(".timer-object-date")
        if timer:
            raw = timer.get_text(strip=True)
            date_part = raw.split(" - ")[0].strip()
            for fmt in ["%B %d, %Y", "%B %d %Y"]:
                try:
                    return datetime.datetime.strptime(date_part, fmt)
                except ValueError:
                    continue
        raw = clean_text(popup.get_text(" ", strip=True))
        match = re.search(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b", raw)
        if match:
            try:
                return datetime.datetime.strptime(match.group(1), "%B %d, %Y")
            except ValueError:
                pass

    if fallback_month_day:
        year = SEASON_YEARS.get(season_num, 2024)
        for fmt in ["%B %d", "%b %d"]:
            try:
                d = datetime.datetime.strptime(fallback_month_day.strip(), fmt)
                return d.replace(year=year)
            except ValueError:
                continue

    return datetime.datetime(SEASON_YEARS.get(season_num, 2024), 1, 1)


def imgs_to_names(container) -> list:
    if not container:
        return []
    return [
        img["alt"].strip()
        for img in container.find_all("img")
        if img.get("alt") and img["alt"].strip()
    ]


def upsert_hero(db, name: str):
    if not name:
        return
    try:
        hero = db.query(models.Hero).filter_by(name=name).first()
        if not hero:
            db.add(models.Hero(name=name))
            db.commit()
    except Exception:
        db.rollback()
        pass


PATCH_TYPES = {
    "BUFF": "BUFF",
    "BUFFED": "BUFF",
    "NERF": "NERF",
    "NERFED": "NERF",
    "ADJUST": "ADJUST",
    "ADJUSTED": "ADJUST",
    "ADJUSTMENT": "ADJUST",
    "REVAMP": "ADJUST",
    "REVAMPED": "ADJUST",
    "REWORK": "ADJUST",
    "REWORKED": "ADJUST",
    "NEW": "NEW",
}


def clean_text(txt: str) -> str:
    return " ".join((txt or "").replace("\xa0", " ").split())


def clean_heading_name(txt: str) -> str:
    txt = clean_text(txt)
    txt = re.sub(r"^\s*[IVXLCM\d]+[.)]?\s+", "", txt, flags=re.I)
    txt = re.sub(r"\s*\[[^\]]+\]\s*$", "", txt)
    txt = re.sub(r"\s*\([^)]*\)\s*$", "", txt)
    return txt.strip(" :-")


def normalize_patch_type(txt: str):
    key = clean_text(txt).upper()
    key = re.sub(r"[^A-Z ]", "", key).strip()
    return PATCH_TYPES.get(key)


def is_heading_tag(node, names=("h1", "h2", "h3")) -> bool:
    if not getattr(node, "name", None):
        return False
    if node.name in names:
        return True
    if "mw-heading" in (node.get("class") or []):
        return bool(node.find(names))
    return False


def heading_text(node) -> str:
    if not node:
        return ""
    heading = node.find(["h1", "h2", "h3"]) if node.name != "h2" and node.name != "h3" else node
    return clean_heading_name(heading.get_text(" ", strip=True) if heading else node.get_text(" ", strip=True))


def iter_between_headings(heading):
    start = heading.parent if heading.parent and "mw-heading" in (heading.parent.get("class") or []) else heading
    for sibling in start.next_siblings:
        if is_heading_tag(sibling):
            break
        yield sibling


def first_patch_type_after_heading(heading):
    for node in iter_between_headings(heading):
        if not hasattr(node, "find_all"):
            continue
        for el in node.find_all(["span", "b", "strong", "div"], recursive=True):
            patch_type = normalize_patch_type(el.get_text(" ", strip=True))
            if patch_type:
                return patch_type
        patch_type = normalize_patch_type(node.get_text(" ", strip=True))
        if patch_type:
            return patch_type
    return None


def extract_infobox_value(soup, label: str) -> str:
    """Read a Liquipedia infobox value even when label/value cells use different classes."""
    label = label.lower().rstrip(":")
    for cell in soup.select(".fo-nttax-infobox-wrapper div, .fo-nttax-infobox div, .infobox div"):
        txt = clean_text(cell.get_text(" ", strip=True)).lower().rstrip(":")
        if txt != label:
            continue
        sibling = cell.find_next_sibling()
        while sibling:
            sibling_text = clean_text(sibling.get_text(" ", strip=True))
            sibling_key = sibling_text.lower().rstrip(":")
            sibling_classes = sibling.get("class") or []
            if sibling_text and not sibling_key.endswith(":") and "infobox-description" not in sibling_classes:
                return sibling_text
            if sibling_key.endswith(":") or "infobox-description" in sibling_classes:
                break
            sibling = sibling.find_next_sibling()
    return ""


def patch_data_needs_refresh(hero_adjustments) -> bool:
    if not hero_adjustments:
        return True
    for hero, patch_type in hero_adjustments.items():
        if patch_type not in {"BUFF", "NERF", "ADJUST", "NEW"}:
            return True
        if not hero or hero.isupper() or " - " in hero or hero.upper() in {"LORD", "MINIONS", "CREEPS"}:
            return True
    return False


def duration_to_seconds(txt: str) -> int:
    if not txt:
        return None
    txt = txt.strip()
    parts = txt.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        pass
    return None


def parse_draft(popup, t1, t2, s1, s2, match_id, db):
    if not popup: return
    t1, t2 = normalize_team(t1), normalize_team(t2)
    veto = popup.select_one(".brkts-popup-mapveto")
    ban_rounds = []
    if veto:
        for ban_row in veto.select(".brkts-popup-mapveto__ban-round"):
            cells = ban_row.select(".brkts-popup-mapveto__ban-round-picks")
            ban_rounds.append((
                imgs_to_names(cells[0]) if len(cells) > 0 else [],
                imgs_to_names(cells[1]) if len(cells) > 1 else [],
            ))

    game_rows = popup.select(".brkts-popup-body-grid-row")
    for i, row in enumerate(game_rows):
        thumbs = row.select(".brkts-popup-body-element-thumbs")
        left_picks  = imgs_to_names(thumbs[0]) if len(thumbs) > 0 else []
        right_picks = imgs_to_names(thumbs[1]) if len(thumbs) > 1 else []
        left_bans, right_bans = ban_rounds[i] if i < len(ban_rounds) else ([], [])

        blue_team  = t1 if i % 2 == 0 else t2
        red_team   = t2 if i % 2 == 0 else t1
        blue_picks = left_picks  if i % 2 == 0 else right_picks
        red_picks  = right_picks if i % 2 == 0 else left_picks
        blue_bans  = left_bans   if i % 2 == 0 else right_bans
        red_bans   = right_bans  if i % 2 == 0 else left_bans
        # Determine actual game winner using the generic-label indicators
        g_winner = None
        labels = row.select(".generic-label")
        if len(labels) >= 2:
            left_type = labels[0].get("data-label-type", "") or ""
            right_type = labels[-1].get("data-label-type", "") or ""
            if "win" in left_type:
                g_winner = t1
            elif "win" in right_type:
                g_winner = t2
        
        if g_winner is None:
            # Fallback to series score procedural assignment
            g_winner = t1 if i < s1 else t2


        duration_seconds = None
        detail_block = row.select_one(".brkts-popup-body-grid-row-detail")
        if detail_block:
            for el in detail_block.find_all(["span", "div"]):
                t = el.get_text(strip=True)
                if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', t):
                    duration_seconds = duration_to_seconds(t)
                    break
        
        if duration_seconds is None:
            for element in row.select(".brkts-popup-spaced, .brkts-popup-body-element-center, .brkts-popup-body-element, .brkts-popup-body-element-text"):
                txt = element.get_text(strip=True)
                if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', txt):
                    duration_seconds = duration_to_seconds(txt)
                    break

        for h in blue_picks + red_picks:
            upsert_hero(db, h)

        db.add(models.Game(
            match_id=match_id,
            game_number=i + 1,
            game_duration_seconds=duration_seconds,
            blue_side_team=blue_team,
            red_side_team=red_team,
            map_winner=g_winner,
            bans={"blue": blue_bans, "red": red_bans},
            picks={"blue": blue_picks, "red": red_picks},
            pick_ban_sequence=[],
            blue_roster=[],
            red_roster=[],
        ))


# ─── Patch scraper ────────────────────────────────────────────────────────────

def scrape_detailed_patch(version: str):
    """Fetch an individual patch page and extract hero-only balance changes."""
    clean_v = version.replace("Patch ", "").strip()
    url = f"{BASE_URL}/Patch_{clean_v}"
    soup = get_soup(url, min_d=1.0, max_d=2.0)
    if not soup: return {}
    adjustments = {}

    current_section = None
    headings = soup.select(".mw-parser-output h2, .mw-parser-output h3")
    if not headings:
        headings = soup.find_all(["h2", "h3"])

    for heading in headings:
        title = heading_text(heading)
        title_key = title.lower()

        if heading.name == "h2":
            if "new hero" in title_key:
                current_section = "new"
            elif "hero" in title_key and any(word in title_key for word in ["adjust", "balance", "revamp"]):
                current_section = "hero"
            else:
                current_section = None
            continue

        if heading.name != "h3" or current_section not in {"new", "hero"}:
            continue

        hero_name = clean_heading_name(title)
        if len(hero_name) < 2 or len(hero_name) > 32:
            continue

        if current_section == "new":
            adjustments[hero_name] = "NEW"
            continue

        patch_type = first_patch_type_after_heading(heading)
        if patch_type in {"BUFF", "NERF", "ADJUST", "NEW"}:
            adjustments[hero_name] = patch_type

    return adjustments

def scrape_patches(db):
    log.info("Scraping patches...")
    soup = get_soup(f"{BASE_URL}/Portal:Patches")
    if not soup: return
    patch_list = []
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for table in soup.select("table.wikitable"):
        rows = table.find_all("tr")
        if not rows: continue
        header_row = rows[0].get_text().strip()
        if "Patch" not in header_row or "Release Date" not in header_row: continue
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 2: continue
            raw_v = cols[0].get_text(strip=True)
            if not raw_v or raw_v in months_abbr: continue
            version = raw_v.split("(")[0].replace("Patch", "").strip()
            date_str = cols[1].get_text(strip=True)
            if not version or len(version) < 3: continue
            patch_list.append((version, date_str))

    for version, date_str in patch_list:
        date = None
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"]:
            try:
                date = datetime.datetime.strptime(date_str, fmt)
                break
            except ValueError: continue
        if not date: date = datetime.datetime.now(datetime.timezone.utc)
        existing = db.query(models.Patch).filter_by(patch_version=version).first()
        if not existing or patch_data_needs_refresh(existing.hero_adjustments):
            log.info(f"  Fetching details for patch {version}...")
            hero_adjs = scrape_detailed_patch(version)
            for hero in hero_adjs:
                upsert_hero(db, hero)
            if not existing:
                db.add(models.Patch(patch_version=version, release_timestamp=date, hero_adjustments=hero_adjs))
            else:
                existing.hero_adjustments, existing.release_timestamp = hero_adjs, date
            db.commit()
        else: log.info(f"  Patch {version} already in DB.")
    log.info(f"  Patch scraping complete.")


# ─── Playoffs bracket scraper ─────────────────────────────────────────────────

def scrape_playoffs(soup, season_num: int, default_patch: str, db) -> int:
    bracket_matches = soup.select(".brkts-match")
    log.info(f"  Playoffs: {len(bracket_matches)} bracket matches")
    ok = 0
    for m_box in bracket_matches:
        group_label = None
        p = m_box.parent
        while p and p.name != 'body':
            header = p.select_one(".brkts-header")
            if header:
                txt = header.get_text(strip=True).upper()
                if "UPPER" in txt: group_label = "Upper Bracket"
                elif "LOWER" in txt: group_label = "Lower Bracket"
                elif "GRAND FINAL" in txt or "FINAL" in txt: group_label = "Finals"
                break
            p = p.parent
        try:
            opponents = m_box.select(".brkts-opponent-entry")
            if len(opponents) < 2: continue
            t1 = normalize_team((opponents[0].select_one(".name") or opponents[0]).get_text(strip=True))
            t2 = normalize_team((opponents[1].select_one(".name") or opponents[1]).get_text(strip=True))
            if not t1 or not t2: continue
            def get_score(opp):
                el = opp.select_one(".brkts-opponent-score-inner")
                txt = el.get_text(strip=True) if el else ""
                return int(txt) if txt.isdigit() else 0
            s1, s2 = get_score(opponents[0]), get_score(opponents[1])
            popup = m_box.select_one(".brkts-popup")
            match_date = parse_real_date(m_box, season_num)
            existing = db.query(models.Match).filter_by(season=season_num, team_a_name=t1, team_b_name=t2, match_timestamp=match_date).first()
            if existing:
                if group_label and not existing.group:
                    existing.group = group_label
                    db.commit()
                continue
            match_id = str(uuid.uuid4())
            db.add(models.Match(match_id=match_id, season=season_num, stage="Playoffs", group=group_label, match_timestamp=match_date, patch_version=default_patch if default_patch != "Unknown" else None, team_a_name=t1, team_b_name=t2, series_score_a=s1, series_score_b=s2))
            parse_draft(popup, t1, t2, s1, s2, match_id, db)
            db.commit()
            ok += 1
        except Exception as e:
            log.error(f"    Bracket match error: {e}"); db.rollback()
    return ok


# ─── Regular Season scraper ───────────────────────────────────────────────────

def scrape_regular_season(soup_rs, season_num: int, default_patch: str, db) -> int:
    matchlists = soup_rs.select(".brkts-matchlist")
    log.info(f"  Regular Season: {len(matchlists)} weeks")
    ok = 0
    for ml in matchlists:
        current_date_str = None
        for child in ml.children:
            if not hasattr(child, "select"): continue
            for header in child.select(".brkts-matchlist-header"):
                raw = header.get_text(strip=True)
                if raw: current_date_str = raw
            for m_box in child.select(".brkts-matchlist-match"):
                try:
                    opponents = m_box.select(".brkts-matchlist-opponent")
                    if len(opponents) < 2: continue
                    t1 = normalize_team(opponents[0].get("aria-label", "").strip() or (opponents[0].select_one(".name") or opponents[0]).get_text(strip=True))
                    t2 = normalize_team(opponents[1].get("aria-label", "").strip() or (opponents[1].select_one(".name") or opponents[1]).get_text(strip=True))
                    if not t1 or not t2: continue
                    score_divs = m_box.select(".brkts-matchlist-score")
                    s1 = int(score_divs[0].get_text(strip=True)) if score_divs and score_divs[0].get_text(strip=True).isdigit() else 0
                    s2 = int(score_divs[1].get_text(strip=True)) if len(score_divs) > 1 and score_divs[1].get_text(strip=True).isdigit() else 0
                    popup = m_box.select_one(".brkts-popup")
                    match_date = parse_real_date(m_box, season_num, current_date_str)
                    existing = db.query(models.Match).filter_by(season=season_num, team_a_name=t1, team_b_name=t2, match_timestamp=match_date).first()
                    if existing: continue
                    match_id = str(uuid.uuid4())
                    db.add(models.Match(match_id=match_id, season=season_num, stage="Regular Season", match_timestamp=match_date, patch_version=default_patch if default_patch != "Unknown" else None, team_a_name=t1, team_b_name=t2, series_score_a=s1, series_score_b=s2))
                    parse_draft(popup, t1, t2, s1, s2, match_id, db)
                    db.commit()
                    ok += 1
                except Exception as e:
                    log.error(f"    RS match error: {e}"); db.rollback()
    return ok


# ─── Roster scraper ───────────────────────────────────────────────────────────

def extract_canonical_ign(a_tag) -> str:
    if not a_tag: return ""
    title = a_tag.get("title", "")
    if title:
        # Extract "KarlTzy" from "KarlTzy (page does not exist)"
        title = title.replace("(page does not exist)", "").strip()
        # Clean up any leftover disambiguation like "(Player)"
        title = re.sub(r'\(.*?\)', '', title).strip()
        return title
    
    # Fallback to text
    txt = a_tag.get_text(strip=True)
    txt = re.sub(r'\(.*?\)', '', txt).replace("DNP", "").replace("SUB", "").strip()
    return txt

def scrape_rosters(soup, season_num: int, db):
    """Scrape participating teams' rosters (players and staff)."""
    log.info(f"  Scraping rosters for Season {season_num}...")
    
    # Support both old and new Liquipedia layouts
    team_cards = soup.select(".teamcard, .participant-card, .team-participant-card")
    log.info(f"    Found {len(team_cards)} team cards.")
    if not team_cards: return

    for card in team_cards:
        try:
            # 1. Team Name
            # Old: center a, New: .team-participant-card__opponent-full .name a
            name_el = card.select_one(".teamcard-inner center b a, .teamcard-inner center a, center a, .participant-card-header a, .team-participant-card__opponent-full .name a")
            if not name_el: name_el = card.find("a")
            if not name_el: continue
            team_name = normalize_team(name_el.get_text(strip=True))
            
            players, staff = [], []

            # --- CASE A: Modern Layout (.team-participant-card) ---
            if "team-participant-card" in card.get("class", []):
                members = card.select(".team-participant-card__member")
                for m in members:
                    # Role is in left img title or right text
                    role_left = m.select_one(".team-participant-card__member-role-left img")
                    role_txt = role_left.get("title") or role_left.get("alt") if role_left else ""
                    if not role_txt:
                        role_right = m.select_one(".team-participant-card__member-role-right")
                        role_txt = role_right.get_text(strip=True) if role_right else ""
                    
                    # IGN via Canonical extraction
                    ign_el = m.select_one(".team-participant-card__member-name .name a")
                    if not ign_el: continue
                    ign = extract_canonical_ign(ign_el)
                    if not ign or ign == "New Entry" or len(ign) < 2: continue
                    
                    entry = {"ign": ign, "role": role_txt}
                    if any(x in role_txt.upper() for x in ["COACH", "MANAGER", "ANALYST", "OWNER", "FOUNDER", "DIRECTOR"]):
                        staff.append(entry)
                    else:
                        players.append(entry)

            # --- CASE B: Classic Layout (.teamcard) ---
            else:
                for row in card.find_all("tr"):
                    th, td = row.find("th"), row.find("td")
                    if not th or not td: continue
                    role_img = th.select_one("img")
                    role_txt = role_img.get("title") or role_img.get("alt") if role_img else th.get_text(strip=True)
                    
                    ign = ""
                    # Find the first valid anchor that's not a Category link
                    for a in td.find_all("a"):
                        if "Category" not in a.get("href", ""):
                            ign = extract_canonical_ign(a)
                            if ign: break
                            
                    # If no anchor, fallback to raw td text
                    if not ign:
                        ign = td.get_text(strip=True)
                        ign = re.sub(r'\(.*?\)', '', ign).replace("DNP", "").replace("SUB", "").strip()
                        
                    if not ign or ign == "New Entry" or len(ign) < 2: continue
                    
                    entry = {"ign": ign, "role": role_txt}
                    if any(x in role_txt.upper() for x in ["COACH", "MANAGER", "ANALYST", "OWNER", "FOUNDER", "DIRECTOR"]):
                        staff.append(entry)
                    else:
                        players.append(entry)

            if not players and not staff: continue
            
            # Deduplicate
            seen_ign = set()
            unique_players = [p for p in players if not (p["ign"] in seen_ign or seen_ign.add(p["ign"]))]
            seen_staff = set()
            unique_staff = [s for s in staff if not (s["ign"] in seen_staff or seen_staff.add(s["ign"]))]
            
            existing = db.query(models.SeasonRoster).filter_by(season=season_num, team_name=team_name).first()
            if existing: existing.players, existing.staff = unique_players, unique_staff
            else: db.add(models.SeasonRoster(season=season_num, team_name=team_name, players=unique_players, staff=unique_staff))
            db.commit()
            log.info(f"    ✓ Roster: {team_name} ({len(unique_players)} players, {len(unique_staff)} staff)")
        except Exception as e:
            log.error(f"    Roster error: {e}"); db.rollback()


# ─── Season scraper ───────────────────────────────────────────────────────────

def scrape_season(season_num: int):
    db = SessionLocal()
    log.info(f"\n{'='*55}\n  SEASON {season_num}\n{'='*55}")
    soup_main = get_soup(f"{BASE_URL}/MPL/Philippines/Season_{season_num}")
    default_patch = "Unknown"
    if soup_main:
        patch_txt = extract_infobox_value(soup_main, "Patch")
        if patch_txt and patch_txt.lower() != "unknown":
            default_patch = patch_txt
        scrape_playoffs(soup_main, season_num, default_patch, db)
        scrape_rosters(soup_main, season_num, db)
    soup_rs = get_soup(f"{BASE_URL}/MPL/Philippines/Season_{season_num}/Regular_Season")
    if soup_rs: scrape_regular_season(soup_rs, season_num, default_patch, db)
    db.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        init_db(); db = SessionLocal(); db.close()
        args = sys.argv[1:]
        if not args: seasons = list(range(1, 18))
        elif args[0].lower() in ("all", "1-17"): seasons = list(range(1, 18))
        else: seasons = [int(a) for a in args if a.isdigit()]
        for i, s in enumerate(seasons):
            scrape_season(s)
            if i < len(seasons) - 1: time.sleep(random.uniform(1.0, 2.0))
        log.info("\n✅ All done!")
    except Exception:
        log.error("Fatal error:"); traceback.print_exc()
