from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

team_cards = soup.select(".teamcard, .participant-card")
print(f"Cards: {len(team_cards)}")

for card in team_cards[:1]:
    name_el = card.select_one("center a, b a, .participant-card-header a")
    if not name_el: name_el = card.find("a")
    print(f"Team: {name_el.get_text(strip=True) if name_el else 'None'}")
    
    rows = card.find_all("tr")
    print(f"Rows: {len(rows)}")
    for row in rows:
        th = row.find("th")
        td = row.find("td")
        if th and td:
            img = th.select_one("img")
            role = img.get("title") or img.get("alt") if img else th.get_text(strip=True)
            ign = td.select_one("a").get_text(strip=True) if td.select_one("a") else td.get_text(strip=True)
            print(f"  - {role}: {ign}")
