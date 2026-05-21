from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

cards = soup.select(".teamcard")
if cards:
    card = cards[0]
    print(f"--- TEAM NAME ELEMENT ---")
    name_el = card.select_one("center a, .teamcard-inner center a, .participant-card-header a")
    print(f"Found name element: {name_el}")
    if name_el:
        print(f"Team Name text: {name_el.get_text(strip=True)}")
    
    print("\n--- TABLE ROWS ---")
    rows = card.select("table tr")
    print(f"Total rows in table: {len(rows)}")
    for i, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        txts = [c.get_text(strip=True) for c in cols]
        print(f"Row {i}: {txts}")
else:
    print("No cards found")
