from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Liquipedia brackets are often structured with columns (.brkts-bracket-column)
# Each column might have a header or belong to a named section.
for column in soup.select(".brkts-bracket-column"):
    header = column.select_one(".brkts-header-group")
    header_text = header.get_text(strip=True) if header else "No Header"
    print(f"\n--- Column Header: {header_text} ---")
    
    for match in column.select(".brkts-match"):
        # Try to find specific labels inside match
        match_id = match.get('data-bracket-id', 'No ID')
        print(f"  Match ID: {match_id} | Text: {match.get_text(strip=True)[:40]}...")

# Another common structure is to have "Upper Bracket" / "Lower Bracket" as H3/H2
for h in soup.find_all(["h2", "h3"]):
    t = h.get_text(strip=True).upper()
    if "UPPER" in t or "LOWER" in t or "PLAYOFFS" in t:
        print(f"\nSection Header: {t}")
        # Look at the next element to see if it's a bracket
        nxt = h.find_next_sibling()
        if nxt and "brkts-main" in nxt.get("class", []):
            print("  -> Followed by a bracket container")
