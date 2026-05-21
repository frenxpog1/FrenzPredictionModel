from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_13"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Look at the layout of round-body and round-center
for i, match in enumerate(soup.select(".brkts-match")):
    print(f"\n--- MATCH {i} ---")
    p = match.parent
    found = False
    while p and p.name != 'body':
        # Brackets usually use brkts-round-body
        # Let's see if we can find a header relative to the round
        header = p.find_previous_sibling(class_="brkts-header")
        if header:
            print(f"FOUND SIBLING HEADER: {header.get_text(strip=True)}")
            found = True
            break
        # Sometimes it's in a parent container of the round
        if "brkts-round" in p.get("class", []):
            round_parent = p.parent
            if round_parent:
                header = round_parent.select_one(".brkts-header")
                if header:
                    print(f"FOUND ROUND PARENT HEADER: {header.get_text(strip=True)}")
                    found = True
                    break
        p = p.parent
