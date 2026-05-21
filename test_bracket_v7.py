from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for bracket in soup.select(".brkts-main"):
    print("\n--- NEW BRACKET ---")
    # A bracket has multiple rounds
    for round_div in bracket.select(".brkts-round"):
        # Look for a header in this round's parent (.brkts-bracket)
        # or within the round itself
        header = round_div.select_one(".brkts-header")
        header_text = header.get_text(strip=True) if header else "No Header"
        print(f"Round Header: {header_text}")
        
        for match in round_div.select(".brkts-match"):
            print(f"  Match: {match.get_text(strip=True)[:40]}...")
