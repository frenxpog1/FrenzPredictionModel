from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for i, match in enumerate(soup.select(".brkts-match")):
    # Try to find a header above the match by going up to round and then searching
    print(f"\n--- MATCH {i} ---")
    
    # Try traversing ancestors
    p = match.parent
    found = False
    while p and p.name != 'body':
        # Check all headers in this container
        header = p.select_one(".brkts-header")
        if header:
            print(f"FOUND HEADER in ancestor [{p.name}, {p.get('class')}]: {header.get_text(strip=True)}")
            found = True
            break
        p = p.parent
    
    if not found:
        print("No header found in ancestors.")
