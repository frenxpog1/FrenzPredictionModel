from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Liquipedia brackets are built using a system where headers are siblings to the bracket container
# or within a specific layout.
print("--- Searching for Headers using CSS Selectors ---")
headers = soup.select(".brkts-header-group, .brkts-header, .brkts-round-title")
for h in headers:
    print(f"Selector Found: {h.get_text(strip=True)} | Class: {h.get('class')}")

# Let's find matches and see their containers
matches = soup.select(".brkts-match")
if matches:
    print("\n--- Examining Match Parents ---")
    parent = matches[0].parent
    while parent and parent.name != 'body':
        print(f"Parent Tag: {parent.name} | Class: {parent.get('class')}")
        # Look for headers within this parent
        h = parent.select_one(".brkts-header-group, .brkts-header")
        if h:
            print(f"  -> Found Header in Parent: {h.get_text(strip=True)}")
        parent = parent.parent
        if parent.name == 'div' and 'brkts-main' in parent.get('class', []):
            break
