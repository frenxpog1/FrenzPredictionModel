from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_13"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for i, match in enumerate(soup.select(".brkts-match")):
    print(f"\n--- MATCH {i} ---")
    p = match.parent
    while p and p.name != 'body':
        # Check all direct children of every ancestor for a header
        for child in p.find_all(class_="brkts-header", recursive=False):
            print(f"  FOUND HEADER in [{p.name}, {p.get('class')}]: {child.get_text(strip=True)}")
        p = p.parent
