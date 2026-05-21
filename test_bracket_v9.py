from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_13"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for i, match in enumerate(soup.select(".brkts-match")):
    print(f"\n--- MATCH {i}: {match.get_text(strip=True)[:40]} ---")
    # Print all ancestor headers
    p = match.parent
    while p and p.name != 'body':
        headers = p.select(".brkts-header")
        if headers:
            for h in headers:
                print(f"  HEADER in ancestor [{p.name}, {p.get('class')}]: {h.get_text(strip=True)}")
        p = p.parent
