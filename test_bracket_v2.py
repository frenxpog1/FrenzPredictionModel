from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for column in soup.select(".brkts-bracket-column"):
    header = column.select_one(".brkts-header-group")
    if header:
        print(f"\n--- COLUMN HEADER: {header.get_text(strip=True)} ---")
        for match in column.select(".brkts-match"):
            print(f"  Match: {match.get_text(strip=True)[:50]}...")
