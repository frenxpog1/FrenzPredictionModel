from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for i, match in enumerate(soup.select(".brkts-match")):
    popup = match.select_one(".brkts-popup")
    if popup:
        # Check all headers and divs in the popup for bracket info
        print(f"\n--- MATCH {i} ---")
        for tag in popup.find_all(["div", "span", "h3"]):
            text = tag.get_text(strip=True).upper()
            if any(k in text for k in ["UPPER", "LOWER", "GRAND", "FINAL", "ROUND"]):
                print(f"Tag [{tag.name}]: {text}")
