from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Look for ALL brkts-header-group elements anywhere in the page
headers = soup.select(".brkts-header-group")
print(f"Total headers found: {len(headers)}")
for h in headers:
    print(f"Header: {h.get_text(strip=True)}")

# Look for match info that might be stored as comments or hidden data
matches = soup.select(".brkts-match")
print(f"\nTotal matches found: {len(matches)}")
if matches:
    print("\n--- FIRST MATCH FULL HTML ---")
    print(matches[0].prettify())
