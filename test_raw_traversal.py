from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/Patch_1.9.42"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

print("--- RAW TEXT BETWEEN H3s ---")
for h3 in soup.find_all("h3"):
    name = h3.get_text().strip()
    print(f"\nH3: {name}")
    
    curr = h3.next_sibling
    while curr and curr.name not in ["h1", "h2", "h3"]:
        if hasattr(curr, "get_text"):
            t = curr.get_text().strip()
            if t: print(f"  [{curr.name}] {t[:100]}")
        curr = curr.next_sibling
