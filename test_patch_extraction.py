from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/Patch_1.9.42"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Get list of all heroes from DB if possible, or use a sample
heroes_to_find = ["SUN", "ARGUS", "AULUS", "CECILION", "LUKAS", "FANNY"]

for h3 in soup.find_all("h3"):
    name = h3.get_text().strip()
    print(f"\n--- H3: {name} ---")
    
    # Check if this h3 is a hero
    # (In real scraper, we'd check against models.Hero names)
    
    # Find the next h2 or h3 to limit search
    # All hero info is BETWEEN this h3 and the next h3 (or end of section)
    content_parts = []
    curr = h3.next_sibling
    while curr and curr.name not in ["h1", "h2", "h3"]:
        if hasattr(curr, "get_text"):
            content_parts.append(curr.get_text().upper())
        curr = curr.next_sibling
    
    full_content = " ".join(content_parts)
    if "BUFF" in full_content: print("  RESULT: BUFF")
    elif "NERF" in full_content: print("  RESULT: NERF")
    elif "ADJUST" in full_content: print("  RESULT: ADJUST")
    elif "NEW" in full_content: print("  RESULT: NEW")
    elif "REVAMP" in full_content: print("  RESULT: REVAMP")
