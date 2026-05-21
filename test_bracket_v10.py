from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_13"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for i, match in enumerate(soup.select(".brkts-match")):
    print(f"\n--- MATCH {i}: {match.get_text(strip=True)[:30]} ---")
    
    # Each match is usually in a column or round body.
    # Let's see if we can find a header specifically ABOVE this match inside its own container.
    
    # Search for siblings that are headers
    prev = match.find_previous_sibling()
    while prev:
        if "brkts-header" in prev.get("class", []):
            print(f"  Sibling Header Found: {prev.get_text(strip=True)}")
            break
        prev = prev.find_previous_sibling()
    
    # Try searching children of ancestors specifically
    curr = match.parent
    found = False
    while curr and curr.name != 'body' and not found:
        # Check all children of this container
        headers = curr.find_all(class_=re.compile("brkts-header"), recursive=False)
        for h in headers:
            print(f"  Container Header Found [{curr.name}, {curr.get('class')}]: {h.get_text(strip=True)}")
            found = True
            break
        curr = curr.parent
