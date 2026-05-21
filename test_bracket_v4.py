from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Look for bracket containers
for main in soup.select(".brkts-main"):
    # Brackets often have multiple sections (Upper, Lower)
    # Let's try to find headers that are siblings or nearby
    print("\n--- NEW BRACKET CONTAINER ---")
    
    # Try searching for headers above this bracket
    prev = main.find_previous(["h2", "h3"])
    if prev:
        print(f"Nearest Preceding Header: {prev.get_text(strip=True)}")
    
    # Check if matches inside have any indicators
    for match in main.select(".brkts-match")[:2]:
        # Liquipedia sometimes encodes bracket info in data attributes of the column
        col = match.find_parent(class_=re.compile("column"))
        if col:
            col_class = col.get("class", [])
            print(f"  Match in column with classes: {col_class}")
        
        # Check the popup title again, very carefully
        popup = match.select_one(".brkts-popup")
        if popup:
            # Print ALL text from the top of the popup
            header = popup.select_one(".match-info-header")
            if header:
                # Sometimes there's a "Upper Bracket Round 1" type text above the teams
                # Let's check siblings of match-info-header
                sibling = header.find_previous_sibling()
                if sibling:
                    print(f"  Popup Sibling Text: {sibling.get_text(strip=True)}")
