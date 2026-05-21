from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for match in soup.select(".brkts-match"):
    # Look for any text indicating the bracket
    # Sometimes it's in the popup title or as a small text above the match
    header = match.select_one(".brkts-match-header")
    header_text = header.get_text(strip=True) if header else "No Header"
    
    # Check if there's a popup with more info
    popup = match.select_one(".brkts-popup")
    popup_title = ""
    if popup:
        # In popups, look for "Upper Bracket" or "Lower Bracket"
        popup_text = popup.get_text(strip=True).upper()
        if "UPPER BRACKET" in popup_text: popup_title = "Upper"
        elif "LOWER BRACKET" in popup_text: popup_title = "Lower"
        elif "GRAND FINAL" in popup_text: popup_title = "Grand Final"

    print(f"Match: {match.get_text(strip=True)[:30]}... | Header: {header_text} | Popup Bracket: {popup_title}")
