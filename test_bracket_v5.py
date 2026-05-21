from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Liquipedia brackets are often built using complex CSS-based layouts.
# Let's check for any <div> that has a 'title' or 'bracket' related class.
for div in soup.find_all("div", class_=re.compile("brkts")):
    classes = div.get("class", [])
    if any(c in ["brkts-header-group", "brkts-round-title", "brkts-bracket-title"] for c in classes):
        print(f"Found Bracket Label: {div.get_text(strip=True)} | Class: {classes}")

# Let's search for "Lower Bracket" or "Upper Bracket" in the ENTIRE page text
# and see where it appears relative to matches.
for text in soup.find_all(string=re.compile("Bracket|Final")):
    parent = text.parent
    print(f"Text: {text.strip()} | Parent Tag: {parent.name} | Classes: {parent.get('class', [])}")
