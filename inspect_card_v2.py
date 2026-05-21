from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

cards = soup.select(".teamcard")
if cards:
    card = cards[0]
    print("--- CARD STRUCTURE ---")
    for child in card.descendants:
        if child.name in ["table", "tr", "th", "td", "a", "img"]:
            classes = child.get("class", [])
            text = child.get_text(strip=True)[:30]
            print(f"Tag: {child.name} | Classes: {classes} | Text: {text}")
else:
    print("No teamcards found")
