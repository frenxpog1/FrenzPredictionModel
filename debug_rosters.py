from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

# Look for team cards or participant lists
cards = soup.select(".teamcard")
print(f"Cards found with .teamcard: {len(cards)}")

if not cards:
    # Try another common class for participants
    cards = soup.select(".participant-card")
    print(f"Cards found with .participant-card: {len(cards)}")

if cards:
    print("\n--- FIRST CARD HTML ---")
    print(cards[0].prettify()[:2000])
else:
    # Print a snippet of the page to find the container
    print("\n--- PAGE SNIPPET ---")
    print(soup.get_text()[:1000])
