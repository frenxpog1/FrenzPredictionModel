from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

cards = soup.select(".teamcard")
if cards:
    card = cards[0]
    print(card.prettify()[:5000])
else:
    print("No teamcards found")
