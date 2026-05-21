from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14/Regular_Season"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")
popups = soup.select(".brkts-popup")
if popups:
    # Look for the first row with picks/bans
    rows = popups[0].select(".brkts-popup-body-grid-row")
    for row in rows:
        print("--- ROW ---")
        print(row.prettify())
else:
    print("No popups found")
