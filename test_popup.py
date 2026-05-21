from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url_rs = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14/Regular_Season"
r_rs = fetcher.fetch(url_rs)
soup_rs = BeautifulSoup(r_rs.html_content, "html.parser")
ml_matches = soup_rs.select(".brkts-matchlist-match")
if ml_matches:
    print("\n--- Matchlist Match ---")
    print(ml_matches[0].prettify()[:5000])
