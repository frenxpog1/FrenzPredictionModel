from scrapling import DynamicFetcher
from bs4 import BeautifulSoup

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/Portal:Patches"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for table in soup.select("table.wikitable"):
    print("\n--- NEW TABLE ---")
    rows = table.find_all("tr")
    for row in rows[:5]: # Show first 5 rows of each table
        cols = row.find_all(["th", "td"])
        txt = [c.get_text(strip=True) for c in cols]
        print(f"Row: {txt}")
