from database import SessionLocal
import models
from scraper import get_soup, scrape_rosters

db = SessionLocal()
url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_14"
soup = get_soup(url)
if soup:
    scrape_rosters(soup, 14, db)
db.close()
