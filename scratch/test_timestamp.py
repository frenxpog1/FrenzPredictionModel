import requests
from bs4 import BeautifulSoup
import re

url = "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_17"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

print("Fetching Season 17 page...")
r = requests.get(url, headers=headers)
if r.status_code == 200:
    soup = BeautifulSoup(r.text, "html.parser")
    print("Page fetched successfully.")
    
    # Let's find bracket matches
    bracket_matches = soup.select(".brkts-match")
    print(f"Found {len(bracket_matches)} bracket matches.")
    
    # Inspect a few bracket matches
    for i, m_box in enumerate(bracket_matches[:3]):
        print(f"\n--- Match {i+1} ---")
        timer = m_box.select_one(".timer-object")
        timer_date = m_box.select_one(".timer-object-date")
        
        if timer:
            print("  .timer-object found:")
            print("    Text:", timer.get_text(strip=True))
            print("    Attributes:", timer.attrs)
        else:
            print("  .timer-object not found.")
            
        if timer_date:
            print("  .timer-object-date found:")
            print("    Text:", timer_date.get_text(strip=True))
            print("    Attributes:", timer_date.attrs)
        else:
            print("  .timer-object-date not found.")
else:
    print(f"Failed to fetch: {r.status_code}")
