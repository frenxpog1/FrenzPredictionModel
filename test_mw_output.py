from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/Patch_1.9.42"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

main_content = soup.find("div", class_="mw-parser-output")
if main_content:
    print("Found mw-parser-output")
    # All h3s and their info are children of this div
    for child in main_content.children:
        if child.name == "h3":
            print(f"H3: {child.get_text().strip()}")
        elif child.name in ["div", "p", "ul"]:
            txt = child.get_text().strip()
            if txt: print(f"  [{child.name}] {txt[:50]}...")
else:
    print("Main content not found")
