from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
# Testing a patch that definitely has hero adjustments
url = "https://liquipedia.net/mobilelegends/Patch_1.9.42"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

print("--- ANALYZING HEADERS ---")
for header in soup.find_all(["h3", "h4"]):
    text = header.get_text().strip()
    print(f"Header: {text}")

print("\n--- ANALYZING SPANS ---")
for span in soup.find_all("span"):
    text = span.get_text().strip().upper()
    if text in ["BUFF", "NERF", "ADJUSTED", "REVAMPED", "NEW"]:
        print(f"Label: {text} | Parent: {span.parent.get_text().strip()}")
