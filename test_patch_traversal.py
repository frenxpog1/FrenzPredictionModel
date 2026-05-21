from scrapling import DynamicFetcher
from bs4 import BeautifulSoup
import re

fetcher = DynamicFetcher()
url = "https://liquipedia.net/mobilelegends/Patch_1.9.42"
r = fetcher.fetch(url)
soup = BeautifulSoup(r.html_content, "html.parser")

for h3 in soup.find_all("h3"):
    print(f"\n--- H3: {h3.get_text()} ---")
    # Check if a BUFF/NERF label is nearby
    # Liquipedia structure: <h3>Hero Name</h3> <div/p>... <span class="patch-label">BUFF</span>
    next_node = h3.next_sibling
    # Look through next few siblings to find adjustment labels
    count = 0
    while next_node and count < 10:
        if hasattr(next_node, "get_text"):
            text = next_node.get_text().upper()
            if "BUFF" in text or "NERF" in text or "ADJUST" in text:
                print(f"FOUND LABEL in sibling: {text[:50]}...")
                # break
        next_node = next_node.next_sibling
        count += 1
