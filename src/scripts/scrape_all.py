import requests
from bs4 import BeautifulSoup
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

response = requests.get("https://ultimateframedata.com/sf6/")
soup = BeautifulSoup(response.text, "html.parser")
chars = []
for a in soup.find_all("a"):
    href = a.get("href")
    if href and not href.startswith("http") and not href.startswith("#") and href not in ["/", "/sf6", "/sf6/"]:
        # usually just the character name
        name = href.strip("/")
        if name and name not in chars and "hitbox" not in name:
            chars.append(name)

logging.info(f"Found characters: {chars}")
subprocess.run(["python", "src/scripts/scrape_ufd.py", "--characters"] + chars)
