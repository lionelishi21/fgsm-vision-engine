import argparse
import json
import logging
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_URL = "https://ultimateframedata.com/sf6/"

def scrape_character(character: str) -> dict:
    url = f"{BASE_URL}{character.lower()}"
    logging.info(f"Fetching {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Failed to fetch {character} (status code {response.status_code})")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    move_containers = soup.find_all("div", class_="movecontainer")
    
    char_data = {}
    for container in move_containers:
        name_div = container.find("div", class_="movename")
        if not name_div:
            continue
        move_name = name_div.text.strip().lower().replace(" ", "_")
        
        startup_div = container.find("div", class_="startup")
        active_div = container.find("div", class_="activeframes")
        recovery_div = container.find("div", class_="recovery")
        total_div = container.find("div", class_="totalframes")

        def parse_int(text: str):
            if not text or text == "--" or text == "~":
                return None
            # Extract first number found
            match = re.search(r'\d+', text)
            if match:
                return int(match.group(0))
            return None
        
        def parse_str(text: str):
            if not text or text == "--" or text == "~":
                return None
            return text.strip()

        startup = parse_int(startup_div.text) if startup_div else None
        active = parse_str(active_div.text) if active_div else None
        recovery = parse_int(recovery_div.text) if recovery_div else None
        total = parse_int(total_div.text) if total_div else None
        
        key = f"{character.lower()}:{move_name}"
        char_data[key] = {
            "startup": startup,
            "active": active,
            "recovery": recovery,
            "total": total or 0
        }
    
    return char_data

def main():
    parser = argparse.ArgumentParser(description="Scrape UFD for SF6 frame data")
    parser.add_argument("--characters", nargs="+", default=["ryu", "ken", "luke"], help="Characters to scrape")
    parser.add_argument("--out", default="data/ufd/frame_data_oracle.json", help="Output JSON path")
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if any
    oracle_data = {}
    if out_path.exists():
        with open(out_path, "r") as f:
            try:
                oracle_data = json.load(f)
            except json.JSONDecodeError:
                pass
    
    for char in args.characters:
        char_data = scrape_character(char)
        oracle_data.update(char_data)
        logging.info(f"Scraped {len(char_data)} moves for {char}")
        
    with open(out_path, "w") as f:
        json.dump(oracle_data, f, indent=2)
    logging.info(f"Saved {len(oracle_data)} total moves to {out_path}")

if __name__ == "__main__":
    main()
