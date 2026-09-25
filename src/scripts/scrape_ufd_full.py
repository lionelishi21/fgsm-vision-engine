"""Scrape full SF6 move tables (with sections, on-block/on-hit, damage) from
ultimateframedata.com for the backend character encyclopedia.

Separate from scrape_ufd.py, which only keeps the fields the vision models need.
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_URL = "https://ultimateframedata.com/sf6/"

FIELDS = {
    "startup": "startup",
    "active": "activeframes",
    "recovery": "recovery",
    "total": "totalframes",
    "on_hit": "onhit",
    "on_block": "onblock",
    "damage": "basedamage",
    "attack_type": "attacktype",
    "cancellable": "cancellable",
    "notes": "notes",
}


def first_int(text):
    m = re.search(r"[+-]?\d+", text or "")
    return int(m.group(0)) if m else None


def scrape_character(slug: str) -> list:
    response = requests.get(f"{BASE_URL}{slug}", timeout=30)
    if response.status_code != 200:
        logging.error(f"{slug}: HTTP {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    moves, section = [], None
    for el in soup.find_all(["h2", "div"]):
        if el.name == "h2":
            section = el.get_text(strip=True)
            continue
        if "movecontainer" not in (el.get("class") or []):
            continue
        name_div = el.find("div", class_="movename")
        if not name_div:
            continue
        move = {"name": name_div.get_text(strip=True), "section": section}
        for key, css in FIELDS.items():
            div = el.find("div", class_=css)
            move[key] = div.get_text(" ", strip=True) if div else None
        moves.append(move)
    return moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--characters", nargs="+", required=True)
    parser.add_argument("--out", default="data/ufd/sf6_full_frame_data.json")
    args = parser.parse_args()

    out = {}
    for slug in args.characters:
        out[slug] = scrape_character(slug)
        logging.info(f"{slug}: {len(out[slug])} moves")
        time.sleep(0.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    logging.info(f"Saved {sum(len(v) for v in out.values())} moves to {args.out}")


if __name__ == "__main__":
    main()
