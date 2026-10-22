"""Scrape move GIFs for the whole SF6 roster on ultimateframedata.com into
data/ufd/ (per-character metadata.json + move_classification_dataset/), for
training the temporal move classifier. Leaves segmentation_dataset/ alone.

Unlike scrape_ufd.py / scrape_ufd_full.py (frame-data tables only), this
downloads every move GIF and keeps its frames, so it needs several GB of disk
for the full roster - run it on the training box, not a nearly-full laptop.

Usage (from the repo root):
    python src/scripts/scrape_ufd_gifs.py              # every character UFD lists
    python src/scripts/scrape_ufd_gifs.py ryu ken      # just these
"""
import argparse
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ufd_scraper import UFDPipeline  # noqa: E402


def list_roster() -> list:
    resp = requests.get(UFDPipeline.BASE_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    roster = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip("/")
        if href.startswith("sf6/"):
            slug = href.split("/", 1)[1]
            if slug and slug != "stats" and slug not in roster:
                roster.append(slug)
    return roster


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("characters", nargs="*", help="UFD slugs (default: whole roster)")
    args = parser.parse_args()

    characters = args.characters or list_roster()
    print(f"Scraping {len(characters)} characters: {', '.join(characters)}")
    UFDPipeline().generate_move_dataset(characters)


if __name__ == "__main__":
    main()
