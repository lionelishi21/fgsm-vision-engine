# src/data/ufd_scraper.py
"""
Scrape and parse Ultimate Frame Data for SF6.
Extracts GIFs, frame data tables, and generates segmentation masks.
"""
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
from PIL import Image
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

MIN_CONTOUR_AREA = 20  # drop speck-sized contours from anti-aliased mask edges


@dataclass
class FrameData:
    move_name: str
    startup: Optional[int]
    active: Optional[str]  # e.g., "3" or "2...5" for meaty
    recovery: Optional[int]
    on_hit: Optional[str]
    on_block: Optional[str]
    damage: Optional[int]
    stun: Optional[int]
    total_frames: Optional[int]


@dataclass
class UFDMove:
    character: str
    move_name: str
    frame_data: FrameData
    gif_url: str  # source GIF is deleted after frame extraction to save disk
    frames: List[np.ndarray]  # RGB frames extracted from GIF
    masks: List[np.ndarray]   # Segmentation masks per frame


class UFDColorMap:
    """
    UFD hitbox color legend:
    Red = Hitbox
    Green = Hurtbox
    Teal = Projectile Invincible
    Dark Green = Air Strike Invincible
    Pinkish White = Strike Invincible
    Purple = Armor
    Transparent Box w/ White Outline = Mystery Box
    """
    HITBOX = 1
    HURTBOX = 2
    PROJ_INV = 3
    AIR_STRIKE_INV = 4
    STRIKE_INV = 5
    ARMOR = 6
    MYSTERY = 7

    # Approximate RGB values (may need tuning per GIF)
    COLOR_RANGES = {
        HITBOX: [(200, 255), (0, 50), (0, 50)],        # Red
        HURTBOX: [(0, 100), (150, 255), (0, 100)],      # Green
        PROJ_INV: [(0, 100), (150, 255), (150, 255)],   # Teal/Cyan
        AIR_STRIKE_INV: [(0, 80), (100, 200), (0, 80)], # Dark Green
        STRIKE_INV: [(200, 255), (200, 255), (200, 255)], # Pinkish White
        ARMOR: [(150, 200), (0, 100), (150, 200)],      # Purple
    }


class UFDPipeline:
    BASE_URL = "https://ultimateframedata.com/sf6/"
    OUTPUT_DIR = Path("data/ufd")

    def __init__(self):
        self.output_dir = self.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        })

    def scrape_character(self, character_slug: str) -> List[UFDMove]:
        """
        Scrape all moves for a character.
        character_slug: e.g., 'ryu', 'ken', 'chunli'
        """
        url = f"{self.BASE_URL}{character_slug}"
        char_dir = self.output_dir / character_slug
        char_dir.mkdir(exist_ok=True)

        print(f"Scraping {character_slug}...")
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        moves = []
        
        # UFD structure: each move is in a div with class like 'movecontainer'
        move_containers = soup.find_all("div", class_=re.compile(r"move(container|row)"))
        
        for container in move_containers:
            move_name = self._extract_move_name(container)
            frame_data = self._extract_frame_data(container)
            gif_url = self._extract_gif_url(container)
            
            if not gif_url:
                continue

            # Download GIF
            gif_path = char_dir / f"{self._sanitize(move_name)}.gif"
            try:
                self._download_gif(gif_url, gif_path)
            except requests.exceptions.HTTPError as e:
                print(f"  ! Skipping '{move_name}': {e}")
                continue

            # Parse GIF into frames and masks, then drop the raw file - a
            # single animated GIF can be 5-20MB+ and we only need the
            # extracted frames/masks, not the original asset.
            frames, masks = self._parse_gif(gif_path)
            gif_path.unlink(missing_ok=True)

            moves.append(UFDMove(
                character=character_slug,
                move_name=move_name,
                frame_data=frame_data,
                gif_url=gif_url,
                frames=frames,
                masks=masks,
            ))

        # Save metadata
        meta_path = char_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump([self._move_to_dict(m) for m in moves], f, indent=2)

        print(f"  → Extracted {len(moves)} moves")
        return moves

    def _extract_move_name(self, container) -> str:
        # UFD renders each stat as its own <div class="...">, not a table.
        name_tag = container.find("div", class_="movename")
        return name_tag.get_text(strip=True) if name_tag else "unknown"

    def _extract_frame_data(self, container) -> FrameData:
        def field(cls_name: str) -> Optional[str]:
            tag = container.find("div", class_=cls_name)
            if not tag:
                return None
            text = tag.get_text(strip=True)
            return text if text and text not in ("-", "--", "~") else None

        def parse_int(text: Optional[str]) -> Optional[int]:
            if not text:
                return None
            match = re.search(r"\d+", text)
            return int(match.group(0)) if match else None

        return FrameData(
            move_name=self._extract_move_name(container),
            startup=parse_int(field("startup")),
            active=field("activeframes"),
            recovery=parse_int(field("recovery")),
            on_hit=field("onhit"),
            on_block=field("onblock"),
            damage=parse_int(field("basedamage")),
            stun=None,  # not published on the SF6 UFD page
            total_frames=parse_int(field("totalframes")),
        )

    def _extract_gif_url(self, container) -> Optional[str]:
        img = container.find("img", src=re.compile(r"\.gif$"))
        if img:
            src = img.get("src")
            from urllib.parse import urljoin
            return urljoin(self.BASE_URL, src)
        return None

    def _download_gif(self, url: str, path: Path):
        if path.exists():
            return
        print(f"Downloading {url} to {path}")
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        path.write_bytes(resp.content)

    def _parse_gif(self, gif_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gif = Image.open(gif_path)
        frames = []
        masks = []

        for frame_idx in range(gif.n_frames):
            gif.seek(frame_idx)
            frame = np.array(gif.convert("RGBA"))
            rgb = frame[:, :, :3]
            alpha = frame[:, :, 3]

            mask = self._color_to_mask(rgb, alpha)
            
            frames.append(rgb)
            masks.append(mask)

        return frames, masks

    def _color_to_mask(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        for class_id, (r_range, g_range, b_range) in UFDColorMap.COLOR_RANGES.items():
            r_min, r_max = r_range
            g_min, g_max = g_range
            b_min, b_max = b_range

            matches = (
                (rgb[:, :, 0] >= r_min) & (rgb[:, :, 0] <= r_max) &
                (rgb[:, :, 1] >= g_min) & (rgb[:, :, 1] <= g_max) &
                (rgb[:, :, 2] >= b_min) & (rgb[:, :, 2] <= b_max) &
                (alpha > 128)
            )
            mask[matches] = class_id

        return mask

    def _sanitize(self, name: str) -> str:
        return re.sub(r'[^\w\-_]', '_', name.lower())

    def _move_to_dict(self, move: UFDMove) -> dict:
        return {
            "character": move.character,
            "move_name": move.move_name,
            "frame_data": {
                "startup": move.frame_data.startup,
                "active": move.frame_data.active,
                "recovery": move.frame_data.recovery,
                "on_hit": move.frame_data.on_hit,
                "on_block": move.frame_data.on_block,
                "damage": move.frame_data.damage,
                "stun": move.frame_data.stun,
            },
            "num_frames": len(move.frames),
            "gif_source_url": move.gif_url,
        }

    def generate_training_datasets(self, characters: List[str]):
        seg_dir = self.output_dir / "segmentation_dataset"
        move_dir = self.output_dir / "move_classification_dataset"
        seg_dir.mkdir(exist_ok=True)
        move_dir.mkdir(exist_ok=True)

        all_moves = []
        for char in characters:
            moves = self.scrape_character(char)
            all_moves.extend(moves)

        self._build_segmentation_dataset(all_moves, seg_dir)
        self._build_move_dataset(all_moves, move_dir)
        self._build_frame_data_oracle(all_moves)

    def _build_segmentation_dataset(self, moves: List[UFDMove], out_dir: Path):
        import json
        
        images = []
        annotations = []
        categories = [
            {"id": 1, "name": "hitbox"},
            {"id": 2, "name": "hurtbox"},
            {"id": 3, "name": "projectile_invincible"},
            {"id": 4, "name": "air_strike_invincible"},
            {"id": 5, "name": "strike_invincible"},
            {"id": 6, "name": "armor"},
            {"id": 7, "name": "mystery_box"},
        ]

        ann_id = 0
        img_dir = out_dir / "images"
        mask_dir = out_dir / "masks"
        img_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)

        for move in moves:
            for i, (frame, mask) in enumerate(zip(move.frames, move.masks)):
                img_id = len(images)
                img_name = f"{move.character}_{self._sanitize(move.move_name)}_f{i:03d}.png"
                mask_name = f"{move.character}_{self._sanitize(move.move_name)}_f{i:03d}_mask.png"

                Image.fromarray(frame).save(img_dir / img_name)
                Image.fromarray(mask).save(mask_dir / mask_name)

                images.append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                })

                height, width = mask.shape[:2]
                for class_id in np.unique(mask):
                    if class_id == 0:
                        continue

                    binary = (mask == class_id).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        if len(contour) < 3 or cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                            continue

                        polygon = contour.reshape(-1).astype(float).tolist()
                        rle = mask_utils.frPyObjects([polygon], height, width)[0]
                        area = float(mask_utils.area(rle))
                        bbox = mask_utils.toBbox(rle).tolist()

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(class_id),
                            "segmentation": [polygon],
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        })
                        ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(out_dir / "annotations.json", "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Segmentation dataset: {len(images)} images, {len(annotations)} annotations")

    def _build_move_dataset(self, moves: List[UFDMove], out_dir: Path):
        for move in moves:
            move_class_dir = out_dir / move.character / self._sanitize(move.move_name)
            move_class_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(move.frames):
                Image.fromarray(frame).save(move_class_dir / f"frame_{i:03d}.png")

        print(f"Move classification dataset created at {out_dir}")

    def _build_frame_data_oracle(self, moves: List[UFDMove]):
        oracle = {}
        for move in moves:
            key = f"{move.character}:{self._sanitize(move.move_name)}"
            oracle[key] = {
                "startup": move.frame_data.startup,
                "active": move.frame_data.active,
                "recovery": move.frame_data.recovery,
                "total": (move.frame_data.startup or 0) + 
                         (self._parse_active_frames(move.frame_data.active)) + 
                         (move.frame_data.recovery or 0),
            }

        oracle_path = self.output_dir / "frame_data_oracle.json"
        with open(oracle_path, "w") as f:
            json.dump(oracle, f, indent=2)

        print(f"Frame data oracle: {len(oracle)} moves")

    def _parse_active_frames(self, active: Optional[str]) -> int:
        if not active:
            return 0
        clean = active.split("...")[0].split("(")[0].split("[")[0].strip()
        try:
            return int(clean)
        except ValueError:
            return 0
