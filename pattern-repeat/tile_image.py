#!/usr/bin/env python3
"""Tile an image horizontally n times."""

import sys
from pathlib import Path
from PIL import Image


def tile_image(image_path: str, n: int) -> str:
    src = Path(image_path)
    img = Image.open(src)
    w, h = img.size

    tiled = Image.new(img.mode, (w * n, h))
    for i in range(n):
        tiled.paste(img, (w * i, 0))

    out_path = src.with_name(f"{src.stem}_x{n}{src.suffix}")
    tiled.save(out_path)
    print(f"Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tile_image.py <image_path> <n>")
        sys.exit(1)

    tile_image(sys.argv[1], int(sys.argv[2]))
