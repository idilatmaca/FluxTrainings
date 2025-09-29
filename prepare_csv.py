#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build image-caption CSV")
    parser.add_argument("input_dir", help="Folder containing images (searched recursively)")
    parser.add_argument("-o", "--output", default="images.csv", help="Output CSV path")
    parser.add_argument("-t", "--text", default="sks-idil person", help="Caption text to use for every image")
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}
    root = Path(args.input_dir)

    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            # Always use absolute path
            image_name = str(p.resolve())
            rows.append((image_name, args.text))

    # Sort for determinism
    rows.sort(key=lambda r: r[0].lower())

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
