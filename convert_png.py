import os
from PIL import Image
import pillow_heif

# Register HEIF/HEIC format
pillow_heif.register_heif_opener()

input_dir = "/home/syntonym4090/dreambooth/idil dataset-20250923T122934Z-1-001/idil dataset"
output_dir = "/home/syntonym4090/dreambooth/idil-dataset"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    fpath = os.path.join(input_dir, fname)
    try:
        img = Image.open(fpath)
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_dir, base + ".png")
        img.save(out_path, "PNG")
        print(f"Converted: {fname} -> {out_path}")
    except Exception as e:
        print(f"Skipping {fname}: {e}")
