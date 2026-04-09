"""
step1_extract.py
================
Extract and inspect a Roboflow dataset zip before annotation.
Discovers folder structure, counts images and annotations,
and lists all class names found in VOC XML files.

Usage:
    python step1_extract.py                         # auto-finds zip in current dir
    python step1_extract.py --zip my_dataset.zip
    python step1_extract.py --zip my_dataset.zip --out /content/dataset
"""

import os
import glob
import zipfile
import shutil
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def find_zip_in_cwd():
    """Auto-detect a single .zip file in the current directory."""
    zips = [f for f in os.listdir('.') if f.endswith('.zip')]
    if not zips:
        return None
    if len(zips) == 1:
        return zips[0]
    print(f"Multiple zips found: {zips}")
    return zips[0]


def inspect_zip(zip_path):
    """Print a summary of what is inside the zip without extracting."""
    print("=" * 60)
    print(f"Inspecting: {zip_path}")
    print("=" * 60)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = zf.namelist()

    images = [f for f in all_files
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    xmls   = [f for f in all_files if f.endswith('.xml')]
    txts   = [f for f in all_files if f.endswith('.txt')]
    jsons  = [f for f in all_files if f.endswith('.json')]
    other  = [f for f in all_files
              if not any(f.endswith(e) for e in
                         ('.jpg', '.jpeg', '.png', '.xml', '.txt', '.json'))]

    print(f"\nTotal files     : {len(all_files)}")
    print(f"Images          : {len(images)}")
    print(f"XML annotations : {len(xmls)}")
    print(f"TXT files       : {len(txts)}")
    print(f"JSON files      : {len(jsons)}")
    print(f"Other           : {len(other)}")

    # Top-level folder structure
    top_folders = set()
    for f in all_files:
        parts = f.replace('\\', '/').split('/')
        if len(parts) > 1:
            top_folders.add(parts[0])
    print(f"\nTop-level folders: {sorted(top_folders) or ['(flat — no subfolders)']}")

    # Detect splits
    splits = [s for s in ['train', 'valid', 'val', 'test']
              if any(s in f.lower() for f in all_files)]
    print(f"Splits detected  : {splits or ['none']}")

    # Read class names from XML (sample up to 300)
    print("\nReading class names from XML annotations (sampling) ...")
    class_counter = Counter()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        sample = [f for f in zf.namelist() if f.endswith('.xml')][:300]
        for xml_name in sample:
            try:
                with zf.open(xml_name) as f:
                    tree = ET.parse(f)
                    for obj in tree.findall('.//object'):
                        name = obj.find('name')
                        if name is not None:
                            class_counter[name.text.strip()] += 1
            except Exception:
                pass

    print(f"\nClasses found ({len(class_counter)} unique):")
    for cls, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {cls:<40s}  {count:>6d} instances")


def extract_and_organize(zip_path, output_dir):
    """Extract zip and organize into images/ and annotations/ folders."""
    print(f"\nExtracting to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    print("Extraction complete.")

    # Count what we got
    imgs  = glob.glob(output_dir + '/**/*.jpg',  recursive=True) + \
            glob.glob(output_dir + '/**/*.png',  recursive=True)
    xmls  = glob.glob(output_dir + '/**/*.xml',  recursive=True)
    jsons = glob.glob(output_dir + '/**/*.json', recursive=True)

    print(f"\nExtracted:")
    print(f"  Images : {len(imgs)}")
    print(f"  XMLs   : {len(xmls)}")
    print(f"  JSONs  : {len(jsons)}")

    if imgs:
        print(f"\nExample image: {imgs[0]}")
    if xmls:
        print(f"Example XML:   {xmls[0]}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if xmls:
        print("XMLs found — dataset has existing annotations.")
        print("Run: python step2_split.py --root", output_dir)
    elif jsons:
        print("JSON found — COCO format dataset.")
        print("Run: python coco_to_voc.py --json <json_path>",
              "--images", output_dir, "--output", output_dir + "_voc")
    else:
        print("No annotations found.")
        print("Run auto_annotate.py to generate boxes with Grounding DINO:")
        print(f"  python auto_annotate.py --images {output_dir}/images",
              f"--output {output_dir}/annotations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default=None,
                        help='Path to dataset zip (auto-detected if omitted)')
    parser.add_argument('--out', default='dataset',
                        help='Output directory for extraction')
    parser.add_argument('--inspect-only', action='store_true',
                        help='Only inspect zip contents without extracting')
    args = parser.parse_args()

    zip_path = args.zip or find_zip_in_cwd()
    if zip_path is None:
        print("ERROR: No .zip file found. Use --zip to specify the path.")
        exit(1)
    if not os.path.exists(zip_path):
        print(f"ERROR: File not found: {zip_path}")
        exit(1)

    inspect_zip(zip_path)

    if not args.inspect_only:
        extract_and_organize(zip_path, args.out)
