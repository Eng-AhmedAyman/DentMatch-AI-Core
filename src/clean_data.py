"""
==============================================================================
FILE: clean_data.py
DESCRIPTION:
Data Preprocessing Engine.
Reads raw nested image directories, filters valid formats, and splits them
into Train (70%), Validation (15%), and Test (15%) subsets.

NOTE: Run ONCE before training. Edit BASE_SOURCE_DIR to point to your raw data.

USAGE:
    python src/clean_data.py
    python src/clean_data.py --source /path/to/raw/data

DATASET STRUCTURE EXPECTED (inside BASE_SOURCE_DIR):
    BASE_SOURCE_DIR/
    ├── Dental_Caries/  ...
    ├── Hypodontia/     ...
    ├── Mouth_Ulcer/    ...
    ├── Periodontal_Disease/ ...
    └── Tooth_Discoloration/ ...

OUTPUT:
    data/stage3_disease/
    ├── train/  (70%)
    ├── val/    (15%)
    └── test/   (15%)

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 1.1.0  (fix — removed Calculus/Gingivitis, aligned to 5-class pipeline)
==============================================================================
"""

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# TARGET_DIR: dynamically points to HealthySmile_AI_Core/data/stage3_disease
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "stage3_disease")

# ==============================================================================
# 2. ARGUMENT PARSING
# Allows overriding BASE_SOURCE_DIR from the command line without editing code.
# Example: python src/clean_data.py --source D:\raw_data\stage3
# ==============================================================================
parser = argparse.ArgumentParser(description="Stage 3 Data Preprocessing Engine")
parser.add_argument(
    "--source",
    type=str,
    default=None,
    help="Absolute path to the raw data folder. Overrides BASE_SOURCE_DIR.",
)
args = parser.parse_args()

# BASE_SOURCE_DIR: Edit this to your local raw data path, or pass via --source
BASE_SOURCE_DIR = args.source or os.path.join(PROJECT_ROOT, "data", "raw")

print("=" * 60)
print("🦷  DentMatch AI — Data Preprocessing Engine  v1.1.0")
print("=" * 60)
print(f"   Source  : {BASE_SOURCE_DIR}")
print(f"   Target  : {TARGET_DIR}")
print("=" * 60 + "\n")

# ==============================================================================
# 3. DATA MAPPING — 5 CLASSES (aligned with master_pipeline & train_stage3.py)
# Each class maps to one or more raw source folders.
# Add/remove paths here if your raw folder structure differs.
# ==============================================================================
data_map = {
    "Dental_Caries": [
        os.path.join(
            BASE_SOURCE_DIR,
            "Data caries",
            "Data caries",
            "caries augmented data set",
            "preview",
        ),
        os.path.join(
            BASE_SOURCE_DIR,
            "Data caries",
            "Data caries",
            "caries orignal data set",
            "done",
        ),
    ],
    "Hypodontia": [
        os.path.join(BASE_SOURCE_DIR, "hypodontia", "hypodontia"),
    ],
    "Mouth_Ulcer": [
        os.path.join(
            BASE_SOURCE_DIR,
            "Mouth Ulcer",
            "Mouth Ulcer",
            "Mouth_Ulcer_augmented_DataSet",
            "preview",
        ),
        os.path.join(
            BASE_SOURCE_DIR,
            "Mouth Ulcer",
            "Mouth Ulcer",
            "ulcer original dataset",
            "ulcer original dataset",
        ),
    ],
    "Periodontal_Disease": [
        os.path.join(BASE_SOURCE_DIR, "Periodontal_Disease"),
    ],
    "Tooth_Discoloration": [
        os.path.join(
            BASE_SOURCE_DIR,
            "Tooth Discoloration",
            "Tooth Discoloration",
            "Tooth_discoloration_augmented_dataser",
            "preview",
        ),
        os.path.join(
            BASE_SOURCE_DIR,
            "Tooth Discoloration",
            "Tooth Discoloration",
            "tooth discoloration original dataset",
            "tooth discoloration original dataset",
        ),
    ],
}

SPLITS = ["train", "val", "test"]
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")
RANDOM_STATE = 42

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
print("[INIT] Creating clean directory structure...")
for split in SPLITS:
    for class_name in data_map:
        os.makedirs(os.path.join(TARGET_DIR, split, class_name), exist_ok=True)
print(f"✅ Directories ready at: {TARGET_DIR}\n")

total_copied = 0
errors = []

for class_name, folder_paths in data_map.items():
    all_images = []

    for folder in folder_paths:
        if os.path.exists(folder):
            found = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(VALID_EXTENSIONS)
            ]
            all_images.extend(found)
        else:
            print(f"   ⚠️  Path not found (skipping): {folder}")

    if not all_images:
        errors.append(class_name)
        print(f"   ❌ No images found for {class_name} — check BASE_SOURCE_DIR.\n")
        continue

    print(f"   📂 {class_name}: {len(all_images)} images — splitting 70/15/15...")

    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs = train_test_split(
        all_images, test_size=0.30, random_state=RANDOM_STATE
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.50, random_state=RANDOM_STATE
    )

    split_dict = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    for split_name, imgs in split_dict.items():
        for img_path in imgs:
            # Unique filename: parent_folder_originalname to avoid overwrites
            parent = os.path.basename(os.path.dirname(img_path))
            safe_filename = f"{parent}_{os.path.basename(img_path)}"
            dest = os.path.join(TARGET_DIR, split_name, class_name, safe_filename)
            shutil.copy2(img_path, dest)

        print(f"      {split_name:5s}: {len(imgs)} images")

    total_copied += len(all_images)
    print()

# ==============================================================================
# 5. SUMMARY
# ==============================================================================
print("=" * 60)
if errors:
    print(f"⚠️  Completed with warnings. Missing classes: {errors}")
else:
    print(f"✅ SUCCESS — {total_copied} images processed across 5 classes.")
print(f"   Dataset ready at: {TARGET_DIR}")
print("=" * 60)
