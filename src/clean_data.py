"""
==============================================================================
FILE: clean_data.py
DESCRIPTION:
This script acts as the Data Preprocessing Engine. It reads raw, nested image
directories, filters valid image formats, and systematically splits them into
Train (70%), Validation (15%), and Test (15%) subsets.

NOTE: This script should be executed ONLY ONCE before training the models.
==============================================================================
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
# This automatically finds the 'src' folder and steps back to the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# BASE_SOURCE_DIR: Keep this as your absolute path where the messy RAW downloaded folders are.
# You only need to change this if your raw downloads move to another flash drive or folder.
BASE_SOURCE_DIR = (
    r"C:\Users\ms\OneDrive\Documents\Healthy_Smile_System\data\stage3_disease"
)

# TARGET_DIR: This will dynamically point to HealthySmile_AI_Core/data/stage3_disease
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "stage3_disease")

# ==============================================================================
# 2. DATA MAPPING (RAW FOLDERS)
# ==============================================================================
data_map = {
    "Calculus": [os.path.join(BASE_SOURCE_DIR, "Calculus", "Calculus")],
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
    "Gingivitis": [os.path.join(BASE_SOURCE_DIR, "Gingivitis", "Gingivitis")],
    "Hypodontia": [os.path.join(BASE_SOURCE_DIR, "hypodontia", "hypodontia")],
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

splits = ["train", "val", "test"]

# ==============================================================================
# 3. EXECUTION LOGIC
# ==============================================================================
print("[INIT] Creating clean directory structure...")
for split in splits:
    for class_name in data_map.keys():
        os.makedirs(os.path.join(TARGET_DIR, split, class_name), exist_ok=True)

print(f"Target Directory configured at: {TARGET_DIR}\n")

for class_name, folder_paths in data_map.items():
    all_images = []

    for folder in folder_paths:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_images.append(os.path.join(folder, file))
        else:
            print(f"Warning: Raw path not found -> {folder}")

    if not all_images:
        print(
            f"[ERROR] No images found for {class_name}! Please check your BASE_SOURCE_DIR."
        )
        continue

    print(f" {class_name}: Found {len(all_images)} total valid images. Splitting...")

    # Data Splitting (70% Train, 15% Val, 15% Test)
    train_imgs, temp_imgs = train_test_split(
        all_images, test_size=0.30, random_state=42
    )
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)

    split_dict = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    # Copy files to the structured target directory
    for split_name, imgs in split_dict.items():
        for img_path in imgs:
            # Create a safe, unique filename to prevent overwriting
            safe_filename = f"{os.path.basename(os.path.dirname(img_path))}_{os.path.basename(img_path)}"
            dest_path = os.path.join(TARGET_DIR, split_name, class_name, safe_filename)
            shutil.copy2(img_path, dest_path)

print(f"\n[SUCCESS] All done! Your clean, production-ready dataset is generated.")
