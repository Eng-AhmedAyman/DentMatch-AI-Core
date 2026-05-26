# Data Directory

This directory contains the datasets used to train and evaluate the HealthySmile AI 3-Stage Diagnostic Pipeline.

> **Note:** The actual image data is **not included** in this repository due to size constraints.  
> Follow the instructions below to reconstruct the dataset locally.

---

## Directory Structure

```
data/
├── stage1_binary/          ← Binary classification (Healthy vs Dental Issue)
│   ├── train/
│   ├── val/
│   └── test/
│
├── stage3_disease/         ← Multi-class disease classification (5 classes)
│   ├── train/
│   ├── val/
│   └── test/
│       ├── Dental_Caries/
│       ├── Hypodontia/
│       ├── Mouth_Ulcer/
│       ├── Periodontal_Disease/
│       └── Tooth_Discoloration/
│
└── test_samples/           ← Sample images for quick demo & smoke testing
```

---

## How to Reconstruct the Dataset

### Step 1 — Collect Raw Images
Gather raw images for each disease class from your local sources or public datasets.

### Step 2 — Run the Preprocessing Script
The `src/clean_data.py` script handles everything automatically:
- Filters valid image formats (`.jpg`, `.jpeg`, `.png`)
- Splits data into **Train (70%) / Val (15%) / Test (15%)**
- Organizes into the structure above

```bash
# Update BASE_SOURCE_DIR in src/clean_data.py to point to your raw data
python src/clean_data.py
```

### Step 3 — Verify
After running, confirm the structure matches the tree above.

---

## Dataset Statistics

| Stage | Task | Classes | Total Images |
|-------|------|---------|--------------|
| Stage 1 | Binary (Healthy / Issue) | 2 | ~12,000 |
| Stage 3 | Disease Classification | 5 | ~12,000 |

---

## Disease Classes (Stage 3)

| Class | Description |
|-------|-------------|
| Dental_Caries | Tooth decay caused by bacterial acids |
| Hypodontia | Congenitally missing teeth |
| Mouth_Ulcer | Painful sores on oral mucosa |
| Periodontal_Disease | Gum and bone infection around teeth |
| Tooth_Discoloration | Staining or color changes in teeth |

---

## test_samples/
Contains a small set of sample images (one per class) used for:
- Quick smoke testing after deployment
- Demo purposes without needing the full dataset
