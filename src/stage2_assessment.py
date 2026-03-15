"""
==============================================================================
FILE: stage2_assessment.py
DESCRIPTION:
Health Assessment Engine (Stage 2: The Triage Doctor).
This module uses an offline, pre-trained Vision Transformer (ViT) to rapidly
classify validated dental images into "Good Teeth" (Healthy) or "Bad Teeth"
(Requires further diagnosis). Operating offline ensures zero-latency and
maximum reliability in clinical production environments.
==============================================================================
"""

import os
import warnings
from PIL import Image
from transformers import pipeline
import logging

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION & CONFIGURATION
# ==============================================================================
# Automatically locate the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Dynamically point to the offline Stage 2 model folder
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stage2")

# Suppress Hugging Face warnings for a cleaner production console
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================================================================
# 2. MODEL INITIALIZATION (OFFLINE MODE)
# ==============================================================================
print(
    f" [INIT] Loading Stage 2 (Triage Doctor) from local storage:\n   -> {LOCAL_MODEL_PATH}"
)

try:
    # Initialize the Hugging Face Pipeline using the local directory
    health_classifier = pipeline("image-classification", model=LOCAL_MODEL_PATH)
    print(" [SUCCESS] Stage 2 ViT Model initialized successfully!\n")
except Exception as e:
    raise RuntimeError(
        f" [FATAL] Failed to load Stage 2 model. Did you run the download script? Details: {e}"
    )


# ==============================================================================
# 3. CORE INFERENCE LOGIC
# ==============================================================================
def assess_tooth_health(img_path):
    """
    Analyzes a valid dental image to determine overall health status.

    Args:
        img_path (str): Absolute path to the dental image.

    Returns:
        tuple: (status_label (str), confidence_score (float))
    """
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # ViT architecture requires PIL Image format natively
        img = Image.open(img_path).convert("RGB")

        # Run inference through the transformer pipeline
        results = health_classifier(img)

        # Extract best prediction (Pipeline returns a sorted list of dicts)
        best_prediction = results[0]
        label = best_prediction["label"]
        confidence = best_prediction["score"] * 100

        return label, confidence

    except Exception as e:
        print(
            f" [ERROR] Stage 2 Processing Failed for {os.path.basename(img_path)}: {e}"
        )
        return "ERROR", 0.0


if __name__ == "__main__":
    # Quick module test
    print("Module is ready to be imported by the Master Pipeline.")
