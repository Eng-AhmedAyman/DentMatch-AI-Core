"""
==============================================================================
FILE: stage1_inference.py
DESCRIPTION:
Production-ready inference engine for Stage 1 (Security Guard).
It handles image preprocessing (CLAHE), strict security guardrails
(Face Detection to protect privacy), CNN model inference, and detailed
error logging for auditing purposes.
==============================================================================
"""

import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION & CONFIGURATION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Dynamically point to the model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stage1_mobilenet.keras")

# Create a structured directory for logs
LOG_DIR = os.path.join(PROJECT_ROOT, "reports", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "system_logs.txt")

# Security Threshold (Synchronized with Master Pipeline)
SECURITY_THRESHOLD = 0.85

# ==============================================================================
# 2. INITIALIZATION (LOAD MODELS ONCE)
# ==============================================================================
print(f" [INIT] Loading Security Guard Model from:\n   -> {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f" Failed to load model. Ensure path is correct. Details: {e}")

# Load Haar Cascade for face detection to ensure patient privacy (Guardrail)
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)


# ==============================================================================
# 3. CORE INFERENCE LOGIC
# ==============================================================================
def log_action(filename, status, conf):
    """Logs system decisions with a timestamp for auditing and transparency."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(
            f"[{timestamp}] Image: {filename} | Status: {status} | Confidence: {conf:.2f}%\n"
        )


def analyze_image(img_path):
    """
    Executes the 4-step filtering pipeline:
    1. Integrity: Checks file existence.
    2. Pre-processing: Normalizes lighting using CLAHE.
    3. Privacy Guardrail: Detects faces to prevent processing sensitive data.
    4. Inference: Predicts using CNN with strict thresholding.
    """
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise ValueError("Corrupted image or invalid format.")

        # --- Step 1: Pre-processing (Contrast Enhancement) ---
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img_cv_enhanced = cv2.merge((cl, a, b))
        img_cv = cv2.cvtColor(img_cv_enhanced, cv2.COLOR_LAB2BGR)

        # --- Step 2: Privacy Guardrail (Face Detection) ---
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # --- Step 3: CNN Inference ---
        img_prep = image.load_img(img_path, target_size=(224, 224))
        arr = np.expand_dims(image.img_to_array(img_prep) / 255.0, axis=0)
        pred = model.predict(arr, verbose=0)[0][0]

        # --- Step 4: Adaptive Decision Logic ---
        confidence = float(pred * 100) if pred >= 0.5 else float((1 - pred) * 100)

        if pred >= SECURITY_THRESHOLD:
            status = "ACCEPTED: Valid Dental Image"
        elif len(faces) > 0:
            status = "REJECTED: Privacy Alert (Face Detected)"
        else:
            status = "REJECTED: Invalid Structure (Not Teeth)"

        log_action(os.path.basename(img_path), status, confidence)
        return status, confidence

    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        log_action(os.path.basename(img_path), error_msg, 0)
        return "ERROR", 0.0


# ==============================================================================
# 4. BATCH PROCESSING CAPABILITY
# ==============================================================================
def run_batch_test(folder_path):
    """Processes images in bulk with a live progress indicator."""
    if not os.path.exists(folder_path):
        print(f" [ERROR] Directory not found: {folder_path}")
        return

    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not files:
        print(f" [WARNING] No images found in: {folder_path}")
        return

    print(
        f"\n Starting batch processing for {len(files)} images in:\n   -> {folder_path}"
    )

    for filename in tqdm(files, desc="Processing Images", unit="img"):
        path = os.path.join(folder_path, filename)
        analyze_image(path)

    print("\n Batch processing completed successfully!")
    print(f" Check the detailed logs at:\n   -> {LOG_FILE}")


if __name__ == "__main__":
    # Test on the dynamic test_samples directory
    TEST_FOLDER = os.path.join(PROJECT_ROOT, "data", "test_samples")

    # Create the folder if it doesn't exist just in case
    os.makedirs(TEST_FOLDER, exist_ok=True)

    run_batch_test(TEST_FOLDER)
