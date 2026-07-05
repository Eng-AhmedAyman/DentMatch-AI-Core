"""
==============================================================================
FILE: stage1_evaluation.py
DESCRIPTION:
Evaluation module for Stage 1 (Security Guard).
This script loads the trained MobileNetV2 model and runs inference on the
unseen test dataset. It calculates precision, recall, and F1-score, and
generates a visually appealing Confusion Matrix saved directly to the
reports directory.
==============================================================================
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Dynamically point to Model, Test Data, and Output folders
# NOTE: path MUST match STAGE1_PATH in deployment/master_pipeline.py
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stage1", "stage1_mobilenet.keras")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "stage1_binary", "test")

# Save outputs to the structured reports folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the figures folder exists

CM_SAVE_PATH = os.path.join(OUTPUT_DIR, "stage1_confusion_matrix.png")


# ==============================================================================
# 2. EVALUATION LOGIC
# ==============================================================================
def evaluate_model():
    print(f"⏳ [INIT] Loading Stage 1 Security Guard from:\n   -> {MODEL_PATH}")

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f" [ERROR] Failed to load model. Ensure it exists. Details: {e}")
        return

    y_true, y_pred = [], []

    print(f" [INFO] Scanning test directory:\n   -> {TEST_DIR}")
    print(" Running inference on test dataset. Please wait...")

    for root, dirs, files in os.walk(TEST_DIR):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, filename)

                # Assign label based on folder name (teeth vs not)
                # This ensures we don't accidentally match 'teeth' in the root path
                folder_name = os.path.basename(root).lower()
                label = 1 if folder_name == "teeth" else 0

                try:
                    # Preprocess Image
                    img = image.load_img(path, target_size=(224, 224))
                    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

                    # Predict
                    pred = model.predict(arr, verbose=0)[0][0]

                    y_true.append(label)
                    # We use 0.85 as our strict security threshold to match the Master Pipeline
                    y_pred.append(1 if pred >= 0.85 else 0)
                except Exception as e:
                    print(f" [WARNING] Failed to process image {filename}: {e}")

    # ==============================================================================
    # 3. REPORT GENERATION & VISUALIZATION
    # ==============================================================================
    print("\n" + "=" * 50)
    print(" CLINICAL CLASSIFICATION REPORT")
    print("=" * 50)
    # 0 = Not_Teeth, 1 = Teeth
    print(classification_report(y_true, y_pred, target_names=["Not_Teeth", "Teeth"]))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # Stylish Seaborn Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not_Teeth", "Teeth"],
        yticklabels=["Not_Teeth", "Teeth"],
    )

    plt.title("Stage 1 (Security Guard) - Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=300)

    print("=" * 50)
    print(f"[SUCCESS] Confusion matrix saved beautifully at:\n   -> {CM_SAVE_PATH}")


if __name__ == "__main__":
    evaluate_model()
