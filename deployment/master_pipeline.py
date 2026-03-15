"""
==============================================================================
FILE: master_pipeline.py
DESCRIPTION:
The Master Production Pipeline for the Healthy Smile System.
This module orchestrates the 3 AI models (Security, Triage, Specialist).
It processes the patient's image securely, displays a beautiful console UI,
and exports a structured JSON medical report to the /reports directory.
==============================================================================
"""

import os
import json
import time
import datetime
import warnings
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Suppress Hugging Face warnings for a clean production console
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Model Paths
STAGE1_PATH = os.path.join(PROJECT_ROOT, "models/stage1", "stage1_mobilenet.keras")
STAGE2_PATH = os.path.join(
    PROJECT_ROOT, "models", "stage2"
)  # Updated to match your folder
STAGE3_PATH = os.path.join(
    PROJECT_ROOT, "models/stage3", "stage3_efficientnet_finetuned_best.keras"
)

# Output Directory for JSON Reports
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "patient_results")
os.makedirs(REPORTS_DIR, exist_ok=True)


class DentalAI_System:
    def __init__(self):
        print("\n" + "=" * 60)
        print("🏥 BOOTING HEALTHY SMILE AI CLINIC...")
        print("=" * 60)

        try:
            print("⏳ Loading Stage 1 (Security Guard)...")
            self.stage1_guard = tf.keras.models.load_model(STAGE1_PATH)

            # Load Privacy Guard (Face Cascade)
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(haar_path)

            print("⏳ Loading Stage 2 (Triage Doctor - Offline)...")
            from transformers import pipeline

            self.stage2_triage = pipeline("image-classification", model=STAGE2_PATH)

            print("⏳ Loading Stage 3 (Specialist)...")
            self.stage3_specialist = tf.keras.models.load_model(STAGE3_PATH)

            self.disease_names = [
                "Calculus",
                "Dental_Caries",
                "Gingivitis",
                "Hypodontia",
                "Mouth_Ulcer",
                "Tooth_Discoloration",
            ]
            self.safety_threshold = 80.0

            print("✅ [SYSTEM READY] All AI Models successfully loaded.\n")

        except Exception as e:
            print(f"❌ [FATAL ERROR] Initialization failed: {e}")
            raise

    def preprocess_image(self, img_path, needs_rescale=False):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        if needs_rescale:
            img_array = img_array / 255.0
        return img_array

    def analyze_patient(self, img_path):
        start_time = time.time()
        filename = os.path.basename(img_path)

        print("-" * 50)
        print(f"📥 New Patient Image Received: {filename}")
        print("-" * 50)

        report = {
            "patient_image": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
            "stages_passed": 0,
            "diagnosis": None,
            "confidence_score": 0.0,
            "requires_human_review": False,
            "rejection_reason": None,
            "message": "",
            "processing_time_seconds": 0.0,
        }

        if not os.path.exists(img_path):
            print("❌ [ERROR] Image file not found.")
            report["status"] = "error"
            report["message"] = "File not found."
            return report

        try:
            # ==========================================
            # STAGE 1: SECURITY & PRIVACY CHECK
            # ==========================================
            print("🛡️  [STAGE 1] Security & Privacy Check...")

            # Privacy Check (Face Detection)
            img_cv = cv2.imread(img_path)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                print("   ❌ REJECTED: Human Face Detected (Privacy Violation)!")
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Privacy Violation",
                        "message": "Upload rejected. Image contains a human face. Please upload teeth only.",
                    }
                )
                return report

            # Security Inference
            img_stage1 = self.preprocess_image(img_path, needs_rescale=True)
            pred_stage1 = self.stage1_guard.predict(img_stage1, verbose=0)[0][0]

            # Adaptive Threshold
            confidence = (
                float(pred_stage1 * 100)
                if pred_stage1 >= 0.5
                else float((1 - pred_stage1) * 100)
            )

            if pred_stage1 < 0.85:
                print(
                    f"   ❌ REJECTED: Not a valid dental image (Confidence: {confidence:.2f}%)"
                )
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Invalid Structure",
                        "message": "Upload rejected. Image does not clearly show dental structure.",
                    }
                )
                return report

            print(f"   ✅ PASSED: Valid Dental Image (Confidence: {confidence:.2f}%)")
            report["stages_passed"] = 1

            # ==========================================
            # STAGE 2: TRIAGE CHECK (Healthy vs. Sick)
            # ==========================================
            print("🩺 [STAGE 2] General Health Assessment...")

            img_stage2 = Image.open(img_path).convert("RGB")
            results_stage2 = self.stage2_triage(img_stage2)
            label = results_stage2[0]["label"]
            conf_stage2 = results_stage2[0]["score"] * 100

            if label == "Good Teeth":
                # 🚀 التعديل العبقري: حد الأمان المتشدد للمرضى الأصحاء
                if conf_stage2 >= 86.0:
                    print(
                        f"   ✅ RESULT: Healthy Teeth Detected (Confidence: {conf_stage2:.2f}%)"
                    )
                    report.update(
                        {
                            "status": "success",
                            "diagnosis": "Healthy",
                            "confidence_score": round(conf_stage2, 2),
                            "stages_passed": 2,
                            "message": "Patient is healthy. Routine checkup recommended.",
                            "processing_time_seconds": round(
                                time.time() - start_time, 2
                            ),
                        }
                    )
                    return report
                else:
                    # لو الثقة أقل من 85%، الموديل هيشك في نفسه ويكمل للمرحلة التالتة
                    print(
                        f"   ⚠️ ALERT: 'Healthy' confidence is only {conf_stage2:.2f}% (< 85%). Routing to Specialist for deep check..."
                    )
            else:
                print(
                    f"   ⚠️ ALERT: Dental Issue Detected (Confidence: {conf_stage2:.2f}%). Routing to Specialist..."
                )

            report["stages_passed"] = 2

            # ==========================================
            # STAGE 3: SPECIALIST DIAGNOSIS
            # ==========================================
            print("🔬 [STAGE 3] Specialist Diagnosis...")

            img_stage3 = self.preprocess_image(img_path, needs_rescale=False)
            predictions = self.stage3_specialist.predict(img_stage3, verbose=0)[0]

            # --- التعديل السحري: كشف كل احتمالات الموديل ---
            print("\n   🧠 [AI BRAIN SCAN] What the model actually sees:")
            for i, disease in enumerate(self.disease_names):
                print(f"      - {disease}: {predictions[i]*100:.2f}%")
            print("   " + "-" * 40)
            # ----------------------------------------------

            disease_index = np.argmax(predictions)
            final_conf = float(np.max(predictions) * 100)
            diagnosis = self.disease_names[disease_index]

            report.update(
                {
                    "status": "success",
                    "diagnosis": diagnosis,
                    "confidence_score": round(final_conf, 2),
                    "stages_passed": 3,
                }
            )

            if final_conf < self.safety_threshold:
                print(
                    f"   ⚠️ WARNING: Low Confidence ({final_conf:.2f}%). Flagged for human review."
                )
                report["requires_human_review"] = True
                report["message"] = (
                    f"Diagnosed with {diagnosis}, but confidence is low. Human review required."
                )
            else:
                print(f"   ✅ DIAGNOSIS: {diagnosis} (Confidence: {final_conf:.2f}%)")
                report["message"] = (
                    f"Diagnosed with {diagnosis} with high clinical certainty."
                )

            report["processing_time_seconds"] = round(time.time() - start_time, 2)
            return report

        # هنا القفلة بتاعت الـ try اللي كانت ممسوحة
        except Exception as e:
            print(f"❌ [INTERNAL ERROR] {e}")
            report["status"] = "error"
            report["message"] = f"Internal processing error: {str(e)}"
            return report


# ==============================================================================
# 🛠️ HOW TO TEST THE PIPELINE
# ==============================================================================
if __name__ == "__main__":
    clinic = DentalAI_System()

    # Get an image from the test_samples folder dynamically
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")

    # Check if the folder exists and has images
    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        # Just grab the first image in the folder for testing
        test_image_name = os.listdir(TEST_DIR)[0]
        test_image_path = os.path.join(TEST_DIR, test_image_name)

        # 1. Run Analysis
        final_result = clinic.analyze_patient(test_image_path)

        # 2. Save JSON to a file (Hide from Terminal!)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp_str}.json"
        report_filepath = os.path.join(REPORTS_DIR, report_filename)

        with open(report_filepath, "w", encoding="utf-8") as json_file:
            json.dump(final_result, json_file, indent=4, ensure_ascii=False)

        print("\n" + "=" * 50)
        print(f"📁 MEDICAL REPORT SAVED SECURELY!")
        print(f"📍 Location: {report_filepath}")
        print("=" * 50 + "\n")

    else:
        print(f"\n⚠️ Please put at least one test image inside: {TEST_DIR}")
