"""
==============================================================================
FILE: master_pipeline.py
DESCRIPTION:
    Master Production Pipeline for the DentMatch AI (Healthy Smile) System.

    Orchestrates two sequential AI models to analyse a patient's dental image:

        Stage 1 — MobileNet Security & Privacy Guard
                  Rejects (a) images containing a detected human face (privacy)
                  and (b) images that do not look like dental images (score < 0.85
                  on the binary dental-vs-non-dental classifier).

        Stage 2 — HuggingFace Transformers Triage Doctor (offline)
                  Binary healthy-vs-diseased classifier.  Returns "Healthy" early
                  when confidence ≥ 86 %, otherwise routes to the EfficientNetB4
                  specialist for multi-class disease classification.

        Stage 3 — EfficientNetB4 Specialist  (internal; reached via Stage 2)
                  Multi-class classifier for 5 disease categories:
                      Dental_Caries, Hypodontia, Mouth_Ulcer,
                      Periodontal_Disease, Tooth_Discoloration.
                  ``stages_passed`` is reported as 2 (public pipeline depth).

    Both a disk-based entry point (``analyze_patient``) and an in-memory entry
    point (``analyze_patient_from_bytes``) are provided.  The FastAPI gateway
    and the Streamlit dashboard both call ``analyze_patient_from_bytes``
    exclusively, ensuring zero temporary-file overhead.

MODEL PATHS (relative to project root):
    Stage 1 — models/stage1/stage1_mobilenet.keras
    Stage 2 — models/stage2/   (HuggingFace local model directory)
    Stage 3 — models/stage3/stage3_efficientnet_finetuned_best.keras

OUTPUT SCHEMA:
    Both ``analyze_*`` methods return a JSON-serialisable dict:

        patient_image           str   — original filename
        timestamp               str   — ISO-style datetime string
        status                  str   — "success" | "rejected" | "error"
        stages_passed           int   — 0–2
        diagnosis               str | None — e.g. "Dental_Caries" or "Healthy"
        confidence_score        float — 0.0–100.0
        disease_probabilities   dict  — {disease_name: probability_%}
        requires_human_review   bool  — True when confidence < SAFETY_THRESHOLD
        rejection_reason        str | None
        message                 str
        processing_time_seconds float

USAGE (CLI smoke-test):
    python master_pipeline.py
    Requires at least one JPEG/PNG image in data/test_samples/.

DEPENDENCIES:
    tensorflow, transformers, opencv-python, Pillow, numpy

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 2.1.0  (audit fix — out-of-scope guard, Healthy disease_probabilities,
                 removed duplicate REPORTS_DIR side-effect at import time)
==============================================================================
"""

# ==============================================================================
# ZONE 1: IMPORTS
# ==============================================================================
import io
import os
import json
import time
import datetime
import warnings
import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Suppress TF / Transformers verbosity in production
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==============================================================================
# ZONE 2: DYNAMIC PATH RESOLUTION
# All paths are relative to the project root so the module works regardless
# of the working directory from which it is imported.
# ==============================================================================
CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(CURRENT_DIR)

# Model file / directory paths
STAGE1_PATH: str = os.path.join(
    PROJECT_ROOT, "models", "stage1", "stage1_mobilenet.keras"
)
STAGE2_PATH: str = os.path.join(PROJECT_ROOT, "models", "stage2")
STAGE3_PATH: str = os.path.join(
    PROJECT_ROOT, "models", "stage3", "stage3_efficientnet_finetuned_best.keras"
)

# JSON report output directory — created lazily inside the CLI block only.
# NOTE: os.makedirs() is intentionally NOT called at module level so that
#       importing this module from FastAPI or Streamlit does NOT create
#       directories on the server's filesystem as a side-effect.
REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports", "patient_results")

# Disease labels for the specialist model — ORDER MUST MATCH the model's output layer
DISEASE_NAMES: list[str] = [
    "Dental_Caries",
    "Hypodontia",
    "Mouth_Ulcer",
    "Periodontal_Disease",
    "Tooth_Discoloration",
]

# All valid output labels (used for out-of-scope guard in _run_stage3)
VALID_LABELS: frozenset[str] = frozenset(DISEASE_NAMES + ["Healthy"])

# Minimum confidence % before a "requires_human_review" flag is raised
SAFETY_THRESHOLD: float = 80.0


# ==============================================================================
# ZONE 3: HELPER — EMPTY REPORT TEMPLATE
# Centralised so both analyze_* methods return an identical schema, even on
# failure.
# ==============================================================================


def _empty_report(filename: str) -> dict:
    """
    Return a zeroed-out report dict with the given filename and current timestamp.

    This guarantees that both ``analyze_patient`` and
    ``analyze_patient_from_bytes`` always return a JSON-serialisable dict with
    a consistent schema, even when an early error occurs.

    Parameters
    ----------
    filename : str
        Original image filename — used for audit tracing only.

    Returns
    -------
    dict
        Report skeleton with ``status="pending"`` and all numeric fields at
        their zero defaults.
    """
    return {
        "patient_image": filename,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending",
        "stages_passed": 0,
        "diagnosis": None,
        "confidence_score": 0.0,
        "disease_probabilities": {},
        "requires_human_review": False,
        "rejection_reason": None,
        "message": "",
        "processing_time_seconds": 0.0,
    }


# ==============================================================================
# ZONE 4: MAIN CLASS
# ==============================================================================


class DentalAI_System:
    """
    Singleton-style orchestrator for the 3-stage DentMatch AI pipeline.

    Intended to be instantiated **once** at application startup — either via
    ``@st.cache_resource`` in Streamlit or at module level in FastAPI — and
    then reused for every patient request.  All heavy models live as instance
    attributes so they are loaded only once.

    Attributes
    ----------
    stage1_guard : tf.keras.Model
        Binary dental/non-dental MobileNet guard (Stage 1).
    face_cascade : cv2.CascadeClassifier
        OpenCV Haar Cascade face detector (Stage 1 — privacy enforcement).
    stage2_triage : transformers.Pipeline
        HuggingFace healthy/diseased binary triage classifier (Stage 2).
    stage3_specialist : tf.keras.Model
        EfficientNetB4 5-class disease specialist (Stage 3).
    disease_names : list[str]
        Ordered disease label list — matches the specialist's output layer.
    safety_threshold : float
        Minimum confidence % before a ``requires_human_review`` flag is set.
    """

    def __init__(self) -> None:
        print("\n" + "=" * 60)
        print("🏥 BOOTING HEALTHY SMILE AI CLINIC...")
        print("=" * 60)

        try:
            # ---- Stage 1: MobileNet Security Guard ----
            print("⏳ Loading Stage 1 (Security Guard)...")
            self.stage1_guard = tf.keras.models.load_model(STAGE1_PATH)

            # Haar Cascade face detector — bundled with OpenCV
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(haar_path)
            print("   ✅ [STAGE 1 READY]")

            # ---- Stage 2: HuggingFace Triage Doctor (offline) ----
            print("⏳ Loading Stage 2 (Triage Doctor — Offline)...")
            from transformers import pipeline as hf_pipeline

            self.stage2_triage = hf_pipeline("image-classification", model=STAGE2_PATH)
            print("   ✅ [STAGE 2 READY]")

            # ---- Stage 3: EfficientNetB4 Specialist ----
            print("⏳ Loading EfficientNetB4 Specialist...")
            self.stage3_specialist = tf.keras.models.load_model(
                STAGE3_PATH,
                compile=False,  # Skip focal-loss deserialisation (inference-only)
            )
            print("   ✅ [SPECIALIST READY]")

            # Shared configuration
            self.disease_names: list[str] = DISEASE_NAMES
            self.safety_threshold: float = SAFETY_THRESHOLD

            print("✅ [SYSTEM READY] All AI models loaded successfully.\n")

        except Exception as exc:
            print(f"❌ [FATAL] Initialisation failed: {exc}")
            raise

    # --------------------------------------------------------------------------
    # PREPROCESSING HELPERS
    # --------------------------------------------------------------------------

    def preprocess_image(
        self, img_path: str, needs_rescale: bool = False
    ) -> np.ndarray:
        """
        Load an image from disk and prepare it for a Keras model.

        Parameters
        ----------
        img_path : str
            Absolute path to a JPEG or PNG image file.
        needs_rescale : bool, optional
            If ``True``, pixel values are divided by 255.0 after loading.
            Use ``True`` for MobileNet (Stage 1) which expects ``[0, 1]``
            inputs; ``False`` for EfficientNetB4 (Stage 3) which has its own
            internal rescaling layer.

        Returns
        -------
        np.ndarray
            Shape ``(1, 224, 224, 3)``, dtype ``float32``.
        """
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(keras_image.img_to_array(img), axis=0)
        if needs_rescale:
            img_array = img_array / 255.0
        return img_array

    def preprocess_image_from_memory(
        self, pil_img: Image.Image, needs_rescale: bool = False
    ) -> np.ndarray:
        """
        Prepare a PIL image (already in RAM) for a Keras model.

        Identical contract to :meth:`preprocess_image` but avoids all disk I/O.
        Used by :meth:`analyze_patient_from_bytes` for the in-memory API path.

        Parameters
        ----------
        pil_img : PIL.Image.Image
            Source PIL image (any size / colour mode).
        needs_rescale : bool, optional
            See :meth:`preprocess_image` for semantics.

        Returns
        -------
        np.ndarray
            Shape ``(1, 224, 224, 3)``, dtype ``float32``.
        """
        img = pil_img.resize((224, 224))
        img_array = np.expand_dims(keras_image.img_to_array(img), axis=0)
        if needs_rescale:
            img_array = img_array / 255.0
        return img_array

    # --------------------------------------------------------------------------
    # PRIVATE: STAGE 3 SHARED LOGIC
    # --------------------------------------------------------------------------

    def _run_stage3(
        self, pil_img: Image.Image, report: dict, start_time: float
    ) -> dict:
        """
        Run the EfficientNetB4 specialist on a PIL image and update the report.

        Extracted as a private helper to avoid duplicating the specialist logic
        in both :meth:`analyze_patient` and :meth:`analyze_patient_from_bytes`.

        Fixes applied in v2.1
        ----------------------
        * **Out-of-scope guard**: if ``np.argmax`` returns an index beyond
          ``len(self.disease_names)`` (can happen if the model's output layer
          has extra neurons), the pipeline now returns a safe ``"Healthy"``
          fallback with ``requires_human_review=True`` instead of an
          ``IndexError``.
        * **Healthy ``disease_probabilities``**: the ``disease_probabilities``
          dict is now always populated (even for low-confidence edge cases) so
          the Streamlit bar chart renders correctly for all paths.

        Parameters
        ----------
        pil_img : PIL.Image.Image
            Image to classify (any size; resized internally to 224×224).
        report : dict
            Report dict to update in-place (from :func:`_empty_report`).
        start_time : float
            ``time.time()`` value from the caller's start — used to compute
            ``processing_time_seconds``.

        Returns
        -------
        dict
            Updated report with ``status``, ``diagnosis``, ``confidence_score``,
            ``disease_probabilities``, ``stages_passed=2``, and either a message
            or a ``requires_human_review`` flag.
        """
        print("🔬 [SPECIALIST] Disease Classification...")

        img_stage3 = self.preprocess_image_from_memory(pil_img, needs_rescale=False)
        predictions = self.stage3_specialist.predict(img_stage3, verbose=0)[0]

        print("\n   🧠 [AI BRAIN SCAN] Disease probability breakdown:")
        for name, prob in zip(self.disease_names, predictions):
            print(f"      - {name}: {prob * 100:.2f}%")
        print("   " + "-" * 40)

        prob_dict: dict[str, float] = {
            self.disease_names[i]: round(float(predictions[i] * 100), 2)
            for i in range(len(self.disease_names))
        }

        disease_index = int(np.argmax(predictions))

        # Out-of-scope guard — protects against unexpected model output shapes
        if disease_index >= len(self.disease_names):
            print(
                f"   ⚠️ [GUARD] argmax index {disease_index} out of range "
                f"(expected 0–{len(self.disease_names) - 1}). "
                "Defaulting to Healthy."
            )
            report.update(
                {
                    "status": "success",
                    "diagnosis": "Healthy",
                    "confidence_score": 0.0,
                    "disease_probabilities": prob_dict,
                    "stages_passed": 2,
                    "requires_human_review": True,
                    "message": (
                        "Model output index out of range. "
                        "Defaulted to Healthy — human review required."
                    ),
                    "processing_time_seconds": round(time.time() - start_time, 2),
                }
            )
            return report

        final_conf = round(float(np.max(predictions) * 100), 2)
        diagnosis = self.disease_names[disease_index]

        report.update(
            {
                "status": "success",
                "diagnosis": diagnosis,
                "confidence_score": final_conf,
                "disease_probabilities": prob_dict,
                "stages_passed": 2,
            }
        )

        if final_conf < self.safety_threshold:
            print(
                f"   ⚠️ Low confidence ({final_conf:.2f}%). "
                "Flagging for human review."
            )
            report.update(
                {
                    "requires_human_review": True,
                    "message": (
                        f"Diagnosed with {diagnosis}, but confidence is low "
                        f"({final_conf:.2f}%). Human review required."
                    ),
                }
            )
        else:
            print(f"   ✅ DIAGNOSIS: {diagnosis} (Confidence: {final_conf:.2f}%)")
            report["message"] = (
                f"Diagnosed with {diagnosis} with high clinical certainty."
            )

        report["processing_time_seconds"] = round(time.time() - start_time, 2)
        return report

    # --------------------------------------------------------------------------
    # PUBLIC API — DISK-BASED ENTRY POINT
    # --------------------------------------------------------------------------

    def analyze_patient(self, img_path: str) -> dict:
        """
        Full 3-stage analysis for an image stored on disk.

        Useful for batch processing, CLI testing, and offline report generation.
        For the web API and Streamlit dashboard, prefer
        :meth:`analyze_patient_from_bytes` to avoid temporary-file overhead.

        Parameters
        ----------
        img_path : str
            Absolute path to a JPEG or PNG dental image.

        Returns
        -------
        dict
            Structured diagnostic report (see module docstring for schema).
        """
        start_time = time.time()
        filename = os.path.basename(img_path)

        print("-" * 50)
        print(f"📥 New Patient Image (Disk): {filename}")
        print("-" * 50)

        report = _empty_report(filename)

        if not os.path.exists(img_path):
            print("❌ [ERROR] Image file not found.")
            report.update({"status": "error", "message": "File not found."})
            return report

        try:
            # ---- Stage 1: Security & Privacy ----
            print("🛡️  [STAGE 1] Security & Privacy Check...")
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                report.update(
                    {"status": "error", "message": "Could not decode image file."}
                )
                return report

            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=10, minSize=(150, 150)
            )

            if len(faces) > 0:
                print("   ❌ REJECTED: Human face detected (privacy violation).")
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Privacy Violation",
                        "message": (
                            "Upload rejected. Image contains a human face. "
                            "Please upload teeth only."
                        ),
                    }
                )
                return report

            img_stage1 = self.preprocess_image(img_path, needs_rescale=True)
            pred_stage1 = self.stage1_guard.predict(img_stage1, verbose=0)[0][0]

            if pred_stage1 < 0.85:
                print(
                    f"   ❌ REJECTED: Not a valid dental image "
                    f"(score: {pred_stage1:.4f})"
                )
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Invalid Structure",
                        "message": (
                            "Upload rejected. Image does not clearly show "
                            "dental structure."
                        ),
                    }
                )
                return report

            confidence = (
                float(pred_stage1 * 100)
                if pred_stage1 >= 0.5
                else float((1 - pred_stage1) * 100)
            )
            print(f"   ✅ PASSED: Valid dental image (confidence: {confidence:.2f}%)")
            report["stages_passed"] = 1

            # ---- Stage 2: Healthy vs. Diseased Triage ----
            print("🩺 [STAGE 2] General Health Assessment...")
            img_stage2 = Image.open(img_path).convert("RGB")
            results_s2 = self.stage2_triage(img_stage2)
            label = results_s2[0]["label"]
            conf_stage2 = results_s2[0]["score"] * 100

            if label == "Good Teeth" and conf_stage2 >= 86.0:
                print(f"   ✅ RESULT: Healthy (confidence: {conf_stage2:.2f}%)")
                healthy_probs: dict[str, float] = {
                    name: 0.0 for name in self.disease_names
                }
                report.update(
                    {
                        "status": "success",
                        "diagnosis": "Healthy",
                        "confidence_score": round(conf_stage2, 2),
                        "disease_probabilities": healthy_probs,
                        "stages_passed": 1,
                        "message": "Patient is healthy. Routine check-up recommended.",
                        "processing_time_seconds": round(time.time() - start_time, 2),
                    }
                )
                return report

            print(
                f"   ⚠️ Routing to specialist "
                f"(label={label}, conf={conf_stage2:.2f}%)"
            )
            report["stages_passed"] = 2

            # ---- Stage 3: EfficientNetB4 Specialist ----
            return self._run_stage3(img_stage2, report, start_time)

        except Exception as exc:
            print(f"❌ [INTERNAL ERROR] {exc}")
            report.update(
                {"status": "error", "message": f"Internal processing error: {exc}"}
            )
            return report

    # --------------------------------------------------------------------------
    # PUBLIC API — IN-MEMORY ENTRY POINT  (FastAPI & Streamlit)
    # --------------------------------------------------------------------------

    def analyze_patient_from_bytes(
        self, image_bytes: bytes, filename: str = "uploaded_image.jpg"
    ) -> dict:
        """
        Full 3-stage analysis for an image provided as raw bytes.

        Zero disk I/O — all processing happens in RAM.  This is the preferred
        entry point for the FastAPI gateway and Streamlit dashboard.

        Parameters
        ----------
        image_bytes : bytes
            Raw JPEG or PNG bytes (e.g. from ``UploadFile.read()``).
        filename : str, optional
            Original filename used only for the report audit trail.

        Returns
        -------
        dict
            Structured diagnostic report (see module docstring for schema).
        """
        start_time = time.time()

        print("-" * 50)
        print(f"📥 New Patient Image (In-Memory): {filename}")
        print("-" * 50)

        report = _empty_report(filename)

        try:
            # Decode bytes to PIL (Keras) and OpenCV (face detection)
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_img = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Guard: imdecode returns None for corrupt / non-image bytes.
            # Without this, cvtColor crashes with an unhelpful NoneType error.
            if img_cv is None:
                report.update(
                    {
                        "status": "error",
                        "message": (
                            "Could not decode image bytes. "
                            "File may be corrupt or unsupported."
                        ),
                    }
                )
                return report

            # ---- Stage 1: Security & Privacy ----
            print("🛡️  [STAGE 1] Security & Privacy Check...")
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=10, minSize=(150, 150)
            )

            if len(faces) > 0:
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Privacy Violation",
                        "message": (
                            "Human face detected. " "Please upload a teeth-only image."
                        ),
                    }
                )
                return report

            img_stage1 = self.preprocess_image_from_memory(pil_img, needs_rescale=True)
            pred_stage1 = self.stage1_guard.predict(img_stage1, verbose=0)[0][0]

            if pred_stage1 < 0.85:
                report.update(
                    {
                        "status": "rejected",
                        "rejection_reason": "Invalid Structure",
                        "message": "Not a valid dental image.",
                    }
                )
                return report

            report["stages_passed"] = 1

            # ---- Stage 2: Healthy vs. Diseased Triage ----
            print("🩺 [STAGE 2] General Health Assessment...")
            results_s2 = self.stage2_triage(pil_img)
            label = results_s2[0]["label"]
            conf_stage2 = results_s2[0]["score"] * 100

            if label == "Good Teeth" and conf_stage2 >= 86.0:
                healthy_probs: dict[str, float] = {
                    name: 0.0 for name in self.disease_names
                }
                report.update(
                    {
                        "status": "success",
                        "diagnosis": "Healthy",
                        "confidence_score": round(conf_stage2, 2),
                        "disease_probabilities": healthy_probs,
                        "stages_passed": 1,
                        "message": "Patient is healthy. Routine check-up recommended.",
                        "processing_time_seconds": round(time.time() - start_time, 2),
                    }
                )
                return report

            report["stages_passed"] = 2

            # ---- Stage 3: EfficientNetB4 Specialist ----
            return self._run_stage3(pil_img, report, start_time)

        except Exception as exc:
            report.update(
                {"status": "error", "message": f"Internal processing error: {exc}"}
            )
            return report


# ==============================================================================
# ZONE 5: CLI SMOKE TEST
# Run directly:  python master_pipeline.py
# REPORTS_DIR is created HERE (not at import time) to avoid filesystem
# side-effects when the module is imported by FastAPI or Streamlit.
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    clinic = DentalAI_System()

    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")

    if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
        test_image_path = os.path.join(TEST_DIR, os.listdir(TEST_DIR)[0])
        final_result = clinic.analyze_patient(test_image_path)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(REPORTS_DIR, f"report_{timestamp_str}.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 50)
        print("📁 MEDICAL REPORT SAVED!")
        print(f"📍 Location: {report_path}")
        print("=" * 50 + "\n")
    else:
        print(f"\n⚠️  No test images found in: {TEST_DIR}")
        print("    Add at least one JPEG/PNG to run the smoke test.")
