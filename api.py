"""
==============================================================================
FILE: api.py
DESCRIPTION:
    FastAPI Gateway for the DentMatch AI (Healthy Smile) System.

    Exposes two primary inference endpoints:

        1. POST /analyze/
           Image-based dental diagnosis via the 3-stage CNN deep-learning
           pipeline (MobileNetV2 domain gate → HF ViT healthy/diseased
           triage → EfficientNetB4 disease classifier). Accepts optional
           form fields:
               - pain_duration    (str, default "غير محدد")
               - chronic_diseases (str, default "لا يوجد")

        2. POST /triage-symptoms/
           Text-based patient symptom triage routed to the correct university
           clinic department via Gemini 2.5 Flash (LLM, temperature=0).

    Both endpoints return the SAME top-level report schema so the Streamlit
    dashboard can render them with a single ``_render_report()`` function.

REPORT SCHEMA (identical for both endpoints):
    document_info
    symptoms_and_history
    initial_medical_assessment   ← contains confidence_score, stages_passed, etc.
    care_and_referral_plan
    legal_disclaimer

DEPENDENCIES:
    fastapi, uvicorn, python-multipart, pydantic, google-genai,
    python-dotenv, deployment.master_pipeline

ENVIRONMENT VARIABLES (required in .env):
    GEMINI_API_KEY — Google Generative AI API key.

USAGE:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

    Swagger UI : http://127.0.0.1:8000/docs
    ReDoc      : http://127.0.0.1:8000/redoc

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 2.2.0  (out-of-scope short-circuit, error-message hardening, docstring fixes)
==============================================================================
"""

# ==============================================================================
# ZONE 1: IMPORTS
# ==============================================================================
import os
import json
import time
from datetime import datetime
import keras
import keras.src.saving.serialization_lib as serialization_lib

original_deserialize = serialization_lib.deserialize_keras_object


def safe_deserialize(*args, **kwargs):
    try:
        return original_deserialize(*args, **kwargs)
    except TypeError as e:
        if "quantization_config" in str(e):
            if len(args) > 0 and "config" in args[0]:
                args[0]["config"].pop("quantization_config", None)
            elif "config" in kwargs:
                kwargs["config"].pop("quantization_config", None)
            return original_deserialize(*args, **kwargs)
        raise e


serialization_lib.deserialize_keras_object = safe_deserialize
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

from deployment.master_pipeline import DentalAI_System

# ==============================================================================
# ZONE 2: DIAGNOSIS MAP
# Single source-of-truth for all disease metadata used by BOTH endpoints.
#
# Keys MUST match the labels emitted by master_pipeline.DISEASE_NAMES + "Healthy".
# All entries have the SAME set of keys so callers never need .get() fallbacks.
#
# DISPLAY FORMAT CONTRACT:
#   "ar_name" is the human-readable Arabic label shown in the dashboard under
#   🎯 Diagnosis Result — format:  "<EnglishName> - <Arabic description>"
#   e.g. "Periodontal Disease - التهابات لثة وتراكم جير"
# ==============================================================================
DIAGNOSIS_MAP: dict[str, dict] = {
    "Dental_Caries": {
        "ar_name": "Dental Caries - تسوس في الأسنان",
        "dept_eng": "Operative",
        "dept_ar": "قسم الحشوات التجميلية والتحفظية",
        "ai_diagnosis": (
            "تسوس في الأسنان — يحتاج إلى تدخل وعمل حشو. "
            "(يتم تحديد نوع الحشو المناسب بواسطة الطبيب/الطالب المعالج "
            "بعد إجراء الفحص السريري المباشر)."
        ),
        "action": (
            "سيتم عرض حالتك على طلاب التخرج في قسم الحشوات التجميلية والتحفظية "
            "للتواصل معك وعلاج الحالة تحت إشراف الأطباء المتخصصين."
        ),
        "priority": "يحتاج فحص وتدخل طبي 🟡",
    },
    "Mouth_Ulcer": {
        "ar_name": "Mouth Ulcer - قرحة في الفم",
        "dept_eng": "Oral Medicine",
        "dept_ar": "قسم أمراض الفم والتشخيص",
        "ai_diagnosis": (
            "قرحة فموية — تحتاج تقييم سريري لتحديد السبب والعلاج المناسب."
        ),
        "action": (
            "سيتم عرض حالتك على طلاب التخرج في قسم أمراض الفم والتشخيص "
            "للتواصل معك وعلاج الحالة تحت إشراف الأطباء المتخصصين."
        ),
        "priority": "يحتاج فحص وتدخل طبي 🟡",
    },
    "Tooth_Discoloration": {
        "ar_name": "Tooth Discoloration - تصبغات واصفرار في الأسنان",
        "dept_eng": "Operative / Aesthetic",
        "dept_ar": "قسم الحشوات التجميلية والتحفظية",
        "ai_diagnosis": (
            "تغير لون في الأسنان — يحتاج فحص لتحديد السبب (خارجي أو داخلي) "
            "وخيارات التبييض أو العلاج المناسب."
        ),
        "action": (
            "سيتم عرض حالتك على طلاب التخرج في قسم الحشوات التجميلية والتحفظية "
            "للتواصل معك وعلاج الحالة تحت إشراف الأطباء المتخصصين."
        ),
        "priority": "حالة روتينية 🟢",
    },
    "Periodontal_Disease": {
        "ar_name": "Periodontal Disease - التهابات لثة وتراكم جير",
        "dept_eng": "Perio",
        "dept_ar": "قسم أمراض اللثة",
        "ai_diagnosis": (
            "مؤشرات على مرض في اللثة — يحتاج فحص وتدخل طبي في أقرب وقت "
            "لتجنب تطور الحالة."
        ),
        "action": (
            "سيتم عرض حالتك على طلاب التخرج في قسم أمراض اللثة "
            "للتواصل معك وعلاج الحالة تحت إشراف الأطباء المتخصصين."
        ),
        "priority": "أولوية عالية 🟠",
    },
    "Hypodontia": {
        "ar_name": "Hypodontia - نقص أو فقدان في الأسنان",
        "dept_eng": "Fixed / Removable",
        "dept_ar": "قسم تقويم الأسنان والتعويضات السنية",
        "ai_diagnosis": (
            "نقص في عدد الأسنان — يحتاج تقييم شامل لتحديد خطة العلاج "
            "التعويضي أو التقويمي المناسبة."
        ),
        "action": (
            "سيتم عرض حالتك على طلاب التخرج في قسم تقويم الأسنان والتعويضات السنية "
            "للتواصل معك وعلاج الحالة تحت إشراف الأطباء المتخصصين."
        ),
        "priority": "حالة روتينية 🟢",
    },
    "Healthy": {
        "ar_name": "Healthy - أسنان ولثة سليمة تماماً",
        "dept_eng": "N/A",
        "dept_ar": "لا توجد حالة مرضية",
        "ai_diagnosis": "لا توجد مؤشرات مرئية لأمراض في الأسنان أو اللثة.",
        "action": (
            "حالتك ممتازة، لا توجد حاجة للتحويل لأي قسم طبي في الوقت الحالي. "
            "يرجى الحفاظ على النظافة الدورية للأسنان."
        ),
        "priority": "حالة روتينية 🟢",
    },
}

# ==============================================================================
# ZONE 2b: DEPARTMENT → DIAGNOSIS MAP
# Maps the LLM's target_department_eng codes to the correct DIAGNOSIS_MAP entry.
# This ensures /triage-symptoms/ produces the same تصنيف_الحالة display name
# as /analyze/ (Arabic + English label, not a raw English dept code).
# ==============================================================================
_DEPT_TO_DIAGNOSIS_KEY: dict[str, str] = {
    "Operative": "Dental_Caries",  # حشوات → تسوس (most common)
    "Endo": "Dental_Caries",  # عصب → تسوس شديد
    "Perio": "Periodontal_Disease",  # لثة
    "Fixed": "Hypodontia",  # تعويض ثابت → نقص أسنان
    "Remove": "Hypodontia",  # تعويض متحرك → نقص أسنان
    "Surgery": "Dental_Caries",  # جراحة → تسوس/جذور
    "Needs_Clarification": "Healthy",  # fallback — no disease confirmed
}


def _dept_to_diagnosis_info(dept_eng: str) -> dict:
    """Return the DIAGNOSIS_MAP entry that best represents a triage department."""
    key = _DEPT_TO_DIAGNOSIS_KEY.get(dept_eng, "Healthy")
    return DIAGNOSIS_MAP.get(key, DIAGNOSIS_MAP["Healthy"])


# Fixed, hand-written apology for complaints unrelated to dentistry — see
# usage in triage_patient_symptoms() for why this isn't LLM-generated.
OUT_OF_SCOPE_MESSAGE: str = (
    "شكرًا على تواصلك معنا. منصة DentMatch AI متخصصة في تقييم مشاكل الأسنان "
    "واللثة فقط، وشكواك دي مش متعلقة بالأسنان على ما يبدو، فننصحك تتوجه "
    "لطبيب التخصص المناسب لحالتك عشان تاخد التقييم والرعاية الصح."
)


# ==============================================================================
# ZONE 3: CONFIGURATION & SECURITY
# ==============================================================================

load_dotenv()

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("🚨 GEMINI_API_KEY is missing! Please add it to your .env file.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Hard limit on image upload size (5 MB)
MAX_FILE_SIZE: int = 5 * 1024 * 1024

app = FastAPI(
    title="🦷 DentMatch AI Core API",
    description=(
        "Enterprise REST API for dental triage and deep-learning diagnostics. "
        "Integrates a 3-stage CNN vision pipeline with a "
        "Gemini-powered LLM triage engine."
    ),
    version="2.2.0",
)

# CORS — allow all origins for development; tighten for production.
# NOTE: allow_credentials=True is INCOMPATIBLE with allow_origins=["*"].
#       Keep allow_credentials=False whenever using the wildcard origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to ["https://yourdomain.com"] in prod
    allow_credentials=False,  # Intentional — incompatible with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Boot AI models once at startup (avoids per-request cold-start latency)
print("⏳ Initialising DentMatch AI models...")
clinic = DentalAI_System()
print("✅ Models ready.")


# ==============================================================================
# ZONE 4: SHARED REPORT BUILDER
# Both endpoints call _build_image_report() so the JSON shape is IDENTICAL.
# app.py (download button) also reads this exact structure.
# ==============================================================================


def _build_image_report(
    raw_report: dict,
    pain_duration: str,
    chronic_diseases: str,
    source_label: str = "المنصة الذكية للتحليل البصري - DentMatch AI",
    file_prefix: str = "DM-IMG",
) -> dict:
    """
    Convert the raw pipeline dict into the unified patient-facing report.

    This is the **single source-of-truth** for the image-analysis report
    schema.  Both the ``/analyze/`` endpoint and ``app.py`` (download button)
    produce this exact structure so the two surfaces are always in sync.

    Parameters
    ----------
    raw_report : dict
        Dict returned by ``DentalAI_System.analyze_patient_from_bytes()``.
    pain_duration : str
        Free-text pain duration supplied by the patient (form field).
    chronic_diseases : str
        Free-text chronic-disease list supplied by the patient (form field).
    source_label : str, optional
        Human-readable provenance string for the document header.
    file_prefix : str, optional
        Prefix for the medical file number (e.g. ``"DM-IMG"``).

    Returns
    -------
    dict
        Unified structured report (JSON-serialisable).
        Keys are English (document_info, symptoms_and_history, etc. — see the
        module docstring's REPORT SCHEMA); most VALUES are Arabic strings for
        direct display to the patient.
    """
    raw_diagnosis = raw_report.get("diagnosis") or "Healthy"
    # Graceful fallback — unknown labels map to the Healthy entry
    info = DIAGNOSIS_MAP.get(raw_diagnosis, DIAGNOSIS_MAP["Healthy"])

    return {
        "document_info": {
            "medical_file_number": f"{file_prefix}-{int(time.time())}",
            "issue_date": raw_report.get(
                "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "report_source": source_label,
        },
        "symptoms_and_history": {
            "recorded_pain_duration": pain_duration,
            "chronic_diseases": chronic_diseases,
        },
        "initial_medical_assessment": {
            # case_classification carries the display name (Arabic + English)
            "case_classification": info["ar_name"],
            "ai_diagnosis": info["ai_diagnosis"],
            "specialized_university_department": info["dept_ar"],
            "case_priority_level": info["priority"],
        },
        "care_and_referral_plan": {
            "next_steps": info["action"],
        },
        "legal_disclaimer": (
            "هذا التقرير استرشادي صادر آلياً عن منظومة DentMatch AI، "
            "ولا يُغني عن الفحص السريري المباشر من طبيب الأسنان المختص."
        ),
        # ── Internal telemetry ── stripped from the downloadable report in app.py.
        # app.py reads these to drive the bar chart and Grad-CAM++ trigger,
        # then excludes this key before handing the JSON to st.download_button.
        "_internal": {
            "confidence_score": raw_report.get("confidence_score", 0.0),
            "requires_human_review": raw_report.get("requires_human_review", False),
            "stages_passed": raw_report.get("stages_passed", 0),
            "processing_time_seconds": raw_report.get("processing_time_seconds", 0.0),
            "disease_probabilities": raw_report.get("disease_probabilities", {}),
        },
    }


# ==============================================================================
# ZONE 5: REQUEST / RESPONSE SCHEMAS
# ==============================================================================


class SymptomRequest(BaseModel):
    """
    JSON payload for the ``/triage-symptoms/`` endpoint.

    Attributes
    ----------
    symptoms_description : str
        Free-text patient complaint (Arabic preferred).
        Min 5 characters (basic sanity check).
        Max 400 characters (controls LLM token cost and prevents abuse).
    include_internal : bool
        When False (default) the ``_internal`` and ``_llm_meta`` telemetry
        blocks are stripped from the response.  Set True only from the
        Streamlit dashboard which needs them for rendering metadata.
    """

    symptoms_description: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description="Patient complaint text (Arabic preferred).",
    )
    include_internal: bool = Field(
        default=False,
        description="Include internal telemetry blocks in the response (dashboard use only).",
    )


# ==============================================================================
# ZONE 6: ENDPOINTS
# ==============================================================================


@app.get("/", summary="Health check", tags=["System"])
def home() -> dict:
    """
    Health-check endpoint.

    Returns a minimal JSON payload confirming the API is online.
    Suitable for use in load-balancer / uptime-monitor health checks.

    Returns
    -------
    dict
        ``{"status": "Online", "message": str}``
    """
    return {
        "status": "Online",
        "message": "Welcome to DentMatch AI Core. Visit /docs for the Swagger UI.",
    }


@app.post("/analyze/", summary="Dental image diagnosis", tags=["Vision Pipeline"])
async def analyze_dental_image(
    file: UploadFile = File(
        ...,
        description="Dental X-Ray or clinical photo (JPEG/PNG, max 5 MB).",
    ),
    pain_duration: str = Form(
        default="غير محدد",
        description=(
            "Patient-reported pain duration " "(e.g. 'أيام قليلة', 'أسبوع إلى شهر')."
        ),
    ),
    chronic_diseases: str = Form(
        default="لا يوجد",
        description=("Patient-reported chronic conditions (e.g. 'مرض السكري')."),
    ),
    include_internal: bool = Form(default=False),
) -> JSONResponse:
    """
    Image-based dental diagnosis via the 3-stage CNN pipeline.

    Accepts a ``multipart/form-data`` request containing:
        - ``file``             : Dental image (JPEG or PNG, max 5 MB).
        - ``pain_duration``    : (optional) How long the patient has had pain.
        - ``chronic_diseases`` : (optional) Relevant chronic conditions.

    Pipeline stages
    ---------------
    1. MobileNet Security Guard — rejects faces and non-dental images.
    2. HuggingFace Triage Doctor — healthy vs. diseased binary classifier.
    3. EfficientNetB4 Specialist — 5-class disease classifier (when needed).

    The response is a unified Arabic structured report identical in schema
    to the report produced by the Streamlit dashboard download button.

    Returns
    -------
    JSONResponse 200 : Structured patient report (see :func:`_build_image_report`).
    JSONResponse 400 : Image rejected by the pipeline (face / non-dental).
    JSONResponse 413 : File exceeds the 5 MB size limit.
    JSONResponse 500 : Internal server / model error.

    Example (cURL)::

        curl -X POST "http://127.0.0.1:8000/analyze/" \\
             -F "file=@tooth.jpg" \\
             -F "pain_duration=أسبوع إلى شهر" \\
             -F "chronic_diseases=مرض السكري"
    """
    try:
        image_bytes = await file.read()

        # File-size guard (5 MB hard limit)
        if len(image_bytes) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "message": (
                        "حجم الصورة يتجاوز الحد المسموح (5 ميجابايت). "
                        "يرجى ضغط الصورة وإعادة المحاولة."
                    ),
                },
            )

        # Run the 3-stage pipeline
        raw_report = clinic.analyze_patient_from_bytes(
            image_bytes, filename=file.filename or "upload.jpg"
        )

        if raw_report["status"] == "error":
            return JSONResponse(status_code=500, content=raw_report)

        if raw_report["status"] == "rejected":
            return JSONResponse(status_code=400, content=raw_report)

        # Build unified formatted report
        formatted_report = _build_image_report(
            raw_report=raw_report,
            pain_duration=pain_duration,
            chronic_diseases=chronic_diseases,
        )
        if not include_internal:
            formatted_report.pop("_internal", None)
        return JSONResponse(status_code=200, content=formatted_report)

    except Exception as exc:
        print(f"❌ [/analyze/ ERROR] {exc}")  # full detail stays server-side only
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "حدث خطأ داخلي أثناء تحليل الصورة. يرجى المحاولة مرة أخرى.",
            },
        )


@app.post(
    "/triage-symptoms/",
    summary="Text-based patient triage",
    tags=["LLM Triage"],
)
async def triage_patient_symptoms(request: SymptomRequest) -> JSONResponse:
    """
    LLM-powered triage endpoint for text-based patient complaints.

    Uses Gemini 2.5 Flash (``temperature=0``) to parse the complaint and route
    it to one of the six university dental clinic departments:
        ``Endo``, ``Operative``, ``Perio``, ``Fixed``, ``Remove``, ``Surgery``.

    Special return states
    ---------------------
    ``Needs_Clarification`` — Complaint is too vague for confident routing.
                              Still returns the full unified report (asks for
                              more detail in ``next_steps``).
    Out-of-scope complaints — Short-circuited BEFORE the full report is built:
                              returns ``{"status": "out_of_scope", "message": ...}``
                              instead, to avoid spending output tokens on
                              report fields that don't apply to a non-dental
                              complaint, and to avoid mislabeling it as a
                              dental department or "Healthy".

    The full-report response schema is **identical** to ``/analyze/`` so both
    endpoints can be consumed by the same client-side rendering logic.

    Parameters
    ----------
    request : SymptomRequest
        JSON body with ``"symptoms_description"`` (Arabic text).

    Returns
    -------
    JSONResponse 200 : Unified structured patient report.
    JSONResponse 400 : Complaint too short (< 10 characters after strip).
    JSONResponse 422 : Pydantic validation error (Starlette default).
    JSONResponse 500 : LLM / JSON parsing error.

    Example (cURL)::

        curl -X POST "http://127.0.0.1:8000/triage-symptoms/" \\
             -H "Content-Type: application/json" \\
             -d '{"symptoms_description": "عندي ألم شديد في ضرسي بقاله أسبوع"}'
    """
    # Secondary server-side length guard (Pydantic min_length=5 is the first gate)
    if len(request.symptoms_description.strip()) < 10:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": (
                    "الشكوى قصيرة جداً. " "المرجو كتابة 10 حروف على الأقل من فضلك."
                ),
            },
        )

    # Strict prompt — zero temperature for clinical consistency
    prompt = f"""
    أنت مساعد ذكي لفرز حالات طب الأسنان في منصة "DentMatch AI".

    ══════════════════════════════════════════
    ⚡ الخطوة الأولى (الأهم) — قرار النطاق:
    اقرأ الشكوى وقرر فورًا: هل هي متعلقة بطب الأسنان (أسنان/لثة/فم/فك) بأي شكل،
    ولو بشكل غير مباشر؟ أم أنها لا علاقة لها بطب الأسنان إطلاقًا
    (مثل: ألم ركبة، صداع، مشكلة في المعدة، سؤال عام، دردشة)؟

    - إذا كانت الشكوى لا علاقة لها بطب الأسنان إطلاقًا:
      أرجع فقط الحقل التالي ولا شيء غيره (توفيرًا للمعالجة):
      {{
        "is_out_of_scope": true
      }}
      توقف هنا ولا تكمل باقي الحقول أدناه إطلاقًا.

    - إذا كانت متعلقة بطب الأسنان (حتى لو غامضة): أضف "is_out_of_scope": false
      واستكمل توليد كل الحقول الموصوفة أدناه بالكامل.
    ══════════════════════════════════════════

    ⚠️ قاعدة ذهبية للحالات السنية:
    أي شكوى فيها ذكر لـ (ألم، وجع، ضرس، سن، لثة، أسنان، تسوس، جير، خلع، تركيبة)
    = شكوى واضحة تماماً → يجب توجيهها لقسم مباشرةً.
    إياك تستخدم Needs_Clarification إلا لو الكلام عشوائي 100% مثل "asdfgh" أو "لا شيء".
    ══════════════════════════════════════════

    دليل توجيه الأقسام:
    - Endo      : ألم شديد، ألم ليلي يوقظ من النوم، ألم لا يزول بالمسكنات → (حشو عصب).
    - Operative : أي ألم في الضرس أو السن (خفيف أو متوسط)، تسوس، كسر سن، ألم مع الحلو أو البارد → (حشو عادي).
    - Perio     : التهاب لثة، نزيف لثة، جير، تخلخل أسنان، قرحة في الفم، حبوب على الشفاه.
    - Fixed     : تعويض سن مفقود بتركيبة ثابتة (جسر أو تاج).
    - Remove    : تعويض أسنان مفقودة بطقم متحرك.
    - Surgery   : خلع ضرس، بقايا جذور، ضرس العقل.
    - Needs_Clarification: فقط إذا كان الكلام حروف عشوائية أو جمل غير مفهومة لا علاقة لها بالأسنان.

    ══════════════════════════════════════════
    أمثلة توضيحية (few-shot):

    شكوى: "عندي ألم في ضرسي أديله 3 أيام"
    → target_department_eng: "Operative"  ✅ (ألم في ضرس = واضح)

    شكوى: "ضرسي وجعني من 3 أيام وعندي ضغط"
    → target_department_eng: "Operative"  ✅ (ذكر ضرس + ألم = واضح، الضغط معلومة طبية إضافية فقط)

    شكوى: "ألم شديد في ضرسي بيصحيني من النوم"
    → target_department_eng: "Endo"  ✅ (ألم ليلي شديد = عصب)

    شكوى: "لثتي بتنزف لما باتفرشن"
    → target_department_eng: "Perio"  ✅

    شكوى: "عايز أخلع ضرسي"
    → target_department_eng: "Surgery"  ✅

    شكوى: "اسنان" (كلمة واحدة بدون تفاصيل)
    → target_department_eng: "Needs_Clarification"  (لا يوجد أي وصف للمشكلة)

    شكوى: "asdfghjk"
    → target_department_eng: "Needs_Clarification"  (كلام عشوائي)
    ══════════════════════════════════════════

    قواعد إضافية:
    - إذا ذكر المريض أمراضاً مزمنة (ضغط، سكر، الخ) مع شكوى سنية → وجّه للقسم المناسب وسجّل المرض في extracted_data.
    - اجعل "is_emergency": false دائماً.

    شكوى المريض: "{request.symptoms_description}"

    قم بإرجاع رد بصيغة JSON فقط يحتوي على المفاتيح التالية (لأن الشكوى متعلقة
    بطب الأسنان — راجع قرار النطاق في الخطوة الأولى):
    - "is_out_of_scope": false (لأننا وصلنا هنا فقط إذا كانت الشكوى سنية).
    - "is_emergency": (اجعلها false دائماً).
    - "extracted_data": JSON يحتوي على ("pain_duration", "chronic_conditions"). اكتب "غير محدد" إذا لم يذكرها.
    - "complexity_level": (Easy أو Medium أو Complex).
    - "recommended_student_level": (طالب بكالوريوس / طبيب امتياز / دراسات عليا).
    - "patient_friendly_diagnosis": تشخيص مبدئي بالعامية المصرية يطمئن المريض.
    - "target_department_eng": (Endo, Operative, Perio, Fixed, Remove, Surgery, Needs_Clarification).
    - "target_department_ar": اسم القسم بالعربية.
    - "action_plan": خطة العمل بالعامية. قواعد:
        * إذا كانت الحالة Needs_Clarification: اطلب منه بلطف إعادة كتابة الشكوى بتفاصيل أكثر في مربع النص.
          إياك أن تخبره أنك ستطرح عليه أسئلة لأن النظام لا يدعم المحادثة.
        * إذا كانت الحالة واضحة: اشرح له أننا سنعرض حالته على طلاب القسم المختص ليتواصلوا معه.

    ملاحظة هامة: لا تكتب أي نصوص خارج الـ JSON.
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,  # Zero creativity — strict rule adherence
            ),
        )

        # response_mime_type="application/json" returns clean JSON in most cases.
        # The strip/replace is a defensive fallback for occasional markdown wrapping.
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

    except json.JSONDecodeError as exc:
        print(
            f"❌ [/triage-symptoms/ JSON ERROR] {exc}"
        )  # full detail server-side only
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "حدث خطأ أثناء معالجة رد الذكاء الاصطناعي. يرجى المحاولة مرة أخرى.",
            },
        )
    except Exception as exc:
        print(f"❌ [/triage-symptoms/ ERROR] {exc}")  # full detail server-side only
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "حدث خطأ داخلي أثناء معالجة الشكوى. يرجى المحاولة مرة أخرى.",
            },
        )

    # ── Early exit: complaint is unrelated to dentistry ──────────────────────
    # The prompt instructs the model to stop after just these two fields when
    # out of scope, so this path never builds the full unified report (saves
    # output tokens and avoids mislabeling an unrelated complaint as "Healthy"
    # or any dental department).
    if data.get("is_out_of_scope"):
        return JSONResponse(
            status_code=200,
            content={
                "status": "out_of_scope",
                # Fixed, hand-written message instead of an LLM-generated one —
                # guarantees consistent, natural phrasing every time (the LLM's
                # own phrasing varied in quality call-to-call), avoids it
                # guessing a specific medical specialty that might be wrong,
                # and saves the output tokens it would've spent writing this.
                "message": OUT_OF_SCOPE_MESSAGE,
            },
        )

    # Build unified report (identical schema to /analyze/)
    is_emergency: bool = data.get("is_emergency", False)  # Always False per prompt
    extracted: dict = data.get("extracted_data", {})
    dept_eng: str = data.get("target_department_eng", "غير محدد")

    # ── Resolve display info from DIAGNOSIS_MAP (same source-of-truth as /analyze/) ──
    # This guarantees تصنيف_الحالة is the Arabic+English label, never a raw dept code.
    diag_info = _dept_to_diagnosis_info(dept_eng)

    # For Needs_Clarification, override the action with the LLM's patient-
    # friendly plan (it asks for more details), not the generic Healthy action.
    # NOTE: "Out_of_Domain" never reaches here — it's caught by the
    # is_out_of_scope short-circuit above before dept_eng is even read.
    is_special = dept_eng == "Needs_Clarification"
    action_text = (
        data.get("action_plan", diag_info["action"])
        if is_special
        else diag_info["action"]
    )

    # Use the LLM's patient_friendly_diagnosis for clinical note (keeps LLM nuance),
    # but fall back to DIAGNOSIS_MAP ai_diagnosis if the LLM returns nothing useful.
    llm_diagnosis_text = data.get("patient_friendly_diagnosis", "").strip()
    ai_diagnosis_text = (
        llm_diagnosis_text if llm_diagnosis_text else diag_info["ai_diagnosis"]
    )

    formatted_report = {
        "document_info": {
            "medical_file_number": f"DM-TXT-{int(time.time())}",
            "issue_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_source": "المنصة الذكية للفرز الطبي - DentMatch AI",
        },
        "symptoms_and_history": {
            "recorded_pain_duration": extracted.get("pain_duration", "غير محدد"),
            "chronic_diseases": extracted.get("chronic_conditions", "لا يوجد"),
        },
        "initial_medical_assessment": {
            # ── Unified display name — same format as /analyze/ ──
            # e.g. "Dental Caries - تسوس في الأسنان"  (never a raw dept code)
            "case_classification": diag_info["ar_name"],
            "ai_diagnosis": ai_diagnosis_text,
            "specialized_university_department": diag_info["dept_ar"],
            "case_priority_level": diag_info["priority"],
        },
        "care_and_referral_plan": {
            "next_steps": action_text,
        },
        "legal_disclaimer": (
            "هذا التقرير استرشادي صادر آلياً عن منظومة DentMatch AI، "
            "ولا يُغني عن الفحص السريري المباشر من طبيب الأسنان المختص."
        ),
        # ── Internal telemetry ── stripped from the downloadable report in app.py.
        # N/A for text-triage path; kept for schema parity with /analyze/.
        "_internal": {
            "confidence_score": None,
            "requires_human_review": False,
            "stages_passed": None,
            "processing_time_seconds": None,
            "disease_probabilities": {},
        },
        # LLM-specific metadata — useful for API consumers that need raw LLM output
        "_llm_meta": {
            "is_emergency": is_emergency,
            "complexity_level": data.get("complexity_level", "غير محدد"),
            "recommended_student_level": data.get(
                "recommended_student_level", "غير محدد"
            ),
            "target_department_eng": dept_eng,
            "target_department_ar": data.get("target_department_ar", "غير محدد"),
        },
    }

    if not request.include_internal:
        formatted_report.pop("_internal", None)
        formatted_report.pop("_llm_meta", None)

    return JSONResponse(status_code=200, content=formatted_report)
