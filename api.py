"""
==============================================================================
FILE: api.py
DESCRIPTION:
Production REST API for the Healthy Smile AI System using FastAPI.
This allows any external application (Web, Mobile, Desktop) to send an image
and receive a structured JSON medical report instantly.
==============================================================================
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# استدعاء المايسترو بتاعنا
from deployment.master_pipeline import DentalAI_System

# ==============================================================================
# 1. API INITIALIZATION & CONFIGURATION
# ==============================================================================
app = FastAPI(
    title="🦷 Healthy Smile AI Engine",
    description="Enterprise REST API for 3-Stage Deep Learning Dental Diagnostics.",
    version="1.0.0",
)

# السماح لأي موقع ويب أو تطبيق موبايل إنه يكلم الـ API ده (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل موديلات الذكاء الاصطناعي مرة واحدة عند تشغيل السيرفر
print("⏳ Initializing AI Models for API...")
clinic = DentalAI_System()

# ==============================================================================
# 2. API ENDPOINTS
# ==============================================================================


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "Online",
        "message": "Welcome to Healthy Smile AI Core. Visit /docs for the Swagger Interface.",
    }


@app.post("/analyze/")
async def analyze_dental_image(file: UploadFile = File(...)):
    """
    Core Endpoint: Accepts a dental image and returns a detailed AI diagnosis.
    """
    try:
        # حفظ الصورة المرفوعة في ملف مؤقت عشان المايسترو يقدر يقرأها
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        # تمرير الصورة لـ AI Pipeline
        report = clinic.analyze_patient(tmp_path)

        # مسح الصورة المؤقتة بعد الانتهاء لحماية مساحة السيرفر
        os.remove(tmp_path)

        # إرجاع التقرير كـ JSON Response
        if report["status"] == "error":
            return JSONResponse(status_code=500, content=report)
        elif report["status"] == "rejected":
            return JSONResponse(status_code=400, content=report)
        else:
            return JSONResponse(status_code=200, content=report)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
