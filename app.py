"""
==============================================================================
FILE: app.py  ✦ DentMatch AI (Healthy Smile Core)
DESCRIPTION:
    Interactive Web Dashboard for the DentMatch AI System (Healthy Smile).
    Visually redesigned with a futuristic, cinematic aesthetic:
    glassmorphism, animated gradients, neon glow, premium typography,
    immersive hero section, and elite UI/UX.

ARCHITECTURE:
    Tab 1 — Clinical Diagnosis
        Section A : Image upload → POST /analyze/ → display report + Grad-CAM++
        Section B : Symptom text → POST /triage-symptoms/ → display report
    Tab 2 — AI Analytics
        Static evaluation figures (confusion matrix, ROC, t-SNE, etc.)
    Tab 3 — API Hub
        cURL / Python code snippets + Swagger link

AUTHOR:  Eng. Ahmed Ayman — AI & Data Science Engineer
VERSION: 1.0.0  ( DentMatch )
==============================================================================
"""

# ==============================================================================
# ZONE 1: IMPORTS
# ==============================================================================
import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from deployment.explainability import get_gradcampp_heatmap, overlay_heatmap_on_image

# ==============================================================================
# ZONE 2: PAGE CONFIG — Must be first Streamlit call
# ==============================================================================
st.set_page_config(
    page_title="DentMatch - Healthy Smile AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# ZONE 3: GLOBAL CSS — The full cinematic design system
# ==============================================================================
st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   DENTMATCH — ELITE DESIGN SYSTEM v4.0
   Typography: Inter Display (hero) + DM Sans (body) + JetBrains Mono (code)
   Rationale: Inter is the gold standard for premium SaaS (Linear, Vercel,
   Notion). DM Sans has optimal legibility at small sizes with friendly
   geometry. Both are zero-tension — no compressed letterforms.
═══════════════════════════════════════════════════════════════════════════ */

/* ── FONT IMPORTS ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Inter:wght@400;500;600;700;800;900&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── DESIGN TOKENS ─────────────────────────────────────────────────────── */
:root {
  /* Backgrounds */
  --bg-void:        #04070f;
  --bg-surface:     #070c18;
  --bg-card:        rgba(8, 14, 28, 0.88);
  --bg-glass:       rgba(255,255,255,0.028);
  --bg-glass-hover: rgba(255,255,255,0.055);

  /* Accent palette */
  --cyan:    #00d4ff;
  --teal:    #00e5c3;
  --rose:    #ff4d7e;
  --amber:   #f59e0b;
  --purple:  #a78bfa;
  --green:   #10b981;

  /* Glow shadows */
  --glow-cyan:   0 0 24px rgba(0,212,255,0.28), 0 0 64px rgba(0,212,255,0.10);
  --glow-teal:   0 0 24px rgba(0,229,195,0.28), 0 0 64px rgba(0,229,195,0.10);
  --glow-rose:   0 0 24px rgba(255,77,126,0.28), 0 0 64px rgba(255,77,126,0.10);

  /* Borders */
  --border-subtle: 1px solid rgba(255,255,255,0.06);
  --border-glass:  1px solid rgba(255,255,255,0.09);
  --border-cyan:   1px solid rgba(0,212,255,0.28);

  /* Text */
  --text-100: #f1f5ff;
  --text-200: #a0aec8;
  --text-300: #5a6a8a;
  --text-400: #334466;

  /* Radius */
  --r-sm:   10px;
  --r-md:   16px;
  --r-lg:   22px;
  --r-pill: 100px;

  /* Typography — NO tight letter-spacing on body */
  --font-hero:    'Inter', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-arabic:  'Cairo', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
}

/* ── GLOBAL RESET ──────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  background-color: var(--bg-void) !important;
  font-family: var(--font-body) !important;
  color: var(--text-100) !important;
  -webkit-font-smoothing: antialiased !important;
  text-rendering: optimizeLegibility !important;
}

/* ── HIDE STREAMLIT CHROME ─────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 2.5rem 5rem 2.5rem !important;
  max-width: 1520px !important;
}

/* ── HIDE SIDEBAR COMPLETELY ───────────────────────────────────────────── */
/* ── SIDEBAR ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#050a18 0%,#070d1e 100%) !important;
  border-right: 1px solid rgba(0,212,255,0.09) !important;
  box-shadow: 4px 0 40px rgba(0,0,0,0.7) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
button[data-testid="baseButton-headerNoPadding"] {
  visibility: visible !important; display: flex !important;
  background: rgba(0,212,255,0.07) !important;
  border: 1px solid rgba(0,212,255,0.20) !important;
  border-radius: 8px !important; color: #00d4ff !important;
}
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"] {
  visibility: visible !important; display: flex !important; opacity: 1 !important;
  background: rgba(5,10,24,0.95) !important;
  border: 1px solid rgba(0,212,255,0.28) !important;
  border-left: none !important; border-radius: 0 10px 10px 0 !important;
  box-shadow: 4px 0 20px rgba(0,212,255,0.18) !important;
  color: #00d4ff !important;
}

/* ── ATMOSPHERIC BACKGROUND ────────────────────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 70% 45% at 15% -5%,  rgba(0,212,255,0.07) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 85% 105%, rgba(167,139,250,0.06) 0%, transparent 65%),
    radial-gradient(ellipse 40% 55% at 55%  55%, rgba(0,229,195,0.025) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
  animation: bgBreathe 16s ease-in-out infinite alternate;
}
@keyframes bgBreathe {
  0%   { opacity: 0.65; transform: scale(1);    }
  100% { opacity: 1;    transform: scale(1.04); }
}

/* Subtle dot-grid */
.stApp::after {
  content: '';
  position: fixed; inset: 0;
  background-image: radial-gradient(rgba(0,212,255,0.07) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none; z-index: 0;
}

/* ── TABS — completely kill default underline indicator ────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: rgba(7,12,24,0.95) !important;
  border-radius: var(--r-md);
  padding: 5px;
  border: var(--border-subtle);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  margin-bottom: 32px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
/* Kill the default BaseUI ink-bar / underline */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [role="tablist"] > div:last-child {
  display: none !important;
  background: transparent !important;
  height: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  height: 44px;
  background: transparent !important;
  border-radius: var(--r-sm) !important;
  padding: 0 22px !important;
  color: var(--text-300) !important;
  font-weight: 600 !important;
  font-family: var(--font-body) !important;
  font-size: 13.5px !important;
  letter-spacing: 0.1px !important;
  border: none !important;
  transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
  border-bottom: none !important;
  box-shadow: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-100) !important;
  background: rgba(255,255,255,0.05) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.14) 0%, rgba(0,229,195,0.09) 100%) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,212,255,0.22) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.18), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

/* ── BUTTONS ───────────────────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--font-body) !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.2px !important;
  border-radius: var(--r-sm) !important;
  transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
  height: 44px !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00d4ff 0%, #00c4e8 40%, #00e5c3 100%) !important;
  color: #03080f !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(0,212,255,0.32), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 8px 28px rgba(0,212,255,0.50) !important;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0) scale(0.99) !important;
}

/* ── DOWNLOAD BUTTON ───────────────────────────────────────────────────── */
.stDownloadButton > button {
  font-family: var(--font-body) !important;
  font-weight: 700 !important;
  border-radius: var(--r-sm) !important;
  height: 44px !important;
}
.stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00d4ff 0%, #00e5c3 100%) !important;
  color: #03080f !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(0,212,255,0.28) !important;
  transition: all 0.18s ease !important;
}
.stDownloadButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(0,212,255,0.48) !important;
}

/* ── METRICS — elevated glass cards ───────────────────────────────────── */
[data-testid="metric-container"] {
  background: linear-gradient(145deg, rgba(8,14,28,0.92), rgba(5,10,20,0.96)) !important;
  border: var(--border-subtle) !important;
  border-radius: var(--r-lg) !important;
  padding: 22px 26px !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(0,212,255,0.6) 50%, transparent 100%);
}
[data-testid="metric-container"]:hover {
  border-color: rgba(0,212,255,0.18) !important;
  box-shadow: var(--glow-cyan), 0 8px 32px rgba(0,0,0,0.4) !important;
  transform: translateY(-3px) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--text-300) !important;
  font-family: var(--font-body) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;  /* ← reduced from 1.5px — breathes better */
}
[data-testid="stMetricValue"] {
  color: var(--text-100) !important;
  font-family: var(--font-hero) !important;
  font-size: 26px !important;
  font-weight: 800 !important;
  letter-spacing: -0.5px !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-body) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}

/* ── INPUTS & TEXTAREAS ────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
  background: rgba(7,12,24,0.85) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-100) !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: rgba(0,212,255,0.38) !important;
  box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
  outline: none !important;
}

/* ── SELECT / DROPDOWN ─────────────────────────────────────────────────── */
[data-baseweb="select"] > div {
  background: rgba(7,12,24,0.85) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: var(--r-sm) !important;
  color: var(--text-100) !important;
  font-family: var(--font-body) !important;
  transition: border-color 0.18s ease !important;
}
[data-baseweb="select"] > div:focus-within {
  border-color: rgba(0,212,255,0.35) !important;
  box-shadow: 0 0 0 3px rgba(0,212,255,0.08) !important;
}
/* Dropdown menu panel */
[data-baseweb="menu"] {
  background: #07101f !important;
  border: var(--border-glass) !important;
  border-radius: var(--r-md) !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.7) !important;
  overflow: hidden !important;
}
[data-baseweb="menu"] li {
  color: var(--text-200) !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
  transition: background 0.15s ease !important;
}
[data-baseweb="menu"] li:hover {
  background: rgba(0,212,255,0.09) !important;
  color: var(--cyan) !important;
}
[data-baseweb="menu"] [aria-selected="true"] {
  background: rgba(0,212,255,0.12) !important;
  color: var(--cyan) !important;
}

/* ── FILE UPLOADER ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] > div > div {
  background: rgba(7,12,24,0.7) !important;
  border: 1.5px dashed rgba(0,212,255,0.22) !important;
  border-radius: var(--r-md) !important;
  transition: border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="stFileUploader"] > div > div:hover {
  border-color: rgba(0,212,255,0.5) !important;
  background: rgba(0,212,255,0.035) !important;
}
[data-testid="stFileUploader"] * {
  font-family: var(--font-body) !important;
}

/* ── CODE BLOCKS ───────────────────────────────────────────────────────── */
.stCode, pre, code {
  background: rgba(3,6,16,0.97) !important;
  border: var(--border-subtle) !important;
  border-radius: var(--r-md) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
  border-left: 2px solid rgba(0,212,255,0.4) !important;
}

/* ── ALERTS ────────────────────────────────────────────────────────────── */
.stAlert {
  border-radius: var(--r-md) !important;
  border: var(--border-subtle) !important;
  backdrop-filter: blur(12px) !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
}

/* ── SPINNER ───────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── SCROLLBAR ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(0,212,255,0.18);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.38); }

/* ── ARABIC TEXT ───────────────────────────────────────────────────────── */
[dir="rtl"], [dir="rtl"] * { font-family: var(--font-arabic) !important; }

/* ── UTILITY CLASSES ───────────────────────────────────────────────────── */
.eyebrow {
  /* Replaces: text-transform uppercase + tight letter-spacing on JetBrains */
  font-family: var(--font-body) !important;
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;   /* ← 1.2 not 2.5 — readable, not barcode */
  color: var(--text-400);
}
.report-section-label {
  font-family: var(--font-body);
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--text-400);
  margin-bottom: 8px;
}
.diagnosis-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 20px;
  border-radius: var(--r-pill);
  font-size: 14.5px;
  font-weight: 700;
  letter-spacing: 0.1px;
  margin-bottom: 6px;
  font-family: var(--font-body);
}
.divider-thin {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.05);
  margin: 20px 0;
}

/* ── ANIMATIONS ────────────────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes bgBreathe {
  0%   { opacity: 0.65; transform: scale(1); }
  100% { opacity: 1;    transform: scale(1.04); }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
@keyframes dotPulse {
  0%, 100% { opacity: 1;   box-shadow: 0 0 6px rgba(16,185,129,0.9); }
  50%       { opacity: 0.5; box-shadow: 0 0 14px rgba(16,185,129,0.4); }
}
.animate-in { animation: fadeUp 0.55s cubic-bezier(0.4,0,0.2,1) both; }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# ZONE 3B: SIDEBAR JS FIX — must use components.html (st.markdown strips <script>)
# ==============================================================================
components.html(
    """
<script>
(function () {
function fixBtn() {
var w = window.parent.document.querySelector(
'[data-testid="stSidebarCollapsedControl"]'
);
if (!w) return;
w.style.cssText = "visibility:visible!important;opacity:1!important;" +
"display:flex!important;align-items:center!important;" +
"position:fixed!important;top:50%!important;left:0!important;" +
"transform:translateY(-50%)!important;z-index:99999!important;" +
"pointer-events:all!important;background:transparent!important;" +
"border:none!important;box-shadow:none!important;";
var btn = w.querySelector("button");
if (!btn) return;
btn.style.cssText = "visibility:visible!important;opacity:1!important;" +
"background:rgba(5,8,16,0.95)!important;" +
"border:1px solid rgba(0,212,255,0.4)!important;" +
"border-left:none!important;border-radius:0 12px 12px 0!important;" +
"box-shadow:4px 0 24px rgba(0,212,255,0.25)!important;" +
"color:#00d4ff!important;padding:16px 8px!important;" +
"cursor:pointer!important;pointer-events:all!important;";
}
fixBtn();
setInterval(fixBtn, 400);
})();
</script>
""",
    height=0,
)

# ==============================================================================
# ZONE 4: CONSTANTS
# ==============================================================================
# Defaults to the deployed public API on Hugging Face — this is what's
# actually reachable, since the Space's Docker image runs FastAPI directly
# on the public port (colleagues call /analyze/, /triage-symptoms/, /docs
# on this URL). Override with an environment variable only if you're running
# BOTH api.py and app.py together on your own machine via start.sh, e.g.:
#   set API_BASE_URL=http://127.0.0.1:8000
API_BASE_URL: str = os.environ.get(
    "API_BASE_URL", "https://eng-ahmedayman10-healthysmile-ai.hf.space"
)
ANALYZE_ENDPOINT: str = f"{API_BASE_URL}/analyze/"
TRIAGE_ENDPOINT: str = f"{API_BASE_URL}/triage-symptoms/"


# ==============================================================================
# ZONE 5: CACHED RESOURCE LOADERS
# ==============================================================================
# NOTE: DentalAI_System is NOT loaded here — the dashboard calls the FastAPI
# backend via HTTP (requests). Loading it here would waste RAM by booting all
# three AI models a second time. The only model loaded client-side is the
# Stage 3 specialist, used exclusively for the Grad-CAM++ overlay.
@st.cache_resource
def load_cam_model():
    import tensorflow as tf
    from deployment.master_pipeline import STAGE3_PATH

    return tf.keras.models.load_model(STAGE3_PATH, compile=False)


# ==============================================================================
# ZONE 5B: REAL-TIME API HEALTH CHECK
# Pings the FastAPI /  health endpoint to determine live status.
# Cached for 10 seconds (ttl) so it doesn't fire on every re-render.
# ==============================================================================
@st.cache_data(ttl=10)
def _check_api_status() -> bool:
    """Return True if the FastAPI backend is reachable, False otherwise."""
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ==============================================================================
# ZONE 6: SHARED REPORT RENDERER
# ==============================================================================
def _render_report(
    report: dict,
    pain_duration: str = "",
    chronic_disease: str = "",
) -> None:
    assessment = report.get("initial_medical_assessment", {})
    history = report.get("symptoms_and_history", {})
    care_plan = report.get("care_and_referral_plan", {})
    doc_info = report.get("document_info", {})
    llm_meta = report.get("_llm_meta", {})
    internal = report.get("_internal", {})
    probs = internal.get("disease_probabilities", {})

    diagnosis = assessment.get("case_classification", "")
    ai_text = assessment.get("ai_diagnosis", "")
    dept_ar = assessment.get("specialized_university_department", "")
    priority = assessment.get("case_priority_level", "")
    action = care_plan.get("next_steps", "")
    disclaimer = report.get("legal_disclaimer", "")
    file_no = doc_info.get("medical_file_number", "—")
    issued_at = doc_info.get("issue_date", "—")

    display_pain = history.get("recorded_pain_duration", pain_duration or "غير محدد")
    display_chronic = history.get("chronic_diseases", chronic_disease or "لا يوجد")

    dept_eng_llm = llm_meta.get("target_department_eng", "")
    is_out_of_domain = dept_eng_llm == "Out_of_Domain"
    is_needs_clarif = dept_eng_llm == "Needs_Clarification"
    is_healthy = diagnosis.startswith("Healthy") and not (
        is_needs_clarif or is_out_of_domain
    )

    # Stage 3 is a closed 5-class classifier — it has no "other disease" option,
    # so it cannot literally say an image is "out of scope". Instead, when its
    # top-class confidence falls below master_pipeline.py's SAFETY_THRESHOLD,
    # it sets requires_human_review=True. Previously this flag was computed
    # but never shown to the patient — surfaced here as a clear caveat instead
    # of silently presenting a low-confidence guess as a certain diagnosis.
    requires_review = bool(internal.get("requires_human_review", False))

    if requires_review:
        ai_text = (
            "⚠️ الصورة دي مش واضح إنها بتطابق حالة من الحالات الخمس اللي النظام "
            "متدرب عليها (تسوس، نقص أسنان، تقرح فم، التهاب لثة، تغيّر لون الأسنان). "
            "يُرجى مراجعة طبيب أسنان مباشرة للتقييم الدقيق، والتشخيص التالي "
            "للاسترشاد فقط وليس تأكيدًا:<br><br>" + ai_text
        )

    if is_needs_clarif:
        accent = "#ffbb00"
        badge_bg = "rgba(255, 187, 0, 0.08)"
        glow_color = "rgba(255, 187, 0, 0.25)"
        icon = "◈"
        header_label = "NEEDS CLARIFICATION"
        stripe_grad = (
            "linear-gradient(135deg, rgba(255,187,0,0.15), rgba(255,187,0,0.03))"
        )
    elif is_out_of_domain:
        accent = "#8899bb"
        badge_bg = "rgba(136, 153, 187, 0.08)"
        glow_color = "rgba(136, 153, 187, 0.25)"
        icon = "○"
        header_label = "OUT OF SCOPE — DENTAL ONLY"
        stripe_grad = (
            "linear-gradient(135deg, rgba(136,153,187,0.15), rgba(136,153,187,0.03))"
        )
    elif requires_review:
        accent = "#ffbb00"
        badge_bg = "rgba(255, 187, 0, 0.08)"
        glow_color = "rgba(255, 187, 0, 0.25)"
        icon = "◈"
        header_label = "LOW CONFIDENCE — HUMAN REVIEW NEEDED"
        stripe_grad = (
            "linear-gradient(135deg, rgba(255,187,0,0.15), rgba(255,187,0,0.03))"
        )
    elif is_healthy:
        accent = "#00e5c3"
        badge_bg = "rgba(0, 229, 195, 0.08)"
        glow_color = "rgba(0, 229, 195, 0.25)"
        icon = "✦"
        header_label = "CLINICAL DIAGNOSIS"
        stripe_grad = (
            "linear-gradient(135deg, rgba(0,229,195,0.12), rgba(0,229,195,0.02))"
        )
    else:
        accent = "#ff4d7e"
        badge_bg = "rgba(255, 77, 126, 0.08)"
        glow_color = "rgba(255, 77, 126, 0.25)"
        icon = "⬡"
        header_label = "CLINICAL DIAGNOSIS"
        stripe_grad = (
            "linear-gradient(135deg, rgba(255,77,126,0.12), rgba(255,77,126,0.02))"
        )

    st.markdown(
        f"""
<div class="animate-in" style="
background: linear-gradient(160deg, rgba(8,13,26,0.95) 0%, rgba(6,10,20,0.98) 100%);
padding: 32px 36px;
border-radius: 24px;
border: 1px solid rgba(255,255,255,0.07);
box-shadow: 0 24px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04), inset 0 1px 0 rgba(255,255,255,0.06);
margin-bottom: 28px;
position: relative;
overflow: hidden;
backdrop-filter: blur(20px);
-webkit-backdrop-filter: blur(20px);
">
<!-- Decorative glow accent -->
<div style="
position: absolute; top: 0; left: 0; right: 0; height: 3px;
background: linear-gradient(90deg, transparent 0%, {accent} 40%, {accent} 60%, transparent 100%);
box-shadow: 0 0 20px {glow_color};
"></div>

<!-- Corner accent blob -->
<div style="
position: absolute; top: -40px; right: -40px;
width: 140px; height: 140px;
background: radial-gradient(circle, {glow_color} 0%, transparent 70%);
pointer-events: none;
"></div>

<!-- Header row -->
<div style="
display: flex; justify-content: space-between; align-items: center;
border-bottom: 1px solid rgba(255,255,255,0.06);
padding-bottom: 18px; margin-bottom: 24px;
">
<div style="display: flex; align-items: center; gap: 10px;">
    <span style="color: {accent}; font-size: 20px;">{icon}</span>
    <span style="
    color: {accent}; font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 3px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    ">{header_label}</span>
</div>
<div style="
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 6px 14px; border-radius: 100px;
    font-size: 11px; color: #556688;
    font-family: 'JetBrains Mono', monospace;
">
    {file_no} &nbsp;·&nbsp; {issued_at}
</div>
</div>

<!-- Diagnosis Result -->
<div class="report-section-label">Diagnosis Result</div>
<div style="margin-bottom: 24px;">
<span class="diagnosis-badge" style="
    background: {badge_bg};
    color: {accent};
    border: 1px solid {accent}44;
    box-shadow: 0 0 20px {glow_color};
    font-family: 'Plus Jakarta Sans', sans-serif;
">
    {icon} &nbsp; {diagnosis}
</span>
</div>

<!-- Clinical Assessment -->
<div class="report-section-label">AI Clinical Assessment</div>
<div dir="rtl" style="
text-align: right; font-size: 16px; line-height: 1.9;
color: #c5d5ee; margin-bottom: 24px;
font-family: 'Cairo', sans-serif;
background: rgba(255,255,255,0.025);
border: 1px solid rgba(255,255,255,0.05);
border-radius: 14px;
padding: 18px 22px;
">
{ai_text}
</div>

<hr class="divider-thin"/>

<!-- Patient history chips -->
<div dir="rtl" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 24px;">
<div style="
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.15);
    color: #7ab8d4; padding: 7px 18px;
    border-radius: 100px; font-size: 13px;
    font-family: 'Cairo', sans-serif;
">
    ⏱ مدة الألم: <strong style="color: #a8d8f0;">{display_pain}</strong>
</div>
<div style="
    background: rgba(168,85,247,0.05);
    border: 1px solid rgba(168,85,247,0.15);
    color: #b094d4; padding: 7px 18px;
    border-radius: 100px; font-size: 13px;
    font-family: 'Cairo', sans-serif;
">
    🩸 أمراض مزمنة: <strong style="color: #cdb4f0;">{display_chronic}</strong>
</div>
</div>

<!-- Target Department -->
<div class="report-section-label">Target Department</div>
<div dir="rtl" style="
text-align: right; font-size: 18px; font-weight: 700;
margin-bottom: 24px;
font-family: 'Cairo', sans-serif;
">
<span style="color: #00e5c3;">{dept_ar}</span>
<span style="color: #445577; font-size: 13px; font-weight: 400;"> — {priority}</span>
</div>

<hr class="divider-thin"/>

<!-- Action Plan -->
<div class="report-section-label">Action Plan</div>
<div dir="rtl" style="
text-align: right; font-size: 15px; line-height: 1.9;
color: #8899bb;
font-family: 'Cairo', sans-serif;
">
{action}
</div>

<!-- Legal disclaimer -->
<div style="
margin-top: 28px; padding-top: 18px;
border-top: 1px solid rgba(255,255,255,0.04);
">
<div dir="rtl" style="
    font-size: 11px; color: #334455;
    text-align: right; line-height: 1.8;
    font-family: 'Cairo', sans-serif;
">
    ⚖ {disclaimer}
</div>
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    # ── Disease probability bar chart ──
    if probs:
        st.markdown(
            """
<div style="font-size:11px;font-weight:700;text-transform:uppercase;
        letter-spacing:2.5px;color:#445577;margin-bottom:12px;
        font-family:'Plus Jakarta Sans',sans-serif;">
Disease Probability Distribution
</div>
""",
            unsafe_allow_html=True,
        )
        chart_data = pd.DataFrame(
            {
                "Disease": list(probs.keys()),
                "Probability (%)": list(probs.values()),
            }
        ).sort_values(by="Probability (%)", ascending=True)
        st.bar_chart(chart_data.set_index("Disease"), horizontal=True)

    # ── Downloadable report ──
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    file_prefix = "Clinical" if probs else "Triage"
    report_for_download = {
        k: v for k, v in report.items() if k not in ("_internal", "_llm_meta")
    }
    st.download_button(
        label="↓  تحميل التقرير الطبي  (JSON)",
        file_name=f"DentMatch_{file_prefix}_Report_{int(time.time())}.json",
        mime="application/json",
        data=json.dumps(report_for_download, indent=4, ensure_ascii=False),
        type="primary",
        on_click=lambda: st.toast("تم تحميل التقرير بنجاح ✦", icon="📥"),
        use_container_width=True,
    )


# ==============================================================================
# ZONE 7: SIDEBAR — native Streamlit components (guaranteed to render)
# ==============================================================================
with st.sidebar:

    # ── TOP: inject sidebar CSS via st.markdown (styles apply to sidebar scope)
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800;900&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Inter', sans-serif !important;
}
/* Status dots animation */
@keyframes sdot {
    0%,100% { opacity:1; box-shadow:0 0 8px rgba(16,185,129,0.9); }
    50%      { opacity:0.5; box-shadow:0 0 16px rgba(16,185,129,0.4); }
}
.sdot { animation: sdot 2s ease-in-out infinite; }
</style>
""",
        unsafe_allow_html=True,
    )

    # ── LOGO BLOCK ──
    st.markdown(
        """
<div style="padding:28px 18px 20px 18px;border-bottom:1px solid rgba(0,212,255,0.08);margin-bottom:4px;">
  <div style="width:46px;height:46px;background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(0,229,195,0.12));border:1px solid rgba(0,212,255,0.32);border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:14px;box-shadow:0 0 20px rgba(0,212,255,0.20);">&#129463;</div>
  <div style="font-size:18px;font-weight:800;font-family:'Inter',sans-serif;letter-spacing:-0.3px;background:linear-gradient(135deg,#00d4ff,#00e5c3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2;margin-bottom:5px;">DentMatch</div>
  <div style="font-size:9px;color:#2d4466;text-transform:uppercase;letter-spacing:1.8px;font-family:'DM Sans',sans-serif;font-weight:600;">Core Engine v2.2 · Healthy Smile AI</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── SYSTEM STATUS (real-time health check) ──
    _api_online = _check_api_status()
    _api_color = "#10b981" if _api_online else "#ef4444"
    _api_label = "Active" if _api_online else "Offline"
    _api_glow = "rgba(16,185,129,0.9)" if _api_online else "rgba(239,68,68,0.9)"
    _api_bg = "rgba(16,185,129,0.06)" if _api_online else "rgba(239,68,68,0.06)"
    _api_border = "rgba(16,185,129,0.14)" if _api_online else "rgba(239,68,68,0.14)"
    # Vision CNN & NLP are loaded by the API server — they follow its status
    _model_color = _api_color
    _model_label = "Online" if _api_online else "Offline"
    _model_glow = _api_glow
    _model_bg = _api_bg
    _model_border = _api_border

    st.markdown(
        f"""
<div style="padding:18px 18px 4px 18px;">
  <div style="font-size:9px;color:#2d4466;text-transform:uppercase;letter-spacing:1.8px;font-weight:700;font-family:'DM Sans',sans-serif;margin-bottom:12px;">⬡ System Status</div>
  <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 14px;background:{_model_bg};border:1px solid {_model_border};border-radius:10px;margin-bottom:6px;">
    <span style="font-size:12px;color:#8ba8c8;font-family:'DM Sans',sans-serif;">Vision CNN</span>
    <span style="display:flex;align-items:center;gap:5px;">
      <span class="sdot" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{_model_color};box-shadow:0 0 8px {_model_glow};"></span>
      <span style="font-size:11px;color:{_model_color};font-family:'DM Sans',sans-serif;font-weight:600;">{_model_label}</span>
    </span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 14px;background:{_model_bg};border:1px solid {_model_border};border-radius:10px;margin-bottom:6px;">
    <span style="font-size:12px;color:#8ba8c8;font-family:'DM Sans',sans-serif;">NLP &amp; LLM</span>
    <span style="display:flex;align-items:center;gap:5px;">
      <span class="sdot" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{_model_color};box-shadow:0 0 8px {_model_glow};"></span>
      <span style="font-size:11px;color:{_model_color};font-family:'DM Sans',sans-serif;font-weight:600;">{_model_label}</span>
    </span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 14px;background:{_api_bg};border:1px solid {_api_border};border-radius:10px;margin-bottom:6px;">
    <span style="font-size:12px;color:#8ba8c8;font-family:'DM Sans',sans-serif;">API Gateway</span>
    <span style="display:flex;align-items:center;gap:5px;">
      <span class="sdot" style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{_api_color};box-shadow:0 0 8px {_api_glow};"></span>
      <span style="font-size:11px;color:{_api_color};font-family:'DM Sans',sans-serif;font-weight:600;">{_api_label}</span>
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── AUTHOR CARD ──
    st.markdown(
        """
<div style="margin:16px 14px;background:linear-gradient(135deg,rgba(0,212,255,0.07),rgba(0,229,195,0.04));border:1px solid rgba(0,212,255,0.15);border-radius:14px;padding:16px 18px;">
  <div style="font-size:9px;color:#00d4ff;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-family:'DM Sans',sans-serif;">✦ AI Engineer</div>
  <div style="font-size:16px;font-weight:800;font-family:'Inter',sans-serif;letter-spacing:-0.2px;background:linear-gradient(135deg,#e8f4ff,#a8d4f0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.25;margin-bottom:4px;">Eng. Ahmed Ayman</div>
  <div style="font-size:11px;color:#445577;font-family:'DM Sans',sans-serif;">AI &amp; Data Science Engineer</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── DIVIDER ──
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,212,255,0.07);margin:4px 14px 16px 14px;'>",
        unsafe_allow_html=True,
    )

    # ── TECH STACK ──
    st.markdown(
        """
<div style="padding:0 14px 8px 14px;">
  <div style="font-size:9px;color:#2d4466;text-transform:uppercase;letter-spacing:1.8px;font-weight:700;font-family:'DM Sans',sans-serif;margin-bottom:12px;">◈ Core Technology</div>
</div>
""",
        unsafe_allow_html=True,
    )

    _tech = [
        ("TensorFlow", "#ff6f42"),
        ("FastAPI", "#00d4ff"),
        ("Gemini 2.5", "#a78bfa"),
        ("EfficientNetB4", "#00e5c3"),
        ("Grad-CAM++", "#f59e0b"),
        ("OpenCV", "#10b981"),
        ("Streamlit", "#ff4b6e"),
        ("Clean Arch", "#6b8cba"),
    ]
    _tech_html = "".join(
        f'<div style="display:flex;align-items:center;gap:9px;padding:8px 14px;'
        f"margin:0 14px 5px 14px;background:rgba(255,255,255,0.025);"
        f'border:1px solid rgba(255,255,255,0.05);border-radius:10px;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{col};'
        f'box-shadow:0 0 8px {col}99;display:inline-block;flex-shrink:0;"></span>'
        f'<span style="font-size:12px;color:#8ba8c8;font-family:DM Sans,sans-serif;'
        f'font-weight:500;">{name}</span></div>'
        for name, col in _tech
    )
    st.markdown(_tech_html, unsafe_allow_html=True)

    # ── BOTTOM SPACER ──
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ==============================================================================
# ZONE 8: HERO HEADER
# ==============================================================================
st.markdown(
    """
<div class="animate-in" style="
padding: 28px 0 28px 0;
position: relative;
">
<!-- Decorative left border line -->
<div style="
    position: absolute; left: -2rem; top: 52px; bottom: 40px;
    width: 2px;
    background: linear-gradient(180deg, transparent, #00d4ff55, transparent);
"></div>

<div style="
    font-size: 11px; color: #00d4ff;
    font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; gap: 10px;
">
    <span style="
        display:inline-block;width:6px;height:6px;border-radius:50%;
        background:#00d4ff;box-shadow:0 0 10px rgba(0,212,255,0.8);
        animation: pulse-dot 2s ease-in-out infinite;
    "></span>
    Dental AI · Clinical Intelligence Platform
</div>

<h1 style="
    font-size: clamp(36px, 4.5vw, 60px);
    font-weight: 800;
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.05;
    margin: 0 0 16px 0;
    background: linear-gradient(135deg, #ffffff 20%, #e0f4ff 50%, #00d4ff 80%, #00e5c3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
">
    Healthy Smile AI
</h1>

<p style="
    font-size: 15.5px; color: #5d7099;
    max-width: 580px; line-height: 1.7;
    font-family: 'Plus Jakarta Sans', sans-serif;
    margin: 0;
">
    AI-powered dental diagnostics with Grad-CAM++ explainability,
    Arabic NLP triage, and clinical-grade reporting.
</p>
</div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# ZONE 9: HEADER METRICS
# ==============================================================================
hm1, hm2, hm3, hm4 = st.columns(4)
hm1.metric("AI Models Active", "3 Stages", "CNN Pipeline")
hm2.metric("API Latency", "< 1.5s", "-0.2s", delta_color="inverse")
hm3.metric("System Uptime", "99.99%", "Stable")
hm4.metric("Processing Mode", "In-Memory", "Zero Disk I/O")
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ==============================================================================
# ZONE 10: MAIN TABS
# ==============================================================================
tab_clinic, tab_analytics, tab_api = st.tabs(
    [
        "⬡  Clinical Diagnosis",
        "◈  AI Performance & Analytics",
        "✦  API Integration Hub",
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLINICAL DIAGNOSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab_clinic:
    col1, col2 = st.columns([1, 1.2], gap="large")

    is_valid_image: bool = False

    # ── LEFT COLUMN: upload + medical history ──
    with col1:
        # Upload panel
        st.markdown(
            """
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;margin-bottom:12px;
    font-family:'Plus Jakarta Sans',sans-serif;
">Patient Image Upload</div>
""",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop a dental X-Ray or clinical photo here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.markdown(
                    """
<div style="
    font-size:10px;color:#00d4ff;font-weight:600;
    text-transform:uppercase;letter-spacing:2px;
    margin:12px 0 8px;font-family:'JetBrains Mono',monospace;
">✓ Image Loaded</div>
""",
                    unsafe_allow_html=True,
                )
                st.image(image, use_container_width=True, clamp=True)
                is_valid_image = True
            except Exception:
                st.error("⚠️ الملف تالف أو ليس صورة. يرجى رفع ملف صورة سليم.")
                is_valid_image = False

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Medical history panel
        st.markdown(
            """
<div style="
background: rgba(255,255,255,0.02);
border: 1px solid rgba(255,255,255,0.05);
border-radius: 20px;
padding: 24px;
margin-bottom: 4px;
">
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;margin-bottom:18px;
    font-family:'Plus Jakarta Sans',sans-serif;
">Medical History</div>
""",
            unsafe_allow_html=True,
        )

        selected_disease = st.selectbox(
            "هل تعاني من أمراض مزمنة؟",
            [
                "لا يوجد",
                "مرض السكري (Diabetes)",
                "ضغط الدم (Hypertension)",
                "سيولة في الدم",
                "أخرى",
            ],
        )
        if selected_disease == "أخرى":
            custom_disease = st.text_input(
                "📝 يرجى كتابة اسم المرض المزمن:",
                placeholder="مثال: حساسية من البنسلين...",
            )
            chronic_disease = (
                custom_disease.strip() if custom_disease.strip() else "أخرى (غير محدد)"
            )
        else:
            chronic_disease = selected_disease

        pain_duration = st.selectbox(
            "⏱ مدة الألم؟",
            ["بدون ألم (فحص روتيني)", "أيام قليلة", "أسبوع إلى شهر", "أكثر من شهر"],
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN: AI report ──
    with col2:
        st.markdown(
            """
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;margin-bottom:12px;
    font-family:'Plus Jakarta Sans',sans-serif;
">AI Diagnostic Report</div>
""",
            unsafe_allow_html=True,
        )

        if uploaded_file is None:
            st.markdown(
                """
<div style="
background: rgba(255,255,255,0.02);
border: 1px solid rgba(255,255,255,0.05);
border-radius: 20px;
padding: 48px 36px;
text-align: center;
">
<div style="font-size:48px;margin-bottom:16px;opacity:0.3;">⬡</div>
<div style="
    color:#445577;font-size:14px;line-height:1.7;
    font-family:'Plus Jakarta Sans',sans-serif;
">
    Upload a dental image on the left<br>to generate an AI diagnostic report.
</div>
</div>
""",
                unsafe_allow_html=True,
            )
        elif not is_valid_image:
            pass
        else:
            if st.button(
                "⬡  Analyze with AI Doctor", use_container_width=True, type="primary"
            ):
                image_bytes = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                payload_data = {
                    "chronic_diseases": chronic_disease,
                    "pain_duration": pain_duration,
                }

                with st.spinner("Running DentMatch AI Pipeline..."):
                    try:
                        # include_internal is a Form field — must be in `data`, not `params`
                        payload_data["include_internal"] = "true"
                        res = requests.post(
                            ANALYZE_ENDPOINT,
                            files=files,
                            data=payload_data,
                            timeout=30,
                        )
                        if res.status_code == 200:
                            result = res.json()
                            if result.get("status") == "error":
                                st.error(f"❌ {result.get('message', 'Unknown error')}")
                                result = {}
                        elif res.status_code in (400, 413):
                            body = res.json()
                            st.warning(
                                f"⚠️ {body.get('message', 'الصورة غير مقبولة.')}\n\n📸 يرجى رفع صورة أسنان واضحة."
                            )
                            result = {}
                        elif res.status_code == 500:
                            body = res.json()
                            st.error(
                                f"❌ خطأ داخلي: {body.get('message', 'Unknown error')}"
                            )
                            result = {}
                        else:
                            st.error(f"❌ استجابة غير متوقعة (HTTP {res.status_code}).")
                            result = {}
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ Could not reach the backend API. Make sure `uvicorn api:app` is running."
                        )
                        result = {}
                    except requests.exceptions.Timeout:
                        st.error("❌ Request timed out.")
                        result = {}
                    except Exception as exc:
                        st.error(f"❌ Unexpected error: {exc}")
                        result = {}

                if result:
                    st.success("✦  AI Pipeline Execution Complete")
                    _render_report(result)

                    stages_passed = result.get("_internal", {}).get("stages_passed", 0)
                    diagnosis_key = result.get("initial_medical_assessment", {}).get(
                        "case_classification", ""
                    )
                    is_healthy = diagnosis_key.startswith("Healthy")

                    if stages_passed == 2 and not is_healthy:
                        st.markdown(
                            """
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;
    margin:28px 0 12px;font-family:'Plus Jakarta Sans',sans-serif;
">Explainable AI — Grad-CAM++ Heatmap</div>
<div style="font-size:13px;color:#556688;margin-bottom:16px;font-family:'Plus Jakarta Sans',sans-serif;">
    Red/yellow areas show the exact clinical features the AI used to reach its diagnosis.
</div>
""",
                            unsafe_allow_html=True,
                        )
                        with st.spinner("Generating Explainable AI scan..."):
                            try:
                                from tensorflow.keras.preprocessing import (
                                    image as keras_image,
                                )

                                cam_model = load_cam_model()
                                original_pil = Image.open(
                                    io.BytesIO(image_bytes)
                                ).convert("RGB")
                                img_resized = original_pil.resize((224, 224))
                                img_array = np.expand_dims(
                                    keras_image.img_to_array(img_resized), axis=0
                                )
                                heatmap = get_gradcampp_heatmap(img_array, cam_model)
                                if heatmap is not None and np.max(heatmap) > 0:
                                    cam_image_pil = overlay_heatmap_on_image(
                                        original_pil, heatmap, alpha=0.55
                                    )
                                else:
                                    cam_image_pil = original_pil
                                    st.warning(
                                        "⚠️ Grad-CAM++ could not generate a heatmap. Showing original."
                                    )
                                hc1, hc2 = st.columns(2)
                                hc1.image(
                                    original_pil,
                                    caption="Original (AI Input)",
                                    use_container_width=True,
                                )
                                hc2.image(
                                    cam_image_pil,
                                    caption="Disease Heatmap (Grad-CAM++)",
                                    use_container_width=True,
                                )
                            except Exception as cam_err:
                                st.warning(
                                    f"⚠️ Could not generate Grad-CAM++ heatmap: {cam_err}"
                                )

    # ── SYMPTOM TRIAGE SECTION ──
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div style="
background: linear-gradient(135deg, rgba(168,85,247,0.06), rgba(0,212,255,0.04));
border: 1px solid rgba(168,85,247,0.12);
border-radius: 24px;
padding: 32px 36px;
margin-top: 8px;
">
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#a855f766;margin-bottom:10px;
    font-family:'Plus Jakarta Sans',sans-serif;
">AI Patient Triage</div>
<div style="
    font-size:22px;font-weight:800;
    font-family:'Syne',sans-serif;
    letter-spacing:-0.5px;
    background:linear-gradient(135deg,#e0d0ff,#a855f7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
    margin-bottom:8px;
">Symptom Analyzer</div>
<div style="font-size:14px;color:#667799;font-family:'Plus Jakarta Sans',sans-serif;">
    Describe the patient's complaint in Arabic — the AI will triage it to the correct university department.
</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    user_symptoms = st.text_area(
        "Patient Complaint (Arabic)",
        placeholder="مثال: عندي ألم شديد في ضرسي بقاله 3 أيام ومش عارف أنام، مع العلم إني مريض سكر...",
        max_chars=400,
        height=110,
        label_visibility="collapsed",
    )
    st.caption(
        "💡 يفضل تذكر **مدة الألم** وهل بتعاني من أي **أمراض مزمنة** لنتيجة أدق."
    )

    if st.button(
        "◈  Analyze Complaint & Triage", type="primary", use_container_width=True
    ):
        if not user_symptoms:
            st.warning("⚠️ Please type the patient's complaint first.")
        elif len(user_symptoms.strip()) < 10:
            st.warning(
                "⚠️ الشكوى قصيرة جداً! يرجى كتابة تفاصيل أكثر (10 حروف على الأقل)."
            )
        else:
            with st.spinner("Analysing patient complaint..."):
                try:
                    res = requests.post(
                        TRIAGE_ENDPOINT,
                        json={
                            "symptoms_description": user_symptoms,
                            "include_internal": True,
                        },
                        timeout=30,
                    )
                    if res.status_code == 200:
                        triage_data = res.json()
                        if triage_data.get("status") == "error":
                            st.error(
                                f"❌ {triage_data.get('message', 'Unknown error')}"
                            )
                            triage_data = {}
                    elif res.status_code == 422:
                        body = res.json()
                        detail = body.get("detail", "")
                        if isinstance(detail, list):
                            detail = detail[0].get("msg", "خطأ في التحقق من البيانات.")
                        st.warning(f"⚠️ {detail}")
                        triage_data = {}
                    elif res.status_code == 400:
                        body = res.json()
                        st.warning(f"⚠️ {body.get('message', 'الشكوى قصيرة جداً.')}")
                        triage_data = {}
                    else:
                        st.error(f"❌ استجابة غير متوقعة (HTTP {res.status_code}).")
                        triage_data = {}
                except requests.exceptions.ConnectionError:
                    st.error("❌ Failed to connect to the FastAPI backend.")
                    triage_data = {}
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out.")
                    triage_data = {}
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")
                    triage_data = {}

            if triage_data:
                # New minimal response for out-of-scope complaints — api.py
                # skips generating the full report entirely for these (saves
                # tokens), so render it as a plain info message, not a card.
                if triage_data.get("status") == "out_of_scope":
                    st.info(
                        f"🦷 {triage_data.get('message', 'الشكوى غير متعلقة بطب الأسنان.')}"
                    )
                else:
                    llm_meta = triage_data.get("_llm_meta", {})
                    dept_eng = llm_meta.get("target_department_eng", "")
                    is_emergency = llm_meta.get("is_emergency", False)

                    if is_emergency:
                        st.error("🚨 حالة طوارئ طبية!")
                    elif dept_eng == "Needs_Clarification":
                        st.warning("⚠️ محتاجين تفاصيل أكتر.")
                    else:
                        st.success("✦  تم توجيه الحالة بنجاح")

                    _render_report(triage_data)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI PERFORMANCE & ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown(
        """
<div class="animate-in" style="margin-bottom:32px;">
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;margin-bottom:10px;
    font-family:'Plus Jakarta Sans',sans-serif;
">Model Evaluation</div>
<div style="
    font-size:28px;font-weight:800;
    font-family:'Syne',sans-serif;letter-spacing:-0.5px;
    color:#f0f6ff;margin-bottom:10px;
">AI Performance & Analytics</div>
<div style="
    font-size:14px;color:#556688;max-width:620px;line-height:1.7;
    font-family:'Plus Jakarta Sans',sans-serif;
">
    Rigorous evaluation of the Stage-3 Clinical Engine (EfficientNetB4)
    on the held-out test dataset, ensuring clinical safety and reliability.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        met1, met2 = st.columns(2, gap="large")
        with met1:
            st.markdown(
                "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:10px;font-family:\"Plus Jakarta Sans\",sans-serif;'>Confusion Matrix · Stage 3</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:rgba(4,7,15,0.98);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:10px;overflow:hidden;">',
                unsafe_allow_html=True,
            )
            st.image("reports/figures/confusion_matrix.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with met2:
            st.markdown(
                "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:10px;font-family:\"Plus Jakarta Sans\",sans-serif;'>ROC Curves</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:rgba(4,7,15,0.98);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:10px;overflow:hidden;">',
                unsafe_allow_html=True,
            )
            st.image("reports/figures/roc_curves.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:10px;font-family:\"Plus Jakarta Sans\",sans-serif;'>t-SNE Feature Space Visualisation</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:13px;color:#556688;margin-bottom:12px;font-family:\"Plus Jakarta Sans\",sans-serif;'>Proves the AI genuinely learned distinct clinical features rather than memorising training images.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="background:rgba(4,7,15,0.98);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:10px;overflow:hidden;">',
            unsafe_allow_html=True,
        )
        st.image("reports/figures/tsne.png", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        met3, met4 = st.columns(2, gap="large")
        with met3:
            st.markdown(
                "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:10px;font-family:\"Plus Jakarta Sans\",sans-serif;'>AI Confidence Distribution</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:rgba(4,7,15,0.98);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:10px;overflow:hidden;">',
                unsafe_allow_html=True,
            )
            st.image("reports/figures/confidence_plot.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with met4:
            st.markdown(
                "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:10px;font-family:\"Plus Jakarta Sans\",sans-serif;'>Sample AI Predictions</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:rgba(4,7,15,0.98);border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:10px;overflow:hidden;">',
                unsafe_allow_html=True,
            )
            st.image("reports/figures/predictions_grid.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.warning(
            "⚠️ Analytics images not found. Please ensure evaluation images are saved in `reports/figures/`."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — API INTEGRATION HUB
# ──────────────────────────────────────────────────────────────────────────────
with tab_api:
    st.markdown(
        """
<div class="animate-in" style="margin-bottom:32px;">
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#445577;margin-bottom:10px;
    font-family:'Plus Jakarta Sans',sans-serif;
">REST API</div>
<div style="
    font-size:28px;font-weight:800;
    font-family:'Syne',sans-serif;letter-spacing:-0.5px;
    color:#f0f6ff;margin-bottom:10px;
">FastAPI Integration Hub</div>
<div style="
    font-size:14px;color:#556688;max-width:680px;line-height:1.7;
    font-family:'Plus Jakarta Sans',sans-serif;
">
    DentMatch AI is a fully containerised REST API, ready to be integrated
    into any hospital EHR system, mobile app, or web portal.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Endpoint 1
    st.markdown(
        """
<div style="
background: rgba(0,212,255,0.04);
border: 1px solid rgba(0,212,255,0.12);
border-radius: 16px;
padding: 20px 24px;
margin-bottom: 24px;
">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="
        background:rgba(0,229,195,0.12);color:#00e5c3;
        font-size:11px;font-weight:700;padding:3px 10px;border-radius:100px;
        font-family:'JetBrains Mono',monospace;border:1px solid rgba(0,229,195,0.2);
    ">POST</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:14px;color:#00d4ff;">/analyze/</span>
</div>
<div style="font-size:13px;color:#556688;font-family:'Plus Jakarta Sans',sans-serif;">
    Send a dental image (+ optional <code style="color:#00d4ff;background:rgba(0,212,255,0.08);padding:1px 6px;border-radius:4px;">pain_duration</code>
    & <code style="color:#00d4ff;background:rgba(0,212,255,0.08);padding:1px 6px;border-radius:4px;">chronic_diseases</code>) to receive a structured JSON diagnostic report.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_code1, col_code2 = st.columns(2, gap="large")
    col_code1.markdown(
        "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:8px;font-family:\"Plus Jakarta Sans\",sans-serif;'>cURL</div>",
        unsafe_allow_html=True,
    )
    col_code1.code(
        'curl -X POST "https://eng-ahmedayman10-healthysmile-ai.hf.space/analyze/" \\\n'
        '     -H "accept: application/json" \\\n'
        '     -F "file=@patient_tooth.jpg" \\\n'
        '     -F "pain_duration=أسبوع إلى شهر" \\\n'
        '     -F "chronic_diseases=مرض السكري"',
        language="bash",
    )
    col_code2.markdown(
        "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:8px;font-family:\"Plus Jakarta Sans\",sans-serif;'>Python · requests</div>",
        unsafe_allow_html=True,
    )
    col_code2.code(
        "import requests\n\n"
        'url   = "https://eng-ahmedayman10-healthysmile-ai.hf.space/analyze/"\n'
        'files = {"file": open("patient_tooth.jpg", "rb")}\n'
        "data  = {\n"
        '    "pain_duration":    "أسبوع إلى شهر",\n'
        '    "chronic_diseases": "مرض السكري",\n'
        "}\n\n"
        "response = requests.post(url, files=files, data=data)\n"
        "print(response.json())",
        language="python",
    )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # Endpoint 2
    st.markdown(
        """
<div style="
background: rgba(168,85,247,0.04);
border: 1px solid rgba(168,85,247,0.12);
border-radius: 16px;
padding: 20px 24px;
margin-bottom: 24px;
">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
<span style="
    background:rgba(168,85,247,0.12);color:#a855f7;
    font-size:11px;font-weight:700;padding:3px 10px;border-radius:100px;
    font-family:'JetBrains Mono',monospace;border:1px solid rgba(168,85,247,0.2);
">POST</span>
<span style="font-family:'JetBrains Mono',monospace;font-size:14px;color:#a855f7;">/triage-symptoms/</span>
</div>
<div style="font-size:13px;color:#556688;font-family:'Plus Jakarta Sans',sans-serif;">
Send an Arabic symptom description to receive AI triage routing to the correct clinical department.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns(2, gap="large")
    col_t1.markdown(
        "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:8px;font-family:\"Plus Jakarta Sans\",sans-serif;'>cURL</div>",
        unsafe_allow_html=True,
    )
    col_t1.code(
        'curl -X POST "https://eng-ahmedayman10-healthysmile-ai.hf.space/triage-symptoms/" \\\n'
        '     -H "Content-Type: application/json" \\\n'
        '     -d \'{"symptoms_description": "عندي ألم شديد في ضرسي بقاله أسبوع"}\'',
        language="bash",
    )
    col_t2.markdown(
        "<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:#445577;margin-bottom:8px;font-family:\"Plus Jakarta Sans\",sans-serif;'>Python · requests</div>",
        unsafe_allow_html=True,
    )
    col_t2.code(
        "import requests\n\n"
        'url     = "https://eng-ahmedayman10-healthysmile-ai.hf.space/triage-symptoms/"\n'
        'payload = {"symptoms_description": "عندي ألم شديد في ضرسي بقاله أسبوع"}\n\n'
        "response = requests.post(url, json=payload)\n"
        "print(response.json())",
        language="python",
    )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # Swagger link
    st.markdown(
        f"""
<div style="
background: linear-gradient(135deg, rgba(0,212,255,0.06), rgba(0,229,195,0.04));
border: 1px solid rgba(0,212,255,0.15);
border-radius: 20px;
padding: 28px 32px;
display: flex; align-items: center; justify-content: space-between;
">
<div>
<div style="
    font-size:10px;font-weight:700;text-transform:uppercase;
    letter-spacing:2.5px;color:#00d4ff88;margin-bottom:8px;
    font-family:'Plus Jakarta Sans',sans-serif;
">Interactive Documentation</div>
<div style="
    font-size:18px;font-weight:800;color:#f0f6ff;
    font-family:'Plus Jakarta Sans',sans-serif;
">Swagger UI · Live API Explorer</div>
<div style="font-size:13px;color:#445577;margin-top:4px;font-family:'Plus Jakarta Sans',sans-serif;">
    Test all endpoints directly from your browser
</div>
</div>
<a href="{API_BASE_URL}/docs" target="_blank" style="text-decoration:none;">
<div style="
    background: linear-gradient(135deg, #00d4ff, #00e5c3);
    color: #020814;
    font-weight: 700; font-size: 13px;
    padding: 12px 24px;
    border-radius: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    white-space: nowrap;
    box-shadow: 0 4px 20px rgba(0,212,255,0.35);
    letter-spacing: 0.3px;
">
    ✦ Open Swagger UI →
</div>
</a>
</div>
""",
        unsafe_allow_html=True,
    )
