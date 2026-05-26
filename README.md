<div align="center">

<br>

```
██████╗ ███████╗███╗   ██╗████████╗███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗     █████╗ ██╗
██╔══██╗██╔════╝████╗  ██║╚══██╔══╝████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║    ██╔══██╗██║
██║  ██║█████╗  ██╔██╗ ██║   ██║   ██╔████╔██║███████║   ██║   ██║     ███████║    ███████║██║
██║  ██║██╔══╝  ██║╚██╗██║   ██║   ██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║    ██╔══██║██║
██████╔╝███████╗██║ ╚████║   ██║   ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║    ██║  ██║██║
╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
```

<img src="reports/figures/DentMatch-Healthy-Smile-AI-05-26-2026_02_52_AM.png" alt="DentMatch AI — Clinical Intelligence Platform" width="92%" style="border-radius:10px;margin:16px 0;" />

<br>

<table border="0" cellspacing="0" cellpadding="0"><tr>
<td align="right" width="50%"><sub><b>WHAT IT IS →</b></sub></td>
<td width="4px"></td>
<td align="left" width="50%"><sub>End-to-end dental triage. CV diagnosis + NLP routing.</sub></td>
</tr><tr>
<td align="right"><sub><b>WHO IT'S FOR →</b></sub></td>
<td></td>
<td align="left"><sub>Dental clinics, university hospitals, and underserved patients in Egypt.</sub></td>
</tr><tr>
<td align="right"><sub><b>WHY IT MATTERS →</b></sub></td>
<td></td>
<td align="left"><sub>Doctors spend hours on triage AI can finish in 2 seconds.</sub></td>
</tr></table>

<br>

[![Status](https://img.shields.io/badge/Status-Production_Ready-00C896?style=flat-square&labelColor=0D0D0D)](.)
[![Accuracy](https://img.shields.io/badge/Stage3_Accuracy-96%25-00C896?style=flat-square&labelColor=0D0D0D)](.)
[![ROC](https://img.shields.io/badge/ROC--AUC-0.9977-7F77DD?style=flat-square&labelColor=0D0D0D)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&labelColor=0D0D0D&logo=python&logoColor=white)](.)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-AI_Engine-FF6F00?style=flat-square&labelColor=0D0D0D&logo=tensorflow&logoColor=white)](.)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-NLP_Core-4285F4?style=flat-square&labelColor=0D0D0D&logo=google&logoColor=white)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&labelColor=0D0D0D&logo=fastapi&logoColor=white)](.)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&labelColor=0D0D0D&logo=docker&logoColor=white)](.)
[![License](https://img.shields.io/badge/License-MIT-F7C948?style=flat-square&labelColor=0D0D0D)](LICENSE)

<br>

[`❤️ Why It Matters`](#-why-it-matters) &nbsp;·&nbsp; [`▶ Live Demo`](#-live-demo) &nbsp;·&nbsp; [`📊 Model Performance`](#-model-performance) &nbsp;·&nbsp; [`⚡ API Docs`](#-api-reference) &nbsp;·&nbsp; [`🚀 Get Started`](#-getting-started) &nbsp;·&nbsp; [`🔮 Roadmap`](#-roadmap)

</div>

<br>

---

<br>

## ◈ Why It Matters

> _This isn't just a technical project — it solves a real-world healthcare accessibility crisis._

```
┌─────────────────────┬──────────────────────────────────────────────────────────┐
│     Stakeholder     │                      Problem Solved                       │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│  🧑‍🦷 Low-Income     │  Access free dental care faster through intelligent AI   │
│     Patients        │  triage — no waiting, no confusion, direct routing        │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│  🎓 Dental          │  Find exact clinical cases (Endo, Perio, Surgery)        │
│     Students        │  needed for graduation — AI-powered case matching         │
├─────────────────────┼──────────────────────────────────────────────────────────┤
│  🏥 University      │  Reduce intake overload through AI-driven routing before  │
│     Hospitals       │  patients arrive — smarter triage, better outcomes        │
└─────────────────────┴──────────────────────────────────────────────────────────┘
```

**DentMatch AI fixes all three at once.**

A production-grade, multi-modal AI engine that handles the entire triage workflow — from patient image or voice complaint, to a structured clinical routing decision landing in the right doctor's queue.

No black boxes. No guessing. **Explainable, auditable, and fast.**

<br>

---

<br>

## ◈ Two Engines. One Pipeline.

<table border="0"><tr>
<td width="49%" valign="top">

### 🔬 Vision Engine

```
  INPUT  ──▶  Dental X-ray or photo
  GATE   ──▶  MobileNetV2 domain check
  ENGINE ──▶  EfficientNetB4 diagnosis
  OUTPUT ──▶  Disease class + heatmap
```

- Accepts X-rays and clinical photos
- Rejects blurry, non-dental, or face-containing inputs
- Classifies **6 oral disease categories**
- Generates **Grad-CAM++ heatmaps** pinpointing the exact lesion
- Medical-grade explainability on every prediction

</td>
<td width="2%" align="center"><sub>│<br>│<br>│<br>│<br>│<br>│<br>│<br>│<br>│</sub></td>
<td width="49%" valign="top">

### 🗣️ NLP Triage Engine

```
  INPUT  ──▶  Voice complaint (Arabic)
  NLP    ──▶  Gemini 2.5 Flash
  PARSE  ──▶  Symptoms + chronic cond.
  OUTPUT ──▶  Department + urgency JSON
```

- Understands **Egyptian Arabic** natively
- Extracts symptoms, pain duration, chronic conditions
- Routes automatically: Endodontics · Surgery · Periodontology
- Outputs structured JSON ready for EHR integration
- Powered by advanced medical prompt engineering

</td>
</tr></table>

<br>

---

<br>

## ◈ Full AI Pipeline

```
╔══════════════════════════════════════════════════════════════════════════╗
║                           PATIENT ENTRY POINT                           ║
╚═══════════════════════════════════╤════════════════════════════════════╝
                                    │
                    ┌───────────────▼───────────────┐
                    │     SECURITY & VALIDATION      │
                    │  Face detect · Format · Size   │
                    └───────┬───────────────┬────────┘
                            │               │
              ┌─────────────▼──┐       ┌────▼──────────────────┐
              │  IMAGE / X-RAY │       │    TEXT / VOICE        │
              └──────┬─────────┘       └────────────┬──────────┘
                     │                              │
              ┌──────▼─────────┐       ┌────────────▼──────────┐
              │  STAGE 1       │       │  Gemini 2.5 Flash      │
              │  MobileNetV2   │       │  Medical Prompt Engine │
              │  Domain Gate   │       │  Symptom Extraction    │
              └──────┬─────────┘       └────────────┬──────────┘
              ✓ dental confirmed                     │
              ✗ rejected → error                ┌────▼──────────┐
                     │                          │  Dept Routing  │
              ┌──────▼─────────┐               │  Urgency Class │
              │  STAGE 2       │               │  JSON Report   │
              │  EfficientNetB4│               └────────────────┘
              │  6-class Diag  │
              └──────┬─────────┘
                     │
              ┌──────▼─────────┐
              │  STAGE 3       │
              │  Grad-CAM++    │
              │  Lesion XAI    │
              └──────┬─────────┘
                     │
              ┌──────▼──────────────────────────┐
              │  STRUCTURED CLINICAL REPORT      │
              │  Prediction · Confidence · Map   │
              └──────────────────────────────────┘
```

<br>

### Stage Performance

```
  Haar-Cascade   ·  Privacy Guard       ──────────────────────  98% ████████████████████ ✓
  MobileNetV2    ·  Domain Gatekeeper   ──────────────────────  96% ███████████████████░ ✓
  EfficientNetB4 ·  Core Diagnostic     ──────────────────────  96% ███████████████████░ ✓  ← weighted avg
  ROC-AUC Score  ·  Mean Macro          ──────────────────────  99.77% ████████████████████ ✓
```

|   #   | Model            | Role                                             |     Accuracy     |
| :---: | :--------------- | :----------------------------------------------- | :--------------: |
|  `G`  | Haar-Cascade     | Privacy guard — rejects identifiable face images |     **98%**      |
| `S1`  | MobileNetV2      | Domain gatekeeper — blocks non-dental inputs     |     **96%+**     |
| `S2`  | EfficientNetB4   | Core diagnostic engine — 6 disease classes       | **96% weighted** |
| `S3`  | Grad-CAM++       | Pixel-level lesion explainability                |      ✦ XAI       |
| `NLP` | Gemini 2.5 Flash | Symptom extraction + department routing          |      ✦ LLM       |

<br>

**Per-class breakdown (Stage 3 — EfficientNetB4):**

| Disease Class       | Precision |  Recall  | F1-Score | Support  |
| :------------------ | :-------: | :------: | :------: | :------: |
| Dental Caries       |   0.97    |   0.91   |   0.94   |   391    |
| Hypodontia          |   0.97    |   0.93   |   0.95   |   188    |
| Mouth Ulcer         |   0.99    |   0.98   | **0.99** |   421    |
| Periodontal Disease |   0.92    |   0.99   |   0.96   |   548    |
| Tooth Discoloration |   0.94    |   0.93   |   0.94   |   303    |
| **Weighted Avg**    | **0.96**  | **0.96** | **0.96** | **1851** |

<sub>Mean Macro ROC-AUC: **0.9977**</sub>

<br>

---

<br>

## ◈ Real Clinical Scenario

_This is what actually happens when a patient submits a complaint:_

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  🎙️  PATIENT INPUT (Egyptian Arabic)                            │
  │  "عندي ألم شديد في سني من 3 أيام وعندي سكر"                    │
  │  "I've had severe tooth pain for 3 days and I have diabetes."   │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  🧠  AI PROCESSING                                              │
  │  ├── ✅  Chronic condition detected  →  Diabetes                │
  │  ├── ✅  Pain duration extracted     →  3 days                  │
  │  ├── ✅  Severity classified         →  High urgency            │
  │  └── ✅  Department matched          →  Endodontics             │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  📍  ROUTING DECISION (JSON — EHR ready)                        │
  │  ├── department         :  Endodontics                          │
  │  ├── urgency            :  High                                 │
  │  ├── chronic_conditions :  ["Diabetes"]                         │
  │  ├── pain_duration      :  3 days                               │
  │  └── recommended_level  :  Postgraduate Student                 │
  └─────────────────────────────────────────────────────────────────┘

  ✅  From voice note to routed clinical report — in under 2 seconds.
```

<br>

_Here's the same flow live in the dashboard — patient types a complaint in Arabic, AI routes it instantly:_

<div align="center">
<img src="reports/figures/DentMatch-Healthy-Smile-AI-05-26-2026_02_54_AM.png" width="88%" alt="Live Arabic NLP triage — Symptom Analyzer result" />
<br><sub>Live symptom analysis: Arabic complaint → Dental Caries diagnosis → department routing, all in one report.</sub>
</div>

<br>

---

<br>

## ◈ Explainability

> _Medical AI that can't explain itself isn't medical AI — it's liability._

<table border="0"><tr>
<td width="52%">

DentMatch AI never produces a bare prediction.

Every image diagnosis is paired with a **Grad-CAM++ heatmap** — a pixel-level visualization showing exactly which region drove the classification decision.

```
  Clinician receives:
  ├── Disease classification
  ├── Confidence score per class
  ├── Grad-CAM++ heatmap overlay
  └── Structured clinical report

  Not just: "It's caries."
  But:      "It's caries — here."
                              ↑
                  (with the pixel evidence)
```

This means:

- Clinicians verify AI reasoning at a glance
- Misdiagnosis risks are visible and auditable
- Regulatory compliance is built in from day one

</td>
<td width="48%" align="center">

<img src="reports/figures/gradcam.png" width="100%" alt="Grad-CAM++ heatmap — disease probability + lesion location" />
<sub><i>Confidence distribution + Grad-CAM++ heatmap: the model shows exactly where and how confident.</i></sub>

</td>
</tr></table>

<br>

_Two different cases — both explained at the pixel level:_

<div align="center">

<table border="0"><tr>
<td align="center" width="50%">
<img src="reports/figures/heatmap.png" width="100%" alt="Grad-CAM heatmap — Dental Caries case 2" />
<sub>Caries case — AI Focus Area highlighted in red/yellow</sub>
</td>
<td align="center" width="50%">
<img src="reports/figures/confidence_plot.png" width="100%" alt="Stage 3 AI Confidence & Reliability Analysis" />
<sub>Confidence analysis: 1769 correct vs 82 incorrect — clustered above 80% threshold</sub>
</td>
</tr></table>

</div>

<br>

---

<br>

## ◈ Live Demo

_The full dashboard has three tabs — each one a complete AI workflow:_

<div align="center">

<img src="reports/figures/DentMatch-Healthy-Smile-AI-05-26-2026_02_53_AM.png" width="92%" alt="DentMatch AI — Live diagnosis result with Grad-CAM heatmap" />
<br><sub>Tab 1 — Clinical Diagnosis: image uploaded → Dental Caries detected at 94% confidence → Grad-CAM++ heatmap generated → routing decision issued.</sub>

</div>

<br>

<table border="0"><tr>
<td align="center" width="33%">

**🖼️ Image Track**
Upload a dental photo
→ disease classification
→ confidence scores
→ Grad-CAM heatmap

</td>
<td align="center" width="33%">

**🎙️ Voice / Text Track**
Type or record complaint
→ Arabic NLP analysis
→ symptom extraction
→ department routing

</td>
<td align="center" width="33%">

**📋 API Integration Hub**
Full REST API docs
→ curl examples
→ Python snippets
→ Open Swagger UI

</td>
</tr></table>

<br>

_Tab 2 — AI Performance & Analytics: full model evaluation dashboard with confusion matrix, ROC curves, t-SNE, and sample predictions:_

<div align="center">
<img src="reports/figures/DentMatch-Healthy-Smile-AI-05-26-2026_02_52_AM (1).png" width="72%" alt="AI Performance & Analytics tab — confusion matrix, t-SNE, sample predictions" />
</div>

<br>

_Tab 3 — FastAPI Integration Hub: ready-to-use curl and Python code for every endpoint:_

<div align="center">
<img src="reports/figures/DentMatch-Healthy-Smile-AI-05-26-2026_02_52_AM (2).png" width="88%" alt="API Integration Hub — curl and Python examples" />
</div>

<br>

> **Try it yourself** — run the system locally and access the full interactive dashboard at `http://localhost:8501`. Interactive API docs (Swagger UI) at `http://127.0.0.1:8000/docs`.

<br>

---

<br>

## ◈ Sample AI Predictions

_9 real model predictions from the test set — all correctly classified at high confidence:_

<div align="center">
<img src="reports/figures/predictions_grid.png" width="90%" alt="Sample AI predictions — 9 disease classes with confidence scores" />
<br><sub>Dental Caries · Gingivitis · Tooth Discoloration · Calculus · Mouth Ulcer · Hypodontia — all correctly identified at 97–100% confidence.</sub>
</div>

<br>

---

<br>

## ◈ Security Architecture

_Every request passes through five layers before inference ever runs._

```
┌─────────────┬──────────────────────────────┬───────────────────────────────────┐
│    Layer    │         Mechanism             │          What It Stops            │
├─────────────┼──────────────────────────────┼───────────────────────────────────┤
│  PRIVACY    │  Haar-Cascade face detection  │  Identifiable patient faces       │
│  DOMAIN     │  MobileNetV2 binary cls.      │  Non-dental images pre-inference  │
│  PAYLOAD    │  FastAPI middleware limits    │  Malformed files, audio > 5 MB    │
│  MEMORY     │  io.BytesIO — zero disk I/O  │  Patient data ever touching disk  │
│  FORMAT     │  Deep content inspection      │  Malicious file-type spoofing     │
└─────────────┴──────────────────────────────┴───────────────────────────────────┘
```

_Stage 1 domain gate — binary classifier confusion matrix (498 Not_Teeth correctly rejected, 773 Teeth correctly passed, 0 false positives):_

<div align="center">
<img src="reports/figures/stage1_confusion_matrix.png" width="55%" alt="Stage 1 Security Guard — Confusion Matrix" />
</div>

<br>

---

<br>

## ◈ Stack

<table border="0"><tr>
<td valign="top" width="50%">

**AI Layer**

```
Vision  ── TensorFlow · Keras
            EfficientNetB4 · MobileNetV2
            OpenCV · Grad-CAM++

NLP     ── Google Gemini 2.5 Flash
            Medical Prompt Engineering
            Egyptian Arabic support
```

</td>
<td valign="top" width="50%">

**Infrastructure Layer**

```
Backend ── FastAPI · Uvicorn · Pydantic
Frontend── Streamlit
Deploy  ── Docker · ngrok
Runtime ── Python 3.10+
```

</td>
</tr></table>

<br>

---

<br>

## ◈ Getting Started

**Prerequisites**

```
Python 3.10+   ·   4 GB RAM minimum (8 GB recommended)   ·   Git
Windows / Linux / macOS
```

**Install**

```bash
git clone https://github.com/your-username/DentMatch-AI.git
cd DentMatch-AI
pip install -r requirements.txt
cp .env.example .env   # → add your GEMINI_API_KEY
```

**Launch (two terminals)**

```bash
# ─── Terminal 1 — AI Backend ─────────────────────────────────────────
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# ─── Terminal 2 — Dashboard ──────────────────────────────────────────
streamlit run app.py
```

```
Dashboard  →  http://localhost:8501
Swagger UI →  http://127.0.0.1:8000/docs
```

**Or Docker**

```bash
docker build -t dentmatch-ai .
docker run -p 8000:8000 -p 8501:8501 --env-file .env dentmatch-ai
```

<br>

---

<br>

## ◈ API Reference

<details>
<summary><code>POST /analyze/</code> &nbsp;—&nbsp; Image Diagnosis</summary>
<br>

Accepts a dental image. Returns disease classification, per-class confidence, and a base64 Grad-CAM++ heatmap.

```bash
curl -X POST "http://127.0.0.1:8000/analyze/" \
  -H "accept: application/json" \
  -F "file=@patient_tooth.jpg" \
  -F "pain_duration=اسبوع الى شهر" \
  -F "chronic_diseases=مرض السكري"
```

```json
{
  "status": "success",
  "prediction": "Caries",
  "confidence": 0.94,
  "all_scores": {
    "Dental_Caries": 0.94,
    "Hypodontia": 0.02,
    "Mouth_Ulcer": 0.01,
    "Periodontal_Disease": 0.02,
    "Tooth_Discoloration": 0.01
  },
  "heatmap_base64": "...",
  "model_stage": "EfficientNetB4"
}
```

</details>

<details>
<summary><code>POST /triage-audio/</code> &nbsp;—&nbsp; Voice Triage</summary>
<br>

Accepts a patient audio complaint. Transcribes, extracts clinical data, returns routing decision.

```bash
curl -X POST http://127.0.0.1:8000/triage-audio/ \
  -F "file=@complaint.mp3"
```

```json
{
  "transcript": "عندي ألم شديد في سني من 3 أيام وعندي سكر",
  "department": "Endodontics",
  "urgency": "High",
  "chronic_conditions": ["Diabetes"],
  "pain_duration": "3 days",
  "recommended_level": "Postgraduate Student"
}
```

</details>

<details>
<summary><code>POST /triage-symptoms/</code> &nbsp;—&nbsp; Text Triage</summary>
<br>

Accepts typed complaint in Egyptian Arabic. Returns structured routing report.

```bash
curl -X POST http://127.0.0.1:8000/triage-symptoms/ \
  -H "Content-Type: application/json" \
  -d '{"symptoms_description": "عندي ألم شديد في ضرسي بقاله اسبوع"}'
```

```json
{
  "department": "Periodontology",
  "urgency": "Medium",
  "extracted_symptoms": ["gum pain", "bleeding on brushing"],
  "duration": "1 week",
  "recommended_level": "Undergraduate Student"
}
```

</details>

<br>

> 📖 Full interactive docs available at `http://127.0.0.1:8000/docs` — test every endpoint directly in the browser.

<br>

---

<br>

## ◈ Project Structure

```
DentMatch-AI/
│
├── 📄  api.py                        ← FastAPI backend — all inference endpoints
├── 📄  app.py                        ← Streamlit frontend dashboard
├── 📄  start.sh                      ← One-command launcher
├── 📄  requirements.txt
├── 📄  Dockerfile
├── 📄  .env.example
│
├── 📁  deployment/
│   ├── master_pipeline.py            ← End-to-end inference orchestrator
│   └── explainability.py             ← Grad-CAM++ heatmap engine
│
├── 📁  src/
│   ├── clean_data.py                 ← Dataset preprocessing
│   ├── stage1_train.py               ← MobileNetV2 training
│   ├── train_stage3.py               ← EfficientNetB4 fine-tuning
│   ├── stage1_inference.py
│   ├── stage2_assessment.py
│   └── stage3_evaluation.py
│
├── 📁  models/
│   ├── stage1/  →  stage1_mobilenet.keras
│   ├── stage2/  →  pytorch_model.bin + config.json
│   └── stage3/  →  best.keras + model_config.json
│
├── 📁  data/
│   ├── stage1_binary/   train/ · val/ · test/
│   ├── stage3_disease/  train/ · val/ · test/
│   └── test_samples/
│
├── 📁  reports/
│   ├── training/                     ← Training curves & logs
│   └── figures/                      ← Confusion matrix · ROC · t-SNE · Grad-CAM
│
└── 📁  .streamlit/
    └── config.toml
```

> ⚠️ Add `.env` and `models/` to `.gitignore` — never commit API keys or model weights.

<br>

---

<br>

## ◈ Model Performance

<div align="center">

<table border="0"><tr>
<td align="center" width="50%">
<b>Stage 3 — Confusion Matrix</b><br><br>
<img src="reports/figures/confusion_matrix.png" width="100%" alt="Stage 3 Confusion Matrix" />
</td>
<td align="center" width="50%">
<b>Per-Class ROC Curves</b><br><br>
<img src="reports/figures/roc_curves.png" width="100%" alt="ROC Curves — Mean AUC 0.9977" />
</td>
</tr></table>

<br>

**t-SNE Feature Space Visualisation**

<sub>Proves the AI genuinely learned distinct clinical features — not pattern shortcuts.</sub><br><br>

<img src="reports/figures/tsne.png" width="80%" alt="t-SNE — 5 distinct clinical clusters" />

<br>

**Stage 3 — AI Confidence & Reliability Analysis**

<sub>1769 correct predictions vs 82 incorrect — the vast majority of errors occur below the 80% confidence threshold, meaning the model knows when it's uncertain.</sub><br><br>

<img src="reports/figures/confidence_plot.png" width="90%" alt="Confidence distribution — correct vs incorrect predictions" />

</div>

<br>

---

<br>

## ◈ System Impact

```
  ⏱️  Triage Time        ──────────────────────────────  ~70% reduction per patient
  🎯  Routing Accuracy   ──────────────────────────────  Consistent, fatigue-free
  🏥  Unnecessary Visits ──────────────────────────────  Significantly reduced
  🎓  Student Matching   ──────────────────────────────  Automated case-to-student
  📊  ROC-AUC            ──────────────────────────────  0.9977 mean macro
```

<br>

---

<br>

## ◈ Use Cases

<table border="0"><tr>
<td align="center" width="33%">

**🏥 University Clinics**
AI-powered pre-screening
and smart case routing
before patients arrive

</td>
<td align="center" width="33%">

**🎓 Dental Education**
Intelligent case-student
matching for graduation
requirements (Endo/Perio/Surgery)

</td>
<td align="center" width="33%">

**🧑‍⚕️ Remote Pre-Screening**
Accessible triage for
patients in underserved areas
before the clinic visit

</td>
</tr></table>

<br>

---

<br>

## ◈ Roadmap

<table border="0"><tr>
<td valign="top" width="50%">

**Planned**

| Feature          | Description                        |
| :--------------- | :--------------------------------- |
| EHR Integration  | HL7/FHIR-compliant endpoints       |
| Arabic Voice NLP | Full dialect-aware Arabic ASR      |
| Async Queue      | Celery + Redis for concurrent load |
| Mobile App       | Flutter/React Native patient app   |

</td>
<td valign="top" width="50%">

**Research**

| Feature            | Description                                           |
| :----------------- | :---------------------------------------------------- |
| Federated Learning | Train across hospital nodes without centralizing data |
| X-ray Module       | CBCT + panoramic X-ray pipeline support               |

</td>
</tr></table>

<br>

---

<br>

## ◈ Contributing

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change clearly"
git push origin feature/your-feature-name
# → Open a Pull Request
```

Open an issue first for significant changes. Follow existing code style and include tests where relevant.

<br>

---

<br>

## ◈ License

MIT License — see [LICENSE](LICENSE) for details.

<br>

---

<br>

<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Built with focus on  impact · scalability · trust in AI    ║
║                                                               ║
║              Ahmed Ayman — AI & Data Science Engineer         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-ayman-10b966292/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Eng-AhmedAyman)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](https://mail.google.com/mail/?view=cm&fs=1&to=ahmedayman162210@gmail.com)
<br>

_"The best medical AI makes a doctor's judgment sharper — not obsolete."_

<br>

⭐ **If this project helped you, please give it a star!** ⭐

<br>

<sub>DentMatch AI &nbsp;·&nbsp; MIT License &nbsp;·&nbsp; Built for impact, designed for trust</sub>

</div>
