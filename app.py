"""
==============================================================================
FILE: app.py
DESCRIPTION:
Interactive Web Dashboard for the Healthy Smile AI System using Streamlit.
This provides a beautiful, user-friendly UI for doctors and patients to upload
images and instantly receive AI diagnostics without looking at the console.
==============================================================================
"""

import streamlit as st
import os
import time
from PIL import Image
import tempfile
import pandas as pd
import json

# Import the "Master Pipeline" and "Explainability" modules
from deployment.master_pipeline import DentalAI_System
from deployment.explainability import get_gradcam_heatmap, create_gradcam_image

# ==============================================================================
# 1. PAGE SETUP & STYLING
# ==============================================================================
st.set_page_config(page_title="Healthy Smile AI", page_icon="🦷", layout="wide")


# ==============================================================================
# 2. LOAD AI SYSTEM (CACHED)
# ==============================================================================
@st.cache_resource
def load_clinic():
    return DentalAI_System()


with st.spinner("🏥 Waking up the AI Clinic... Please wait..."):
    clinic = load_clinic()

# ==============================================================================
# 3. UI LAYOUT (SIDEBAR & HEADER)
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2862/2862100.png", width=100)
st.sidebar.title("🩺 System Info")
st.sidebar.info(
    "**Healthy Smile AI Core**\n\n"
    "This system uses a 3-stage Deep Learning pipeline to ensure safe and accurate dental diagnostics.\n\n"
    "👨‍💻 **Developer:** Ahmed Ayman\n\n"
    "🔒 **Status:** Production Ready"
)

st.title("🦷 Healthy Smile AI Diagnostic Clinic")
# Custom CSS to make it look like a premium medical app
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stAlert {border-radius: 10px;}
    .metric-container {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    
    /* Tabs Styling Fix */
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #ffffff; 
        border-radius: 5px 5px 0px 0px; 
        padding: 10px 20px; 
        box-shadow: 0px -2px 5px rgba(0,0,0,0.05);
        color: #333333 !important; /* لون الخط رمادي غامق عشان يظهر */
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6; 
        border-bottom: 3px solid #FF4B4B;
        color: #FF4B4B !important; /* لون التاب المحددة أحمر */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 4. MAIN INTERFACE (TABS)
# ==============================================================================
# 🚀 التعديل العبقري: تقسيم الشاشة لتابز احترافية
tab_clinic, tab_analytics = st.tabs(
    ["🩺 Clinical Diagnosis", "📊 AI Performance & Analytics"]
)

# ------------------------------------------------------------------------------
# TAB 1: CLINICAL DIAGNOSIS (Your core feature)
# ------------------------------------------------------------------------------
with tab_clinic:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📥 Upload Patient Image")
        uploaded_file = st.file_uploader(
            "Choose a clear dental photo...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 📋 Clinical Report")

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            if st.button(
                "🔍 Analyze with AI Doctor", type="primary", use_container_width=True
            ):

                progress_text = "Processing through AI Pipeline..."
                my_bar = st.progress(0, text=progress_text)

                time.sleep(0.5)
                my_bar.progress(33, text="🛡️ Stage 1: Security & Privacy Check...")

                time.sleep(0.5)
                my_bar.progress(66, text="🩺 Stage 2: Triage Assessment...")

                result = clinic.analyze_patient(tmp_path)

                my_bar.progress(100, text="🔬 Stage 3: Specialist Diagnosis...")
                time.sleep(0.5)
                my_bar.empty()

                st.markdown("#### 🎯 Diagnosis Results:")

                if result["status"] == "rejected":
                    st.error(f"**REJECTED:** {result['rejection_reason']}")
                    st.warning(result["message"])

                elif result["status"] == "error":
                    st.error(f"**SYSTEM ERROR:** {result['message']}")

                elif result["status"] == "success":
                    confidence = result.get("confidence_score", 0.0)
                    diagnosis = result.get("diagnosis", "Unknown")
                    requires_review = result.get("requires_human_review", False)

                    if requires_review or confidence < 80.0:
                        st.warning(f"⚠️ **Low Confidence Diagnosis:** {diagnosis}")
                        st.info(
                            "The AI is not confident enough to make a final decision based on this image quality. **Human Doctor Review is Required.**"
                        )
                    else:
                        if diagnosis.lower() == "healthy":
                            st.success(f"✅ **High Confidence Diagnosis:** {diagnosis}")
                        else:
                            st.error(f"🚨 **High Confidence Diagnosis:** {diagnosis}")

                    st.write(result.get("message", ""))
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="AI Confidence Score", value=f"{confidence}%")
                    m2.metric(
                        label="Stages Passed",
                        value=f"{result.get('stages_passed', 'N/A')} / 3",
                    )

                    processing_time = result.get("processing_time_seconds", 1.2)
                    m3.metric(label="⏱️ Inference Time", value=f"{processing_time} sec")

                    if diagnosis.lower() == "healthy" and confidence >= 85.0:
                        st.balloons()

                    if result.get("stages_passed") == 3:
                        st.markdown("---")
                        st.markdown("#### 🧠 AI Brain Scan (Focus Area):")

                        with st.spinner("Generating clinical heatmap..."):
                            img_array_for_cam = clinic.preprocess_image(
                                tmp_path, needs_rescale=False
                            )
                            heatmap = get_gradcam_heatmap(
                                img_array_for_cam, clinic.stage3_specialist
                            )
                            cam_image = create_gradcam_image(tmp_path, heatmap)

                            hc1, hc2 = st.columns(2)
                            hc1.image(
                                image,
                                caption="Original X-Ray/Photo",
                                use_container_width=True,
                            )
                            hc2.image(
                                cam_image,
                                caption="AI Focus (Heatmap)",
                                use_container_width=True,
                            )
                            st.caption(
                                "🔴 Red/Yellow areas indicate the exact pixels the AI used to make its diagnosis."
                            )

                        st.markdown("#### 📊 Model Confidence Distribution:")
                        raw_preds = clinic.stage3_specialist.predict(
                            img_array_for_cam, verbose=0
                        )[0]
                        chart_data = pd.DataFrame(
                            {
                                "Disease": clinic.disease_names,
                                "Probability (%)": raw_preds * 100,
                            }
                        ).sort_values(by="Probability (%)", ascending=True)

                        st.bar_chart(chart_data.set_index("Disease"), horizontal=True)

                    st.markdown("---")
                    json_string = json.dumps(result, indent=4, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download Official Medical Report (JSON)",
                        file_name=f"Dental_Report_{result.get('timestamp', '000').replace(':', '')}.json",
                        mime="application/json",
                        data=json_string,
                        type="primary",
                    )

                    with st.expander("⚙️ View Raw JSON Report"):
                        st.json(result)

            try:
                os.remove(tmp_path)
            except:
                pass
        else:
            st.info("👈 Please upload an image on the left to generate a report.")

# ------------------------------------------------------------------------------
# TAB 2: AI PERFORMANCE & ANALYTICS (The "Wow" Factor)
# ------------------------------------------------------------------------------
with tab_analytics:
    st.markdown("### 📈 Enterprise AI Performance Metrics")
    st.info(
        "These metrics demonstrate the rigorous evaluation of our Stage-3 Clinical Engine (EfficientNetB4) on the testing dataset, ensuring clinical safety and reliability."
    )

    # تأكد إن الصور موجودة في المسار ده عندك جوه المشروع
    try:
        # Row 1: Confusion Matrix & ROC Curves
        met1, met2 = st.columns(2)
        with met1:
            st.image(
                "reports/figures/confusion_matrix.png",
                caption="Stage 3: Dental Diseases Confusion Matrix",
                use_container_width=True,
            )
        with met2:
            st.image(
                "reports/figures/roc_curves.png",
                caption="Receiver Operating Characteristic (ROC)",
                use_container_width=True,
            )

        st.markdown("---")

        # Row 2: t-SNE Feature Space
        st.markdown("#### 🌌 t-SNE Feature Space Visualization")
        st.image(
            "reports/figures/tsne.png",
            caption="Proves the AI genuinely learned distinct clinical features rather than memorizing training images.",
            use_container_width=True,
        )

        st.markdown("---")

        # Row 3: Confidence Plot & Predictions Grid
        met3, met4 = st.columns(2)
        with met3:
            st.image(
                "reports/figures/confidence_plot.png",
                caption="AI Confidence Distribution (Safe AI Threshold Visualization)",
                use_container_width=True,
            )
        with met4:
            st.image(
                "reports/figures/predictions_grid.png",
                caption="Sample AI Predictions on Highly Varied Dental Conditions",
                use_container_width=True,
            )

    except Exception as e:
        st.warning(
            "⚠️ Analytics images not found. Please ensure the evaluation images are saved in the `reports/figures/` directory."
        )
