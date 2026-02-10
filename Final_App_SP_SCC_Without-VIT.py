import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import base64
import gdown

# -----------------------------
# YOLO Imports
# -----------------------------
from ultralytics import YOLO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SOMAEYE: Vision-Based Somatic Cell Count",
    layout="wide",
    page_icon="🔬"
)

# =====================================================
# SESSION STATE (UPLOADER VERSIONING)
# =====================================================
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0

# =====================================================
# LOAD LOGO
# =====================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("SOMAEYE.jpeg")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(90deg, #d8f1ff 0%, #eef8ff 100%);
}
h1 { font-weight: 800; color: #1f2a6d; }

.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.08);
    border-left: 6px solid;
    margin: 12px 0;
}

.card.clean { border-color: #22c55e; }
.card.warn { border-color: #f97316; }
.card.critical { border-color: #ef4444; }

.card-title {
    font-size: 15px;
    font-weight: 700;
    color: #475569;
}

.card-value {
    font-size: 28px;
    font-weight: 900;
    color: #0f172a;
}

/* Change Browse button text */
[data-testid="stFileUploader"] button {
    font-size: 0px;
}
[data-testid="stFileUploader"] button::after {
    content: "Capture Images";
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    f"""
    <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:20px;">
        <img src="data:image/png;base64,{logo_base64}" style="width:360px;margin-bottom:10px;">
        <h1>AI-Powered Somatic Cell Count With Complete Traceability</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# CONFIG
# =====================================================
DETECTION_CONF = 0.5

SCC_MULTIPLIER_1 = 10
SCC_MULTIPLIER_2 = 1000
SCC_DIVISOR = 1.2

# =====================================================
# LOAD YOLO MODEL FROM GOOGLE DRIVE (SINGLE FILE)
# =====================================================
GDRIVE_FILE_ID = "15dGBp4DIKu2VLNlMdTFRU4_dPeN4ehwp"
MODEL_NAME = "best.pt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def load_detection_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading SCC detection model from Google Drive..."):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    model = YOLO(MODEL_PATH)
    return model, model.names

with st.spinner("Loading detection model..."):
    detection_model, detection_class_names = load_detection_model()

# =====================================================
# FILE UPLOADER
# =====================================================
uploaded_files = st.file_uploader(
    "Capture exactly 3 images for detection",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"scc_uploader_{st.session_state.uploader_version}"
)

if not uploaded_files or len(uploaded_files) != 3:
    st.warning("Please Capture exactly 3 images.")
    st.stop()

# =====================================================
# PROCESS IMAGES
# =====================================================
total_cells = 0
cols = st.columns(3)

for idx, (file, col) in enumerate(zip(uploaded_files, cols), 1):

    original_pil = Image.open(file).convert("RGB")
    original_rgb = np.array(original_pil)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        original_pil.save(tmp_path)

    results = detection_model.predict(tmp_path, conf=DETECTION_CONF, verbose=False)
    os.remove(tmp_path)

    img_out = original_rgb.copy()
    cells = 0

    for r in results:
        if r.boxes is None:
            continue
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if detection_class_names[int(cls)].lower() == "cells":
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cells += 1

    total_cells += cells
    col.image(img_out, caption=f"Image {idx} | Cells: {cells}", use_container_width=True)

# =====================================================
# FINAL RESULT CARD
# =====================================================
scc_count = (total_cells * SCC_MULTIPLIER_1 * SCC_MULTIPLIER_2) / SCC_DIVISOR

st.markdown("## 🧪 Detected Somatic Cells")

if scc_count >= 1_000_000:
    result = "SCC > 1,000,000 cells/ml"
    card = "critical"

elif scc_count <= 200_000:
    result = "SCC < 200,000 cells/ml"
    card = "clean"

else:
    result = f"SCC = {int(scc_count)} cells/ml"
    card = "warn"

st.markdown(
    f"""
    <div class="card {card}">
        <div class="card-title">Somatic Cell Count</div>
        <div class="card-value">{result}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TEST NEXT SAMPLE BUTTON
# =====================================================
st.markdown("---")
if st.button("🔄 Test Next Sample"):
    st.session_state.uploader_version += 1
    st.rerun()
