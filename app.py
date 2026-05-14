# ============================================
# Rice Quality Detection System
# Premium UI Version
# ============================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Rice Quality Detection",
    page_icon="🌾",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""

<style>

html, body, [class*="css"]  {
    background-color: #0B1120;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #7CFFB2;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #CCCCCC;
    margin-bottom: 40px;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #2D3748;
}

.metric-title {
    font-size: 20px;
    color: #DDDDDD;
}

.metric-value {
    font-size: 40px;
    font-weight: bold;
}

.footer-box {
    background: rgba(0,255,170,0.08);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #00FFAA;
}

</style>

""", unsafe_allow_html=True)

# ============================================
# TITLE
# ============================================

st.markdown(
    "<div class='main-title'>🌾 Rice Quality Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload a rice image to detect rice categories and count grains.</div>",
    unsafe_allow_html=True
)

# ============================================
# LOAD MODEL
# ============================================

model = YOLO("best.pt")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("Project Info")

st.sidebar.success("AI Powered Rice Quality Detection")

st.sidebar.write("Detects:")

st.sidebar.write("✅ Head Rice")
st.sidebar.write("✅ Broken Rice")
st.sidebar.write("✅ Damaged Rice")
st.sidebar.write("✅ Red Rice")
st.sidebar.write("✅ Unhulled Rice")
st.sidebar.write("✅ Foreign Objects")

# ============================================
# FILE UPLOADER
# ============================================

uploaded_file = st.file_uploader(
    "📤 Upload Rice Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# IF IMAGE UPLOADED
# ============================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    # ============================================
    # PREDICTION
    # ============================================

    with st.spinner("Detecting Rice Objects..."):

        results = model.predict(
            source=image_np,
            conf=0.25
        )

    # ============================================
    # ANNOTATED IMAGE
    # ============================================

    annotated_frame = results[0].plot()

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    # ============================================
    # IMAGE COLUMNS
    # ============================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Detected Image")

        st.image(
            annotated_frame,
            use_container_width=True
        )

    # ============================================
    # COUNTING
    # ============================================

    class_names = model.names

    detected_classes = []

    boxes = results[0].boxes

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = class_names[class_id]

        detected_classes.append(class_name)

    counts = Counter(detected_classes)

    total_objects = sum(counts.values())

    # ============================================
    # COLOR MAPPING
    # ============================================

    color_map = {

        "Head rice": "#6DFF8B",
        "Broken rice": "#4D96FF",
        "Damaged rice": "#FFC857",
        "Red rice": "#FF914D",
        "Unhulled rice": "#B980FF",
        "Foreign objects": "#FF4B4B"

    }

    # ============================================
    # METRICS
    # ============================================

    st.write("")
    st.subheader("📊 Detection Summary")

    metric_cols = st.columns(len(counts) + 1)

    metric_cols[0].markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Grains</div>
        <div class="metric-value" style="color:#7CFFB2;">{total_objects}</div>
    </div>
    """, unsafe_allow_html=True)

    for i, (cls, count) in enumerate(counts.items()):

        color = color_map.get(cls, "#FFFFFF")

        metric_cols[i + 1].markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{cls}</div>
            <div class="metric-value" style="color:{color};">{count}</div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================
    # PIE CHART
    # ============================================

    st.write("")
    st.subheader("🌾 Rice Composition")

    labels = list(counts.keys())

    values = list(counts.values())

    colors = [color_map.get(label, "#FFFFFF") for label in labels]

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(

        values,

        labels=None,

        colors=colors,

        autopct=lambda pct: f"{int(round(pct/100.*sum(values)))}",

        startangle=90,

        textprops={'color': "white", 'fontsize': 16}

    )

    ax.legend(

        wedges,

        labels,

        title="Rice Types",

        loc="center left",

        bbox_to_anchor=(1, 0.5),

        fontsize=14,

        labelcolor="white"

    )

    fig.patch.set_facecolor('#0B1120')

    ax.set_facecolor('#0B1120')

    st.pyplot(fig)

    # ============================================
    # QUALITY INSIGHT
    # ============================================

    st.write("")

    st.markdown(f"""
    <div class="footer-box">

    <h3 style="color:#7CFFB2;">
    ✅ Quality Insight
    </h3>

    <p style="font-size:20px;">

    Higher <b>Head Rice</b> generally indicates better quality rice.

    Large amounts of <b>Foreign Objects</b>, <b>Broken Rice</b>,
    or <b>Damaged Rice</b> may indicate lower quality.

    </p>

    </div>
    """, unsafe_allow_html=True)