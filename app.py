# ============================================
# Rice Quality Detection System
# Advanced UI Version
# ============================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from collections import Counter

# ============================================
# Page Config
# ============================================

st.set_page_config(

    page_title="Rice Quality Detector",

    page_icon="🌾",

    layout="wide"

)

# ============================================
# Custom CSS
# ============================================

st.markdown("""

<style>

.main {

    background-color: #0E1117;

}

h1 {

    text-align: center;

    color: #00FFAA;

    font-size: 45px;

}

.stMarkdown {

    font-size: 18px;

}

.result-box {

    background-color: #1E1E1E;

    padding: 20px;

    border-radius: 15px;

    border: 1px solid #333;

}

</style>

""", unsafe_allow_html=True)

# ============================================
# Title
# ============================================

st.title("🌾 Rice Quality Detection System")

st.markdown(

    "<center>Upload a rice image to detect rice categories and count grains.</center>",

    unsafe_allow_html=True

)

st.write("")

# ============================================
# Load YOLO Model
# ============================================

model = YOLO("best.pt")

# ============================================
# Sidebar
# ============================================

st.sidebar.title("About Project")

st.sidebar.info(

    """

This AI model detects:

- Head Rice

- Broken Rice

- Damaged Rice

- Red Rice

- Unhulled Rice

- Foreign Objects

Built using:

- YOLOv8

- Streamlit

- OpenCV

"""

)

# ============================================
# Upload Image
# ============================================

uploaded_file = st.file_uploader(

    "📤 Upload Rice Image",

    type=["jpg", "jpeg", "png"]

)

# ============================================
# If Image Uploaded
# ============================================

if uploaded_file is not None:

    # PIL image
    image = Image.open(uploaded_file)

    # Convert to numpy
    image_np = np.array(image)

    # ============================================
    # Create Columns
    # ============================================

    col1, col2 = st.columns(2)

    # ============================================
    # Original Image
    # ============================================

    with col1:

        st.subheader("Original Image")

        st.image(

            image,

            use_container_width=True

        )

    # ============================================
    # YOLO Prediction
    # ============================================

    with st.spinner("Detecting Rice Objects..."):

        results = model.predict(

            source=image_np,

            conf=0.25

        )

    # ============================================
    # Annotated Image
    # ============================================

    annotated_frame = results[0].plot()

    annotated_frame = cv2.cvtColor(

        annotated_frame,

        cv2.COLOR_BGR2RGB

    )

    # ============================================
    # Show Prediction Image
    # ============================================

    with col2:

        st.subheader("Detection Result")

        st.image(

            annotated_frame,

            use_container_width=True

        )

    # ============================================
    # Counting Classes
    # ============================================

    class_names = model.names

    detected_classes = []

    boxes = results[0].boxes

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = class_names[class_id]

        detected_classes.append(class_name)

    counts = Counter(detected_classes)

    # ============================================
    # Results Section
    # ============================================

    st.write("")

    st.subheader("📊 Detection Summary")

    total_objects = sum(counts.values())

    st.success(f"Total Objects Detected: {total_objects}")

    # ============================================
    # Metrics
    # ============================================

    cols = st.columns(len(counts))

    for i, (cls, count) in enumerate(counts.items()):

        cols[i].metric(

            label=cls,

            value=count

        )

    # ============================================
    # Footer
    # ============================================

    st.write("")

    st.markdown("---")

    st.markdown(

        "<center>Built with YOLOv8 + Streamlit</center>",

        unsafe_allow_html=True

    )