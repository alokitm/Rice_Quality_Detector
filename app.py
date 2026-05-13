# ============================================
# Rice Quality Detection Web App
# ============================================

# Streamlit -> web app banane ke liye
import streamlit as st

# YOLO model
from ultralytics import YOLO

# Image handling
from PIL import Image

# OpenCV
import cv2

# Numpy
import numpy as np

# Counter for counting rice classes
from collections import Counter

# ============================================
# Page Title
# ============================================

st.title("Rice Quality Detection System")

st.write("Upload an image to detect and count rice categories.")

# ============================================
# Load YOLO Model
# ============================================

model = YOLO("best.pt")

# ============================================
# Upload Image
# ============================================

uploaded_file = st.file_uploader(

    "Upload Rice Image",

    type=["jpg", "jpeg", "png"]

)

# ============================================
# If Image Uploaded
# ============================================

if uploaded_file is not None:

    # PIL image
    image = Image.open(uploaded_file)

    # Convert image to numpy array
    image_np = np.array(image)

    # ============================================
    # Run YOLO Prediction
    # ============================================

    results = model.predict(

        source=image_np,

        conf=0.25

    )

    # ============================================
    # Get Annotated Image
    # ============================================

    annotated_frame = results[0].plot()

    # Convert BGR to RGB
    annotated_frame = cv2.cvtColor(

        annotated_frame,

        cv2.COLOR_BGR2RGB

    )

    # ============================================
    # Display Output Image
    # ============================================

    st.image(

        annotated_frame,

        caption="Detected Rice Objects",

        use_container_width=True

    )

    # ============================================
    # Rice Counting
    # ============================================

    class_names = model.names

    detected_classes = []

    boxes = results[0].boxes

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = class_names[class_id]

        detected_classes.append(class_name)

    # Count classes
    counts = Counter(detected_classes)

    # ============================================
    # Show Counts
    # ============================================

    st.subheader("Rice Counts")

    for cls, count in counts.items():

        st.write(f"{cls} : {count}")