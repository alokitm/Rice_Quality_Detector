# 🌾 Rice Quality Detection using YOLOv8

An AI-powered Rice Quality Detection System built using YOLOv8 and Streamlit. This project detects and classifies different types of rice grains and impurities from uploaded images.

---

## 🚀 Features

* Detects multiple rice grain categories
* Automatic rice grain counting
* Real-time image prediction
* Bounding box visualization
* Streamlit-based web interface
* YOLOv8 object detection model

---

## 🧾 Classes Detected

* 🍚 Head Rice
* 🌾 Broken Rice
* 🌱 Unhulled Rice
* ⚠️ Damaged Rice
* 🔴 Red Rice
* 🪨 Foreign Objects

---

## 🛠️ Tech Stack

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* Streamlit

---

## 🧠 Model Details

* Model Architecture: YOLOv8
* Task Type: Object Detection
* Training Platform: Kaggle
* Framework: Ultralytics

---

## 📂 Dataset

The dataset contains annotated rice grain images with YOLO bounding box annotations for different rice quality categories.

Dataset Includes:

* Train Images
* Validation Images
* Test Images
* YOLO Annotation Files
* data.yaml Configuration

---

## ⚙️ Installation

Clone the repository:

```bash id="ukhrpn"
git clone https://github.com/your-username/rice-quality-detection.git
cd rice-quality-detection
```

Install dependencies:

```bash id="rjvnsz"
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash id="kh0d2k"
streamlit run app.py
```

---

## 📁 Project Structure

```text id="4v3t74"
Rice_Quality_Project/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── images/
├── predictions/
└── notebooks/
```

---

## 🔄 Workflow

1. 📤 Upload rice image
2. 🤖 Model detects rice categories
3. 📦 Bounding boxes are generated
4. 🔢 Rice counts are displayed
5. 🖼️ Final detected image is shown

---

## 📈 Future Improvements

* Increase dataset size
* Improve fine-grained classification
* Better detection accuracy
* Mobile deployment
* Real-time camera detection

---

## 👨‍💻 Author

**Alokit Mishra**

---

## 📜 License

This project is for educational and research purposes.
