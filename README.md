# SECURE-EYE // CodeAlpha Object Detection and Tracking Dashboard

Welcome to **CodeAlpha_ObjectDetectionTracking** (Task 4 for the CodeAlpha Artificial Intelligence Internship). 

**SECURE-EYE** is a real-time tactical surveillance HUD and object tracking dashboard. It integrates state-of-the-art deep learning architectures—**YOLOv8** (detector) and **DeepSORT** (tracker)—to detect, identify, and track targets across frames. The frontend is built on **Streamlit** styled with a dark, glassmorphic sci-fi HUD theme.

---

## 👁️ System Capabilities & Features

### Core Tracking Features
* **Real-Time YOLOv8 Inference**: Automatic target acquisition using YOLOv8 nano, small, or medium weights.
* **DeepSORT Re-Identification**: Motion tracking using Kalman filtering coupled with visual appearance features extracted via a pre-trained MobileNet CNN.
* **Dynamic Target ID Mapping**: Persistent, unique track ID assignment to distinguish individual targets.
* **Visual Trajectory Trails**: Draws glowing movement vectors indicating target pathways over the last 30 frames.

### Advanced Analytical Features
* **Double-Count Prevention**: An intelligent session state logging system that records a target once upon visual lock rather than flooding log histories.
* **Live HUD Metrics**: Key Performance Indicators (KPIs) showing Active Targets, Cumulative Unique Targets, dynamic FPS, and Average Confidence.
* **Plotly Interactive Charts**: Real-time updating line charts for target density and FPS, plus horizontal bar charts displaying cumulative class counts.
* **Telemetry Data Exporter**: Quick session data exports to CSV files.
* **Instant Snapshot Captures**: Action buttons to download the current frame's visual annotations.

### UI/UX Design System
* **Futuristic Sci-Fi HUD Theme**: Custom dark theme overrides using deep slate, cyber cyan (`#00f3ff`), neon purple (`#b026ff`), and jade green.
* **Glassmorphic Panels**: Semi-transparent containers featuring backdrop filters and subtle glow animations.
* **Cyber Brackets Bounding Boxes**: Futuristic corner bracket overlays instead of generic rectangles.

---

## 📐 Architecture Data Flow

```
                      +-----------------------------+
                      |    Video / Camera Input     |
                      |   (Webcam / MP4 / Sample)   |
                      +--------------+--------------+
                                     |
                                     v [BGR Frame]
                      +--------------+--------------+
                      |        OpenCV Stream        |
                      +--------------+--------------+
                                     |
                                     v [Frame Array]
                      +--------------+--------------+
                      |   YOLOv8 Object Detection   |
                      | (Class, Conf, Bbox Coordinates)
                      +--------------+--------------+
                                     |
                                     v [Left, Top, Width, Height]
                      +--------------+--------------+
                      |  DeepSORT Tracking Engine   |
                      | (Kalman Filters & Re-ID)    |
                      +--------------+--------------+
                                     |
                                     v [Track ID + LTRB coordinates]
                      +--------------+--------------+
                      |      Analytics Manager      |
                      |  (Path Trails & Aggregates) |
                      +--------------+--------------+
                                     |
                                     v [Annotated Frame & Live Metrics]
                      +--------------+--------------+
                      |   Streamlit Futuristic HUD  |
                      |    (Plotly Charts & Logs)   |
                      +-----------------------------+
```

---

## 📂 Project Structure

* **`app.py`**: The Streamlit user interface, custom CSS stylesheet injector, Plotly graphing utilities, and primary execution loop.
* **`detector.py`**: Lightweight wrapper around the `ultralytics` YOLOv8 model handling frame inference and detection parsing.
* **`tracker.py`**: Integration layer for `deep-sort-realtime` wrapping the Kalman Filtering and CNN association algorithms.
* **`analytics.py`**: Tracks cumulative and active object counts, keeps track of coordinate histories for motion trails, and maintains detection logs.
* **`requirements.txt`**: Lists all external Python dependencies.

---

## 🛠️ Setup & Installation

Follow these steps to set up and run the application on your local machine:

### Prerequisites
Make sure you have **Python 3.8 to 3.11** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows (PowerShell/CMD)
.\venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Streamlit, OpenCV, Ultralytics, and PyTorch will download automatically. If you have a CUDA-enabled GPU and wish to leverage CUDA acceleration, ensure you have PyTorch installed with matching CUDA version).*

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 🕹️ Operations & Usage Guide

1. **Start the Interface**: Open the Streamlit URL (typically `http://localhost:8501`) inside your browser.
2. **Choose Source**: 
   * **Sample Video**: Click "INITIALIZE" to automatically download and process the Intel traffic sample.
   * **Upload Video**: Drag and drop your local `.mp4`, `.avi`, or `.mov` file and select "INITIALIZE".
   * **Webcam Device**: Ensure a webcam is connected, choose the index (0 for default built-in camera), and initialize.
3. **Refine Filters**: Add or remove objects (e.g. `person`, `car`, `motorcycle`) in the sidebar target class filter. Adjust the detection confidence to ignore low-quality matches.
4. **Acquire Reports**:
   * Scroll to the **Export Panel** to download target records in `.csv` format.
   * Toggle off "INITIALIZE" or hit "SHUTDOWN" to view the last recorded frame and click "DOWNLOAD CURRENT FRAME SCREENSHOT".

---

## 🚀 Deployment Guide

### Deploying to Streamlit Community Cloud
You can deploy this dashboard directly on **Streamlit Community Cloud** for online demonstrations:

1. Push your code repository to **GitHub**.
2. Visit [Streamlit Share](https://share.streamlit.io/) and log in using your GitHub account.
3. Click **New App**, select your repository `CodeAlpha_ObjectDetectionTracking`, branch `main`, and file path `app.py`.
4. Click **Deploy**.
*(Note: Streamlit Community Cloud instances run on CPU-only containers. The tracker will automatically fall back to CPU MobileNet embeddings as configured in `tracker.py`).*
