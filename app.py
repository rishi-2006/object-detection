import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import tempfile
import os
import io
from PIL import Image

# Import custom modules
from detector import YOLODetector
from tracker import ObjectTracker
from analytics import AnalyticsManager

# Set page config
st.set_page_config(
    page_title="SECURE-EYE // AI Object Detection & Tracking HUD",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Futuristic Dark Theme & Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #030712;
        color: #f3f4f6;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Cyber Title */
    .cyber-title {
        font-family: 'Orbitron', sans-serif;
        color: #00f3ff;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.4), 0 0 20px rgba(0, 243, 255, 0.1);
        font-weight: 900;
        letter-spacing: 2px;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(0, 243, 255, 0.3);
        padding-bottom: 0.5rem;
    }
    
    /* Blinking recording indicator */
    .blink {
        animation: blink-animation 1.5s steps(5, start) infinite;
        -webkit-animation: blink-animation 1.5s steps(5, start) infinite;
        color: #ff0055;
        font-weight: bold;
        margin-right: 8px;
    }
    @keyframes blink-animation {
        to { visibility: hidden; }
    }
    @-webkit-keyframes blink-animation {
        to { visibility: hidden; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 243, 255, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 10px rgba(0, 243, 255, 0.03);
        margin-bottom: 1rem;
        transition: all 0.3s ease-in-out;
    }
    .glass-card:hover {
        border: 1px solid rgba(0, 243, 255, 0.35);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 243, 255, 0.08);
    }
    
    .card-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        color: #9ca3af;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .card-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        color: #00f3ff;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(0, 243, 255, 0.3);
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(0, 243, 255, 0.1);
    }
    
    /* Tab overrides */
    button[data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
        letter-spacing: 1px;
        color: #9ca3af;
    }
    button[aria-selected="true"] {
        color: #00f3ff !important;
        border-color: #00f3ff !important;
    }
    
    /* Streamlit input custom typography */
    label, p, span, li {
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Status indicators */
    .status-ok {
        color: #10b981;
        font-family: 'Share Tech Mono', monospace;
    }
    .status-warning {
        color: #f59e0b;
        font-family: 'Share Tech Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "running" not in st.session_state:
    st.session_state.running = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "yolov8n.pt"

# Cache YOLO models to avoid loading them repeatedly
@st.cache_resource
def load_detector(model_name):
    return YOLODetector(model_name)

# Cyberpunk Custom Bounding Box & HUD Drawings
def draw_cyberpunk_box(img, bbox, track_id, class_name, confidence):
    left, top, right, bottom = bbox
    
    # Generate unique color hash based on ID
    np.random.seed(int(track_id.split('_')[0]) if '_' in track_id else int(track_id) if track_id.isdigit() else 0)
    color = tuple(int(c) for c in np.random.randint(50, 255, size=3)) # BGR
    # Ensure color is somewhat bright/neon
    
    # Draw corners (HUD bracket ticks)
    length = max(10, int((right - left) * 0.15))
    thickness = 2
    
    # Top-Left Corner
    cv2.line(img, (left, top), (left + length, top), color, thickness)
    cv2.line(img, (left, top), (left, top + length), color, thickness)
    # Top-Right Corner
    cv2.line(img, (right, top), (right - length, top), color, thickness)
    cv2.line(img, (right, top), (right, top + length), color, thickness)
    # Bottom-Left Corner
    cv2.line(img, (left, bottom), (left + length, bottom), color, thickness)
    cv2.line(img, (left, bottom), (left, bottom - length), color, thickness)
    # Bottom-Right Corner
    cv2.line(img, (right, bottom), (right - length, bottom), color, thickness)
    cv2.line(img, (right, bottom), (right, bottom - length), color, thickness)
    
    # Thin overall outline
    cv2.rectangle(img, (left, top), (right, bottom), color, 1)
    
    # HUD Label tag background
    label = f"ID: {track_id} | {class_name.upper()} {confidence:.0%}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    
    # Draw dark label background box
    cv2.rectangle(img, (left, top - h - 8), (left + w + 10, top), (7, 11, 20), -1)
    cv2.rectangle(img, (left, top - h - 8), (left + w + 10, top), color, 1)
    
    # Write Label Text
    cv2.putText(img, label, (left + 5, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("<h2 class='cyber-title' style='font-size:1.4rem; border-bottom:1px solid rgba(0, 243, 255, 0.2);'>SYSTEM CONTROL</h2>", unsafe_allow_html=True)

# 1. Input Source Selection
input_source_type = st.sidebar.selectbox(
    "INPUT SOURCE TYPE",
    ["Sample Video", "Upload Video", "Webcam Device"],
    index=0
)

video_file_path = None
webcam_id = 0

if input_source_type == "Sample Video":
    # Intel Person-Vehicle-Car detection clip
    video_file_path = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    st.sidebar.caption("💡 Loading pre-configured public tracking sample.")
    
elif input_source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "UPLOAD VIDEO FILE", 
        type=["mp4", "avi", "mov", "mkv"], 
        help="Supported: mp4, avi, mov, mkv"
    )
    if uploaded_file is not None:
        # Save uploaded file temporarily to read via OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_file_path = tfile.name
        st.sidebar.success("Video Uploaded Successfully.")
    else:
        st.sidebar.warning("Please upload an MP4/AVI/MOV file.")

elif input_source_type == "Webcam Device":
    webcam_id = st.sidebar.number_input("WEBCAM INDEX / DEVICE ID", min_value=0, max_value=5, value=0, step=1)
    st.sidebar.caption("📸 Index 0 is typically the default built-in camera.")

# 2. YOLO Model Selection
yolo_model_size = st.sidebar.selectbox(
    "YOLOv8 MODEL PROFILE",
    ["yolov8n.pt (Nano - Fast)", "yolov8s.pt (Small - Balanced)", "yolov8m.pt (Medium - Accurate)"],
    index=0
)
model_name = yolo_model_size.split(" ")[0]

# Initialize detector to cache names list
try:
    detector = load_detector(model_name)
    class_names = list(detector.names.values())
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# 3. Model Parameters
st.sidebar.markdown("<p style='font-family:Orbitron; font-size:0.85rem; margin-top:1rem;'>DETECTION PARAMETERS</p>", unsafe_allow_html=True)
conf_threshold = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.05, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IOU THRESHOLD", 0.05, 1.0, 0.45, 0.05)

# Class Filter Selection
selected_classes = st.sidebar.multiselect(
    "TARGET CLASS FILTER",
    class_names,
    default=["person", "car", "bicycle", "motorcycle", "bus", "truck"]
)

# 4. DeepSORT Parameters
st.sidebar.markdown("<p style='font-family:Orbitron; font-size:0.85rem; margin-top:1rem;'>DEEPSORT PARAMETERS</p>", unsafe_allow_html=True)
max_age = st.sidebar.slider("MAX AGE (MISSED FRAMES)", 10, 100, 30, 5)
max_cosine_distance = st.sidebar.slider("MAX COSINE DISTANCE (RE-ID)", 0.1, 1.0, 0.2, 0.05)

# Surveillance Control buttons
st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)

if col_start.button("🟢 INITIALIZE", use_container_width=True):
    st.session_state.running = True

if col_stop.button("🔴 SHUTDOWN", use_container_width=True):
    st.session_state.running = False

if st.sidebar.button("🧹 RESET METRICS", use_container_width=True):
    st.session_state.analytics_data = None
    st.rerun()

# ----------------- MAIN PANEL -----------------
st.markdown("<h1><span class='blink'>●</span> SECURE-EYE // AI SURVEILLANCE DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6b7280; font-family: Share Tech Mono; margin-top: -1rem;'>TACTICAL MULTI-OBJECT DETECTION & TRACKING CONSOLE v1.0.0</p>", unsafe_allow_html=True)

# Main Grid Row 1: KPI HUD Cards (Glassmorphism layout)
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# Placeholders for KPIs
with kpi_col1:
    kpi_active = st.empty()
with kpi_col2:
    kpi_cumulative = st.empty()
with kpi_col3:
    kpi_fps = st.empty()
with kpi_col4:
    kpi_confidence = st.empty()

# Initialize KPIs defaults if no analytics run
def update_kpi_hud(active=0, cumulative=0, fps=0.0, avg_conf=0.0):
    kpi_active.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Active Targets</div>
        <div class="card-value">{active}</div>
        <div class="status-ok">● SYSTEM TRACKING</div>
    </div>
    """, unsafe_allow_html=True)

    kpi_cumulative.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Cumulative Targets</div>
        <div class="card-value">{cumulative}</div>
        <div class="status-ok">● RECORDED SECURE</div>
    </div>
    """, unsafe_allow_html=True)

    kpi_fps.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Processing Speed</div>
        <div class="card-value">{fps:.1f} FPS</div>
        <div class="status-ok">● REALTIME PROCESS</div>
    </div>
    """, unsafe_allow_html=True)

    kpi_confidence.markdown(f"""
    <div class="glass-card">
        <div class="card-title">Avg Confidence</div>
        <div class="card-value">{avg_conf:.1%}</div>
        <div class="status-ok">● PROBABILISTIC MATCH</div>
    </div>
    """, unsafe_allow_html=True)

# Initial HUD setup
update_kpi_hud()

# Main Grid Row 2: Live feed and Analytics plots
feed_col, plot_col = st.columns([2, 1])

with feed_col:
    st.markdown("<p style='font-family: Orbitron; font-size: 1rem; color: #00f3ff; margin-bottom: 0.5rem;'>👁️ LIVE SURVEILLANCE FEED</p>", unsafe_allow_html=True)
    video_placeholder = st.empty()
    # If not running, show instructions or last frame
    if not st.session_state.running:
        if st.session_state.last_frame is not None:
            video_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
        else:
            video_placeholder.markdown("""
            <div style="background: rgba(17, 24, 39, 0.4); border: 2px dashed rgba(0, 243, 255, 0.3); border-radius: 12px; height: 450px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <h3 style="color: #00f3ff; font-family: Orbitron; margin: 0;">TACTICAL FEED OFFLINE</h3>
                <p style="color: #9ca3af; font-family: Rajdhani; font-size: 1.1rem; margin-top: 0.5rem;">Click 'INITIALIZE' in the system control panel to activate tracking.</p>
            </div>
            """, unsafe_allow_html=True)

with plot_col:
    st.markdown("<p style='font-family: Orbitron; font-size: 1rem; color: #00f3ff; margin-bottom: 0.5rem;'>📊 REAL-TIME ANALYTICS DASHBOARD</p>", unsafe_allow_html=True)
    
    # Placeholders for Plotly charts
    chart_active_targets = st.empty()
    chart_cumulative_targets = st.empty()
    chart_fps = st.empty()

# Main Grid Row 3: Tabs for logs and diagnostics
tab1, tab2, tab3 = st.tabs(["📋 Surveillance logs", "⚙️ Diagnostics", "📥 Export Panel"])

with tab1:
    st.markdown("<p style='font-family: Orbitron; font-size: 0.95rem; color: #00f3ff;'>LIVE TARGET ACQUISITION LOG</p>", unsafe_allow_html=True)
    log_table_placeholder = st.empty()
    # Set default empty table
    log_table_placeholder.markdown("<p style='color:#6b7280; font-family:Share Tech Mono;'>No target logs acquired. Start tracking...</p>", unsafe_allow_html=True)

with tab2:
    st.markdown("<p style='font-family: Orbitron; font-size: 0.95rem; color: #00f3ff;'>DIAGNOSTICS & SYSTEM MATRIX</p>", unsafe_allow_html=True)
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        gpu_status_text = 'AVAILABLE (ENABLED)' if getattr(st.session_state, 'gpu_accel', False) else 'NOT DETECTED (CPU FALLBACK)'
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:0;">
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[SYSTEM] CUDA / GPU ACCELERATION: <span class="status-ok">{gpu_status_text}</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[YOLO] SELECTED DETECTOR PROFILE: <span class="status-ok">{model_name}</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[DEEPSORT] TRACKER RE-ID EMBEDDER: <span class="status-ok">MobileNet (PyTorch)</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[DEEPSORT] TRACKER MAX AGE FRAME: <span class="status-ok">{max_age} frames</span></p>
        </div>
        """, unsafe_allow_html=True)
    with diag_col2:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:0;">
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[HUD] RENDER DRAW BRACKETS: <span class="status-ok">ENABLED</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[HUD] TRAJECTORY MOTION TRAILS: <span class="status-ok">ENABLED (Max 30 centers)</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[FILTER] DETECT CLASSES LOADED: <span class="status-ok">80 classes</span></p>
            <p style="font-family:Share Tech Mono; margin: 2px 0;">[SESSION] TIME INITIATED: <span class="status-ok">Ready to stream</span></p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<p style='font-family: Orbitron; font-size: 0.95rem; color: #00f3ff;'>EXPORT SURVEILLANCE REPORT</p>", unsafe_allow_html=True)
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.write("Generate and download a CSV table of all detected and tracked objects during this tracking session.")
        csv_download_placeholder = st.empty()
    with export_col2:
        st.write("Capture a screenshot of the current processing frame for analysis or inclusion in report briefs.")
        screenshot_placeholder = st.empty()


# ----------------- RUNTIME TRACKING LOOP -----------------
if st.session_state.running:
    # 1. Re-initialize classes and parameters
    selected_class_ids = [detector.get_class_id_from_name(cls) for cls in selected_classes]
    selected_class_ids = [cid for cid in selected_class_ids if cid is not None]
    
    # Check GPU availability for diagnostics display
    try:
        import torch
        st.session_state.gpu_accel = torch.cuda.is_available()
    except:
        st.session_state.gpu_accel = False
        
    # Re-initialize tracker and analytics
    tracker = ObjectTracker(
        max_age=max_age, 
        nms_max_overlap=1.0, 
        max_cosine_distance=max_cosine_distance
    )
    analytics = AnalyticsManager()
    
    # 2. Open stream
    is_webcam = (input_source_type == "Webcam Device")
    if is_webcam:
        cap = cv2.VideoCapture(webcam_id)
    else:
        # For sample URL or uploaded local path
        if not video_file_path:
            st.error("Error: Please select a valid video source.")
            st.session_state.running = False
            st.rerun()
        cap = cv2.VideoCapture(video_file_path)
        
    if not cap.isOpened():
        st.error(f"Error: Unable to connect to source: {input_source_type}")
        st.session_state.running = False
        st.rerun()
        
    # Initialize rolling parameters
    fps_history_list = []
    active_history_list = []
    frame_indices = []
    
    frame_count = 0
    
    # Stream process loop
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Reaching end of stream source or stream interrupted.")
            break
            
        frame_count += 1
        t_start = time.time()
        
        # Keep copy of raw frame for screenshots
        raw_frame = frame.copy()
        
        # Run YOLOv8 Detector
        deepsort_detections, raw_boxes = detector.detect(
            frame, 
            conf_threshold=conf_threshold, 
            iou_threshold=iou_threshold, 
            classes=selected_class_ids
        )
        
        # Run DeepSORT Tracker
        active_tracks = tracker.update(deepsort_detections, frame)
        
        # Compute dynamic frame rate
        t_end = time.time()
        current_fps = 1.0 / (t_end - t_start)
        
        # Update statistics & paths
        analytics.update(active_tracks, current_fps)
        
        # Draw cyber visual additions
        analytics.draw_paths(frame, active_tracks)
        for track in active_tracks:
            draw_cyberpunk_box(
                frame, 
                track["bbox"], 
                track["track_id"], 
                track["class_name"], 
                track["confidence"]
            )
            
        # Convert BGR frame to RGB for Streamlit rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save current frame state for screenshot captures
        st.session_state.last_frame = rgb_frame
        
        # Save analytics statistics in state for exports
        st.session_state.analytics_data = analytics.get_summary_stats()
        
        # Retrieve computed statistics
        stats = analytics.get_summary_stats()
        active_cnt = sum(stats["active_counts"].values())
        cumulative_cnt = stats["total_objects_tracked"]
        avg_conf = stats["average_confidence"]
        
        # 1. Update KPI numbers in real time
        update_kpi_hud(active=active_cnt, cumulative=cumulative_cnt, fps=current_fps, avg_conf=avg_conf)
        
        # 2. Render processed stream frame
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # 3. Update real-time analytics graphs
        # Track statistics history
        fps_history_list.append(current_fps)
        active_history_list.append(active_cnt)
        frame_indices.append(frame_count)
        
        # Keep history lists capped to avoid bloating
        if len(frame_indices) > 50:
            frame_indices.pop(0)
            fps_history_list.pop(0)
            active_history_list.pop(0)
            
        # Draw dynamic charts with custom colors matching theme
        # Plot 1: Target Activity over time
        fig_active = go.Figure()
        fig_active.add_trace(go.Scatter(
            x=frame_indices, y=active_history_list,
            mode='lines', name='Active Targets',
            line=dict(color='#00f3ff', width=2),
            fill='tozeroy', fillcolor='rgba(0, 243, 255, 0.1)'
        ))
        fig_active.update_layout(
            title="ACTIVE TARGETS OVER TIME",
            title_font=dict(family="Orbitron", size=12, color="#00f3ff"),
            paper_bgcolor='rgba(15,23,42,0.6)',
            plot_bgcolor='rgba(15,23,42,0)',
            margin=dict(l=30, r=20, t=40, b=30),
            height=200,
            xaxis=dict(showgrid=False, color="#9ca3af"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#9ca3af"),
            showlegend=False
        )
        chart_active_targets.plotly_chart(fig_active, use_container_width=True)
        
        # Plot 2: Cumulative bar chart by Object Class
        cum_counts = stats["cumulative_counts"]
        if cum_counts:
            df_cum = pd.DataFrame([{"Class": k.capitalize(), "Count": v} for k, v in cum_counts.items()])
            fig_bar = px.bar(
                df_cum, x="Count", y="Class", orientation="h",
                color_discrete_sequence=['#b026ff']
            )
            fig_bar.update_layout(
                title="CUMULATIVE OBJECT COUNTS",
                title_font=dict(family="Orbitron", size=12, color="#b026ff"),
                paper_bgcolor='rgba(15,23,42,0.6)',
                plot_bgcolor='rgba(15,23,42,0)',
                margin=dict(l=50, r=20, t=40, b=30),
                height=180,
                xaxis=dict(showgrid=True, gridcolor="#1e293b", color="#9ca3af"),
                yaxis=dict(showgrid=False, color="#9ca3af")
            )
            chart_cumulative_targets.plotly_chart(fig_bar, use_container_width=True)
        else:
            chart_cumulative_targets.caption("Waiting for object target acquisition...")
            
        # Plot 3: Processing Speed (FPS)
        fig_fps = go.Figure()
        fig_fps.add_trace(go.Scatter(
            x=frame_indices, y=fps_history_list,
            mode='lines', name='FPS',
            line=dict(color='#10b981', width=2)
        ))
        fig_fps.update_layout(
            title="SYSTEM FRAME RATE (FPS)",
            title_font=dict(family="Orbitron", size=12, color="#10b981"),
            paper_bgcolor='rgba(15,23,42,0.6)',
            plot_bgcolor='rgba(15,23,42,0)',
            margin=dict(l=30, r=20, t=40, b=30),
            height=150,
            xaxis=dict(showgrid=False, color="#9ca3af"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#9ca3af"),
            showlegend=False
        )
        chart_fps.plotly_chart(fig_fps, use_container_width=True)
        
        # 4. Render logs table
        if analytics.detection_history:
            log_df = pd.DataFrame(analytics.detection_history)
            # Display latest 8 events in custom dataframe table
            log_table_placeholder.dataframe(
                log_df.tail(8)[::-1], # Reverse order to show newest at top
                use_container_width=True
            )
            
        # Handle Streamlit updates
        time.sleep(0.01) # Yield execution thread slightly
        
    # Release resources when loop ends
    cap.release()
    st.session_state.running = False

# ----------------- ACTIONS & EXPORTS -----------------
# Render download utilities in export panel (shows data from last active session)
if st.session_state.analytics_data is not None:
    stats = st.session_state.analytics_data
    
    # 1. Download logs CSV button
    if len(analytics.detection_history) > 0:
        csv_data = analytics.export_csv()
        csv_download_placeholder.download_button(
            label="📥 DOWNLOAD LOGS AS CSV",
            data=csv_data,
            file_name=f"surveillance_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        csv_download_placeholder.info("No targets recorded to log yet.")
        
    # 2. Download screenshot button
    if st.session_state.last_frame is not None:
        # Convert RGB array back to PIL Image and save bytes
        pil_img = Image.fromarray(st.session_state.last_frame)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        screenshot_placeholder.download_button(
            label="📸 DOWNLOAD CURRENT FRAME SCREENSHOT",
            data=byte_im,
            file_name=f"surveillance_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    else:
        screenshot_placeholder.info("No active frame to capture.")
        
    # Update KPI metric display on UI with last analytics data when stopped
    update_kpi_hud(
        active=sum(stats["active_counts"].values()),
        cumulative=stats["total_objects_tracked"],
        fps=stats["average_fps"],
        avg_conf=stats["average_confidence"]
    )
    
    # Display full summary report in Markdown
    st.markdown("<p style='font-family: Orbitron; font-size: 1rem; color: #00f3ff; margin-top:1.5rem;'>📄 SESSION ANALYTICAL BRIEF</p>", unsafe_allow_html=True)
    summary_md = f"""
    <div class="glass-card">
        <p style="font-family: Share Tech Mono; margin: 4px 0;">[SESSION STATUS] COMPLETE</p>
        <p style="font-family: Share Tech Mono; margin: 4px 0;">[ACTIVE MONITORING DURATION] {stats['duration_seconds']} seconds</p>
        <p style="font-family: Share Tech Mono; margin: 4px 0;">[TOTAL UNIQUE TARGETS IDENTIFIED] {stats['total_objects_tracked']}</p>
        <p style="font-family: Share Tech Mono; margin: 4px 0;">[SYSTEM PERFORMANCE AVERAGE] {stats['average_fps']} frames per second</p>
        <p style="font-family: Share Tech Mono; margin: 4px 0;">[CONFIDENCE FIT RATE AVERAGE] {stats['average_confidence']:.2%}</p>
        <p style="font-family: Share Tech Mono; margin: 4px 0; color:#b026ff;">[OBJECT CLASSES ENCOUNTERED]:</p>
        <ul style="font-family: Share Tech Mono; margin-left: 20px;">
    """
    for cls, count in stats["cumulative_counts"].items():
        summary_md += f"<li>{cls.upper()}: {count} targets</li>"
    summary_md += "</ul></div>"
    st.markdown(summary_md, unsafe_allow_html=True)
