"""
🏗️ AI Construction Safety Monitoring System
Smart City PPE Detection Dashboard powered by YOLOv8
"""

import streamlit as st
import cv2
import os
import glob
import tempfile
import numpy as np
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image

# ╔══════════════════════════════════════════════════════════════╗
# ║                    PAGE CONFIGURATION                       ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="AI Safety Monitor | Smart City",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════════╗
# ║                    CUSTOM CSS THEME                         ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
    /* ── Global Dark Theme ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* ── Header Banner ── */
    .dashboard-header {
        background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(233, 69, 96, 0.3);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .dashboard-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-card .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.3rem 0;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-card .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    /* ── Color Variants ── */
    .metric-blue { border-left: 4px solid #3b82f6; }
    .metric-blue .metric-value { color: #60a5fa; }
    .metric-green { border-left: 4px solid #22c55e; }
    .metric-green .metric-value { color: #4ade80; }
    .metric-orange { border-left: 4px solid #f59e0b; }
    .metric-orange .metric-value { color: #fbbf24; }
    .metric-red { border-left: 4px solid #ef4444; }
    .metric-red .metric-value { color: #f87171; }
    
    /* ── Safety Status Banner ── */
    .safety-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .safety-safe h3 { color: #4ade80; margin: 0; font-size: 1.3rem; }
    .safety-safe p { color: rgba(255,255,255,0.7); margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    
    .safety-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
        animation: pulse-danger 2s ease-in-out infinite;
    }
    .safety-danger h3 { color: #f87171; margin: 0; font-size: 1.3rem; }
    .safety-danger p { color: rgba(255,255,255,0.7); margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 15px rgba(239,68,68,0.2); }
        50% { box-shadow: 0 0 30px rgba(239,68,68,0.4); }
    }
    
    /* ── Section Headers ── */
    .section-header {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0 1rem 0;
    }
    
    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    .sidebar-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green {
        background: rgba(34,197,94,0.2);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .badge-yellow {
        background: rgba(245,158,11,0.2);
        color: #fbbf24;
        border: 1px solid rgba(245,158,11,0.3);
    }
    
    /* ── Glass Container ── */
    .glass-container {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 0.5rem 0;
    }
    
    /* ── Detection Log ── */
    .detection-log {
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* ── Hide default Streamlit elements for cleaner look ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MODEL LOADING                            ║
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_resource
def load_model():
    """Scan for the latest trained model; fall back to generic YOLOv8n (auto-downloads)."""
    import re
    try:
        candidates = glob.glob("runs/detect/train*/weights/best.pt")
        # Sort by train number (e.g. train10 > train9 > train7)
        def train_num(path):
            m = re.search(r"train(\d+)", path)
            return int(m.group(1)) if m else 0
        candidates.sort(key=train_num, reverse=True)
    except Exception:
        candidates = []

    if candidates and os.path.exists(candidates[0]):
        return YOLO(candidates[0]), True, candidates[0]
    else:
        # Ultralytics auto-downloads yolov8n.pt if not present
        return YOLO("yolov8n.pt"), False, "yolov8n.pt"

model, is_custom, model_path = load_model()


# ╔══════════════════════════════════════════════════════════════╗
# ║                    HEADER BANNER                            ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<div class="dashboard-header">
    <h1>🏗️ AI Construction Safety Monitor</h1>
    <p>Real-time PPE compliance detection powered by YOLOv8 · Smart City Initiative</p>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    SIDEBAR                                  ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    # ── ACUD Logo ──
    st.image(
        "assets/acud_logo.png",
        use_container_width=True,
        caption="Inspired by smart city initiatives such as Egypt's New Administrative Capital"
    )
    st.markdown("---")
    
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    # Model status badge
    if model is None:
        st.error("❌ No model file found!")
    elif is_custom:
        st.markdown(f'<span class="sidebar-badge badge-green">● Custom Model Loaded</span>', unsafe_allow_html=True)
        st.caption(f"📂 `{os.path.basename(os.path.dirname(os.path.dirname(model_path)))}`")
    else:
        st.markdown(f'<span class="sidebar-badge badge-yellow">● Fallback Model</span>', unsafe_allow_html=True)
        st.caption("Using generic YOLOv8n — train a custom model for PPE detection.")

    st.markdown("---")

    # Source selection
    source_option = st.radio(
        "📡 Navigation",
        ("ℹ️ About", "📷 Image Upload", "🎥 Video Upload", "🔴 Live Webcam", "📹 CCTV Camera"),
        index=0,
    )

    st.markdown("---")

    # Detection controls
    detection_enabled = st.toggle("🔍 Enable Detection", value=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.75rem;'>"
        "Built with Streamlit + YOLOv8<br>Smart City PPE Monitor v2.0</p>",
        unsafe_allow_html=True,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║                    HELPER FUNCTIONS                         ║
# ╚══════════════════════════════════════════════════════════════╝

def classify_name(raw_name: str) -> str:
    """Map any model class name to one of: helmet, no_helmet, vest, no_vest, person, or other."""
    n = raw_name.lower()
    if "no-hardhat" in n or "no-helmet" in n or "no hardhat" in n:
        return "no_helmet"
    if "hardhat" in n or "helmet" in n:
        return "helmet"
    if "no-safety vest" in n or "no-vest" in n or "no vest" in n or "no-safety" in n:
        return "no_vest"
    if "vest" in n or "safety vest" in n:
        return "vest"
    if n == "person":
        return "person"
    return "other"


def process_frame(img, conf_thresh=0.25):
    """
    Run YOLOv8 inference on a BGR image.
    Returns: (annotated_bgr_image, stats_dict)
    """
    results = model(img, stream=False, verbose=False, conf=conf_thresh)
    annotated = results[0].plot()

    stats = {
        "persons": [], "helmets": [], "vests": [],
        "no_helmets": [], "no_vests": [],
        "violations": 0, "detections": [],
    }

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_name = model.names.get(cls_id, "Unknown")
            category = classify_name(raw_name)

            stats["detections"].append({
                "class": raw_name,
                "category": category,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

            if category == "person":
                stats["persons"].append((x1, y1, x2, y2))
            elif category == "helmet":
                stats["helmets"].append((x1, y1, x2, y2))
            elif category == "vest":
                stats["vests"].append((x1, y1, x2, y2))
            elif category == "no_helmet":
                stats["no_helmets"].append((x1, y1, x2, y2))
            elif category == "no_vest":
                stats["no_vests"].append((x1, y1, x2, y2))

    # ── Violation Logic ──
    # Method 1: explicit "NO-Hardhat" class from dataset
    stats["violations"] += len(stats["no_helmets"])

    # Method 2: spatial check — person bbox without overlapping helmet
    for px1, py1, px2, py2 in stats["persons"]:
        has_helmet = False
        for hx1, hy1, hx2, hy2 in stats["helmets"]:
            hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                has_helmet = True
                break
        # Also check if a "no_helmet" box overlaps this person
        for nhx1, nhy1, nhx2, nhy2 in stats["no_helmets"]:
            ncx, ncy = (nhx1 + nhx2) / 2, (nhy1 + nhy2) / 2
            if px1 <= ncx <= px2 and py1 <= ncy <= py2:
                has_helmet = False  # explicitly marked
                break

        if not has_helmet:
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 3)
            label_y = max(30, py1 - 12)
            cv2.putText(annotated, "!! NO HELMET !!", (px1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            stats["violations"] += 1

    return annotated, stats


def render_metrics(stats):
    """Render the 4 metric cards + safety status banner."""
    n_people = len(stats["persons"])
    n_helmets = len(stats["helmets"])
    n_vests = len(stats["vests"])
    n_violations = stats["violations"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card metric-blue">
            <div class="metric-icon">👷</div>
            <div class="metric-value">{n_people}</div>
            <div class="metric-label">People</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card metric-green">
            <div class="metric-icon">⛑️</div>
            <div class="metric-value">{n_helmets}</div>
            <div class="metric-label">Helmets</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card metric-orange">
            <div class="metric-icon">🦺</div>
            <div class="metric-value">{n_vests}</div>
            <div class="metric-label">Vests</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card metric-red">
            <div class="metric-icon">⚠️</div>
            <div class="metric-value">{n_violations}</div>
            <div class="metric-label">Violations</div>
        </div>""", unsafe_allow_html=True)

    # Safety status banner
    if n_violations == 0 and n_people > 0:
        st.markdown("""
        <div class="safety-safe">
            <h3>✅ ALL CLEAR</h3>
            <p>All detected workers are wearing proper PPE</p>
        </div>""", unsafe_allow_html=True)
    elif n_violations > 0:
        st.markdown(f"""
        <div class="safety-danger">
            <h3>🚨 SAFETY VIOLATION DETECTED</h3>
            <p>{n_violations} worker(s) found without required protective equipment</p>
        </div>""", unsafe_allow_html=True)


def render_charts(stats):
    """Render compliance pie chart and detection bar chart side by side."""
    n_helmets = len(stats["helmets"])
    n_no_helmets = stats["violations"]
    n_vests = len(stats["vests"])
    n_people = len(stats["persons"])

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown('<div class="section-header">📊 PPE Compliance</div>', unsafe_allow_html=True)
        compliant = n_helmets
        non_compliant = n_no_helmets
        if compliant + non_compliant > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Compliant (Helmet)", "Non-Compliant"],
                values=[compliant, non_compliant],
                hole=0.55,
                marker=dict(colors=["#22c55e", "#ef4444"]),
                textfont=dict(size=13, color="white"),
                hoverinfo="label+percent+value",
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.8)"),
                legend=dict(font=dict(color="rgba(255,255,255,0.7)")),
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No helmet data to display yet.")

    with col_bar:
        st.markdown('<div class="section-header">📈 Detection Breakdown</div>', unsafe_allow_html=True)
        categories = ["People", "Helmets", "Vests", "No Helmet"]
        values = [n_people, n_helmets, n_vests, n_no_helmets]
        colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
        fig_bar = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition="auto",
            textfont=dict(size=14, color="white"),
        )])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.8)"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def render_detection_log(stats):
    """Show an expandable detection details table."""
    detections = stats["detections"]
    if not detections:
        return
    with st.expander(f"📋 Detection Log  —  {len(detections)} objects detected", expanded=False):
        header = "| # | Class | Category | Confidence | Bounding Box |"
        sep    = "|---|-------|----------|------------|--------------|"
        rows = [header, sep]
        for i, d in enumerate(detections, 1):
            conf_pct = f"{d['confidence']*100:.1f}%"
            bbox_str = f"({d['bbox'][0]}, {d['bbox'][1]}) → ({d['bbox'][2]}, {d['bbox'][3]})"
            rows.append(f"| {i} | {d['class']} | {d['category']} | {conf_pct} | {bbox_str} |")
        st.markdown("\n".join(rows))


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MAIN CONTENT AREA                        ║
# ╚══════════════════════════════════════════════════════════════╝

# ── ABOUT PAGE ──
if source_option == "ℹ️ About":
    st.title("🏗️ AI Construction Safety Monitoring System")
    st.markdown("---")

    # 1. Overview & Problem Statement
    with st.container():
        st.subheader("📌 Overview")
        st.markdown("""
        The **AI Construction Safety Monitoring System** is an advanced computer vision platform designed to automate safety compliance on large-scale construction sites. 
        By leveraging state-of-the-art deep learning models, this system provides **real-time detection of Personal Protective Equipment (PPE)**, ensuring that workers are equipped with essential safety gear such as helmets and high-visibility vests.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **The Problem:** Manual safety monitoring in sprawling, high-risk construction environments is labor-intensive, prone to human error, and difficult to scale. Undetected safety violations can lead to severe accidents and regulatory penalties.")
        with col2:
            st.success("⚙️ **The Solution:** Automating compliance checks using YOLOv8 object detection. By integrating with existing CCTV or camera infrastructure, the system flags violations instantly, creating a proactive rather than reactive safety culture.")

    st.markdown("---")

    # 2. Key Features & Impact
    with st.container():
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("✨ Core Features")
            st.markdown("""
            - **Real-Time PPE Detection:** Simultaneously detects persons, helmets, and safety vests with high accuracy.
            - **Instant Violation Alerts:** Automatically identifies workers missing required protective gear.
            - **Visual Analytics Dashboard:** Provides live metrics, compliance breakdowns, and detection logs.
            - **Multi-Source Ingestion:** Supports image uploads, pre-recorded video, live webcams, and IP/CCTV camera streams.
            """)
            
        with col4:
            st.subheader("📈 System Impact")
            st.markdown("""
            - 🛡️ **Improves Worker Safety:** Drastically reduces workplace accidents through 24/7 monitoring.
            - ⏱️ **Scalable Monitoring:** A single system can monitor multiple camera feeds simultaneously.
            - 📊 **Data-Driven Insights:** Provides site managers with quantitative data on safety compliance over time.
            - 🏢 **Supports Smart Infrastructure:** Lays the groundwork for fully digitized, intelligent construction management.
            """)

    st.markdown("---")

    # 3. Smart City Relevance
    with st.container():
        st.subheader("🏙️ Smart City Relevance")
        st.markdown("""
        This platform is specifically designed to integrate seamlessly into **Smart City ecosystems**. 
        In landmark developments like **Egypt's New Administrative Capital (developed by ACUD)**, centralized monitoring of mega-construction projects is critical. 
        
        By processing live video streams directly at the edge or via centralized cloud infrastructure, city authorities and primary contractors can ensure standardized safety compliance across dozens of interconnected construction zones simultaneously.
        """)

    st.markdown("---")

    # 4. Future Work
    with st.container():
        st.subheader("🚀 Future Work & Enhancements")
        st.markdown("""
        - **Extended Detection Capabilities:** Adding support for safety gloves, safety boots, and harness detection.
        - **IoT Integration:** Connecting the detection system directly to automated site alarms, access control gates, or supervisor mobile devices.
        - **Edge Deployment:** Optimizing the YOLOv8 model for deployment on edge devices like NVIDIA Jetson Nano for offline, low-latency processing.
        - **Cloud Analytics:** Building a persistent database to track historical compliance trends and generate predictive safety reports.
        """)


if model is None and source_option != "ℹ️ About":
    st.error("🚫 **No model found.** Please place `yolov8n.pt` in the project folder or train a custom model first.")
    st.stop()

# ── IMAGE UPLOAD ──
if source_option == "📷 Image Upload":
    st.markdown('<div class="section-header">📷 Image Analysis</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a construction site image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if detection_enabled:
            with st.spinner("🔍 Analyzing image for safety compliance..."):
                annotated, stats = process_frame(img_bgr)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Metrics row
            render_metrics(stats)

            # Processed image
            st.markdown('<div class="section-header">🖼️ Processed Image</div>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

            # Charts
            render_charts(stats)

            # Detection log
            render_detection_log(stats)
        else:
            st.image(image, caption="Original Image (detection disabled)", use_container_width=True)
    else:
        st.info("👆 Upload a construction site image to begin safety analysis.")


# ── VIDEO UPLOAD ──
elif source_option == "🎥 Video Upload":
    st.markdown('<div class="section-header">🎥 Video Analysis</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader(
        "Drop a construction site video here",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Could not open video file.")
        else:
            # Layout: video on left, live metrics on right
            vid_col, stats_col = st.columns([2, 1])

            with vid_col:
                stframe = st.empty()
            with stats_col:
                st.markdown('<div class="section-header">📊 Live Metrics</div>', unsafe_allow_html=True)
                metrics_placeholder = st.empty()
                status_placeholder = st.empty()

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1

                # Process every frame (or skip for speed)
                if detection_enabled:
                    annotated, stats = process_frame(frame)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

                    # Update live metrics every 3 frames for performance
                    if frame_count % 3 == 0:
                        with metrics_placeholder.container():
                            st.metric("👷 People", len(stats["persons"]))
                            st.metric("⛑️ Helmets", len(stats["helmets"]))
                            st.metric("🦺 Vests", len(stats["vests"]))
                            st.metric("⚠️ Violations", stats["violations"])
                        with status_placeholder.container():
                            if stats["violations"] > 0:
                                st.error(f"🚨 {stats['violations']} violation(s) detected!")
                            elif len(stats["persons"]) > 0:
                                st.success("✅ All workers compliant")
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.info("🏁 Video processing complete.")
    else:
        st.info("👆 Upload a construction site video to begin safety analysis.")


# ── LIVE WEBCAM ──
elif source_option == "🔴 Live Webcam":
    st.markdown('<div class="section-header">🔴 Live Webcam Feed</div>', unsafe_allow_html=True)
    st.caption("Ensure your browser has camera permissions. Click Start to begin.")

    btn_col1, btn_col2 = st.columns(2)
    start_cam = btn_col1.button("▶️  Start Webcam", use_container_width=True)
    stop_cam = btn_col2.button("⏹️  Stop Webcam", use_container_width=True)

    if start_cam:
        # Layout
        cam_col, stats_col = st.columns([2, 1])
        with cam_col:
            frame_placeholder = st.empty()
        with stats_col:
            st.markdown('<div class="section-header">📊 Live Metrics</div>', unsafe_allow_html=True)
            metrics_placeholder = st.empty()
            status_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Make sure no other app is using it!")
        else:
            frame_count = 0
            while cap.isOpened() and not stop_cam:
                success, frame = cap.read()
                if not success:
                    st.error("⚠️ Lost webcam connection.")
                    break
                frame_count += 1

                if detection_enabled:
                    annotated, stats = process_frame(frame)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    if frame_count % 3 == 0:
                        with metrics_placeholder.container():
                            st.metric("👷 People", len(stats["persons"]))
                            st.metric("⛑️ Helmets", len(stats["helmets"]))
                            st.metric("🦺 Vests", len(stats["vests"]))
                            st.metric("⚠️ Violations", stats["violations"])
                        with status_placeholder.container():
                            if stats["violations"] > 0:
                                st.error(f"🚨 {stats['violations']} violation(s)!")
                            elif len(stats["persons"]) > 0:
                                st.success("✅ All clear")
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            cap.release()


# ── CCTV CAMERA ──
elif source_option == "📹 CCTV Camera":
    st.markdown('<div class="section-header">📹 CCTV Camera — 24/7 Monitoring</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-container">
        <p style="color:rgba(255,255,255,0.8); line-height:1.7;">
            Connect to an IP camera or CCTV system for continuous, round-the-clock safety monitoring.
            Enter the RTSP or HTTP stream URL provided by your camera system below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Input fields for CCTV connection
    cctv_url = st.text_input(
        "📡 Camera Stream URL",
        placeholder="rtsp://username:password@192.168.1.100:554/stream1",
        help="Enter the RTSP or HTTP URL of your IP camera / CCTV stream. "
             "Common formats: rtsp://user:pass@IP:554/stream  or  http://IP:port/video"
    )

    with st.expander("💡 Common CCTV URL Formats", expanded=False):
        st.markdown("""
        | Camera Brand | Example URL |
        |---|---|
        | **Hikvision** | `rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101` |
        | **Dahua** | `rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` |
        | **Generic RTSP** | `rtsp://username:password@IP_ADDRESS:554/stream1` |
        | **HTTP Stream** | `http://IP_ADDRESS:8080/video` |
        | **USB Camera** | Enter `0` for default camera, `1` for second camera |
        """)

    btn_col1, btn_col2 = st.columns(2)
    start_cctv = btn_col1.button("▶️  Connect & Start Monitoring", use_container_width=True)
    stop_cctv = btn_col2.button("⏹️  Disconnect", use_container_width=True)

    if start_cctv:
        if not cctv_url:
            st.warning("⚠️ Please enter a camera stream URL above.")
        else:
            # Try numeric input (for USB cameras like 0, 1, 2)
            try:
                stream_source = int(cctv_url)
            except ValueError:
                stream_source = cctv_url

            cap = cv2.VideoCapture(stream_source)
            if not cap.isOpened():
                st.error(
                    "❌ **Could not connect to camera.** Please check:\n"
                    "- The URL is correct and accessible\n"
                    "- The camera is powered on and connected to the network\n"
                    "- Username and password are correct\n"
                    "- Your firewall is not blocking the connection"
                )
            else:
                st.success(f"✅ Connected to camera stream!")

                # Layout: feed on left, live metrics on right
                cam_col, stats_col = st.columns([2, 1])
                with cam_col:
                    frame_placeholder = st.empty()
                with stats_col:
                    st.markdown('<div class="section-header">📊 Live Metrics</div>', unsafe_allow_html=True)
                    metrics_placeholder = st.empty()
                    status_placeholder = st.empty()

                frame_count = 0
                while cap.isOpened() and not stop_cctv:
                    success, frame = cap.read()
                    if not success:
                        st.warning("⚠️ Lost connection to camera. Attempting to reconnect...")
                        cap.release()
                        cap = cv2.VideoCapture(stream_source)
                        continue
                    frame_count += 1

                    if detection_enabled:
                        annotated, stats = process_frame(frame)
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                        if frame_count % 3 == 0:
                            with metrics_placeholder.container():
                                st.metric("👷 People", len(stats["persons"]))
                                st.metric("⛑️ Helmets", len(stats["helmets"]))
                                st.metric("🦺 Vests", len(stats["vests"]))
                                st.metric("⚠️ Violations", stats["violations"])
                            with status_placeholder.container():
                                if stats["violations"] > 0:
                                    st.error(f"🚨 {stats['violations']} violation(s)!")
                                elif len(stats["persons"]) > 0:
                                    st.success("✅ All clear")
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                cap.release()
