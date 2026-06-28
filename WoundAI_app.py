"""
WoundAI — Clinical Wound Detection
Run with: streamlit run app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="WoundAI",
    page_icon="🩹",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "best-8.pt"        # change if your weights live elsewhere
CONFIDENCE_THRESHOLD = 0.20      # below this, a detection is flagged amber as low-confidence

# Initialize a widget version counter in session state if it doesn't exist
if "widget_version" not in st.session_state:
    st.session_state["widget_version"] = 0

# ----------------------------------------------------------------------------
# Reset handler logic
# ----------------------------------------------------------------------------
def reset_all_inputs():
    # Incrementing this counter forces Streamlit to recreate completely clean widgets
    st.session_state["widget_version"] += 1

# ----------------------------------------------------------------------------
# Styling — clinical dark theme, monospace for all data readouts
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #1A2129;
        --bg-card: #212A33;
        --bg-card-hover: #252F39;
        --border: #313D49;
        --text: #E8ECEF;
        --text-dim: #8B97A3;
        --teal: #2DD4BF;
        --teal-dim: #1A3B38;
        --amber: #F59E0B;
        --amber-dim: #3D2F14;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Header */
    .wa-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .wa-header .mark {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
    }
    .wa-header .mark span {
        color: var(--teal);
    }
    .wa-tagline {
        color: var(--text-dim);
        font-size: 0.92rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.4rem;
    }

    /* Step labels */
    .wa-step {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 1.6rem 0 0.7rem 0;
    }
    .wa-step .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--teal);
        flex-shrink: 0;
    }
    .wa-step .dot.idle {
        background: var(--border);
    }
    .wa-step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-dim);
    }

    /* Upload zone override */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed var(--border);
        border-radius: 10px;
        padding: 1.1rem;
        background: var(--bg-card);
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--teal);
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--text) !important;
    }

    /* Custom layout for secondary buttons to play nice with custom css */
    div[data-testid="stButton"] button[pity-button="true"], 
    .stButton button:not([kind="primary"]) {
        background: transparent !important;
        color: var(--text-dim) !important;
        border: 1px solid var(--border) !important;
    }
    .stButton button:not([kind="primary"]):hover {
        border-color: #EF4444 !important;
        color: #EF4444 !important;
        opacity: 1;
    }

    /* Primary button rules */
    .stButton button[kind="primary"] {
        background: var(--teal) !important;
        color: #0B1512 !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.15s ease;
    }
    .stButton button[kind="primary"]:hover {
        opacity: 0.88;
    }

    /* Vitals-style readout card */
    .vitals-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .vitals-label {
        font-size: 0.82rem;
        color: var(--text-dim);
        font-weight: 500;
    }
    .vitals-class {
        font-size: 0.95rem;
        color: var(--text);
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .vitals-reading {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
    }
    .vitals-reading.ok { color: var(--teal); }
    .vitals-reading.low { color: var(--amber); }
    .vitals-unit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-dim);
        margin-left: 0.15rem;
    }
    .vitals-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        padding: 0.18rem 0.5rem;
        border-radius: 4px;
        margin-top: 0.35rem;
        display: inline-block;
    }
    .vitals-tag.ok { background: var(--teal-dim); color: var(--teal); }
    .vitals-tag.low { background: var(--amber-dim); color: var(--amber); }

    /* Summary strip */ 
    .summary-strip {
        display: flex;
        gap: 1.6rem;
        margin: 1rem 0 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-dim);
    }
    .summary-strip b {
        color: var(--text);
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2.4rem 1rem;
        color: var(--text-dim);
        font-size: 0.88rem;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--bg-card);
    }

    .footnote {
        color: var(--text-dim);
        font-size: 0.76rem;
        margin-top: 2.2rem;
        border-top: 1px solid var(--border);
        padding-top: 1rem;
    }

    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="wa-header"><span class="mark">Wound<span>AI</span></span></div>
    <div class="wa-tagline">Image-based wound detection — upload a clinical photo to flag and localize wound regions.</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Step 1 & 2 handle file input and early st.stop() if active_file is None...
# ----------------------------------------------------------------------------

# ... (Image display code happens here) ...

# ----------------------------------------------------------------------------
# Step 3 — Actions (Detect & Clear)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="wa-step"><div class="dot"></div><div class="wa-step-label">03 · Actions</div></div>
    """,
    unsafe_allow_html=True,
)

# 1. First, create the layout columns
col1, col2 = st.columns([1, 4])

# 2. Second, define the 'run' variable inside the column layout
with col1:
    run = st.button("Detect wounds", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear image", type="secondary", on_click=reset_all_inputs)

# 3. Third, check 'if run:' AFTER it has been defined above
if run:
    with st.spinner("Analyzing image..."):
        t0 = time.time()
        
        # 1. Convert PIL image to numpy array
        img_array = np.array(image)
        
        # 2. Convert RGB to BGR for proper model processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 3. Pass the corrected BGR array into the model
        results = model(img_bgr, verbose=False, conf=0.01)
        elapsed = time.time() - t0

        result = results[0]
        
        # Force override the result dictionary for the image annotations
        result.names = {
            0: "Abrasion",
            1: "Laceration",
            2: "Surgical Wound",
            3: "Ulcer"
        }
        
        boxes = result.boxes
        
        # 4. Plot results
        res_plotted = result.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # ... (Rest of your result displaying HTML blocks go here) ...
# ----------------------------------------------------------------------------
# Step 1 — Image Input (Upload or Camera)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="wa-step"><div class="dot"></div><div class="wa-step-label">01 · Input image</div></div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["📁 File Upload", "📷 Use Camera"])

uploaded_file = None
camera_file = None
v = st.session_state["widget_version"]

with tab_upload:
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพ...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"file_uploader_{v}"
    )

with tab_camera:
    camera_file = st.camera_input(
        "Take clinical photo",
        label_visibility="collapsed",
        key=f"camera_input_{v}"
    )

active_file = uploaded_file or camera_file

if active_file is None:
    st.markdown(
        '<div class="empty-state">No image yet. Upload an image file or take a photo to begin.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="footnote">WoundAI is a decision-support tool, not a diagnostic device. '
        "Clinical judgment should always take precedence over model output.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Load image from active file handle
image = Image.open(active_file).convert("RGB")
file_name = getattr(active_file, "name", "Captured Photo.jpg")

st.markdown(
    """
    <div class="wa-step"><div class="dot"></div><div class="wa-step-label">02 · Review image</div></div>
    """,
    unsafe_allow_html=True,
)
st.image(image, caption=file_name, use_container_width=True)

# ----------------------------------------------------------------------------
# Step 3 — Actions (Detect & Clear)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="wa-step"><div class="dot"></div><div class="wa-step-label">03 · Actions</div></div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Detect wounds", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear image", type="secondary", on_click=reset_all_inputs)

if run:
    with st.spinner("Analyzing image..."):
        t0 = time.time()
        
        # 1. Convert PIL image to numpy array
        img_array = np.array(image)
        
        # 2. Convert RGB to BGR for proper model processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 3. Pass the corrected BGR array into the model
        results = model(img_bgr, verbose=False, conf=0.01)
        elapsed = time.time() - t0

        result = results[0]
        boxes = result.boxes
        
        # 4. Plot results (Ultralytics plots natively in BGR, convert it back to RGB for Streamlit)
        res_plotted = result.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Debug trace to console logs
        print("RAW DETECTIONS FOUND:", len(boxes))
        if len(boxes) > 0:
            print("Confidences:", [float(b.conf[0]) for b in boxes])

    st.markdown(
        """
        <div class="wa-step"><div class="dot"></div><div class="wa-step-label">04 · Results</div></div>
        """,
        unsafe_allow_html=True,
    )

    n_detections = len(boxes) if boxes is not None else 0

    st.markdown(
        f"""
        <div class="summary-strip">
            <span><b>{n_detections}</b> detection{'s' if n_detections != 1 else ''}</span>
            <span><b>{elapsed:.2f}s</b> inference time</span>
            <span><b>{image.size[0]}×{image.size[1]}</b> px</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(res_rgb, caption="Detected regions", use_container_width=True)

    if n_detections > 0:
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class {cls_id}") if isinstance(model.names, dict) else str(cls_id)
            is_low = conf < CONFIDENCE_THRESHOLD
            tone = "low" if is_low else "ok"
            tag_text = "LOW CONFIDENCE" if is_low else "ABOVE THRESHOLD"

            st.markdown(
                f"""
                <div class="vitals-card">
                    <div>
                        <div class="vitals-label">Detection #{i + 1}</div>
                        <div class="vitals-class">{cls_name}</div>
                        <div class="vitals-tag {tone}">{tag_text}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="vitals-reading {tone}">{conf * 100:.1f}</span><span class="vitals-unit">% conf</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty-state">No wound regions detected in this image.</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footnote">WoundAI is a decision-support tool, not a diagnostic device. '
    "Clinical judgment should always take precedence over model output.</div>",
    unsafe_allow_html=True,
)
