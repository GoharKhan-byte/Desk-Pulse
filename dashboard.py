import os
import cv2
import time
import csv
import torch
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
from ultralytics import YOLO

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Workplace AI Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# DEEP CSS STYLING OVERRIDES
# -------------------------------------------------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1f2937 !important;
        }
        section[data-testid="stSidebar"] * { color: #f3f4f6 !important; }

        div[data-baseweb="select"], 
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: #1e1b34 !important;
            border-color: #4c1d95 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div { background-color: #1e1b34 !important; }
        span[data-baseweb="tag"] {
            background-color: #7c3aed !important;
            border-radius: 6px !important;
            padding: 2px 8px !important;
            color: #ffffff !important;
        }
        ul[role="listbox"], div[role="option"] {
            background-color: #1e1b34 !important;
            color: #f8fafc !important;
        }
        div[role="option"]:hover { background-color: #7c3aed !important; }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #1f2937 0%, #111827 100%) !important;
            border: 1px solid #374151 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stMetric"]:hover { border-color: #fbbf24 !important; }

        div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
            color: #fbbf24 !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }

        /* Idle-alert row highlight */
        .idle-alert-row {
            background-color: rgba(244, 63, 94, 0.15);
            border-left: 3px solid #f43f5e;
            padding: 6px 10px;
            border-radius: 6px;
            margin-bottom: 4px;
            font-size: 0.85rem;
        }

        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .sub-header { font-size: 0.95rem; color: #9ca3af; margin-bottom: 20px; }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #f3f4f6;
            margin-bottom: 12px;
            border-left: 3px solid #38bdf8;
            padding-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚡ Workplace Activity Intelligence Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time YOLO11 Computer Vision Pipeline & Dynamic Telemetry Analytics</p>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# MODEL INITIALIZATION
# -------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(weights_path="best.pt"):
    device = 0 if torch.cuda.is_available() else "cpu"
    model_obj = YOLO(os.path.abspath(weights_path))
    class_names = list(model_obj.names.values())
    return model_obj, device, class_names

model, device, all_class_names = load_yolo_model("best.pt")

# -------------------------------------------------------------------
# SESSION STATE — persists across reruns/frames
# -------------------------------------------------------------------
if "track_state" not in st.session_state:
    # track_id -> {"class": str, "idle_seconds": float, "last_seen": float, "current_class_since": float}
    st.session_state.track_state = {}

if "telemetry_log" not in st.session_state:
    st.session_state.telemetry_log = []  # buffered rows before flush to disk

LOG_PATH = "floorpulse_telemetry.csv"
FLUSH_EVERY_N_FRAMES = 30
STALE_TRACK_TIMEOUT = 5.0  # seconds — drop a track if unseen this long

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Engine Controls")

video_source = "Prod2.mp4"
st.sidebar.info(f"📹 **Active Source:** `{video_source}`")
st.sidebar.divider()

selected_classes = st.sidebar.multiselect(
    "Target Classes to Detect",
    options=all_class_names,
    default=all_class_names,
    help="Select or remove classes to filter real-time detection telemetry."
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.45, 0.05)
run_stream = st.sidebar.toggle("Enable Live Inference", value=True)

st.sidebar.divider()
st.sidebar.markdown("### ⏱️ Idle Detection")
idle_class = st.sidebar.selectbox(
    "Class treated as 'idle'",
    options=all_class_names,
    help="Track time spent by each worker in this class label."
)
idle_alert_seconds = st.sidebar.number_input(
    "Alert threshold (seconds)", min_value=5, max_value=1800, value=60, step=5
)
log_telemetry = st.sidebar.toggle("Log telemetry to CSV", value=True)

st.sidebar.divider()
st.sidebar.markdown(f"**Hardware Device:** `{'GPU (CUDA)' if device == 0 else 'CPU'}`")

selected_class_indices = [idx for idx, name in model.names.items() if name in selected_classes]

# -------------------------------------------------------------------
# LAYOUT PLACEHOLDERS
# -------------------------------------------------------------------
st.markdown('<div class="section-title">Telemetry & Class Breakdown</div>', unsafe_allow_html=True)
top_kpi_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
col_video, col_chart = st.columns([2.0, 1.2], gap="large")

with col_video:
    st.markdown('<div class="section-title">Live Video Feed</div>', unsafe_allow_html=True)
    frame_window = st.image([])

with col_chart:
    st.markdown('<div class="section-title">Activity Distribution</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()

st.markdown('<div class="section-title">🚨 Idle Alerts</div>', unsafe_allow_html=True)
alert_placeholder = st.empty()

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def update_track_states(results_obj, model_names, idle_class_name, now):
    """Update per-track idle-seconds using tracked box IDs. Returns list of (track_id, class, idle_seconds)."""
    state = st.session_state.track_state
    active_ids = set()

    if results_obj is not None and results_obj.boxes is not None and results_obj.boxes.id is not None:
        ids = results_obj.boxes.id.int().tolist()
        classes = results_obj.boxes.cls.int().tolist()

        for tid, cls_idx in zip(ids, classes):
            cls_name = model_names[cls_idx]
            active_ids.add(tid)
            prev = state.get(tid)

            if prev is None:
                state[tid] = {
                    "class": cls_name,
                    "idle_seconds": 0.0,
                    "last_seen": now,
                    "current_class_since": now,
                }
                continue

            delta = now - prev["last_seen"]

            if cls_name == prev["class"]:
                if cls_name == idle_class_name:
                    prev["idle_seconds"] += delta
            else:
                # class changed — reset idle accumulation for this track
                prev["class"] = cls_name
                prev["current_class_since"] = now
                if cls_name != idle_class_name:
                    prev["idle_seconds"] = 0.0

            prev["last_seen"] = now

    # drop stale tracks that vanished from the feed
    for tid in list(state.keys()):
        if tid not in active_ids and (now - state[tid]["last_seen"]) > STALE_TRACK_TIMEOUT:
            del state[tid]

    return [(tid, d["class"], d["idle_seconds"]) for tid, d in state.items()]


def flush_telemetry_if_needed(frame_count):
    if not st.session_state.telemetry_log:
        return
    if frame_count % FLUSH_EVERY_N_FRAMES != 0:
        return
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "track_id", "class", "idle_seconds"])
        writer.writerows(st.session_state.telemetry_log)
    st.session_state.telemetry_log.clear()


# -------------------------------------------------------------------
# PROCESSING LOOP
# -------------------------------------------------------------------
if run_stream and os.path.exists(video_source):
    cap = cv2.VideoCapture(video_source)
    prev_time = time.time()
    frame_count = 0

    while cap.isOpened() and run_stream:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        now = time.time()

        if selected_class_indices:
            results = model.track(
                source=frame,
                persist=True,
                conf=conf_threshold,
                classes=selected_class_indices,
                device=device,
                verbose=False
            )
            annotated_frame = results[0].plot()
            results0 = results[0]
        else:
            annotated_frame = frame
            results0 = None

        fps = 1 / (now - prev_time + 1e-6)
        prev_time = now

        detected_classes = []
        if results0 is not None and results0.boxes is not None and len(results0.boxes) > 0:
            class_indices = results0.boxes.cls.int().tolist()
            detected_classes = [model.names[idx] for idx in class_indices]
        class_counts = Counter(detected_classes)

        # ---- idle-time tracking ----
        track_rows = update_track_states(results0, model.names, idle_class, now)
        unique_workers = len(track_rows)
        idle_alerts = [(tid, secs) for tid, cls, secs in track_rows
                        if cls == idle_class and secs >= idle_alert_seconds]

        if log_telemetry:
            ts = datetime.now().isoformat(timespec="seconds")
            for tid, cls, secs in track_rows:
                st.session_state.telemetry_log.append([ts, tid, cls, round(secs, 1)])
            flush_telemetry_if_needed(frame_count)

        # ---- KPI cards ----
        with top_kpi_placeholder.container():
            total_card_items = 3 + len(selected_classes)
            cols_per_row = min(total_card_items, 6)
            kpi_cols = st.columns(cols_per_row)

            kpi_cols[0].metric("TOTAL OBJECTS", f"{len(detected_classes)}")
            kpi_cols[1].metric("INFERENCE FPS", f"{fps:.1f}")
            kpi_cols[2].metric("UNIQUE WORKERS", f"{unique_workers}")

            for idx, cls_name in enumerate(selected_classes):
                col_idx = (idx + 3) % cols_per_row
                count = class_counts.get(cls_name, 0)
                kpi_cols[col_idx].metric(f"{cls_name.upper()}", f"{count}")

        # ---- video feed ----
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

        # ---- activity chart ----
        with chart_placeholder.container():
            if class_counts:
                df_counts = pd.DataFrame(
                    list(class_counts.items()), columns=["Activity Class", "Count"]
                ).sort_values(by="Count", ascending=True)

                fig = px.bar(
                    df_counts, x="Count", y="Activity Class", orientation='h',
                    color="Activity Class",
                    color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f43f5e", "#fbbf24"],
                    text="Count"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc", family="Inter, sans-serif"),
                    showlegend=False, height=380, margin=dict(l=0, r=10, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor="#1f2937", zeroline=False),
                    yaxis=dict(showgrid=False)
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True, key=f"activity_chart_{frame_count}")
            else:
                st.info("No active detections for selected classes.")

        # ---- idle alerts ----
        with alert_placeholder.container():
            if idle_alerts:
                for tid, secs in sorted(idle_alerts, key=lambda x: -x[1]):
                    st.markdown(
                        f'<div class="idle-alert-row">Worker #{tid} idle for '
                        f'{secs:.0f}s (threshold {idle_alert_seconds}s)</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No active idle alerts.")

    cap.release()
elif not os.path.exists(video_source):
    st.error(f"❌ Video file `{video_source}` not found in project workspace!")
