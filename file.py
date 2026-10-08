import math
import os
import struct
import time
import wave
import cv2
import pandas as pd
import torch
import streamlit as st
from ultralytics import YOLO

# Threading to prevent audio playback from blocking the main video loop
import threading

# Try importing sound backends (pygame or native winsound)
try:
    import pygame

    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    import winsound

    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False


# -------------------------------------------------------------------
# AUTOMATIC SOUND GENERATOR (SYNTHESIZED BEEP)
# -------------------------------------------------------------------
def generate_default_beep(
    filename="alarm.wav", duration_sec=0.4, freq=1000
):
    """Generates a simple 1000Hz alert beep WAV file if missing."""
    if not os.path.exists(filename):
        try:
            sample_rate = 44100
            num_samples = int(sample_rate * duration_sec)
            with wave.open(filename, "w") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                for i in range(num_samples):
                    value = int(
                        32767
                        * 0.5
                        * math.sin(2 * math.pi * freq * i / sample_rate)
                    )
                    wav_file.writeframes(struct.pack("<h", value))
        except Exception as e:
            st.error(f"Failed to generate default alarm file: {e}")


# Pre-generate alarm.wav on startup if missing
generate_default_beep("alarm.wav")


# -------------------------------------------------------------------
# SOUND ALARM HELPER WITH FALLBACKS
# -------------------------------------------------------------------
def _play_audio_worker(sound_path="alarm.wav"):
    """Internal worker function executed inside a thread."""
    if PYGAME_AVAILABLE and os.path.exists(sound_path):
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            return
        except Exception:
            pass

    # Windows Native Sound Fallback
    if WINSOUND_AVAILABLE:
        try:
            winsound.Beep(1000, 400)  # 1000 Hz for 400 ms
        except Exception:
            pass


def trigger_alarm_async(sound_path="alarm.wav"):
    """Fires alarm sound asynchronously without lagging video stream."""
    thread = threading.Thread(
        target=_play_audio_worker, args=(sound_path,), daemon=True
    )
    thread.start()


# -------------------------------------------------------------------
# PAGE SETUP & MODERN DARK-MODE CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Telemetry & Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
    }
    
    /* Top Header */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0px 20px 0px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .status-badge {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #00E676;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .status-badge-alarm {
        background-color: rgba(255, 23, 68, 0.2);
        color: #FF1744;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #FF1744;
        font-size: 13px;
        font-weight: 600;
        animation: blinker 1.2s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.4; }
    }
    .live-dot {
        height: 8px;
        width: 8px;
        background-color: currentColor;
        border-radius: 50%;
        display: inline-block;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #58A6FF !important;
    }

    /* Video Frame Styling */
    .stImage > img {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161B22;
        border-radius: 8px;
        padding: 0px 20px;
        color: #8B949E;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F242D !important;
        color: #FFFFFF !important;
        border-color: #58A6FF !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# SESSION STATE INIT (FOR INCIDENTS & SNAPSHOTS)
# -------------------------------------------------------------------
if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = []
if "snapshots" not in st.session_state:
    st.session_state.snapshots = []


# -------------------------------------------------------------------
# MODEL INITIALIZATION
# -------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    openvino_path = os.path.abspath("best_openvino_model")
    pt_path = os.path.abspath("best.pt")

    if os.path.exists(openvino_path):
        weights_to_load = openvino_path
        engine_type = "OpenVINO (FP16)"
    else:
        weights_to_load = pt_path
        engine_type = "PyTorch (.pt)"

    model_obj = YOLO(weights_to_load, task="detect")
    class_names = list(model_obj.names.values())
    return model_obj, class_names, engine_type


model, all_class_names, engine_type = load_yolo_model()

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption(f"Hardware Accelerator: `{engine_type}`")
    st.markdown("---")

    selected_classes = st.multiselect(
        "Target Classes",
        options=all_class_names,
        default=all_class_names if all_class_names else [],
    )
    selected_indices = [
        idx
        for idx, name in enumerate(all_class_names)
        if name in selected_classes
    ]

    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    movement_tolerance = st.slider("Movement Tolerance (px)", 5, 100, 35, 5)
    idle_timeout = st.slider("Idle Timeout Threshold (s)", 2, 60, 10, 1)

    st.markdown("---")
    st.markdown("### 🚨 Alarm & Audio Features")
    enable_audio_alarm = st.checkbox("Enable Audio Beeper", value=True)
    alarm_sound_file = st.text_input("Alarm File Path", "alarm.wav")

    st.markdown("---")
    video_source_mode = st.selectbox(
        "Video Source Mode",
        [
            "Built-in / Integrated Webcam",
            "External Camera Index",
            "Video File",
        ],
    )

    cam_index_override = 0
    file_path = "Prod2.mp4"
    if video_source_mode == "External Camera Index":
        cam_index_override = st.number_input(
            "Select Camera Index", min_value=0, max_value=5, value=1, step=1
        )
    elif video_source_mode == "Video File":
        file_path = st.text_input("Video File Path", "Prod2.mp4")

    run_stream = st.checkbox("⚡ Launch System Engine", value=False)


# -------------------------------------------------------------------
# WEBCAM INIT
# -------------------------------------------------------------------
def get_camera_capture(source_mode, override_idx, video_path):
    if source_mode == "Video File":
        if not os.path.exists(video_path):
            st.error(f"❌ File not found: `{video_path}`")
           # --- REPLACE HERE (Lines ~288-296) ---
                        class_name = all_class_names[cls_id] if cls_id < len(all_class_names) else f"Class-{cls_id}"

                        if is_idle:
                            box_color = (68, 23, 255)  # Red
                            status_label = f"ID #{track_id} {class_name} | IDLE ({idle_secs}s)"
                            idle_count_this_frame += 1

                            row_entry = {
                                "Timestamp": time.strftime("%H:%M:%S"),
                                "Target ID": f"#{track_id}",
                                "Class": class_name,
                                "Idle Duration (s)": idle_secs,
                                "Status": "CRITICAL IDLE"
                            }
                            table_rows.append(row_entry)

                            # Trigger Audio Alarm once per breach
                            if not tracker_state[track_id]["alerted"]:
                                if enable_audio_alarm:
                                    play_system_beep(1200, 500)
                                tracker_state[track_id]["alerted"] = True
                                st.session_state.incident_logs.append(row_entry)
                        else:
                            box_color = (118, 230, 0)  # Green
                            status_label = f"ID #{track_id} {class_name} | ACTIVE"
                        # --- END REPLACEMENT ---

# -------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE (COMMAND CENTER)
# -------------------------------------------------------------------
header_placeholder = st.empty()

with header_placeholder.container():
    st.markdown(
        """
        <div class="header-container">
            <div class="header-title">⚡ AI Telemetry & Idle Command Center</div>
            <div class="status-badge"><span class="live-dot"></span> SYSTEM READY</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

tab_live, tab_analytics, tab_logs = st.tabs(
    ["🎥 Live Monitor", "📊 Analytics & Telemetry", "📑 Audit Logs & Exports"]
)

# --- TAB 1: LIVE MONITOR ---
with tab_live:
    col_video, col_side_stats = st.columns([2.4, 1.6], gap="medium")

    with col_video:
        st.markdown("##### Live Video Feed")
        frame_window = st.image([])

    with col_side_stats:
        st.markdown("##### Real-Time Telemetry")
        kpi1, kpi2, kpi3 = st.columns(3)
        metric_total = kpi1.metric("Active Tracks", "0")
        metric_idle = kpi2.metric("Idle Count", "0")
        metric_fps = kpi3.metric("Stream FPS", "0.0")

        st.markdown("---")
        st.markdown("##### ⚠️ Active Violations Table")
        table_placeholder = st.empty()

# --- TAB 2: ANALYTICS & TELEMETRY ---
with tab_analytics:
    st.markdown("##### Idle Duration Distribution Per Target")
    chart_placeholder = st.empty()

# --- TAB 3: AUDIT LOGS & EXPORTS ---
with tab_logs:
    st.markdown("##### Incident History Log")
    logs_table_placeholder = st.empty()

    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        export_btn_placeholder = st.empty()

# -------------------------------------------------------------------
# PROCESSING LOOP
# -------------------------------------------------------------------
tracker_state = {}

if run_stream:
    cap = get_camera_capture(
        video_source_mode,
        cam_index_override,
        file_path if video_source_mode == "Video File" else "",
    )

    if cap is None or not cap.isOpened():
        st.error("❌ Unable to open camera stream.")
    else:
        fps_counter = 0
        start_time = time.time()
        fps_display = 0.0

        while run_stream:
            ret, frame = cap.read()
            if not ret or frame is None:
                if video_source_mode == "Video File":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            current_timestamp = time.time()
            fps_counter += 1
            elapsed = current_timestamp - start_time
            if elapsed >= 1.0:
                fps_display = round(fps_counter / elapsed, 1)
                fps_counter = 0
                start_time = current_timestamp

            if video_source_mode != "Video File":
                frame = cv2.flip(frame, 1)

            results = model.track(
                source=frame,
                persist=True,
                conf=conf_thresh,
                classes=selected_indices if selected_indices else None,
                tracker="bytetrack.yaml",
                verbose=False,
                imgsz=640,
            )

            active_tracks_this_frame = set()
            idle_count_this_frame = 0
            table_rows = []
            chart_data = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxy_coords = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.int().cpu().tolist()

                    for track_id, bbox, cls_id in zip(
                        track_ids, xyxy_coords, class_ids
                    ):
                        active_tracks_this_frame.add(track_id)
                        x1, y1, x2, y2 = map(int, bbox)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        class_name = (
                            all_class_names[cls_id]
                            if cls_id < len(all_class_names)
                            else "Object"
                        )

                        if track_id not in tracker_state:
                            tracker_state[track_id] = {
                                "last_pos": (cx, cy),
                                "stationary_start": current_timestamp,
                                "total_idle": 0.0,
                                "is_idle": False,
                                "alerted": False,
                            }
                        else:
                            prev_x, prev_y = tracker_state[track_id][
                                "last_pos"
                            ]
                            distance = math.sqrt(
                                (cx - prev_x) ** 2 + (cy - prev_y) ** 2
                            )

                            if distance > movement_tolerance:
                                tracker_state[track_id]["last_pos"] = (cx, cy)
                                tracker_state[track_id][
                                    "stationary_start"
                                ] = current_timestamp
                                tracker_state[track_id]["total_idle"] = 0.0
                                tracker_state[track_id]["is_idle"] = False
                                tracker_state[track_id]["alerted"] = False
                            else:
                                idle_duration = (
                                    current_timestamp
                                    - tracker_state[track_id][
                                        "stationary_start"
                                    ]
                                )
                                tracker_state[track_id]["total_idle"] = (
                                    idle_duration
                                )
                                if idle_duration >= idle_timeout:
                                    tracker_state[track_id]["is_idle"] = True

                        is_idle = tracker_state[track_id]["is_idle"]
                        idle_secs = int(tracker_state[track_id]["total_idle"])

                        chart_data.append({
                            "Target": f"ID #{track_id} ({class_name})",
                            "Idle Time (s)": idle_secs,
                        })

                        if is_idle:
                            box_color = (68, 23, 255)  # BGR Red
                            status_label = (
                                f"ID #{track_id} | IDLE: {idle_secs}s"
                            )
                            idle_count_this_frame += 1

                            row_entry = {
                                "Timestamp": time.strftime("%H:%M:%S"),
                                "Target ID": f"#{track_id}",
                                "Class": class_name,
                                "Idle Duration (s)": idle_secs,
                                "Status": "CRITICAL IDLE",
                            }
                            table_rows.append(row_entry)

                            # Trigger Non-Blocking Audio Alarm & Log Event
                            if not tracker_state[track_id]["alerted"]:
                                if enable_audio_alarm:
                                    trigger_alarm_async(alarm_sound_file)
                                tracker_state[track_id]["alerted"] = True
                                st.session_state.incident_logs.append(row_entry)
                        else:
                            box_color = (118, 230, 0)  # BGR Green
                            status_label = f"ID #{track_id} | ACTIVE"

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), box_color, 2
                        )
                        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                        cv2.putText(
                            frame,
                            status_label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_color,
                            2,
                        )

            # Cleanup missing tracks
            stale_tracks = [
                tid
                for tid in tracker_state
                if tid not in active_tracks_this_frame
            ]
            for tid in stale_tracks:
                del tracker_state[tid]

            # Dynamic Alarm Badge in Header
            with header_placeholder.container():
                if idle_count_this_frame > 0:
                    st.markdown(
                        """
                        <div class="header-container">
                            <div class="header-title">⚡ AI Telemetry & Idle Command Center</div>
                            <div class="status-badge-alarm"><span class="live-dot"></span> 🚨 IDLE BREACH DETECTED</div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="header-container">
                            <div class="header-title">⚡ AI Telemetry & Idle Command Center</div>
                            <div class="status-badge"><span class="live-dot"></span> SYSTEM NORMAL</div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # Render Frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

            # Render Metrics
            metric_total.metric("Active Tracks", len(active_tracks_this_frame))
            metric_idle.metric("Idle Count", idle_count_this_frame)
            metric_fps.metric("Stream FPS", f"{fps_display}")

            # Render Active Table
            if table_rows:
                df_active = pd.DataFrame(table_rows)
                table_placeholder.dataframe(
                    df_active, hide_index=True, use_container_width=True
                )
            else:
                table_placeholder.info(
                    "All targets actively moving. Zero active violations."
                )

            # Render Tab 2 Analytics Chart
            if chart_data:
                df_chart = pd.DataFrame(chart_data).set_index("Target")
                chart_placeholder.bar_chart(df_chart)

            # Render Tab 3 Incident Logs & CSV Export
            if st.session_state.incident_logs:
                df_logs = pd.DataFrame(st.session_state.incident_logs)
                logs_table_placeholder.dataframe(
                    df_logs, hide_index=True, use_container_width=True
                )

                csv_data = df_logs.to_csv(index=False).encode("utf-8")
                # Render Tab 3 Incident Logs & CSV Export
if st.session_state.incident_logs:
    df_logs = pd.DataFrame(st.session_state.incident_logs)
    logs_table_placeholder.dataframe(
        df_logs, hide_index=True, use_container_width=True
    )

    csv_data = df_logs.to_csv(index=False).encode("utf-8")
    export_btn_placeholder.download_button(
        label="📥 Download Audit Log (CSV)",
        data=csv_data,
        file_name=f"idle_audit_log_{int(time.time())}.csv",
        mime="text/csv",
        key="download_audit_log_btn",  # <--- PASS A UNIQUE KEY HERE
    )