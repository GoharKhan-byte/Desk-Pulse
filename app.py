import math
import os
import threading
import time
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from alerts import AlertManager

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time AI Idle & Tracking Dashboard (OpenVINO)",
    page_icon="🎥",
    layout="wide",
)


# -------------------------------------------------------------------
# THREADED CAMERA CLASS (HARDWARE DOWNSCALING FOR SPEED)
# -------------------------------------------------------------------
class ThreadedCamera:

  def __init__(self, source_mode, override_idx, video_path):
    self.source_mode = source_mode
    self.cap = None

    if source_mode == "Video File":
      if os.path.exists(video_path):
        self.cap = cv2.VideoCapture(video_path)
    else:
      target_idx = override_idx if source_mode == "External Camera Index" else 0

      # Try DirectShow API first (Fixes Windows camera access lock)
      cap = cv2.VideoCapture(target_idx, cv2.CAP_DSHOW)
      if cap.isOpened():
        self.cap = cap
      else:
        # Fallback to standard MSMF driver
        cap = cv2.VideoCapture(target_idx)
        if cap.isOpened():
          self.cap = cap
        else:
          # Loop through secondary indices if default index fails
          for test_idx in [1, 2, 3]:
            cap = cv2.VideoCapture(test_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
              st.sidebar.warning(
                  f"⚠️ Index 0 failed. Connected to Camera Index {test_idx}"
              )
              self.cap = cap
              break

    if self.cap and self.cap.isOpened():
      # Hardware buffer reduction & stream downscaling for max FPS
      self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
      self.grabbed, self.frame = self.cap.read()
    else:
      self.grabbed, self.frame = False, None

    self.started = False
    self.read_lock = threading.Lock()

  def start(self):
    if self.started or self.cap is None or not self.cap.isOpened():
      return self
    self.started = True
    self.thread = threading.Thread(target=self.update, args=(), daemon=True)
    self.thread.start()
    return self

  def update(self):
    while self.started:
      if self.cap is None or not self.cap.isOpened():
        break
      grabbed, frame = self.cap.read()
      with self.read_lock:
        self.grabbed = grabbed
        self.frame = frame

  def read(self):
    with self.read_lock:
      if not self.grabbed or self.frame is None:
        return False, None
      return True, self.frame.copy()

  def isOpened(self):
    return self.cap is not None and self.cap.isOpened()

  def set_pos_frame(self, pos):
    with self.read_lock:
      if self.cap:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

  def stop(self):
    self.started = False
    if hasattr(self, "thread") and self.thread.is_alive():
      self.thread.join(timeout=1.0)
    if self.cap:
      self.cap.release()


# -------------------------------------------------------------------
# OPENVINO MODEL INITIALIZATION
# -------------------------------------------------------------------
@st.cache_resource
def load_openvino_model():
  openvino_path = os.path.abspath("best_openvino_model")

  if not os.path.exists(openvino_path):
    st.error(
        f"❌ OpenVINO model directory not found at: `{openvino_path}`. "
        "Please run your OpenVINO export script first!"
    )
    st.stop()

  # Load OpenVINO model directory directly
  model_obj = YOLO(openvino_path, task="detect")
  class_names = list(model_obj.names.values())
  engine_type = "Intel OpenVINO (CPU Accelerated)"

  return model_obj, class_names, engine_type


# -------------------------------------------------------------------
# ALERT MANAGER INITIALIZATION (beep + CSV/SQLite logging)
# -------------------------------------------------------------------
@st.cache_resource
def load_alert_manager():
  return AlertManager(
      csv_path="idle_violations.csv",
      sqlite_path="floorpulse.db",   # set to None if you only want CSV
      cooldown_sec=10,                # min seconds between repeat beeps/logs for the SAME track
  )


model, all_class_names, engine_type = load_openvino_model()
alert_mgr = load_alert_manager()

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.title("⚙️ Tracking & Settings")
st.sidebar.info(f"⚡ Loaded Engine: `{engine_type}`")

selected_classes = st.sidebar.multiselect(
    "Select Target Classes",
    options=all_class_names,
    default=all_class_names if all_class_names else [],
)
selected_indices = [
    idx for idx, name in enumerate(all_class_names) if name in selected_classes
]

conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
movement_tolerance = st.sidebar.slider(
    "Movement Tolerance (Pixels)", 5, 100, 35, 5
)
idle_timeout = st.sidebar.slider("Idle Timeout Threshold (Seconds)", 2, 60, 10, 1)

# Frame Skip control for optimizing processing throughput
frame_skip_interval = st.sidebar.slider(
    "Inference Frame Skip Interval",
    min_value=1,
    max_value=5,
    value=2,
    help="Runs inference every Nth frame. Higher values increase video smoothness.",
)

video_source_mode = st.sidebar.selectbox(
    "Video Source Mode",
    ["Built-in / Integrated Webcam", "External Camera Index", "Video File"],
)

cam_index_override = 0
if video_source_mode == "External Camera Index":
  cam_index_override = st.sidebar.number_input(
      "Select Camera Index", min_value=0, max_value=5, value=1, step=1
  )
elif video_source_mode == "Video File":
  file_path = st.sidebar.text_input("Video File Path", "Prod2.mp4")

st.sidebar.markdown("---")
st.sidebar.markdown("**🔔 Alerts**")
alert_cooldown = st.sidebar.slider(
    "Alert Cooldown (seconds)", 3, 60, 10, 1,
    help="Minimum time between repeat beeps/logs for the SAME idle person."
)
alert_mgr.cooldown_sec = alert_cooldown  # live-tunable from the sidebar

run_stream = st.sidebar.checkbox("Start Live Stream", value=False)

# -------------------------------------------------------------------
# MAIN DASHBOARD LAYOUT
# -------------------------------------------------------------------
st.title("🎥 Real-Time Object Tracking & Idle Monitoring")

col_video, col_stats = st.columns([2.5, 1.5])

with col_video:
  st.subheader("Live Feed")
  frame_window = st.image([])

with col_stats:
  st.subheader("Real-Time Telemetry")
  kpi1, kpi2, kpi3 = st.columns(3)
  metric_total = kpi1.metric("Active Tracks", "0")
  metric_idle = kpi2.metric("Idle Count", "0")
  metric_fps = kpi3.metric("Stream FPS", "0")

  st.markdown("### ⚠️ Idle Violations Table")
  table_placeholder = st.empty()

# -------------------------------------------------------------------
# TRACKING STATE DATASTRUCTURE & CACHING
# -------------------------------------------------------------------
tracker_state = {}
cached_detections = []

# -------------------------------------------------------------------
# PROCESSING LOOP
# -------------------------------------------------------------------
if run_stream:
  cam = ThreadedCamera(
      video_source_mode,
      cam_index_override,
      file_path if video_source_mode == "Video File" else "",
  ).start()

  if not cam.isOpened():
    st.error(
        "❌ Unable to open camera stream. Ensure no other application"
        " (Teams, Zoom, Camera app) is using the webcam, or check privacy"
        " settings under Windows Settings > Privacy & Security > Camera."
    )
  else:
    fps_counter = 0
    frame_idx = 0
    start_time = time.time()
    fps_display = 0.0

    while run_stream:
      ret, frame = cam.read()
      if not ret or frame is None:
        if video_source_mode == "Video File":
          cam.set_pos_frame(0)
          time.sleep(0.01)
          continue
        else:
          st.error("⚠️ Stream interrupted or frame read failed.")
          break

      current_timestamp = time.time()
      frame_idx += 1

      # FPS Calculation
      fps_counter += 1
      elapsed = current_timestamp - start_time
      if elapsed >= 1.0:
        fps_display = round(fps_counter / elapsed, 1)
        fps_counter = 0
        start_time = current_timestamp

      # Horizontal flip for front webcams
      if video_source_mode != "Video File":
        frame = cv2.flip(frame, 1)

      # Determine whether to execute OpenVINO model inference or use cached boxes
      should_infer = (frame_idx % frame_skip_interval) == 0

      if should_infer:
        results = model.track(
            source=frame,
            persist=True,
            conf=conf_thresh,
            classes=selected_indices if selected_indices else None,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
        )

        cached_detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
          boxes = results[0].boxes
          if boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            xyxy_coords = boxes.xyxy.numpy()
            class_ids = boxes.cls.int().tolist()

            for track_id, bbox, cls_id in zip(
                track_ids, xyxy_coords, class_ids
            ):
              cached_detections.append((track_id, bbox, cls_id))

      active_tracks_this_frame = set()
      idle_count_this_frame = 0
      table_rows = []

      # Render tracking bounding boxes and state machine logic
      for track_id, bbox, cls_id in cached_detections:
        active_tracks_this_frame.add(track_id)
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        class_name = (
            all_class_names[cls_id]
            if cls_id < len(all_class_names)
            else "Object"
        )

        # Idle State Decision Logic
        if track_id not in tracker_state:
          tracker_state[track_id] = {
              "last_pos": (cx, cy),
              "stationary_start": current_timestamp,
              "total_idle": 0.0,
              "is_idle": False,
              "class_name": class_name,
          }
        else:
          prev_x, prev_y = tracker_state[track_id]["last_pos"]
          distance = math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)

          if distance > movement_tolerance:
            tracker_state[track_id]["last_pos"] = (cx, cy)
            tracker_state[track_id]["stationary_start"] = current_timestamp
            tracker_state[track_id]["total_idle"] = 0.0
            tracker_state[track_id]["is_idle"] = False
            # Person moved again — reset alert cooldown so a NEW idle
            # period starts fresh instead of being suppressed.
            alert_mgr.clear_track(track_id)
          else:
            idle_duration = (
                current_timestamp - tracker_state[track_id]["stationary_start"]
            )
            tracker_state[track_id]["total_idle"] = idle_duration
            if idle_duration >= idle_timeout:
              tracker_state[track_id]["is_idle"] = True

        # Drawing & Annotations
        is_idle = tracker_state[track_id]["is_idle"]
        idle_secs = int(tracker_state[track_id]["total_idle"])

        if is_idle:
          box_color = (0, 0, 255)
          status_label = f"ID #{track_id} {class_name} | IDLE: {idle_secs}s"
          idle_count_this_frame += 1
          table_rows.append({
              "ID": track_id,
              "Class": class_name,
              "Idle Time (s)": idle_secs,
              "Status": "⚠️ VIOLATION",
          })

          # ---- Fire beep + CSV/SQLite log (rate-limited per track_id) ----
          alert_mgr.raise_alert(
              track_id=track_id,
              idle_seconds=idle_secs,
              camera=video_source_mode,
          )
        else:
          box_color = (255, 255, 0)
          status_label = f"ID #{track_id} {class_name} | Active ({idle_secs}s)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        (w, h), _ = cv2.getTextSize(
            status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
        cv2.putText(
            frame,
            status_label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

      # Cleanup missing tracks
      stale_tracks = [
          tid for tid in tracker_state if tid not in active_tracks_this_frame
      ]
      for tid in stale_tracks:
        del tracker_state[tid]
        alert_mgr.clear_track(tid)  # free up cooldown memory for tracks that left the frame

      # Render to Streamlit Canvas
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame_window.image(frame_rgb, channels="RGB")

      # Update Telemetry Metrics only when values shift or on infer cycles
      if should_infer:
        metric_total.metric("Active Tracks", len(active_tracks_this_frame))
        metric_idle.metric("Idle Count", idle_count_this_frame)
        metric_fps.metric("Stream FPS", fps_display)

        if table_rows:
          df_idle = pd.DataFrame(table_rows)
          table_placeholder.dataframe(df_idle, hide_index=True)
        else:
          table_placeholder.info("No active idle violations detected.")

    cam.stop()
