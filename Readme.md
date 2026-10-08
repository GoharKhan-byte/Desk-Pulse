# 🎥 DeskPulse

**Real-time office presence and idle-time analytics, running on a plain CPU.**

DeskPulse detects people from a webcam, external camera, or video file, tracks each one with a persistent ID, and flags anyone who stays stationary beyond a configurable threshold. Inference runs on an OpenVINO-exported YOLO model, so no GPU is required.

<!-- Add a screenshot or GIF of the dashboard here -->
<!-- ![DeskPulse dashboard](assets/dashboard.png) -->

---

## ✨ Features

- **YOLO + ByteTrack** multi-object tracking with persistent IDs
- **Intel OpenVINO** acceleration for real-time CPU inference
- **Idle detection** based on movement tolerance and a configurable timeout
- **Live Streamlit dashboard** with active tracks, idle count, FPS, and a violations table
- **Threaded camera capture** with a minimal buffer for low latency
- **Frame-skip control** to trade inference frequency for smoother video
- **Alerts** with per-track cooldown: audible beep plus CSV and SQLite logging
- **Flexible input**: built-in webcam, external camera index, or video file

---

## 🧠 How It Works

```
Camera / Video  →  Threaded capture  →  YOLO (OpenVINO) + ByteTrack
                                              │
                                              ▼
                              Per-track state machine
                     (position, stationary timer, idle flag)
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
               Annotated live feed     Telemetry + table      Alerts (beep, CSV, SQLite)
```

1. Each frame is read by a background thread so the UI never blocks on the camera.
2. Every Nth frame (configurable), the model detects and tracks people.
3. For each track, the centroid is compared with its last stored position.
4. If movement stays under **Movement Tolerance** for longer than **Idle Timeout**, the track is marked **idle**.
5. Idle events trigger an alert, rate-limited per track by the **Alert Cooldown**.
6. Movement resets the idle timer and the alert cooldown.

---

## 📦 Installation

```bash
git clone https://github.com/GoharKhan-byte/deskpulse.git
cd deskpulse

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

> The camera code uses `cv2.CAP_DSHOW`, so it is tuned for **Windows**. On Linux/macOS, adjust or remove that flag in `ThreadedCamera`.

---

## 🚀 Usage

### 1. Export your model to OpenVINO

Place your trained weights (`best.pt`) in the project root and run:

```bash
python export_model.py
```

This creates the `best_openvino_model/` folder that the app loads. (Under the hood this is `YOLO("best.pt").export(format="openvino")`.)

### 2. Run the dashboard

```bash
streamlit run app.py
```

Then tick **Start Live Stream** in the sidebar.

---

## ⚙️ Configuration

All settings are live-tunable from the sidebar.

| Setting | Description | Default |
|---|---|---|
| Target Classes | Which classes to detect and track | All |
| Confidence Threshold | Minimum detection confidence | 0.45 |
| Movement Tolerance (px) | Max centroid shift still counted as stationary | 35 |
| Idle Timeout (s) | Stationary time before a track is flagged idle | 10 |
| Inference Frame Skip | Run the model every Nth frame | 2 |
| Alert Cooldown (s) | Min time between repeat alerts for the same track | 10 |
| Video Source | Webcam, external camera index, or video file | Webcam |

---

## 📁 Project Structure

```
deskpulse/
├── app.py                      # Streamlit dashboard + tracking loop
├── alerts.py                   # AlertManager (beep, CSV, SQLite)
├── dashboard.py                # Additional dashboard / analytics view
├── export_model.py             # Exports best.pt to OpenVINO
├── data.yaml                   # Dataset config used for training/validation
├── best.pt                     # Trained YOLO weights
├── best_openvino_model/        # Exported OpenVINO model (generated)
├── alarm.wav                   # Alert sound
├── idle_violations.csv         # Generated alert log
├── floorpulse_telemetry.csv    # Generated telemetry log
├── floorpulse.db               # Generated SQLite log
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Logged Data

Idle events are written to `idle_violations.csv` and `floorpulse.db`, so you can analyze them later with pandas, SQL, or a BI tool. Set `sqlite_path=None` in `load_alert_manager()` to log CSV only.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "Unable to open camera stream" | Close Teams/Zoom/Camera app and check Windows camera privacy settings |
| "OpenVINO model directory not found" | Run the export step above |
| Low FPS | Increase frame skip, lower the confidence threshold, or reduce `imgsz` |
| `missing ScriptRunContext` warning | Launch with `streamlit run app.py`, not `python app.py` |

---

## 🔒 Privacy & Responsible Use

DeskPulse is designed as **occupancy and activity analytics**, not individual surveillance.

- Processing runs **locally**; no video is uploaded anywhere
- **No face recognition** or identity tracking, only anonymous track IDs
- Obtain **consent** and follow local workplace and data-protection laws before deploying
- Be transparent with the people being monitored about what is measured and why

---

## 🗺️ Roadmap

- [ ] Dashboard page for historical idle analytics from SQLite
- [ ] Zone-based monitoring (per desk or area)
- [ ] RTSP / IP camera support
- [ ] Docker image
- [ ] Configurable working-hours schedule

---

## 🧰 Tech Stack

Python · Ultralytics YOLO · ByteTrack · Intel OpenVINO · OpenCV · Streamlit · Pandas · SQLite

---

## 👤 Author

**Gohar Ali Khan**, Full Stack AI Engineer
[LinkedIn](https://linkedin.com/in/gohar-ali-khan-144054283) · [GitHub](https://github.com/GoharKhan-byte)

---

## 📄 License

Released under the MIT License. See `LICENSE` for details.
