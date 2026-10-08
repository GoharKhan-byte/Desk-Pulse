import os
import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model weights
WEIGHTS_PATH = os.path.abspath("best.pt")
model = YOLO(WEIGHTS_PATH)

def generate_stream(video_source="Test_Video.mp4"):
    """Reads video, runs YOLO tracking, and streams JPEG frames."""
    cap = cv2.VideoCapture(video_source)
    device = 0 if torch.cuda.is_available() else "cpu"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video stream when finished
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run inference using persistent tracking (ByteTrack)
        results = model.track(source=frame, persist=True, conf=0.45, device=device, verbose=False)
        annotated_frame = results[0].plot()

        # Encode to JPEG for HTTP stream
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    """Endpoint for React <img /> tag."""
    return StreamingResponse(
        generate_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/classes")
def get_classes():
    """Exposes class mapping to the frontend."""
    return {"classes": model.names}