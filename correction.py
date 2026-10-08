from ultralytics import YOLO

model = YOLO("best.pt")
print("Model expects these exact class IDs:", list(model.names.keys()))
print("Model class names mapping:", model.names)