from ultralytics import YOLO

def export_to_openvino():
    print("🚀 Starting OpenVINO Export...")
    
    # Load your trained PyTorch weights
    model = YOLO("best.pt")
    
    # Export to OpenVINO format
    model.export(
        format="openvino",
        half=True,         # FP16 precision — supported on OpenVINO too
        imgsz=640,          # Match input size used during dashboard inference
        device="cpu"        # No CUDA device — OpenVINO handles CPU/iGPU internally
    )
    
    print("✅ Export complete! 'best_openvino_model/' folder is saved in your workspace.")

if __name__ == "__main__":
    export_to_openvino()