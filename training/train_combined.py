import os
from ultralytics import YOLO

def train():
    print("Starting combined training...")
    
    # Load base model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data="datasets/custom.yaml",
        epochs=50,
        imgsz=640,
        device="cpu", # User specified CPU if GPU unavailable
        project="models/checkpoints",
        name="energy_detector_combined",
        exist_ok=True
    )
    
    # Export model
    export_dir = "models/exports"
    os.makedirs(export_dir, exist_ok=True)
    model.export(format="torchscript")
    
    import shutil
    shutil.copy("models/checkpoints/energy_detector_combined/weights/best.pt", os.path.join(export_dir, "energy_detector_combined.pt"))
    print("Training complete and model exported.")

if __name__ == "__main__":
    train()
