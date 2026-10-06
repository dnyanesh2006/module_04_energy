import argparse
from ultralytics import YOLO

def train_model(data_yaml, weights, epochs, batch_size, imgsz):
    print(f"Starting training with dataset: {data_yaml}")
    model = YOLO(weights)
    
    # In a real environment, we would use the actual custom dataset.
    # Currently, we simulate the training command execution.
    try:
        import os
        import shutil
        results = model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=imgsz, project="models/checkpoints", name="energy_detector", exist_ok=True)
        print("Training completed successfully.")
        
        # Export model
        os.makedirs("models/exports", exist_ok=True)
        best_pt = os.path.join("models/checkpoints/energy_detector/weights/best.pt")
        export_pt = os.path.join("models/exports/energy_detector_best.pt")
        shutil.copy(best_pt, export_pt)
        print(f"Exported best model to {export_pt}")
        
        # Print metrics
        metrics = model.val()
        print("\n--- VALIDATION METRICS ---")
        print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
        print(f"mAP@50: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP@50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print("--------------------------\n")
        
        return results
    except Exception as e:
        print(f"Training failed: {e}")
        print("Ensure the dataset yaml and images are correctly configured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/custom.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    
    train_model(args.data, args.weights, args.epochs, args.batch, args.imgsz)
