import os
import torch
import shutil
from ultralytics import YOLO

def main():
    print("Resuming combined_v2 training...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load the LAST saved checkpoint to resume training
    last_checkpoint_path = "models/checkpoints/energy_detector_combined_v2/weights/last.pt"
    if not os.path.exists(last_checkpoint_path):
        print(f"Error: {last_checkpoint_path} not found. Cannot resume.")
        return
        
    model = YOLO(last_checkpoint_path)
    
    # Resume training
    results = model.train(resume=True)
    
    new_model_path = "models/checkpoints/energy_detector_combined_v2/weights/best.pt"
    
    print("\nTraining complete.")
    print("Now evaluating OLD baseline model on combined_v2 TEST split...")
    
    old_model_path = "models/checkpoints/energy_detector_combined/weights/best.pt"
    old_model = YOLO(old_model_path)
    
    old_metrics = old_model.val(data="datasets/combined_v2/dataset.yaml", split="test", device=device)
    
    print("\nNow evaluating NEW model on combined_v2 TEST split...")
    new_model = YOLO(new_model_path)
    new_metrics = new_model.val(data="datasets/combined_v2/dataset.yaml", split="test", device=device)
    
    # 2. Extract metrics
    def extract_metrics(metrics):
        results_dict = metrics.results_dict
        p = metrics.box.p
        r = metrics.box.r
        map50 = metrics.box.map50
        map_95 = metrics.box.map
        
        class_metrics = {}
        for i, c in enumerate(metrics.box.ap_class_index):
            class_metrics[c] = {
                "precision": p[i],
                "recall": r[i],
                "map50": map50[i],
                "map50-95": map_95[i]
            }
        
        return {
            "overall": {
                "precision": results_dict.get('metrics/precision(B)', 0),
                "recall": results_dict.get('metrics/recall(B)', 0),
                "map50": results_dict.get('metrics/mAP50(B)', 0),
                "map50_95": results_dict.get('metrics/mAP50-95(B)', 0),
                "f1": 0 # computed below
            },
            "classes": class_metrics,
            "speed": metrics.speed
        }

    old_m = extract_metrics(old_metrics)
    new_m = extract_metrics(new_metrics)
    
    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0
        
    old_m["overall"]["f1"] = safe_f1(old_m["overall"]["precision"], old_m["overall"]["recall"])
    new_m["overall"]["f1"] = safe_f1(new_m["overall"]["precision"], new_m["overall"]["recall"])
    
    # Write report
    report = f"""# Combined v2 Training & Evaluation Report

## 1. Training Configuration
* **Dataset:** `module_04_energy/datasets/combined_v2/dataset.yaml`
* **Base Model:** YOLOv8n (`yolov8n.pt`) -> Resumed from checkpoint
* **Epochs:** 50
* **Image Size:** 640
* **Batch Size:** Auto/16 (Ultralytics default)
* **Optimizer:** Auto (Ultralytics default)
* **Learning Rate:** Auto (Ultralytics default)
* **Augmentations:** Auto (Ultralytics default)
* **Seed:** 42
* **Device Used:** {device}
* **Model Output Path:** `module_04_energy/models/checkpoints/energy_detector_combined_v2/weights/best.pt`

## 2. Overall Performance Comparison (Held-out TEST split)

| Metric | Old Model (Baseline) | New Model (combined_v2) | Improvement |
|---|---|---|---|
| **Precision** | {old_m['overall']['precision']:.4f} | {new_m['overall']['precision']:.4f} | {new_m['overall']['precision'] - old_m['overall']['precision']:.4f} |
| **Recall** | {old_m['overall']['recall']:.4f} | {new_m['overall']['recall']:.4f} | {new_m['overall']['recall'] - old_m['overall']['recall']:.4f} |
| **F1-Score** | {old_m['overall']['f1']:.4f} | {new_m['overall']['f1']:.4f} | {new_m['overall']['f1'] - old_m['overall']['f1']:.4f} |
| **mAP@50** | {old_m['overall']['map50']:.4f} | {new_m['overall']['map50']:.4f} | {new_m['overall']['map50'] - old_m['overall']['map50']:.4f} |
| **mAP@50:95** | {old_m['overall']['map50_95']:.4f} | {new_m['overall']['map50_95']:.4f} | {new_m['overall']['map50_95'] - old_m['overall']['map50_95']:.4f} |
| **Inference Speed** | {sum(old_m['speed'].values()):.2f}ms/img | {sum(new_m['speed'].values()):.2f}ms/img | - |

## 3. Per-Class Metrics Comparison (mAP@50)

| Class | Old Model mAP@50 | New Model mAP@50 | Improvement |
|---|---|---|---|
"""
    
    for cls_idx, cls_name in {0: "person", 1: "light_on", 2: "light_off"}.items():
        old_val = old_m["classes"].get(cls_idx, {}).get("map50", 0)
        new_val = new_m["classes"].get(cls_idx, {}).get("map50", 0)
        report += f"| **{cls_name}** | {old_val:.4f} | {new_val:.4f} | {new_val - old_val:.4f} |\n"

    report += f"""
## 4. Per-Class Precision & Recall (New Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
"""
    for cls_idx, cls_name in {0: "person", 1: "light_on", 2: "light_off"}.items():
        p = new_m["classes"].get(cls_idx, {}).get("precision", 0)
        r = new_m["classes"].get(cls_idx, {}).get("recall", 0)
        f1 = safe_f1(p, r)
        report += f"| **{cls_name}** | {p:.4f} | {r:.4f} | {f1:.4f} |\n"
        
    report += """
## 5. Summary & Conclusions
* **Did light_on / light_off improve?** See tables above. The massive influx of specific lighting states should yield substantial improvements in mAP for those classes.
* **Did person detection degrade?** See tables above. The sampled AIRC frames should have preserved or improved person detection without allowing it to dominate the loss function.
"""
    
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/combined_v2_training_report.md", "w") as f:
        f.write(report)
        
    print("\nEvaluation metrics written to evaluation/combined_v2_training_report.md")

if __name__ == "__main__":
    main()
