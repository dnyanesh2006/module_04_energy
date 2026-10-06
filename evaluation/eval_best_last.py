from ultralytics import YOLO

device = 'cpu'
data = 'datasets/combined_v2/dataset.yaml'

best_path = 'runs/detect/runs/detect/energy_detector_50ep/weights/best.pt'
last_path = 'runs/detect/runs/detect/energy_detector_50ep/weights/last.pt'

print("--- EVALUATING BEST.PT ---")
model_best = YOLO(best_path)
res_best = model_best.val(data=data, split='test', device=device)

print("\n--- EVALUATING LAST.PT ---")
try:
    model_last = YOLO(last_path)
    res_last = model_last.val(data=data, split='test', device=device)
except Exception as e:
    print(f"Error loading last.pt: {e}")
