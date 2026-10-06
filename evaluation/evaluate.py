import os
import shutil
from ultralytics import YOLO

# Copy the model to correct places
best_src = 'runs/detect/models/checkpoints/energy_detector/weights/best.pt'
os.makedirs('models/checkpoints/energy_detector/weights', exist_ok=True)
os.makedirs('models/exports', exist_ok=True)
shutil.copy(best_src, 'models/checkpoints/energy_detector/weights/best.pt')
export_pt = 'models/exports/energy_detector_best.pt'
shutil.copy(best_src, export_pt)

# Load model
model = YOLO(export_pt)

# Run validation explicitly to get the metrics cleanly
metrics = model.val(data='datasets/custom.yaml')
print('\n--- METRICS ---')
print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"mAP@50: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP@50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

# Run on test set
test_dir = 'datasets/custom_cctv_split/test/images'
test_imgs = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

results = model.predict(source=test_imgs, save=True, project='evaluation', name='results', exist_ok=True)

person_count = 0
light_on_count = 0
light_off_count = 0

for r in results:
    for c in r.boxes.cls:
        if c == 0: person_count += 1
        elif c == 1: light_on_count += 1
        elif c == 2: light_off_count += 1

print(f'\n--- TEST DETECTIONS ---')
print(f'Person: {person_count}')
print(f'Light ON: {light_on_count}')
print(f'Light OFF: {light_off_count}')
