import os
import shutil
import subprocess
from ultralytics import YOLO

# 3. Copy files
best_src = 'runs/detect/runs/detect/energy_detector_50ep/weights/best.pt'
chkpt_dir = 'models/checkpoints/energy_detector_50ep/weights'
os.makedirs(chkpt_dir, exist_ok=True)
os.makedirs('models/exports', exist_ok=True)

final_chkpt = os.path.join(chkpt_dir, 'best.pt')
final_export = 'models/exports/energy_detector_50ep.pt'
shutil.copy(best_src, final_chkpt)
shutil.copy(best_src, final_export)

print(f"MODEL CHECKPOINT: {final_chkpt}")
print(f"MODEL EXPORT: {final_export}")

# Load best model
best_model = YOLO(final_export)

# 4. Metrics
metrics = best_model.val(data='datasets/custom.yaml')
print("\n--- METRICS ---")
print(f"Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"mAP@50: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP@50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

# 5. Inference
def get_images(split, num=5):
    img_dir = f'datasets/custom_cctv_split/{split}/images'
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    imgs.sort()
    return imgs[:num]

train_imgs = get_images('train')
val_imgs = get_images('val')
test_imgs = get_images('test')

all_imgs = train_imgs + val_imgs + test_imgs
out_dir_norm = 'evaluation/results/training_50ep/normal'
out_dir_debug = 'evaluation/results/training_50ep/debug_0.05'
os.makedirs(out_dir_norm, exist_ok=True)
os.makedirs(out_dir_debug, exist_ok=True)

detected_classes = set()

def run_infer(imgs, split_name):
    # normal
    res_norm = best_model.predict(source=imgs, save=True, project='evaluation/results/training_50ep', name=f'normal_{split_name}', exist_ok=True)
    # debug
    res_debug = best_model.predict(source=imgs, conf=0.05, save=True, project='evaluation/results/training_50ep', name=f'debug_{split_name}', exist_ok=True)
    
    total_det = 0
    for r in res_norm:
        total_det += len(r.boxes)
        for c in r.boxes.cls:
            detected_classes.add(int(c.item()))
    print(f"{split_name} IMAGE DETECTIONS (normal conf): {total_det}")
    
    total_det_debug = 0
    for r in res_debug:
        total_det_debug += len(r.boxes)
    print(f"{split_name} IMAGE DETECTIONS (debug conf=0.05): {total_det_debug}")

run_infer(train_imgs, 'TRAIN')
run_infer(val_imgs, 'VAL')
run_infer(test_imgs, 'TEST')

print("\n--- DETECTED CLASSES ---")
print(f"PERSON DETECTION: {0 in detected_classes}")
print(f"LIGHT_ON DETECTION: {1 in detected_classes}")
print(f"LIGHT_OFF DETECTION: {2 in detected_classes}")

# 6. Unit tests
print("\n--- UNIT TESTS ---")
try:
    result = subprocess.run(['pytest', 'tests/'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
except Exception as e:
    print(f"Error running tests: {e}")
