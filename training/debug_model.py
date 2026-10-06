import os
import random
from ultralytics import YOLO

# 1. Load model
model_path = 'models/checkpoints/energy_detector/weights/best.pt'
model = YOLO(model_path)

# 2. Verify classes
print("CLASS VERIFICATION:")
for k, v in model.names.items():
    print(f"{k} = {v}")

# 3. Select images
def get_random_images(split, num=3):
    img_dir = f'datasets/custom_cctv_split/{split}/images'
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    # Use fixed seed for reproducibility, or just take the first few
    imgs.sort()
    return imgs[:num]

train_imgs = get_random_images('train', 3)
val_imgs = get_random_images('val', 3)
test_imgs = get_random_images('test', 3)

output_dir = 'evaluation/results/model_debug/'
os.makedirs(output_dir, exist_ok=True)

# 4. Run inference with low confidence (0.05)
def run_and_report(split_name, img_paths):
    print(f"\n{split_name} IMAGE DETECTIONS:")
    results = model.predict(source=img_paths, conf=0.05, save=True, project='evaluation/results', name=f'model_debug/{split_name}', exist_ok=True)
    
    total_detections = 0
    for r, img_path in zip(results, img_paths):
        img_name = os.path.basename(img_path)
        det_count = len(r.boxes)
        total_detections += det_count
        print(f"Image: {img_name} - Detections: {det_count}")
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            print(f"  Class: {cls_name} ({cls_id}), Conf: {conf:.4f}, Box: [w:{w:.1f}, h:{h:.1f}]")
    
    if total_detections == 0:
        print("  No detections found.")

run_and_report('TRAIN', train_imgs)
run_and_report('VAL', val_imgs)
run_and_report('TEST', test_imgs)
