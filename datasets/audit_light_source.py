import os
import glob
from collections import defaultdict
import cv2
import hashlib

base_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\light_source\Light source detection.v3i.yolov8"

splits = ["train", "valid", "test"]

stats = {
    "images": defaultdict(int),
    "labels": defaultdict(int),
    "annotations_per_class": defaultdict(int),
    "resolutions": defaultdict(int),
    "missing_labels": 0,
    "missing_images": 0,
    "corrupted_images": 0,
    "duplicates": 0,
    "invalid_boxes": 0
}

image_hashes = set()

for split in splits:
    img_dir = os.path.join(base_dir, split, "images")
    lbl_dir = os.path.join(base_dir, split, "labels")
    
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    stats["images"][split] = len(images)
    stats["labels"][split] = len(labels)
    
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
    lbl_map = {os.path.splitext(os.path.basename(p))[0]: p for p in labels}
    
    for base_name in img_map:
        if base_name not in lbl_map:
            stats["missing_labels"] += 1
            
        # check corruption and resolution and duplicates
        try:
            img_path = img_map[base_name]
            img = cv2.imread(img_path)
            if img is None:
                stats["corrupted_images"] += 1
            else:
                h, w, c = img.shape
                stats["resolutions"][f"{w}x{h}"] += 1
                
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in image_hashes:
                        stats["duplicates"] += 1
                    else:
                        image_hashes.add(file_hash)
        except Exception as e:
            stats["corrupted_images"] += 1

    for base_name in lbl_map:
        if base_name not in img_map:
            stats["missing_images"] += 1
            
        # parse label
        try:
            with open(lbl_map[base_name], "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c_id, x, y, w, h = parts
                        c_id = int(c_id)
                        stats["annotations_per_class"][c_id] += 1
                        
                        fx, fy, fw, fh = float(x), float(y), float(w), float(h)
                        if fw <= 0 or fh <= 0 or fw > 1 or fh > 1 or fx < 0 or fx > 1 or fy < 0 or fy > 1:
                            stats["invalid_boxes"] += 1
        except Exception as e:
            pass

print("--- AUDIT RESULTS ---")
print(f"Train images: {stats['images']['train']}")
print(f"Valid images: {stats['images']['valid']}")
print(f"Test images:  {stats['images']['test']}")
print(f"Total images: {sum(stats['images'].values())}")

print("\nAnnotations per class:")
for k, v in dict(stats['annotations_per_class']).items():
    print(f"Class {k}: {v}")

print(f"\nMissing labels: {stats['missing_labels']}")
print(f"Missing images: {stats['missing_images']}")
print(f"Corrupted images: {stats['corrupted_images']}")
print(f"Exact Duplicate images: {stats['duplicates']}")
print(f"Invalid bounding boxes: {stats['invalid_boxes']}")

print("\nResolutions:")
for k, v in dict(stats['resolutions']).items():
    print(f"{k}: {v}")
