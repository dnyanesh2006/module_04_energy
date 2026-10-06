import os
import glob
from collections import defaultdict
import cv2

base_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\person_airc\AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo\AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo\Data"

frames_dir = os.path.join(base_dir, "Frames")
labels_dir = os.path.join(base_dir, "Labels")

images = glob.glob(os.path.join(frames_dir, "**", "*.jpg"), recursive=True)
labels = glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)

image_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
label_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in labels}

images_without_labels = set(image_basenames.keys()) - set(label_basenames.keys())
labels_without_images = set(label_basenames.keys()) - set(image_basenames.keys())

person_count = 0
invalid_boxes = 0
small_boxes = 0

classes_found = set()

for lbl_base, lbl_path in label_basenames.items():
    if lbl_base in image_basenames:
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, x, y, w, h = parts
                        cls_id = int(cls_id)
                        classes_found.add(cls_id)
                        w, h = float(w), float(h)
                        if cls_id == 1:
                            person_count += 1
                        
                        if w <= 0 or h <= 0 or w > 1 or h > 1:
                            invalid_boxes += 1
                        elif w < 0.005 or h < 0.005:
                            small_boxes += 1
        except Exception as e:
            pass

print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")
print(f"Images without labels: {len(images_without_labels)}")
print(f"Labels without images: {len(labels_without_images)}")
print(f"Person annotations (class 1): {person_count}")
print(f"Classes found: {classes_found}")
print(f"Invalid boxes: {invalid_boxes}")
print(f"Very small boxes: {small_boxes}")
