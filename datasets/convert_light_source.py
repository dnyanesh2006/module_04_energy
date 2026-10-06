import os
import glob
import shutil
import hashlib
from collections import defaultdict

base_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\light_source\Light source detection.v3i.yolov8"
out_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\light_source_converted"

splits_map = {"train": "train", "valid": "val", "test": "test"}

# Mappings
# Source -> Target
class_map = {
    1: 2, # Lamp Off -> light_off
    2: 1, # Lamp On -> light_on
    3: 2, # Light Bulb Off -> light_off
    4: 1  # Light Bulb On -> light_on
}

def create_dirs():
    for out_split in splits_map.values():
        os.makedirs(os.path.join(out_dir, "images", out_split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", out_split), exist_ok=True)

def run_conversion():
    stats = {
        "original_images": 0,
        "converted_images": 0,
        "original_splits": {"train": 0, "valid": 0, "test": 0},
        "final_splits": {"train": 0, "val": 0, "test": 0},
        "original_classes": defaultdict(int),
        "converted_classes": defaultdict(int), # 1: light_on, 2: light_off
        "discarded_classes": {0: 0, 5: 0, 6: 0},
        "invalid_removed": 0,
        "duplicates_removed": 0,
        "empty_removed": 0
    }

    seen_hashes = set()
    
    for split_in, split_out in splits_map.items():
        img_dir = os.path.join(base_dir, split_in, "images")
        lbl_dir = os.path.join(base_dir, split_in, "labels")
        
        images = glob.glob(os.path.join(img_dir, "*.jpg"))
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
        
        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
        lbl_map = {os.path.splitext(os.path.basename(p))[0]: p for p in labels}
        
        stats["original_splits"][split_in] += len(images)
        stats["original_images"] += len(images)
        
        for base_name, img_path in img_map.items():
            if base_name not in lbl_map:
                stats["empty_removed"] += 1
                continue
                
            # Check duplicate
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            if file_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
                
            # Process label
            lbl_path = lbl_map[base_name]
            valid_boxes = []
            
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c_id = int(parts[0])
                        stats["original_classes"][c_id] += 1
                        
                        x, y, w, h = map(float, parts[1:])
                        
                        # Validate Box
                        if w <= 0 or h <= 0 or w > 1 or h > 1 or x < 0 or x > 1 or y < 0 or y > 1:
                            stats["invalid_removed"] += 1
                            continue
                            
                        if c_id in class_map:
                            target_c = class_map[c_id]
                            valid_boxes.append(f"{target_c} {x} {y} {w} {h}")
                            stats["converted_classes"][target_c] += 1
                        elif c_id in stats["discarded_classes"]:
                            stats["discarded_classes"][c_id] += 1
                            
            if len(valid_boxes) > 0:
                seen_hashes.add(file_hash)
                
                # Copy files
                safe_name = f"{split_in}_{base_name}"
                out_img = os.path.join(out_dir, "images", split_out, f"{safe_name}.jpg")
                out_lbl = os.path.join(out_dir, "labels", split_out, f"{safe_name}.txt")
                
                shutil.copy2(img_path, out_img)
                with open(out_lbl, "w") as f:
                    f.write("\n".join(valid_boxes) + "\n")
                    
                stats["final_splits"][split_out] += 1
                stats["converted_images"] += 1
            else:
                stats["empty_removed"] += 1

    # Write yaml
    yaml_content = f"""path: ../light_source_converted
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: light_on
  2: light_off
"""
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    return stats

def validate_and_report(stats):
    # Validation logic
    all_imgs = glob.glob(os.path.join(out_dir, "images", "**", "*.jpg"), recursive=True)
    all_lbls = glob.glob(os.path.join(out_dir, "labels", "**", "*.txt"), recursive=True)
    
    assert len(all_imgs) == len(all_lbls), "Image and label count mismatch!"
    
    for lbl_path in all_lbls:
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                c_id, x, y, w, h = map(float, line.strip().split())
                assert int(c_id) in [1, 2], f"Invalid class id {c_id} found in {lbl_path}"
                assert 0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1, f"Invalid box in {lbl_path}"

    report = f"""# Light Source Conversion Report

## 1. Image Counts
* **Original image count:** {stats['original_images']}
* **Converted image count:** {stats['converted_images']}

## 2. Splits
* **Original Train:** {stats['original_splits']['train']} -> **Final Train:** {stats['final_splits']['train']}
* **Original Valid:** {stats['original_splits']['valid']} -> **Final Val:** {stats['final_splits']['val']}
* **Original Test:** {stats['original_splits']['test']} -> **Final Test:** {stats['final_splits']['test']}

## 3. Original Class Counts
"""
    for k, v in dict(stats['original_classes']).items():
        report += f"* **Class {k}:** {v}\n"

    report += f"""
## 4. Converted Class Counts
* **Converted light_on count (Class 1):** {stats['converted_classes'][1]}
* **Converted light_off count (Class 2):** {stats['converted_classes'][2]}

## 5. Discarded and Removed
* **Number of discarded class 0 annotations:** {stats['discarded_classes'][0]}
* **Number of discarded class 5 annotations:** {stats['discarded_classes'][5]}
* **Number of discarded class 6 annotations:** {stats['discarded_classes'][6]}
* **Number of invalid annotations removed:** {stats['invalid_removed']}
* **Number of duplicate images removed:** {stats['duplicates_removed']}
* **Number of empty images removed:** {stats['empty_removed']}

## 6. Final Class Mapping
* **Source `2` (Lamp On)** + **Source `4` (Light Bulb On)** $\\rightarrow$ **M04 `1` (light_on)**
* **Source `1` (Lamp Off)** + **Source `3` (Light Bulb Off)** $\\rightarrow$ **M04 `2` (light_off)**
* M04 Class `0` (`person`) is intentionally not present in this dataset but occupies ID `0` in `dataset.yaml` to ensure structural compatibility for future combination.

## 7. License
* **CC BY 4.0** (Roboflow Universe)

## 8. Dataset Limitations
* Only focuses on light source states (bulbs/lamps). It contains no people, which means it will improve light state detection but will not help with `person` detection unless merged appropriately.
* Close-up object focus on lamps/bulbs might slightly differ in scale from wide-angle CCTV, though YOLO generally handles scale variations well.

## 9. Integrity Confirmation
* The original dataset at `module_04_energy/datasets/public/light_source/` is completely **untouched**.
* Duplicate filenames were prevented by injecting the original split names into the target filename (e.g., `train_001.jpg`).
"""
    os.makedirs(r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\evaluation", exist_ok=True)
    with open(r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\evaluation\light_source_conversion_report.md", "w") as f:
        f.write(report)
        
    print("Report written successfully.")

if __name__ == "__main__":
    create_dirs()
    stats = run_conversion()
    validate_and_report(stats)
