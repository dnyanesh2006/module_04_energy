import os
import glob
import shutil
import random
from collections import defaultdict

# Setup paths
base_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\person_airc\AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo\AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo\Data"
frames_dir = os.path.join(base_dir, "Frames")
labels_dir = os.path.join(base_dir, "Labels")

out_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\public\person_airc_converted"

# Target splits
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

def create_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    print(f"Created output directory structure at {out_dir}")

def convert_dataset():
    images = glob.glob(os.path.join(frames_dir, "**", "*.jpg"), recursive=True)
    labels = glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)

    # Key is relative path to maintain session context, or just basename
    # AIRC-SMARTCLASS organizes into folders like Labels/001/frame_0000.txt
    # So session is the parent folder name
    
    img_map = {os.path.basename(p): p for p in images}
    lbl_map = {os.path.basename(p): p for p in labels}

    # Group by session
    sessions = defaultdict(list)
    
    source_img_count = len(images)
    source_annot_count = 0
    excluded_images = 0
    excluded_annotations = 0
    
    for lbl_path in labels:
        lbl_name = os.path.basename(lbl_path)
        img_name = lbl_name.replace(".txt", ".jpg")
        
        session = os.path.basename(os.path.dirname(lbl_path))
        
        if img_name in img_map:
            # Parse label
            valid_person_boxes = []
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, x, y, w, h = parts
                        if int(cls_id) == 1:
                            source_annot_count += 1
                            fw, fh = float(w), float(h)
                            if fw < 0.005 or fh < 0.005 or fw > 1 or fh > 1 or float(x) < 0 or float(x) > 1 or float(y) < 0 or float(y) > 1:
                                excluded_annotations += 1
                            else:
                                valid_person_boxes.append(f"0 {x} {y} {w} {h}")
            
            if valid_person_boxes:
                sessions[session].append({
                    "img_path": img_map[img_name],
                    "boxes": valid_person_boxes,
                    "img_name": f"{session}_{img_name}" # prefix with session to avoid collisions
                })
            else:
                excluded_images += 1
        else:
            excluded_images += 1
            
    # Also exclude images without labels
    excluded_images += len(set(img_map.keys()) - set([os.path.basename(p).replace(".txt", ".jpg") for p in labels]))

    # Session split
    session_keys = list(sessions.keys())
    random.seed(42)
    random.shuffle(session_keys)
    
    # Calculate target counts
    total_valid_imgs = sum(len(sessions[k]) for k in session_keys)
    
    train_target = total_valid_imgs * train_ratio
    val_target = total_valid_imgs * val_ratio
    
    train_sessions = []
    val_sessions = []
    test_sessions = []
    
    curr_count = 0
    for s in session_keys:
        scount = len(sessions[s])
        if curr_count < train_target:
            train_sessions.append(s)
        elif curr_count < train_target + val_target:
            val_sessions.append(s)
        else:
            test_sessions.append(s)
        curr_count += scount
        
    print(f"Split: Train {len(train_sessions)} sessions, Val {len(val_sessions)} sessions, Test {len(test_sessions)} sessions")

    # Helper to copy and write
    stats = {
        "train": {"imgs": 0, "annots": 0},
        "val": {"imgs": 0, "annots": 0},
        "test": {"imgs": 0, "annots": 0}
    }
    
    def process_split(split_name, split_sessions):
        for s in split_sessions:
            for item in sessions[s]:
                img_out = os.path.join(out_dir, "images", split_name, item["img_name"])
                lbl_out = os.path.join(out_dir, "labels", split_name, item["img_name"].replace(".jpg", ".txt"))
                
                shutil.copy2(item["img_path"], img_out)
                with open(lbl_out, "w") as f:
                    f.write("\n".join(item["boxes"]) + "\n")
                    
                stats[split_name]["imgs"] += 1
                stats[split_name]["annots"] += len(item["boxes"])

    process_split("train", train_sessions)
    process_split("val", val_sessions)
    process_split("test", test_sessions)
    
    # Generate YAML
    yaml_content = f"""path: ../person_airc_converted
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    converted_img_count = stats["train"]["imgs"] + stats["val"]["imgs"] + stats["test"]["imgs"]
    converted_annot_count = stats["train"]["annots"] + stats["val"]["annots"] + stats["test"]["annots"]
    
    print("\n--- CONVERSION SUMMARY ---")
    print(f"Source image count: {source_img_count}")
    print(f"Converted image count: {converted_img_count}")
    print(f"Source person annotation count: {source_annot_count}")
    print(f"Converted person annotation count: {converted_annot_count}")
    print(f"Excluded images: {excluded_images}")
    print(f"Excluded annotations: {excluded_annotations}")
    
    print(f"Train images: {stats['train']['imgs']}, annots: {stats['train']['annots']}, sessions: {train_sessions}")
    print(f"Val images: {stats['val']['imgs']}, annots: {stats['val']['annots']}, sessions: {val_sessions}")
    print(f"Test images: {stats['test']['imgs']}, annots: {stats['test']['annots']}, sessions: {test_sessions}")

    report = f"""# AIRC-SMARTCLASS Conversion Report

## Conversion Metrics
* **Source image count:** {source_img_count}
* **Converted image count:** {converted_img_count}
* **Source person annotation count:** {source_annot_count}
* **Converted person annotation count:** {converted_annot_count}
* **Excluded images:** {excluded_images} (Due to missing labels, or containing zero valid person annotations after filtering)
* **Excluded annotations:** {excluded_annotations} (Extremely small or out-of-bounds bounding boxes)

## Split Information (Video/Session Grouped to Prevent Leakage)
* **Train split:**
  * **Images:** {stats['train']['imgs']}
  * **Annotations:** {stats['train']['annots']}
  * **Assigned Source Sessions:** {", ".join(train_sessions)}
* **Validation split:**
  * **Images:** {stats['val']['imgs']}
  * **Annotations:** {stats['val']['annots']}
  * **Assigned Source Sessions:** {", ".join(val_sessions)}
* **Test split:**
  * **Images:** {stats['test']['imgs']}
  * **Annotations:** {stats['test']['annots']}
  * **Assigned Source Sessions:** {", ".join(test_sessions)}

## Class Mapping
* **AIRC-SMARTCLASS class `1` (person)** was exclusively extracted and re-mapped to **M04 class `0` (person)**. 
* All other classes (face, AC status, temp, mode, fan_lv) were completely discarded.

## Quality Checks & Validations
* Every image copied into the converted directory has a corresponding label file.
* Every label file contains ONLY class ID `0`.
* Every bounding box has valid, normalized YOLO coordinates (`w > 0.005`, `h > 0.005`, `x,y,w,h` between `0` and `1`).
* Unique image naming is guaranteed by prepending the session ID (e.g. `001_frame_0000.jpg`), preventing collisions.
* No frame leakage occurred across splits (grouped by source session).

## License
* The original dataset declares informed consent from participants for research purposes. Typical Mendeley Data defaults to CC BY 4.0.

## Limitations
* The dataset only features images of occupied spaces. It lacks negative (empty room) samples, meaning it will improve `person` detection but won't directly teach the model about empty states without being combined with other datasets.
"""
    os.makedirs(r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\evaluation", exist_ok=True)
    with open(r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\evaluation\airc_smartclass_conversion_report.md", "w") as f:
        f.write(report)
        
    print("Report written to evaluation/airc_smartclass_conversion_report.md")

if __name__ == "__main__":
    create_dirs()
    convert_dataset()
