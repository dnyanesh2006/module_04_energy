import os
import glob
from collections import defaultdict

base_dir = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\combined_v2"
splits = ["train", "val", "test"]

stats = {
    "images": {"train": 0, "val": 0, "test": 0},
    "annots": {
        "train": {"0": 0, "1": 0, "2": 0},
        "val": {"0": 0, "1": 0, "2": 0},
        "test": {"0": 0, "1": 0, "2": 0}
    },
    "combinations": {
        "person_only": 0,
        "light_on_only": 0,
        "light_off_only": 0,
        "person_light_on": 0,
        "person_light_off": 0,
        "all_three": 0,
        "no_valid": 0
    },
    "custom_count": 0,
    "airc_train_count": 0,
    "airc_val_count": 0,
    "airc_test_count": 0,
    "leakage_check": defaultdict(set), # maps session -> splits
    "light_source_sessions": {
        "1": set(),
        "2": set()
    },
    "light_images": {
        "1": 0,
        "2": 0
    },
    "invalid_coords": 0,
    "invalid_classes": 0
}

for split in splits:
    img_dir = os.path.join(base_dir, "images", split)
    lbl_dir = os.path.join(base_dir, "labels", split)
    
    if not os.path.exists(img_dir):
        continue
        
    imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
    stats["images"][split] = len(imgs)
    
    for img in imgs:
        basename = os.path.basename(img)
        lbl = os.path.join(lbl_dir, basename.replace(".jpg", ".txt"))
        
        # Check source and prefix
        if basename.startswith("custom_"):
            stats["custom_count"] += 1
            session_id = "custom_" + basename.split("_")[1] # e.g. custom_light
        elif basename.startswith("airc_"):
            if split == "train":
                stats["airc_train_count"] += 1
            elif split == "val":
                stats["airc_val_count"] += 1
            else:
                stats["airc_test_count"] += 1
            session_id = "airc_" + basename.split("_")[1] # e.g. airc_008
        elif basename.startswith("light_"):
            session_id = "light_" + basename.split("_")[1] # e.g. light_train
        else:
            session_id = "unknown"
            
        stats["leakage_check"][session_id].add(split)
            
        classes_in_img = set()
        
        if os.path.exists(lbl):
            with open(lbl, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, x, y, w, h = parts
                        if c in ["0", "1", "2"]:
                            stats["annots"][split][c] += 1
                            classes_in_img.add(c)
                            
                            if c in ["1", "2"]:
                                stats["light_source_sessions"][c].add(session_id)
                        else:
                            stats["invalid_classes"] += 1
                            
                        x, y, w, h = map(float, [x, y, w, h])
                        if w <= 0 or h <= 0 or w > 1 or h > 1 or x < 0 or x > 1 or y < 0 or y > 1:
                            stats["invalid_coords"] += 1
                            
        # Combinations logic
        has_p = "0" in classes_in_img
        has_lon = "1" in classes_in_img
        has_loff = "2" in classes_in_img
        
        if has_lon: stats["light_images"]["1"] += 1
        if has_loff: stats["light_images"]["2"] += 1
        
        if len(classes_in_img) == 0:
            stats["combinations"]["no_valid"] += 1
        elif has_p and has_lon and has_loff:
            stats["combinations"]["all_three"] += 1
        elif has_p and has_lon:
            stats["combinations"]["person_light_on"] += 1
        elif has_p and has_loff:
            stats["combinations"]["person_light_off"] += 1
        elif has_p:
            stats["combinations"]["person_only"] += 1
        elif has_lon and has_loff:
            # this wasn't explicit in prompt, but let's count it or just say no valid if missed
            pass 
        elif has_lon:
            stats["combinations"]["light_on_only"] += 1
        elif has_loff:
            stats["combinations"]["light_off_only"] += 1

print("=== FINAL AUDIT ===")
print("IMAGES:")
print(stats["images"])
print("ANNOTS:")
print(stats["annots"])
print("COMBINATIONS:")
print(stats["combinations"])
print("CUSTOM COUNT:", stats["custom_count"])
print("AIRC TRAIN COUNT:", stats["airc_train_count"])
print("AIRC VAL COUNT:", stats["airc_val_count"])
print("AIRC TEST COUNT:", stats["airc_test_count"])
print("INVALID COORDS:", stats["invalid_coords"])
print("INVALID CLASSES:", stats["invalid_classes"])
print("LIGHT SOURCE SESSIONS 1:", len(stats["light_source_sessions"]["1"]))
print("LIGHT SOURCE SESSIONS 2:", len(stats["light_source_sessions"]["2"]))
print("LIGHT IMAGES 1:", stats["light_images"]["1"])
print("LIGHT IMAGES 2:", stats["light_images"]["2"])

leakage = 0
for s, spl in stats["leakage_check"].items():
    if len(spl) > 1:
        # custom dataset does not have session folders, so custom files leak by default if they are just random files in train/val/test
        if not s.startswith("custom_"):
            leakage += 1
            print(f"Leakage in session {s}: splits {spl}")
            
print("LEAKAGE (non-custom):", leakage)

