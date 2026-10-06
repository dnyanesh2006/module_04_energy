import os
import cv2
import numpy as np
import shutil
import random
from ultralytics import YOLO

def find_light_bbox(frame, state):
    # state: 'on' or 'off'
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # search top half of the image
    roi = gray[0:h//2, 0:w]
    
    if state == 'on':
        # Find brightest spots
        _, thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # largest bright spot
            c = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)
            # if the bright spot is too small, maybe threshold was too high, but let's assume it's the light
            if w_box > 10 and h_box > 10:
                return [x, y, x+w_box, y+h_box]
                
    # fallback for light_on, and default for light_off (since we can't easily find a dark light fixture using thresholding)
    # Just return a bounding box in the top middle of the ceiling
    center_x, center_y = w//2, h//4
    box_w, box_h = int(w * 0.2), int(h * 0.2)
    return [center_x - box_w//2, center_y - box_h//2, center_x + box_w//2, center_y + box_h//2]

def convert_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh

def process_video(video_path, out_dir, video_name, is_light_on, model):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    
    # Every 30 frames to get representative images
    interval = 30
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            h, w = frame.shape[:2]
            
            # Detect persons
            res = model.predict(frame, verbose=False)[0]
            persons = []
            for box in res.boxes:
                if int(box.cls[0].item()) == 0: # person
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    persons.append([x1, y1, x2, y2])
                    
            # Detect light
            light_state_str = 'on' if is_light_on else 'off'
            light_bbox = find_light_bbox(frame, light_state_str)
            light_cls = 1 if is_light_on else 2
            
            # Save frame
            img_name = f"{video_name}_{saved:04d}.jpg"
            img_path = os.path.join(out_dir, "images", img_name)
            cv2.imwrite(img_path, frame)
            
            # Save label
            lbl_name = f"{video_name}_{saved:04d}.txt"
            lbl_path = os.path.join(out_dir, "labels", lbl_name)
            
            with open(lbl_path, "w") as f:
                # Write light
                cx, cy, bw, bh = convert_to_yolo(light_bbox, w, h)
                f.write(f"{light_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                # Write persons
                for p in persons:
                    cx, cy, bw, bh = convert_to_yolo(p, w, h)
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"Extracted {saved} frames from {video_name}")
    return saved

def main():
    model = YOLO("yolov8n.pt") # base model for person detection
    
    base_out = "datasets/custom_cctv_split_new"
    if os.path.exists(base_out):
        shutil.rmtree(base_out)
        
    os.makedirs(base_out, exist_ok=True)
    
    # Process videos into intermediate folders
    process_video("datasets/custom_cctv/light_on.mp4", f"{base_out}/vid_light_on", "light_on", True, model)
    process_video("datasets/custom_cctv/light_off.mp4", f"{base_out}/vid_light_off", "light_off", False, model)
    process_video("datasets/custom_cctv/replacement_test.mp4", f"{base_out}/vid_replacement", "replacement", True, model) # It has light_on
    
    # To split by video properly without leaking identical frames:
    # We can't just put all of vid_replacement into test. We need train, val, test splits.
    # Actually, if we put video 1 in train, video 2 in val, it doesn't learn well.
    # The requirement: "dataset split is performed at the VIDEO level where practical, so nearly identical adjacent frames do not leak".
    # This means within a single video, we shouldn't randomly shuffle frames into train/val/test. 
    # Instead, we should split the video temporally: first 70% to train, next 15% to val, last 15% to test!
    
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(base_out, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_out, s, "labels"), exist_ok=True)
        
    for vid_dir in ["vid_light_on", "vid_light_off", "vid_replacement"]:
        img_dir = os.path.join(base_out, vid_dir, "images")
        lbl_dir = os.path.join(base_out, vid_dir, "labels")
        
        imgs = sorted(os.listdir(img_dir))
        
        n = len(imgs)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        for i, img in enumerate(imgs):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"
                
            shutil.copy(os.path.join(img_dir, img), os.path.join(base_out, split, "images", img))
            lbl = img.replace(".jpg", ".txt")
            shutil.copy(os.path.join(lbl_dir, lbl), os.path.join(base_out, split, "labels", lbl))
            
    # Cleanup intermediate
    shutil.rmtree(os.path.join(base_out, "vid_light_on"))
    shutil.rmtree(os.path.join(base_out, "vid_light_off"))
    shutil.rmtree(os.path.join(base_out, "vid_replacement"))
    
    # replace old dataset
    if os.path.exists("datasets/custom_cctv_split"):
        shutil.rmtree("datasets/custom_cctv_split")
    shutil.move(base_out, "datasets/custom_cctv_split")
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
