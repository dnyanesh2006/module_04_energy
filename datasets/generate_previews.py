import cv2
import glob
import os

splits = ['train', 'val', 'test']
base_dir = 'datasets/custom_cctv_split'

def draw_boxes():
    images = []
    # Pick a few from train, val, test
    for s in splits:
        img_paths = sorted(glob.glob(os.path.join(base_dir, s, 'images', '*.jpg')))
        if img_paths:
            # Pick first, middle, last
            images.extend([img_paths[0], img_paths[len(img_paths)//2], img_paths[-1]])
            
    # We want exactly 6-10 images. Let's just process the ones we collected.
    print(f"Generating previews for {len(images)} images.")
    for img_path in images:
        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]
        lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        if not os.path.exists(lbl_path):
            continue
            
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    bx = float(parts[1]) * w
                    by = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    
                    x1 = int(bx - bw/2)
                    y1 = int(by - bh/2)
                    x2 = int(bx + bw/2)
                    y2 = int(by + bh/2)
                    
                    if cls_id == 0:
                        color = (0, 0, 255) # Red for person
                        label = "Person"
                    elif cls_id == 1:
                        color = (0, 255, 255) # Yellow for light_on
                        label = "Light_ON"
                    elif cls_id == 2:
                        color = (100, 100, 100) # Gray for light_off
                        label = "Light_OFF"
                    else:
                        color = (255, 255, 255)
                        label = f"Class {cls_id}"
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        out_name = f"final_preview_{os.path.basename(img_path)}"
        cv2.imwrite(os.path.join(base_dir, out_name), frame)

if __name__ == "__main__":
    draw_boxes()
