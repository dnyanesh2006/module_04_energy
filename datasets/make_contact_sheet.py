import cv2
import os
import glob
import numpy as np

def create_contact_sheet(image_dir, output_path, cols=10, row_height=100):
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        print("No images found.")
        return
        
    print(f"Creating contact sheet for {len(images)} images...")
    
    # Read first image to get aspect ratio
    img0 = cv2.imread(images[0])
    if img0 is None:
        return
        
    h, w = img0.shape[:2]
    aspect = w / h
    col_width = int(row_height * aspect)
    
    # Calculate rows
    rows = (len(images) + cols - 1) // cols
    
    # Create blank canvas
    sheet_w = cols * col_width
    sheet_h = rows * row_height
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.resize(img, (col_width, row_height))
        
        # Calculate position
        row = i // cols
        col = i % cols
        y = row * row_height
        x = col * col_width
        
        # Put image on canvas
        sheet[y:y+row_height, x:x+col_width] = img
        
        # Put text
        cv2.putText(sheet, os.path.basename(img_path), (x+5, y+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
    cv2.imwrite(output_path, sheet)
    print(f"Contact sheet saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        create_contact_sheet(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python make_contact_sheet.py <input_dir> <output.jpg>")
