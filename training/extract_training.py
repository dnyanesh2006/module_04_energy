import cv2
import os

def extract(video_path, output_dir, prefix, interval):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            saved += 1
            out_path = os.path.join(output_dir, f"{prefix}_{saved:03d}.jpg")
            cv2.imwrite(out_path, frame)
        count += 1
    cap.release()
    print(f"Extracted {saved} frames from {video_path}")

# light_on.mp4 has 540 frames. Interval 15 => 36 frames
extract('datasets/custom_cctv/light_on.mp4', 'datasets/custom_cctv/frames', 'light_on', 15)

# light_off.mp4 has 425 frames. Interval 12 => 36 frames
extract('datasets/custom_cctv/light_off.mp4', 'datasets/custom_cctv/frames', 'light_off', 12)
