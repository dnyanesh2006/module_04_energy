import os
import cv2
import glob
import json
import shutil
import hashlib
from pathlib import Path

def get_image_hash(image_path):
    """Return MD5 hash of an image file to detect exact duplicates."""
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def extract_frames(video_path, output_dir, frame_interval=30):
    """Extract frames from video at a specified interval."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        count += 1
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_dir}")

def remove_duplicates(image_dir):
    """Remove duplicate images based on MD5 hash."""
    hashes = set()
    removed = 0
    for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):
        img_hash = get_image_hash(img_path)
        if img_hash in hashes:
            os.remove(img_path)
            removed += 1
        else:
            hashes.add(img_hash)
    print(f"Removed {removed} exact duplicate images from {image_dir}")

def generate_statistics(dataset_dir):
    """Generate basic dataset statistics."""
    images = glob.glob(os.path.join(dataset_dir, "**/*.jpg"), recursive=True)
    labels = glob.glob(os.path.join(dataset_dir, "**/*.txt"), recursive=True)
    
    stats = {
        "total_images": len(images),
        "total_labels": len(labels),
    }
    print(json.dumps(stats, indent=4))
    return stats

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """Split dataset into train, val, and test based on camera sessions to avoid leakage."""
    # Assuming images are grouped by session in subfolders or named by session
    # e.g., session1_frame_000.jpg. Here we do a basic random split if no sessions exist,
    # but the requirement states "camera/session-based splitting".
    # For this script, we'll organize by folders.
    sessions = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    if not sessions:
        print("No sessions found. Please group images by session/camera folders for proper splitting.")
        return

    import random
    random.shuffle(sessions)
    train_split = int(len(sessions) * train_ratio)
    val_split = int(len(sessions) * (train_ratio + val_ratio))
    
    splits = {
        "train": sessions[:train_split],
        "val": sessions[train_split:val_split],
        "test": sessions[val_split:]
    }

    for split, split_sessions in splits.items():
        split_img_dir = Path(output_dir) / split / "images"
        split_lbl_dir = Path(output_dir) / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for session in split_sessions:
            for img_path in glob.glob(os.path.join(image_dir, session, "*.jpg")):
                shutil.copy(img_path, split_img_dir)
                lbl_path = os.path.join(label_dir, session, os.path.basename(img_path).replace(".jpg", ".txt"))
                if os.path.exists(lbl_path):
                    shutil.copy(lbl_path, split_lbl_dir)
    print("Dataset split completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Processing Tools for Module 4")
    parser.add_argument("--extract", type=str, help="Path to video to extract frames")
    parser.add_argument("--output", type=str, help="Output directory for extraction or split")
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    parser.add_argument("--dedup", type=str, help="Directory to deduplicate")
    parser.add_argument("--stats", type=str, help="Directory to generate stats for")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    parser.add_argument("--image_dir", type=str, help="Directory containing images grouped by session")
    parser.add_argument("--label_dir", type=str, help="Directory containing labels grouped by session")
    
    args = parser.parse_args()
    
    if args.extract and args.output:
        extract_frames(args.extract, args.output, args.interval)
    if args.dedup:
        remove_duplicates(args.dedup)
    if args.stats:
        generate_statistics(args.stats)
    if args.split and args.image_dir and args.label_dir and args.output:
        split_dataset(args.image_dir, args.label_dir, args.output)
