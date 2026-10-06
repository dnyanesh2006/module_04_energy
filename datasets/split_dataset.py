import os
import shutil
import glob
import random

src_img_dir = 'datasets/custom_cctv_split/images'
src_lbl_dir = 'datasets/custom_cctv_split/labels'
base_dir = 'datasets/custom_cctv_split'

# The required structure
splits = ['train', 'val', 'test']
for s in splits:
    os.makedirs(os.path.join(base_dir, s, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, s, 'labels'), exist_ok=True)

# We only want the 72 frames
light_on_imgs = sorted(glob.glob(os.path.join(src_img_dir, 'light_on_*.jpg')))
light_off_imgs = sorted(glob.glob(os.path.join(src_img_dir, 'light_off_*.jpg')))

# Remove all other files (the ones from test_video.mp4)
for f in glob.glob(os.path.join(src_img_dir, '*.jpg')):
    if 'light_on_' not in os.path.basename(f) and 'light_off_' not in os.path.basename(f):
        os.remove(f)
        os.remove(f.replace('images', 'labels').replace('.jpg', '.txt'))

# We have 36 light_on, 36 light_off
# Let's split them: 
# Train: first 25 of each
# Val: next 6 of each
# Test: last 5 of each
# This ensures we don't randomly split adjacent frames! We keep sequential chunks.

def move_files(img_list, split_name):
    for img_path in img_list:
        lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        dest_img = os.path.join(base_dir, split_name, 'images', os.path.basename(img_path))
        dest_lbl = os.path.join(base_dir, split_name, 'labels', os.path.basename(lbl_path))
        
        shutil.move(img_path, dest_img)
        shutil.move(lbl_path, dest_lbl)

move_files(light_on_imgs[:25], 'train')
move_files(light_on_imgs[25:31], 'val')
move_files(light_on_imgs[31:], 'test')

move_files(light_off_imgs[:25], 'train')
move_files(light_off_imgs[25:31], 'val')
move_files(light_off_imgs[31:], 'test')

# Clean up the original root images/labels folders if empty
try:
    os.rmdir(src_img_dir)
    os.rmdir(src_lbl_dir)
except:
    pass

print("Dataset successfully organized into train/val/test splits!")
