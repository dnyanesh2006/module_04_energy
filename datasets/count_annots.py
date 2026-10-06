import glob
import os

lbls = glob.glob('datasets/custom_cctv_split/*/labels/*.txt')
total_imgs = len(lbls)
person_count = 0
light_on_count = 0
light_off_count = 0

for lbl in lbls:
    with open(lbl, 'r') as f:
        for line in f:
            c = int(line.split()[0])
            if c == 0: person_count += 1
            elif c == 1: light_on_count += 1
            elif c == 2: light_off_count += 1

print(f"Total Images Annotated: {total_imgs}")
print(f"Person: {person_count}")
print(f"Light ON: {light_on_count}")
print(f"Light OFF: {light_off_count}")
