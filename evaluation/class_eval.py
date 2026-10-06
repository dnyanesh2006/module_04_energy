import os
from ultralytics import YOLO

# Evaluate new combined model
model = YOLO('models/checkpoints/energy_detector_combined/weights/best.pt')

print("\n\n=== COMBINED SPLIT METRICS (VAL + TEST) ===")
# We can evaluate on 'val' and 'test'
print("\n--- VAL SPLIT ---")
model.val(data='datasets/custom.yaml', split='val')

print("\n--- TEST SPLIT ---")
model.val(data='datasets/custom.yaml', split='test')
