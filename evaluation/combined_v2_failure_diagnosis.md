# Combined V2 Failure Diagnosis

## 1. Checkpoint Verification
* **Path:** `runs/detect/runs/detect/energy_detector_50ep/weights/best.pt`
* **File Size:** 6,227,818 bytes
* **Modification Time:** 2026-10-03 18:57:10
* **Architecture:** DetectionModel (YOLOv8n)
* **Classes:** `{0: 'person', 1: 'light_on', 2: 'light_off'}`
* **Data trained on:** `datasets/custom.yaml`

**CRITICAL FINDING:** The checkpoint provided as "Combined V2" was **NOT** trained on `datasets/combined_v2/dataset.yaml`. It was trained on `datasets/custom.yaml` (the baseline dataset). 

## 2. Dataset YAML Verification
* **Train Path:** `images/train`
* **Val Path:** `images/val`
* **Test Path:** `images/test`
* **Class 0:** person
* **Class 1:** light_on
* **Class 2:** light_off

## 3. Test Dataset Verification
* **Test Images:** 575
* **Test Labels:** 575 (0 empty labels)
* **Annotation Count:** person (3,448), light_on (25), light_off (17)
* **Source Contribution:** AIRC (533 images), Light Source (28 images), Custom CCTV (14 images)

## 4. Training Results Verification
* **Best Epoch:** Epoch 41 
* **Best mAP@50 (val):** 0.975 (on the *custom.yaml* validation set)
* **Metrics:** The model achieved high metrics (mAP50=0.975) because it was validating against the tiny `custom.yaml` validation set (16 images), completely missing the 575 images from the `combined_v2` dataset.

## 5. Train/Test Leakage & Domain Shift
Because the model was trained exclusively on `custom.yaml` (81 original images) but evaluated on `combined_v2` test split (575 images from diverse domains), the model experienced extreme domain shift. It was asked to predict on AIRC and Light Source data without ever having trained on them.

## 6. Actual Failure Mode
**Primary Cause: C. Wrong checkpoint evaluated**
The model evaluated was the old `energy_detector_50ep` which was trained on `datasets/custom.yaml`, not the new `combined_v2` training. 

**Secondary Effect: E. Severe train/test domain shift**
Evaluating a model trained solely on domain A (Custom CCTV) against a test set containing massive amounts of domain B (AIRC) and domain C (Light Source) caused performance to collapse to ~0.0000 mAP.

## 7. Final Recommendation
* **Likely root cause:** The wrong training script/configuration (`train_50ep.py` with `data='datasets/custom.yaml'`) was used to produce the checkpoint, or the wrong checkpoint path was referenced during evaluation.
* **Confidence:** 100%. The `args.yaml` and `best.pt` metadata definitively prove it was trained on `custom.yaml`.
* **Dataset:** The `combined_v2` dataset itself is perfectly formatted, healthy, and split correctly. Do NOT change it.
* **Public Datasets:** Keep them.
* **Next Steps:** ANOTHER TRAINING ATTEMPT IS REQUIRED. The true `combined_v2` training must be run ensuring `data=datasets/combined_v2/dataset.yaml` is specified in the YOLO config.
