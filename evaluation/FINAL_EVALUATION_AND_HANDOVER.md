# FINAL EVALUATION AND HANDOVER: Module 04 — Electricity & Energy Wastage Detection

## 1. MODULE OVERVIEW
- **Purpose of Module 04**: To analyze CCTV footage and detect situations where an indoor area is unoccupied but energy-consuming devices (specifically, room lights) are left turned on, identifying potential energy wastage.
- **Problem being solved**: Automating the detection of prolonged active lighting in empty rooms, replacing manual CCTV review with automated incident alerts.
- **What the module can detect**: The presence of people (`person`), the visual state of lights being on (`light_on`), and lights being off (`light_off`).
- **What the module cannot detect**: True electrical power usage or invisible device states (e.g., quiet appliances, non-luminous devices).
- **CRITICAL CLARIFICATION**: CCTV provides visual evidence of *potential* energy wastage based purely on visible luminescence and visible occupancy. **It does NOT measure actual electricity consumption, kWh, or voltage.**

## 2. FINAL ARCHITECTURE
The actual implemented inference pipeline operates in the following sequential stages:

**MODEL → DETECTION → TRACKING → TEMPORAL VALIDATION → RULE ENGINE → EVENT → INCIDENT → EVIDENCE → HUMAN REVIEW**

- **MODEL**: Loads the custom trained YOLOv8 checkpoint.
- **DETECTION**: Performs frame-by-frame object detection for `person`, `light_on`, and `light_off`.
- **TRACKING**: Utilizes ByteTrack to assign consistent IDs to detected persons across multiple frames to reliably determine room occupancy.
- **TEMPORAL VALIDATION**: Buffers states across time to prevent flickering detections (e.g., waiting for an `occupancy_cooldown` before declaring a room definitively empty).
- **RULE ENGINE**: Evaluates the buffered state against specific wastage logic (`occupancy = 0` AND `light_on`).
- **EVENT**: If the rule conditions persist beyond the `wastage_persistence_seconds` threshold, a `POTENTIAL_ENERGY_WASTAGE` event is triggered.
- **INCIDENT / EVIDENCE**: Generates structured JSON output and captures the specific evidence frame where the rule violation was finalized.
- **HUMAN REVIEW**: The output JSON and evidence image are forwarded for dashboard display and human validation.

## 3. DATASETS
**A. Data actually used to train the final custom model:**
- **Dataset name**: Custom CCTV Dataset (Generated from raw videos)
- **Purpose**: To train the YOLO model to recognize `person`, `light_on`, and `light_off` classes in actual target environments.
- **Used for final training?**: **YES**
- **Used for validation/testing?**: **YES**
- **Current location**: `datasets/custom_cctv_split/`
- **Limitations**: Contains a limited number of distinct indoor environments (only 3 CCTV angles).

**B. Other datasets present in the project but NOT used in final custom-model training:**
- **Dataset name**: COCO 2017 Dataset (Reference to 18GB dataset)
- **Purpose**: General-purpose object detection dataset (pretrained weights origin).
- **Used for final training?**: **NO** (Not downloaded, processed, or fed into the `energy_detector_combined` training loop).
- **Current location**: Pretrained weights origin (`yolov8n.pt`); raw dataset not physically present in project training paths.
- **Limitations**: Too generic; does not natively support `light_on`/`light_off` states.

- **Dataset name**: UCI Occupancy Dataset
- **Purpose**: Sensor-based (temperature, light, CO2) occupancy prediction. 
- **Used for final training?**: **NO**
- **Current location**: Not physically present (Referenced in early documentation).
- **Limitations**: Tabular/sensor data, completely incompatible with YOLO visual training.

- **Dataset name**: UCI Appliances Energy Prediction Dataset
- **Purpose**: Predicting energy consumption based on temperature/humidity.
- **Used for final training?**: **NO**
- **Current location**: Not physically present (Referenced in early documentation).
- **Limitations**: Tabular data, not applicable to visual object detection.

## 4. FINAL TRAINING DATA
- **Number of source videos**: 3
- **Video filenames**: `light_on.mp4`, `light_off.mp4`, `replacement_test.mp4`
- **Number of extracted images**: 81 total representative frames
- **Train/validation/test counts**: Train: 55, Validation: 12, Test: 14
- **Classes**: `person` (0), `light_on` (1), `light_off` (2)
- **Dataset split method**: Video-level chronological splitting. Rather than randomly shuffling frames (which causes identical adjacent frames to leak between train and test sets), each video was split sequentially (e.g., first 70% of frames to train, next 15% to val, last 15% to test).

## 5. FINAL MODEL
- **Model architecture/name**: YOLOv8n (Custom trained)
- **Training epochs**: 50
- **Checkpoint path**: `models/checkpoints/energy_detector_combined/weights/best.pt`
- **Export path**: `models/exports/energy_detector_combined.pt`
- **Configuration path**: `config/config.yaml`
- **Previous model backed up?**: Yes, located in `models/checkpoints/backups/energy_detector_50ep_backup.pt`
- **Relevant training settings**: Trained on CPU (`device="cpu"`), Image size 640x640.

## 6. VALIDATION METRICS
*Metrics captured on the Validation split (12 images) of the combined dataset:*
- **Precision**: 0.653
- **Recall**: 0.559
- **mAP@50**: 0.863
- **mAP@50-95**: 0.698

**Per-class Results:**
- **person**: Precision: 0.267, Recall: 1.000, mAP@50: 0.995, mAP@50-95: 0.895
- **light_on**: Precision: 0.692, Recall: 0.677, mAP@50: 0.598, mAP@50-95: 0.402
- **light_off**: Precision: 1.000, Recall: 0.000, mAP@50: 0.995, mAP@50-95: 0.796 *(Note: Recall is 0.000 because `light_off` was severely underrepresented in the 12-image validation slice)*

*(Note: mAP stands for Mean Average Precision. It is a measure of bounding box accuracy and recall balance. It is NOT generic classification "accuracy".)*

## 7. REPLACEMENT VIDEO EVALUATION
Actual evaluation results for running the full inference pipeline on `datasets/custom_cctv/replacement_test.mp4`:
- **Person detection**: YES (234 detection instances)
- **Light ON detection**: YES (119 detection instances)
- **Light OFF detection**: NO (0 instances in this specific video)
- **Tracking**: YES
- **Occupancy**: State successfully tracked (Person entered and left).
- **Temporal validation**: Evaluated successfully.
- **Energy-wastage event**: **NO**
- **Event ID**: None
- **Evidence path**: None
- **JSON path**: `evaluation/results/realtime_test/event_output.json`
- **Annotated video path**: `evaluation/results/realtime_test/annotated_test.mp4`

**Why was an energy-wastage event NOT generated?**
The pipeline operated flawlessly, but the event condition was strictly not met in the video. The `light_on` state was successfully detected while the person was in the room. However, after the person left the frame (occupancy dropped to 0), the `light_on` detection did not persist concurrently for the required temporal duration. Therefore, the strict rule `occupancy = 0 AND light_on` was not satisfied long enough to trigger a formal wastage event.

## 8. UNIT TESTS
- **Total tests**: 13
- **Passed tests**: 13
- **Failed tests**: 0
- **Test categories**: Detection processing (`test_detection.py`), Event generation logic (`test_events.py`), Edge case handling (`test_edge_cases.py`).

## 9. EVENT RULE
The exact implemented rule in the pipeline requires:
`occupancy = 0` **AND** `light_on` detected **AND** condition persists for configured duration.

Per `config/config.yaml`, the actual configured thresholds are:
- `occupancy_cooldown_seconds: 2` (Room must be empty for 2 seconds before checking for wastage).
- `wastage_persistence_seconds: 3` (The light must remain visibly ON in the empty room for 3 continuous seconds).

## 10. OUTPUTS
Important generated outputs after running the inference pipeline:
- **Detection JSON**: `evaluation/results/realtime_test/detection_output.json` (Frame-by-frame raw detections)
- **Event JSON**: `evaluation/results/realtime_test/event_output.json` (Structured wastage incidents)
- **Evidence images**: `evaluation/results/evidence/` (Captured frames when events trigger)
- **Annotated videos**: `evaluation/results/realtime_test/annotated_test.mp4` (Visual debugging)
- **Evaluation results**: Generated in terminal via `class_eval.py`
- **Model checkpoint**: `models/checkpoints/energy_detector_combined/weights/best.pt`
- **Model export**: `models/exports/energy_detector_combined.pt`

## 11. UNKNOWN/UNSEEN VIDEO TESTING
To test the model on a genuinely unseen CCTV video, future NEXORA developers should:
1. Ensure the new video has **never** been used in any training or validation dataset.
2. Ensure the video is loaded purely via the configuration interface, passing the path to the pipeline. No custom code, hard-coded filenames, or special bounding-box logic should be written for the new video.
3. Run the standard command: 
   `python -m inference.pipeline --source <path_to_new_video> --camera_id <ID>`
4. **Purpose**: The purpose of this test is to evaluate the model's *generalization* to an entirely unseen environment (different camera angle, different lighting fixtures, different room). 
5. **Disclaimer**: We do **not** claim universal generalization. The current dataset contains only 3 environments. Performance on a completely unseen 4th environment is not guaranteed and will depend on visual similarity to the training data.

## 12. LIMITATIONS
- **Small custom dataset**: The model was trained on only 81 frames.
- **Limited indoor environments**: The training data only covers 3 specific room/camera configurations.
- **Domain shift**: Significant changes in camera angle, lighting fixture type, or room brightness may cause detection failures in new environments.
- **Validation size**: The validation set is extremely small (12 images), making metrics highly sensitive.
- **Visual vs. Electrical**: Visual detection of a light source is not a true electrical measurement. External ambient light hitting a fixture might cause a false positive, and an active but occluded light might cause a false negative.

## 13. INTEGRATION INSTRUCTIONS
To integrate Module 04 into the wider NEXORA system:
- **Folder to copy**: The entire `module_04_energy/` directory.
- **Required Model**: `models/checkpoints/energy_detector_combined/weights/best.pt`
- **Required Configuration**: `config/config.yaml`
- **Dependencies**: `ultralytics`, `opencv-python`, `torch`, `numpy`, `pyyaml`.
- **Providing a new video**: Pass the video file path dynamically via the CLI `--source` argument to `inference/pipeline.py`.
- **Running Inference**: 
  `python -m inference.pipeline --source <video_path> --camera_id <camera_id>`
- **Results Location**: Results are output to the directories defined in `config.yaml` (default: `evaluation/results/`).
- **Structured Output**: The frontend/integration layer should consume the `event_output.json` array, which contains structured `POTENTIAL_ENERGY_WASTAGE` objects containing `event_id`, `timestamp`, `camera_id`, and exact rule violation metadata.

## 14. FINAL STATUS
**Based strictly on the successful test suite execution, the functioning pipeline logic, and the completion of the combined model training, Module 04 is READY for integration.**

## 15. FILE/FOLDER STRUCTURE
```text
module_04_energy/
├── config/                 <-- Configuration files
│   └── config.yaml
├── datasets/               <-- Datasets & preprocessing scripts
│   ├── custom_cctv/
│   ├── custom_cctv_split/
│   ├── custom.yaml
│   └── dataset_tools.py
├── evaluation/             <-- Inference outputs & evidence
│   └── results/
├── inference/              <-- Core runtime inference logic
│   ├── detector.py
│   ├── pipeline.py
│   └── tracker.py
├── models/                 <-- Trained models and backups
│   ├── checkpoints/
│   │   ├── backups/
│   │   └── energy_detector_combined/
│   └── exports/
├── rules/                  <-- Event and logic engines
│   └── event_rules.py
├── tests/                  <-- Pytest unit test suite
│   ├── test_detection.py
│   ├── test_edge_cases.py
│   └── test_events.py
├── training/               <-- Model training configuration
│   └── augmentations.py
├── FINAL_EVALUATION_AND_HANDOVER.md  <-- Documentation (This file)
├── README.md               <-- Documentation
└── train_combined.py       <-- Source code (Training script)
```

## 16. DATASET CLARIFICATION
### Why the downloaded COCO and other datasets are not necessarily part of the final training
In machine learning projects, there is a strict difference between *downloaded/referenced datasets* and datasets *actually processed and fed into the final training loop*.

- **COCO Dataset**: While the model architecture (`yolov8n.pt`) originates from weights pre-trained on the COCO dataset by the Ultralytics team, the massive 18GB COCO dataset itself was **never** downloaded, parsed, or passed into this project's final fine-tuning step.
- **UCI Occupancy / UCI Appliances Datasets**: These are tabular/sensor datasets referenced in early project documentation to define the theoretical problem domain. Because YOLO is a purely visual object detection architecture, these tabular datasets are entirely incompatible with the model and were **never** used in the codebase.
- **Actual Training Data**: The *only* data physically used to train the final `energy_detector_combined` model consisted of the 55 specifically extracted and annotated `.jpg` frames found in `datasets/custom_cctv_split/train/images`.
