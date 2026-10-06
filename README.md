# NEXORA Module 04 — ELECTRICITY & ENERGY WASTAGE DETECTION

## 1. Module Overview
This module analyzes CCTV footage to detect potential energy wastage by identifying conditions where an area is unoccupied but energy-consuming devices (specifically room lights) are visibly active.

**Important Limitations:**
- The model is trained using multiple indoor video environments and is intended to generalize to unseen CCTV footage, but performance on unseen environments is not guaranteed.
- This module detects potential visual energy wastage based on visible occupancy and visible light state. It does not measure actual electrical energy consumption.

## 2. Module ID
`module_04_energy`

## 3. Objective
To automate the visual detection of potential energy wastage in monitored areas, converting prolonged observations of unoccupied spaces with active lights into structured incidents for human review.

## 4. Supported Detection Classes
- person (Class 0)
- light_on (Class 1)
- light_off (Class 2)

## 5. Detection Criteria
- Person detection used to determine room occupancy.
- Light state (ON/OFF) detected directly from visible luminescence.
- Tracking enabled to maintain identities across frames.

## 6. Event Criteria
Detection → Tracking → Occupancy determination (0 people present) → Temporary disappearance validation (cooldown/persistence) → Rule validation (area empty + light_on) → Event generation

## 7. Model Architecture
- Model name: YOLOv8 (Custom trained)
- Model location: `models/checkpoints/energy_detector_combined/weights/best.pt`
- Input resolution: 640x640

## 8. Dataset
- Training videos: `light_off.mp4`, `light_on.mp4`, `replacement_test.mp4` (Combined).
- The dataset split is performed at the video level (temporally) to prevent frame leakage.

## 9. Training Procedure
The model was trained for 50 epochs using the combined dataset generated from the three videos above. It learns the visual classes `person`, `light_on`, and `light_off` directly from actual room appearances.

## 10. Inference Command
To run inference on any video source:
```bash
python -m inference.pipeline --source <video_path> --camera_id CAM-01
# example: python -m inference.pipeline --source datasets/custom_cctv/replacement_test.mp4 --camera_id CAM-01
```
*(You can run `python demo.py --video <video_path>` if supported by the demo wrapper).*

## 11. Output Locations
Structured JSON objects representing frame-level detections and temporal events (potential energy wastage) are saved to:
- `evaluation/results/realtime_test/detection_output.json`
- `evaluation/results/realtime_test/event_output.json`
- Annotated video: `evaluation/results/realtime_test/annotated_test.mp4`
- Evidence images: `evaluation/results/evidence/<event_id>.jpg`

## 12. Event JSON Example
```json
{
  "event_id": "EVT-000123",
  "module": "module_04_energy",
  "event_type": "POTENTIAL_ENERGY_WASTAGE",
  "camera_id": "CAM-001",
  "track_id": null,
  "timestamp": "2026-10-02T14:30:15.200Z",
  "confidence": 0.85,
  "severity": "MEDIUM",
  "status": "PENDING",
  "rule": {
    "name": "unoccupied_with_light_on",
    "device_type": "light_on",
    "persistence_frames": 90
  }
}
```

## 13. Known Limitations
- Cannot directly measure kWh or voltage.
- Relies heavily on accurate person detection to determine occupancy.
- True state of some devices cannot be reliably detected visually.
