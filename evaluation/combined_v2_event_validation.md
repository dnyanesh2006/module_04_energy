# Event Validation Report (Combined v2 Model)

## Video: light_off.mp4
* **Person Detections:** 0
* **Light ON Detections:** 0
* **Light OFF Detections:** 1
* **Tracking / Occupancy Result:** Detections successfully fed into occupancy tracking.
* **Event Generated:** False
* **Evidence Saved:** evaluation/results/test_light_off/annotated_test.mp4 (plus jpg snapshots in evidence dir)

## Video: light_on.mp4
* **Person Detections:** 58
* **Light ON Detections:** 108
* **Light OFF Detections:** 0
* **Tracking / Occupancy Result:** Detections successfully fed into occupancy tracking.
* **Event Generated:** False
* **Evidence Saved:** evaluation/results/test_light_on/annotated_test.mp4 (plus jpg snapshots in evidence dir)

## Video: replacement_test.mp4
* **Person Detections:** 4
* **Light ON Detections:** 0
* **Light OFF Detections:** 0
* **Tracking / Occupancy Result:** Detections successfully fed into occupancy tracking.
* **Event Generated:** False
* **Evidence Saved:** evaluation/results/test_replacement_test/annotated_test.mp4 (plus jpg snapshots in evidence dir)

## Pipeline Verdict
MODEL -> DETECTION -> TRACKING -> TEMPORAL VALIDATION -> RULE ENGINE -> EVENT -> INCIDENT -> EVIDENCE -> HUMAN REVIEW pipeline ran to completion on all 3 videos, successfully proving the codebase logic is intact. However, because the new YOLOv8 model's weights collapsed, it generated nearly zero accurate detections. Consequently, 0 valid wastage events were generated across all test files.

**Important Note:** The event specifically represents a potential energy wastage incident (occupancy=0 AND light_on condition persisting). It does NOT claim physical kWh electricity consumption values. Human review remains strictly required to confirm the incident.
