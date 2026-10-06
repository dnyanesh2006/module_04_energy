# Combined v2 Training & Evaluation Report

## 1. Training Configuration & Paths
* **Dataset:** `module_04_energy/datasets/combined_v2/dataset.yaml`
* **Baseline Model Path:** `models/checkpoints/energy_detector_combined/weights/best.pt`
* **New Model Path:** `runs/detect/runs/detect/energy_detector_50ep/weights/best.pt`
* **Epochs:** 50/50 (Confirmed Complete)
* **Device Used for Eval:** cpu

## 2. Overall Performance Comparison (Held-out TEST split)

| Metric | Old Model (Baseline) | New Model (combined_v2) | Improvement |
|---|---|---|---|
| **Precision** | 0.6532 | 0.0001 | -0.6530 |
| **Recall** | 0.0400 | 0.1915 | 0.1515 |
| **F1-Score** | 0.0754 | 0.0003 | -0.0751 |
| **mAP@50** | 0.0768 | 0.0004 | -0.0764 |
| **mAP@50:95** | 0.0144 | 0.0000 | -0.0143 |
| **Inference Speed (ms/img)** | 82.97 | 71.21 | - |

## 3. Per-Class Metrics Comparison (mAP@50)

| Class | Old Model mAP@50 | New Model mAP@50 | Improvement |
|---|---|---|---|
| **person** | 0.0768 | 0.0004 | -0.0764 |
| **light_on** | 0.0000 | 0.0000 | 0.0000 |
| **light_off** | 0.0000 | 0.0000 | 0.0000 |

## 4. Per-Class Precision & Recall (New Model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| **person** | 0.0000 | 0.0003 | 0.0000 |
| **light_on** | 0.0003 | 0.2800 | 0.0006 |
| **light_off** | 0.0001 | 0.2941 | 0.0003 |

## 5. Summary & Conclusions
* **Did light_on / light_off improve?** No. While the model occasionally guesses the classes (yielding non-zero but abysmal recall and near-zero precision), the mAP@50 remains exactly 0.0000.
* **Did person detection degraded?** Yes. Person detection degraded massively. The baseline model had ~0.0768 mAP@50 for person, but the new model has collapsed completely to 0.0004.
* **Should combined_v2 become the new M04 baseline?** **NO.** The combined_v2 model suffered a catastrophic training collapse and performs significantly worse than the baseline. The baseline model must be retained. Do NOT deploy this model to production.
* **Limitations:** The training appears to have failed to converge or overfit completely (possibly due to learning rate, batch size, or extreme dataset domain mismatch despite downsampling).
* **Final Verdict:** This evaluation task confirms the 50/50 training is complete, but the resulting weights are not viable.
