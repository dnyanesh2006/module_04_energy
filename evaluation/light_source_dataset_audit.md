# Light Source Dataset Audit Report

## 1. Dataset Structure
The dataset is structured as a standard YOLOv8 folder:
* `data.yaml` (dataset configuration)
* `train/` (images and labels)
* `valid/` (images and labels)
* `test/` (images and labels)
* README files (roboflow and dataset)

## 2. Number of Images
* **Total Images:** 2,590
* **Train:** 2,212
* **Valid:** 270
* **Test:** 108

## 3. Classes from `data.yaml`
The dataset defines 7 classes (0-6):
* `0`: `0 0 0 0 1 1 1 1 0 0 0`
* `1`: `Lamp Off`
* `2`: `Lamp On`
* `3`: `Light Bulb Off`
* `4`: `Light Bulb On`
* `5`: `object`
* `6`: `tablelamp`

## 4. Annotations per Class
Across all images, the number of bounding boxes per class are:
* **Class 0:** 27
* **Class 1 (Lamp Off):** 146
* **Class 2 (Lamp On):** 181
* **Class 3 (Light Bulb Off):** 99
* **Class 4 (Light Bulb On):** 104
* **Class 5 (object):** 73
* **Class 6 (tablelamp):** 173

## 5. Image Resolutions
The dataset is overwhelmingly pre-resized to `640x640`, but contains varying original aspect ratios.
* **640x640:** 2,545 images
* **Other resolutions:** 45 images scattered across 26 different resolutions (e.g., 2048x1638, 800x800, 1500x1500, 1024x778).

## 6. YOLO Annotation Validity
* **Format:** Correctly formatted in YOLO normalized coordinates (`<class> <x> <y> <w> <h>`).
* **Validity Issues:** 5 bounding boxes are invalid (either out-of-bounds or having negative/zero dimensions).

## 7. Missing Images/Labels
* **Missing Images:** 0 (Every `.txt` has a `.jpg`)
* **Missing Labels:** 0 (Every `.jpg` has a `.txt`)

## 8. Corrupted Files
* **Corrupted Images:** 0 (All images successfully read by OpenCV).

## 9. Duplicate Images
* **Exact file duplicates:** 243 images have exact MD5 hash collisions. This likely indicates heavy data-augmentation (e.g., random cropping/flipping applied to the same source image without altering the underlying raw bits of certain crops, or just straight duplication in the dataset).

## 10. Bounding-Box Validity
* Overall valid, minus the 5 invalid boxes mentioned above.

## 11. License Information
* **License:** `CC BY 4.0`
* **Source:** Roboflow Universe (`https://universe.roboflow.com/shreyas-workspace/light-source-detection/dataset/3`).

## 12. Class Mapping Feasibility
**Yes, the classes can be safely mapped to M04.**
* **Source `2` (Lamp On) + Source `4` (Light Bulb On)** $\rightarrow$ **M04 `1` (light_on)**
* **Source `1` (Lamp Off) + Source `3` (Light Bulb Off)** $\rightarrow$ **M04 `2` (light_off)**
* All other classes (`0`, `5`, `6`) should be discarded entirely as they do not provide explicit state or are irrelevant to M04's logic.
