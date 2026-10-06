# AIRC-SMARTCLASS Dataset Audit Report

## 1. Dataset Location
The downloaded dataset is already located in the preparation area:
`module_04_energy/datasets/public/person_airc/AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo/AIRC-SMARTCLASS A Multi-Modal Classroom Dataset fo/`

Sub-directories found:
* `Data/Frames/` (Images)
* `Data/Labels/` (Annotations)
* `Data/Videos/` (Source videos)
* `Metadata/` (JSON & CSV descriptions)
* `Consent Forms/` (Legal documentation)

## 2. Dataset Size
Approximately **9,539.89 MB (~9.54 GB)** total.

## 3. Number of Usable Images
* **4,089** total `.jpg` frames found.
* **4,081** frames have matching label files.
* **8** frames are missing label files.

## 4. Number of Person Annotations
* **29,263** bounding boxes correspond to the `person` class across the dataset.

## 5. Annotation Format
* **Format:** YOLO format `.txt` (`<class_id> <x_center> <y_center> <width> <height>`).
* Bounding boxes are normalized relative to image dimensions.

## 6. Image Resolution
* **1920 x 1080** (HD).

## 7. Classes Found
The dataset uses 6 classes (0 through 5). According to the `labels_list.csv` and bounding box characteristics:
* `0` = face
* **`1` = person**
* `2` = status (AC)
* `3` = temp (AC)
* `4` = mode (AC)
* `5` = fan_lv (AC)

## 8. License
* **Source:** https://data.mendeley.com/datasets/5zp8d2wk7b/1
* **License:** The local `metadata.json` explicitly mentions "Informed consent from participating teachers and students for research purposes." Due to Mendeley Data standard defaults, it is likely governed by **CC BY 4.0** (or CC0), but the local files strictly denote it is for "research purposes" under Vietnamese data protection regulations. 

## 9. Data-Quality Findings
* **Missing annotations:** 8 `.jpg` images do not have corresponding `.txt` annotation files.
* **Invalid boxes:** 0 corrupted or out-of-bounds bounding boxes.
* **Extremely small boxes:** 1 bounding box has a width or height `< 0.005` (normalized).
* **Missing classes:** No empty/negative images without people were found (density is very high).
* **Unnecessary classes:** Classes 0, 2, 3, 4, 5 are completely irrelevant for M04.

## 10. Recommended Conversion Method
To prepare this dataset for NEXORA M04 without breaking existing logic:
1. Parse every `.txt` label file.
2. Filter out classes `0`, `2`, `3`, `4`, `5`.
3. **Map AIRC class `1` (person) → M04 class `0` (person).**
4. Discard the 8 images that lack annotation files to prevent YOLO training warnings.
5. Filter out the single sub-0.005 microscopic bounding box.
6. Save the cleaned YOLO `.txt` files into a processed directory, completely separate from the current `custom_cctv_split`.

## 11. Recommended Train/Validation/Test Split
Because the 4,081 frames are extracted from **17 continuous videos**, a standard random 80/20 split will cause severe data leakage (nearly identical frames in train and val). 
* **Recommendation:** Group-based split (GroupKFold) by video folder (`Labels/001/`, `Labels/002/`, etc.). Assign ~13 videos to Train (80%) and ~4 videos to Validation (20%).

## 12. Limitations for NEXORA M04
* **Viewpoint:** The dataset is taken from a top-down, extremely wide-angle camera in a university classroom. While this matches CCTV, the distortion might differ from standard office security cameras.
* **Class Imbalance:** The frames are extremely dense (averaging ~7 people per frame). M04 also needs negative samples (empty rooms) to learn when to trigger "unoccupied" events. AIRC-SMARTCLASS does not appear to provide empty room negative samples.

## 13. Suitability
**Highly Suitable.** With nearly 30,000 accurate person annotations in a surveillance-style environment, it will drastically improve M04's ability to detect people, especially in occluded or seated positions.
