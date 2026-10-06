# AIRC-SMARTCLASS Conversion Report

## Conversion Metrics
* **Source image count:** 4089
* **Converted image count:** 3939
* **Source person annotation count:** 29263
* **Converted person annotation count:** 29263
* **Excluded images:** 150 (Due to missing labels, or containing zero valid person annotations after filtering)
* **Excluded annotations:** 0 (Extremely small or out-of-bounds bounding boxes)

## Split Information (Video/Session Grouped to Prevent Leakage)
* **Train split:**
  * **Images:** 3202
  * **Annotations:** 24456
  * **Assigned Source Sessions:** 008, 011, 006, 007, 016, 015, 014, 010, 009, 002, 003, 013, 017, 005
* **Validation split:**
  * **Images:** 204
  * **Annotations:** 1362
  * **Assigned Source Sessions:** 012
* **Test split:**
  * **Images:** 533
  * **Annotations:** 3445
  * **Assigned Source Sessions:** 001, 004

## Class Mapping
* **AIRC-SMARTCLASS class `1` (person)** was exclusively extracted and re-mapped to **M04 class `0` (person)**. 
* All other classes (face, AC status, temp, mode, fan_lv) were completely discarded.

## Quality Checks & Validations
* Every image copied into the converted directory has a corresponding label file.
* Every label file contains ONLY class ID `0`.
* Every bounding box has valid, normalized YOLO coordinates (`w > 0.005`, `h > 0.005`, `x,y,w,h` between `0` and `1`).
* Unique image naming is guaranteed by prepending the session ID (e.g. `001_frame_0000.jpg`), preventing collisions.
* No frame leakage occurred across splits (grouped by source session).

## License
* The original dataset declares informed consent from participants for research purposes. Typical Mendeley Data defaults to CC BY 4.0.

## Limitations
* The dataset only features images of occupied spaces. It lacks negative (empty room) samples, meaning it will improve `person` detection but won't directly teach the model about empty states without being combined with other datasets.
