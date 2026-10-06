# Light Source Conversion Report

## 1. Image Counts
* **Original image count:** 2590
* **Converted image count:** 449

## 2. Splits
* **Original Train:** 2212 -> **Final Train:** 379
* **Original Valid:** 270 -> **Final Val:** 42
* **Original Test:** 108 -> **Final Test:** 28

## 3. Original Class Counts
* **Class 2:** 176
* **Class 1:** 142
* **Class 6:** 172
* **Class 0:** 26
* **Class 5:** 73
* **Class 4:** 101
* **Class 3:** 94

## 4. Converted Class Counts
* **Converted light_on count (Class 1):** 277
* **Converted light_off count (Class 2):** 236

## 5. Discarded and Removed
* **Number of discarded class 0 annotations:** 26
* **Number of discarded class 5 annotations:** 68
* **Number of discarded class 6 annotations:** 172
* **Number of invalid annotations removed:** 5
* **Number of duplicate images removed:** 27
* **Number of empty images removed:** 2114

## 6. Final Class Mapping
* **Source `2` (Lamp On)** + **Source `4` (Light Bulb On)** $\rightarrow$ **M04 `1` (light_on)**
* **Source `1` (Lamp Off)** + **Source `3` (Light Bulb Off)** $\rightarrow$ **M04 `2` (light_off)**
* M04 Class `0` (`person`) is intentionally not present in this dataset but occupies ID `0` in `dataset.yaml` to ensure structural compatibility for future combination.

## 7. License
* **CC BY 4.0** (Roboflow Universe)

## 8. Dataset Limitations
* Only focuses on light source states (bulbs/lamps). It contains no people, which means it will improve light state detection but will not help with `person` detection unless merged appropriately.
* Close-up object focus on lamps/bulbs might slightly differ in scale from wide-angle CCTV, though YOLO generally handles scale variations well.

## 9. Integrity Confirmation
* The original dataset at `module_04_energy/datasets/public/light_source/` is completely **untouched**.
* Duplicate filenames were prevented by injecting the original split names into the target filename (e.g., `train_001.jpg`).
