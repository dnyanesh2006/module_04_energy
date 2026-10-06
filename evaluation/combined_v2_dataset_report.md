# NEXORA Module 4: Combined v2 Dataset Report

## 1. Existing Data Statistics Before Merging

| Dataset | Split | Images | Person | Light ON | Light OFF |
|---|---|---|---|---|---|
| Existing Custom | Train | 55 | 51 | 66 | 15 |
| Existing Custom | Val | 12 | 51 | 66 | 15 |
| Existing Custom | Test | 14 | 51 | 66 | 15 |
| AIRC Converted | Train | 3202 | 29263 | 0 | 0 |
| AIRC Converted | Val | 204 | 29263 | 0 | 0 |
| AIRC Converted | Test | 533 | 29263 | 0 | 0 |
| Light Source | Train | 379 | 0 | 277 | 236 |
| Light Source | Val | 42 | 0 | 277 | 236 |
| Light Source | Test | 28 | 0 | 277 | 236 |

* **Combined images before sampling:** 4469

## 2. Sampling Strategy
* **Problem:** The AIRC dataset contains an overwhelming ~29,000 person annotations, while light_source only contains ~500. Using all AIRC data would heavily bias the model towards person detection and drown out the light state features.
* **Strategy:** Uniform Temporal Downsampling for AIRC Train Split. 
* **Rule used:** We sampled every 5th frame (stride = 5) from the sorted AIRC train split. This preserves the domain diversity across all 14 video sessions while reducing the annotation count to a more balanced level.
* **Validation & Test Sets:** As instructed, no artificial balancing was applied. All valid and test frames were retained to reflect real-world distributions.

## 3. Final Combined Dataset Statistics

* **Combined images after sampling:** 1908
* **Total Person annotations:** 9748
* **Total Light ON annotations:** 343
* **Total Light OFF annotations:** 251

### Split Breakdown
* **Train images:** 1075
* **Validation images:** 258
* **Test images:** 575

## 4. Source Contributions (Final)
* **Custom CCTV Data:** 81 images
* **AIRC Data:** 1378 images
* **Light Source Data:** 449 images

## 5. Integrity and Quality Validations
* **Leakage Checks:** Group-wise splits for AIRC and Light Source were rigidly maintained. No frames were shifted between Train/Val/Test.
* **Prefixes:** Unique prefixes (`custom_`, `airc_`, `light_`) were applied to all filenames guaranteeing 0 collisions.
* **Class Mapping Verification:** 
  * Class 0: Person
  * Class 1: Light ON
  * Class 2: Light OFF
  * No other class IDs exist in any label file in `combined_v2/`.
* **Corrupted/Missing Files:** Exact 1:1 image-to-label pairing. No orphan labels.

## 6. License & Limitations
* **AIRC-SMARTCLASS:** Vietnamese research consent (typically treated similarly to CC BY / CC0).
* **Light Source:** CC BY 4.0.
* **Custom M04 Data:** Proprietary domain data.
* **Limitations:** The AIRC and Light Source datasets are still somewhat disjoint (images contain either dense people OR close-up lights). The network must rely heavily on the `custom_cctv` dataset to learn the *co-occurrence* of people and lights in the same scene.
