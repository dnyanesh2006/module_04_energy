# Dataset Documentation

## Dataset Name
1. COCO 2017 (Common Objects in Context)
2. UCI Occupancy Detection Data Set
3. UCI Appliances Energy Prediction Data Set
4. Custom NEXORA CCTV Dataset (Pending Collection)

## Dataset Source
1. COCO Consortium
2. UCI Machine Learning Repository
3. UCI Machine Learning Repository
4. Internal NEXORA Collection

## Dataset URL
1. https://cocodataset.org/
2. https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
3. https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
4. N/A (Internal)

## License
1. COCO: Creative Commons Attribution 4.0 License (Images are property of Flickr users)
2. UCI Datasets: Creative Commons Attribution 4.0 International (CC BY 4.0)
3. UCI Appliances: Creative Commons Attribution 4.0 International (CC BY 4.0)
4. Custom NEXORA: Internal Proprietary

## License Verification Date
2026-10-03

## Dataset Purpose
1. **COCO 2017**: General person and object detection to bootstrap the module's YOLO detector. (Note: Only images are present currently; annotations need to be downloaded or a pre-trained model used).
2. **UCI Occupancy Detection**: To analyze sensor-based time-series data related to room occupancy (Light, CO2, etc.) to establish realistic logical rules for occupancy determination.
3. **UCI Appliances Energy Prediction**: To establish a theoretical foundation for integrating actual energy (IoT) data with our visual features.
4. **Custom NEXORA CCTV Dataset**: (Pending) To train and test energy-specific visual features under real CCTV conditions (e.g., empty room + lights ON).

## Classes Used
- Person
- TV/Monitor
- Microwave
- Refrigerator

## Classes Not Used
All other COCO classes (e.g., cars, animals, food) are excluded as they are irrelevant for indoor energy wastage detection.

## Number of Images
- COCO 2017: ~118k train images (unannotated locally).
- Custom CCTV: Pending.

## Number of Videos
- Custom CCTV: Pending.

## Annotation Format
- YOLO format for Custom CCTV.
- COCO format for COCO 2017.

## Train Split
- Custom CCTV: Camera/session based (Pending).

## Validation Split
- Custom CCTV: Camera/session based (Pending).

## Test Split
- Custom CCTV: Camera/session based (Pending).

## Custom Dataset
The custom dataset will contain CCTV footage of typical indoor scenes under various energy states (lights ON/OFF, devices ON/OFF, occupied/unoccupied).

## Custom Data Collection Method
CCTV cameras will be positioned in rooms to capture baseline scenarios, including daytime/nighttime variations, occlusion, and varying angles.

## Data Preprocessing
- Video-to-frame extraction at fixed intervals.
- Exact duplicate removal via MD5 hashing.
- Resizing to 640x640 for YOLO model input.

## Data Augmentation
- Adjustments to brightness and contrast to simulate varying lighting conditions.
- Slight rotations and scaling.

## Hard Negative Samples
- A room with an unilluminated reflective surface mistaken for a light being ON.
- Temporary occlusion of a person creating a false "unoccupied" state.

## Data Quality Issues
- COCO labels are missing from the current local dataset folder.
- True CCTV lighting state can be highly subjective based on camera auto-exposure.

## Known Dataset Limitations
- The current datasets do not contain actual CCTV visual examples of energy wastage.
- The UCI datasets are purely sensor/IoT time-series, lacking visual components.
