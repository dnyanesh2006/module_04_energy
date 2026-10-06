# Augmentations for YOLO training are typically handled via ultralytics hyperparameter configuration.
# This file can serve to document or implement custom albumentations if required.

def get_custom_augmentations():
    """
    Returns a dictionary of recommended augmentation hyperparameters for energy wastage detection.
    Particularly, lighting changes are crucial.
    """
    return {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.6, # High variation in brightness to simulate lights on/off
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1
    }
