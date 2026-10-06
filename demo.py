import json
import os
from datetime import datetime, timezone

def generate_sample_outputs():
    os.makedirs("evaluation/results", exist_ok=True)
    
    # Generate Sample Detection
    sample_detection = {
        "module": "module_04_energy",
        "camera_id": "CAM-001",
        "frame_id": 1250,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_id": 17,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": {
            "x1": 120,
            "y1": 80,
            "x2": 300,
            "y2": 420
        }
    }
    
    with open("evaluation/results/sample_detection.json", "w") as f:
        json.dump(sample_detection, f, indent=4)
        
    # Generate Sample Event
    sample_event = {
        "event_id": "EVT-000123",
        "module": "module_04_energy",
        "event_type": "POTENTIAL_ENERGY_WASTAGE",
        "camera_id": "CAM-001",
        "track_id": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.88,
        "severity": "MEDIUM",
        "status": "PENDING",
        "rule": {
            "name": "unoccupied_with_device_on",
            "device_type": "tv",
            "persistence_frames": 300
        }
    }
    
    with open("evaluation/results/sample_event.json", "w") as f:
        json.dump(sample_event, f, indent=4)
        
    print("Sample JSON outputs generated in evaluation/results/")
    print("NOTE: Real demo video generation requires actual CCTV footage, which is currently missing.")
    print("To run the full demo once footage is available:")
    print("  python inference/pipeline.py --source datasets/custom_cctv/demo.mp4 --camera_id CAM-001")

if __name__ == "__main__":
    generate_sample_outputs()
