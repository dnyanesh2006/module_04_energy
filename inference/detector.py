import cv2
import uuid
from datetime import datetime, timezone
from ultralytics import YOLO

class VisualDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.50):
        # We use a pretrained YOLO model for person and basic object detection.
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        # Define COCO classes relevant to energy/occupancy
        self.target_classes = {
            0: "person",
            1: "light_on",
            2: "light_off"
        }

    def detect_and_track(self, frame):
        """
        Run inference and tracking on a single frame.
        Returns a list of structured detections.
        """
        results = self.model.track(frame, persist=True, conf=self.conf_thresh, verbose=False)
        
        detections = []
        if not results or not results[0].boxes:
            return detections
            
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id in self.target_classes:
                conf = float(boxes.conf[i].item())
                
                # Check for track ID (ByteTrack assigns it)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())
                    
                xyxy = boxes.xyxy[i].tolist()
                bbox = {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3]
                }
                
                detections.append({
                    "class_name": self.target_classes[cls_id],
                    "confidence": conf,
                    "track_id": track_id,
                    "bbox": bbox
                })
                
        return detections

class OccupancyAnalyzer:
    def __init__(self, cooldown_frames=30):
        # Allow temporary occlusion/disappearance of people
        self.cooldown_frames = cooldown_frames
        self.frames_since_last_person = cooldown_frames
        self.is_occupied = False

    def update(self, detections):
        """Update occupancy state based on current detections."""
        person_count = sum(1 for d in detections if d["class_name"] == "person")
        
        if person_count > 0:
            self.frames_since_last_person = 0
            self.is_occupied = True
        else:
            self.frames_since_last_person += 1
            if self.frames_since_last_person >= self.cooldown_frames:
                self.is_occupied = False
                
        return {
            "occupied": self.is_occupied,
            "person_count": person_count,
            "frames_since_last_person": self.frames_since_last_person
        }
