import os
import cv2
import yaml
import json
from datetime import datetime, timezone
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.detector import VisualDetector, OccupancyAnalyzer
from rules.event_rules import EnergyWastageRuleEngine

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_video(source: str, camera_id: str, config_path: str = "config/config.yaml"):
    """
    Main entry point for processing a video source for energy wastage.
    """
    config = load_config(config_path)
    
    # Initialize components
    model_path = config["model"].get("path", "yolov8n.pt")
    conf_thresh = config["model"].get("confidence_threshold", 0.50)
    
    detector = VisualDetector(model_path=model_path, conf_thresh=conf_thresh)
    # Convert seconds to frames using an assumed 30fps for the logic
    cooldown_frames = int(config["event"].get("occupancy_cooldown_seconds", 15) * 30)
    occupancy = OccupancyAnalyzer(cooldown_frames=cooldown_frames)
    rule_engine = EnergyWastageRuleEngine(config)
    
    # Setup video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Update rule engine fps with actual video fps
    rule_engine.fps = fps
    rule_engine.wastage_persistence_frames = config["event"].get("wastage_persistence_seconds", 10) * fps
    rule_engine.event_cooldown_frames = config["event"].get("event_cooldown_seconds", 60) * fps

    # Setup Video Writer for annotated video
    out_dir = "evaluation/results/realtime_test"
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(out_dir, "annotated_test.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_id = 0
    all_events = []
    all_detections = []
    
    # Output directories for evidence
    evidence_dir = config["evidence"].get("evidence_dir", "evaluation/results/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
    print(f"Starting processing for source: {source} on camera: {camera_id}", flush=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id % 5 != 0:
            frame_id += 1
            continue

        if frame_id % 100 == 0:
            print(f"Processing frame {frame_id}...", flush=True)

        annotated_frame = frame.copy()
            
        # 1. Detection & Tracking
        detections = detector.detect_and_track(frame)
        
        # 2. Add standard fields to detections
        timestamp = datetime.now(timezone.utc).isoformat()
        std_detections = []
        for d in detections:
            std_d = {
                "module": "module_04_energy",
                "camera_id": camera_id,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "track_id": d.get("track_id"),
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": d["bbox"]
            }
            std_detections.append(std_d)
            
            # Draw bounding box
            x1, y1, x2, y2 = int(d["bbox"]["x1"]), int(d["bbox"]["y1"]), int(d["bbox"]["x2"]), int(d["bbox"]["y2"])
            label = f"{d['class_name']} {d['confidence']:.2f} ID:{d.get('track_id', 'N/A')}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        all_detections.extend(std_detections)
            
        # 3. Occupancy Determination
        occ_state = occupancy.update(std_detections)
        occ_text = "Occupied" if occ_state["occupied"] else "Unoccupied"
        color = (0, 255, 0) if occ_state["occupied"] else (0, 0, 255)
        cv2.putText(annotated_frame, f"Occupancy: {occ_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 4. Rule Engine Evaluation
        event = rule_engine.evaluate(occ_state, std_detections, frame_id, camera_id)
        
        if event:
            # 5. Evidence Generation
            if config["evidence"].get("save_frame", True):
                evidence_path = os.path.join(evidence_dir, f"{event['event_id']}.jpg")
                cv2.imwrite(evidence_path, annotated_frame)
                
            all_events.append(event)
            print(f"EVENT GENERATED: {json.dumps(event, indent=2)}")
            
        # Draw event status
        if rule_engine.wastage_condition_frames > 0:
            cv2.putText(annotated_frame, f"Wastage Timer: {rule_engine.wastage_condition_frames}/{rule_engine.wastage_persistence_frames}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        out.write(annotated_frame)
        frame_id += 1
        
    cap.release()
    out.release()
    
    # Write JSON outputs
    with open(os.path.join(out_dir, "detection_output.json"), "w") as f:
        json.dump(all_detections, f, indent=4)
        
    with open(os.path.join(out_dir, "event_output.json"), "w") as f:
        json.dump(all_events, f, indent=4)
        
    print(f"Finished processing {source}. Generated {len(all_events)} events.")
    print(f"Outputs saved to {out_dir}")
    return all_events

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Video source path or URL")
    parser.add_argument("--camera_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config")
    
    args = parser.parse_args()
    process_video(args.source, args.camera_id, args.config)
