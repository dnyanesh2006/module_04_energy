import uuid
from datetime import datetime, timezone

class EnergyWastageRuleEngine:
    def __init__(self, config):
        self.config = config
        
        # We need frames to represent time.
        # Assuming ~30 FPS for default logic if we don't have actual timestamps
        self.fps = 30
        
        self.wastage_persistence_frames = self.config["event"].get("wastage_persistence_seconds", 10) * self.fps
        self.event_cooldown_frames = self.config["event"].get("event_cooldown_seconds", 60) * self.fps
        
        self.wastage_condition_frames = 0
        self.frames_since_last_event = self.event_cooldown_frames

    def evaluate(self, occupancy_state, detections, frame_id, camera_id):
        """
        Evaluate if a potential energy wastage event has occurred.
        Returns an event dict if generated, else None.
        """
        self.frames_since_last_event += 1
        
        # A simple visual proxy for energy wastage: 
        # Area is unoccupied AND an energy-consuming device is detected.
        # (Note: For lights, we assume a custom model would output a "lights_on" class. 
        # For now, we use screens/appliances as proxies).
        
        device_active = False
        active_device_name = None
        highest_conf = 0.0
        track_id = None
        
        for d in detections:
            if d["class_name"] == "light_on":
                device_active = True
                if d["confidence"] > highest_conf:
                    highest_conf = d["confidence"]
                    active_device_name = d["class_name"]
                    track_id = d.get("track_id")
        
        # Rule: Unoccupied and device active
        if not occupancy_state["occupied"] and device_active:
            self.wastage_condition_frames += 1
        else:
            # Condition broken, reset counter
            self.wastage_condition_frames = 0
            
        # Check if condition persists and cooldown has passed
        if self.wastage_condition_frames >= self.wastage_persistence_frames:
            if self.frames_since_last_event >= self.event_cooldown_frames:
                # Generate event
                self.frames_since_last_event = 0
                return self._generate_event(camera_id, active_device_name, highest_conf, track_id)
                
        return None

    def _generate_event(self, camera_id, device_name, confidence, track_id):
        return {
            "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
            "module": "module_04_energy",
            "event_type": "POTENTIAL_ENERGY_WASTAGE",
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
            "severity": "MEDIUM",
            "status": "PENDING",
            "rule": {
                "name": "unoccupied_with_light_on",
                "device_type": device_name,
                "persistence_frames": self.wastage_persistence_frames
            }
        }
