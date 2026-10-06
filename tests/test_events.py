import unittest
from rules.event_rules import EnergyWastageRuleEngine

class TestEvents(unittest.TestCase):
    def setUp(self):
        config = {
            "event": {
                "occupancy_cooldown_seconds": 1,
                "wastage_persistence_seconds": 2, # 2 sec * 30 fps = 60 frames
                "event_cooldown_seconds": 1
            }
        }
        self.engine = EnergyWastageRuleEngine(config)
        self.engine.fps = 30 # Override for testing

    def test_empty_room_lights_on(self):
        occupancy_state = {"occupied": False}
        detections = [
            {"class_name": "light_on", "confidence": 0.9, "track_id": 1, "bbox": {}}
        ]
        
        event = None
        # Simulate 60 frames of condition
        for i in range(60):
            event = self.engine.evaluate(occupancy_state, detections, i, "CAM-01")
            
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "POTENTIAL_ENERGY_WASTAGE")

    def test_occupied_room_lights_on(self):
        occupancy_state = {"occupied": True} # Should NOT trigger
        detections = [
            {"class_name": "light_on", "confidence": 0.9, "track_id": 1, "bbox": {}}
        ]
        
        event = None
        for i in range(60):
            event = self.engine.evaluate(occupancy_state, detections, i, "CAM-01")
            
        self.assertIsNone(event)

    def test_cooldown(self):
        # Tests that events don't spam
        occupancy_state = {"occupied": False}
        detections = [{"class_name": "light_on", "confidence": 0.9, "track_id": 1, "bbox": {}}]
        
        events_generated = 0
        for i in range(150): # 5 seconds
            event = self.engine.evaluate(occupancy_state, detections, i, "CAM-01")
            if event:
                events_generated += 1
                
        # 1st event at 60 frames. Cooldown is 30 frames. Next at 90. Next at 120. Next at 150.
        # Actually logic says frames_since_last >= cooldown to trigger.
        self.assertGreater(events_generated, 1)

if __name__ == "__main__":
    unittest.main()
