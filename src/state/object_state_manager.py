import numpy as np
import time
from collections import deque

class ObjectStateManager:
    def __init__(self, fill_thresholds=None, movement_threshold=5):
        self.states = {}  # track_id -> state
        self.fill_thresholds = fill_thresholds or {"empty": 0.2, "full": 0.8}
        self.movement_threshold = movement_threshold  # Pixelbewegung unterhalb dieser Grenze = still
        self.objects = {}

    def update_object(self, track_id, bbox, fill_ratio, mask=None, obj_class=None):
        if track_id not in self.objects:
            self.objects[track_id] = {
                "bbox": bbox,
                "fill_ratio": fill_ratio,
                "mask": mask,
                "status": "NEW",
                "last_seen": time.time(),
                "obj_class": obj_class  # Neu!
            }
        else:
            self.objects[track_id].update({
                "bbox": bbox,
                "fill_ratio": fill_ratio,
                "mask": mask,
                "last_seen": time.time(),
                "obj_class": obj_class  # Aktualisieren
            })

        status = self._determine_status(track_id)
        self.objects[track_id]["status"] = status

        return self.objects[track_id]


    def _calculate_movement(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
        return np.linalg.norm(np.array(center1) - np.array(center2))

    def _determine_status(self, track_id):
        fill_ratio = self.objects[track_id]["fill_ratio"]

        if fill_ratio > 0.8:
            return "FULL"
        elif fill_ratio > 0.3:
            return "HALF"
        else:
            return "EMPTY"
    
    def _update_status(self, track_id):
        obj = self.states[track_id]
        fill_ratio = obj["fill_ratio"]
        movement = obj["movement"]

        if fill_ratio < self.fill_thresholds["empty"]:
            fill_status = "EMPTY"
        elif fill_ratio > self.fill_thresholds["full"]:
            fill_status = "FULL"
        else:
            fill_status = "PARTIAL"

        if movement > self.movement_threshold:
            motion_status = "MOVING"
        else:
            motion_status = "STILL"

        if motion_status == "MOVING":
            if fill_status == "EMPTY":
                obj["status"] = "DRINKING"
            elif fill_status == "FULL":
                obj["status"] = "REFILLING"
            else:
                obj["status"] = "MOVING"
        else:
            obj["status"] = fill_status

    def get_status(self, track_id):
        return self.states.get(track_id, {}).get("status", "UNKNOWN")

    def get_fill_ratio(self, track_id):
        return self.states.get(track_id, {}).get("fill_ratio", 1.0)

    def cleanup_lost_tracks(self, active_track_ids):
        # Löscht alle, die nicht mehr im Frame sind
        for track_id in list(self.states.keys()):
            if track_id not in active_track_ids:
                del self.states[track_id]
