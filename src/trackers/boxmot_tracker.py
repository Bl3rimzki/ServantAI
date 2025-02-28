import numpy as np
import torch
from pathlib import Path
from boxmot import BotSort  # Passe den Import ggf. an deine Repository-Struktur an

class BoxMotTracker:
    def __init__(self, reid_weights="osnet_x1_0_market1501.pt", device=None, half=False):
        """
        Initialisiert den BoxMot Tracker.

        Args:
            reid_weights (str oder Path): Pfad oder Name der ReID-Gewichte. 
                Standardmäßig wird "osnet_x1_0_market1501.pt" verwendet, welches automatisch heruntergeladen werden kann.
            device (str): Gerät ("cuda" oder "cpu"). Wird automatisch ermittelt, falls None.
            half (bool): Ob halbe Präzision genutzt werden soll.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Falls reid_weights als String vorliegt, in ein Path-Objekt umwandeln
        if isinstance(reid_weights, str):
            reid_weights = Path(reid_weights)
        self.tracker = BotSort(reid_weights, device, half)

    def update(self, detections, frame):
        """
        Aktualisiert den Tracker mit den aktuellen Detektionen und verarbeitet
        nur die YOLO-Klassen 39, 40 und 41.

        Args:
            detections (List[Dict]): Liste der Detektionen, z. B.
                [{"class": 39, "confidence": 0.85, "bbox": (x1, y1, x2, y2)}, ...].
            frame (np.ndarray): Der aktuelle Frame (wird intern vom Tracker genutzt).

        Returns:
            List[Dict]: Liste der getrackten Objekte, z. B.
                [{"track_id": 1, "bbox": (x1, y1, x2, y2)}, ...].
        """
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls_id = det["class"]
            det_list.append([x1, y1, x2, y2, conf, cls_id])
        
        if len(det_list) == 0:
            detection_array = np.empty((0, 6))
        else:
            detection_array = np.array(det_list)

        tracked_objects = self.tracker.update(detection_array, frame)

        results = []
        for obj in tracked_objects:
            # Check if obj is a numpy array and extract values accordingly
            if isinstance(obj, np.ndarray):
                # Assuming array format is [x1, y1, x2, y2, track_id, class_id, ...]
                x1, y1, x2, y2 = obj[:4]
                track_id = int(obj[4])
                results.append({
                    "track_id": track_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
            else:
                # Original code for object with attributes
                try:
                    x, y, w, h = obj.tlwh
                    track_id = obj.track_id
                    results.append({
                        "track_id": track_id,
                        "bbox": (int(x), int(y), int(x + w), int(y + h))
                    })
                except AttributeError:
                    print(f"Warning: Unexpected object format: {type(obj)}, {obj}")
        
        return results
