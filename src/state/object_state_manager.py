# state/object_state_manager.py
import time

class ObjectStateManager:
    def __init__(self):
        """
        Initialisiert den State Manager.
        Speichert für jedes Objekt (Track-ID) den aktuellen Zustand und Ereignisse.
        """
        self.objects = {}  # key: track_id, value: Status-Dict

    def update(self, track_id, detection_info):
        """
        Aktualisiert den Zustand für ein bestimmtes Objekt.

        Args:
            track_id (int): Eindeutige ID des getrackten Objekts.
            detection_info (Dict): Informationen, z. B. {"fill_ratio": float}.

        Returns:
            Dict: Das aktuelle Status-Dict für dieses Objekt.
        """
        now = time.time()
        if track_id not in self.objects:
            # Initialisierung eines neuen Objekts
            self.objects[track_id] = {
                "created_at": now,
                "last_update": now,
                "events": [{"event": "served", "timestamp": now}],
                "fill_ratio": detection_info.get("fill_ratio", None),
                "sip_count": 0,
                "status": "unknown"
            }
        else:
            obj = self.objects[track_id]
            obj["last_update"] = now
            new_fill_ratio = detection_info.get("fill_ratio", None)
            if new_fill_ratio is not None:
                old_fill_ratio = obj.get("fill_ratio")
                if old_fill_ratio is not None:
                    delta = old_fill_ratio - new_fill_ratio
                    # Beispiel: Wenn der Füllstand um mehr als 0.1 sinkt, zähle einen "Schluck"
                    if delta > 0.1:
                        obj["sip_count"] += 1
                        obj["events"].append({"event": "sip", "timestamp": now, "delta": delta})
                        # Wenn z. B. 10 Schlucke gezählt wurden, wird der Status als leer gesetzt
                        if obj["sip_count"] >= 10:
                            obj["status"] = "empty"
                obj["fill_ratio"] = new_fill_ratio
        return self.objects[track_id]

    def remove(self, track_id):
        """
        Markiert ein Objekt als entfernt.
        """
        if track_id in self.objects:
            self.objects[track_id]["events"].append({"event": "removed", "timestamp": time.time()})
            # Optional: Entferne das Objekt aus der Datenstruktur
            # del self.objects[track_id]
