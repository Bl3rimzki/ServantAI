import cv2
import time
from cameramanager.camera_manager import CameraManager
from detectors.yolo_detector import YoloDetector
from trackers.boxmot_tracker import BoxMotTracker
from state.object_state_manager import ObjectStateManager
from segmenters.fastsam_segmenter import FastSamSegmenter

def main():
    # Kamera initialisieren
    cam_manager = CameraManager()
    cam_manager.add_camera("cam1", 0)  # Standard-USB-Kamera
    cam_manager.start_camera_stream("cam1")

    # YOLOv8-Detektor initialisieren (nur Klassen 39, 40, 41 = bottle, wine glass, cup)
    detector = YoloDetector(model_path="yolov8n.pt", allowed_classes=[39, 40, 41])

    # BoxMot Tracker initialisieren
    tracker = BoxMotTracker()

    # State Manager für Objekte (z. B. zur Verwaltung von Events wie "sip", "refilled" etc.)
    state_manager = ObjectStateManager()

    # Segmenter initialisieren zum Schätzen der Füllstände
    segmenter = FastSamSegmenter()

    try:
        while True:
            frame = cam_manager.get_frame("cam1")
            if frame is None:
                continue

            # Detektionen mit YOLO
            detections = detector.detect(frame)

            # Tracker aktualisieren (nur erlaubte Klassen 39, 40, 41)
            tracked_objects = tracker.update(detections, frame)
            now = time.time()

            for tracked in tracked_objects:
                track_id = tracked["track_id"]
                x1, y1, x2, y2 = tracked["bbox"]

                # ROI ausschneiden für Segmentierung
                object_roi = frame[y1:y2, x1:x2]

                # Füllstand berechnen
                fill_ratio = segmenter.compute_fill_ratio(object_roi)

                # State Manager aktualisieren
                detection_info = {"fill_ratio": fill_ratio}
                obj_state = state_manager.update(track_id, detection_info)

                # Bounding Box & Infos ins Bild schreiben
                label = f"ID {track_id} - {obj_state['status']} - Fill: {fill_ratio:.2%}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("BoxMot Tracker mit Füllstand", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.02)

    finally:
        cam_manager.stop_camera_stream("cam1")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
