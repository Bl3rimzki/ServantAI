from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", allowed_classes=None):
        """
        YOLOv8 Detector
        Args:
            model_path (str): Pfad zum YOLOv8 Modell.
            allowed_classes (List[int]): Optional, nur diese Klassen werden zurückgegeben.
        """
        self.model = YOLO(model_path)
        self.allowed_classes = allowed_classes if allowed_classes else []

    def detect(self, frame):
        results = self.model(frame)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if self.allowed_classes and cls not in self.allowed_classes:
                continue

            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
