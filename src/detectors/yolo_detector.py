import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", allowed_classes: list = None):
        """
        Initialisiert den YOLOv8-Detektor.

        Args:
            model_path (str): Pfad zu den YOLOv8-Gewichten. Standard ist "yolov8n.pt".
            allowed_classes (list, optional): Liste der erlaubten Klassen (als Integer, z.B. [39, 40, 41]).
                Wenn None, werden alle erkannten Klassen zurückgegeben.
        """
        self.model = YOLO(model_path)
        self.allowed_classes = allowed_classes

    def detect(self, frame):
        """
        Führt die YOLOv8-Erkennung auf dem gegebenen Frame aus.

        Args:
            frame (np.ndarray): Eingabebild im BGR-Format (wie von cv2.read).

        Returns:
            List[Dict]: Liste der Detektionen, z. B. [{"class": 39, "confidence": 0.85, "bbox": (x1, y1, x2, y2)}].
        """
        results = self.model(frame)
        detections = []
        # Iteriere über alle erkannten Boxen im ersten Ergebnis
        for box in results[0].boxes:
            # Extrahiere Bounding Box-Koordinaten, Confidence und Class-ID
            coords = box.xyxy.cpu().numpy()[0].astype(int)
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            # Filtere, falls allowed_classes gesetzt ist
            if self.allowed_classes is not None and cls not in self.allowed_classes:
                continue
            detections.append({
                "class": cls,
                "confidence": conf,
                "bbox": tuple(coords.tolist())
            })
        return detections

# Beispiel zur Nutzung:
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Passe den Pfad zu deinem Testbild an
    frame = cv2.imread(image_path)
    
    # Beispiel: Nur Bottles (39), Wine Glasses (40) und Cups (41) sollen berücksichtigt werden
    allowed = [39, 40, 41]
    detector = YoloDetector(model_path="yolov8n.pt", allowed_classes=allowed)
    detections = detector.detect(frame)
    
    # Zeichne die erkannten Objekte ins Bild
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label_text = f"{det['class']}: {det['confidence']:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("YOLOv8 Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
