from ultralytics import YOLO
import threading

# Create a global lock for CUDA operations
cuda_lock = threading.Lock()

class YoloDetector:
    def __init__(self, model_path="models/yolo11x.pt", allowed_classes=None):
        """
        YOLOv8 Detector
        Args:
            model_path (str): Pfad zum YOLOv8 Modell.
            allowed_classes (List[int]): Optional, nur diese Klassen werden zurückgegeben.
        """
        self.model = YOLO(model_path)
        self.allowed_classes = allowed_classes if allowed_classes else []

    # In your YoloDetector class
    def detect(self, frame):
        """Run detection on a frame with CUDA lock"""
        if frame is None:
            return []
        
        # Use the global CUDA lock when performing GPU operations
        with cuda_lock:
            results = self.model.predict(frame, conf=0.25)
            
            # Convert results to our standard format
            detections = []
            
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    confidence = boxes.conf[i].item()
                    class_id = int(boxes.cls[i].item())
                    
                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class": class_id
                    }
                    detections.append(detection)
            
            return detections
