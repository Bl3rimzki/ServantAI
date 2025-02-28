import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from cameramanager.camera_manager import CameraManager
from ultralytics import YOLO
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class DrinkManager:
    def __init__(self):
        self.camera = CameraManager()
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        self.liquidmodel = YOLO('glass-liquid-level-detection/weights/best.pt') # Using custom liquid level detection model
        self.glasses = {}
        self.fullness_threshold = 0.5  # Fallback threshold

        # Updated segmentation model instantiation with new weights parameter.
        self.water_segmentation_model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        ).eval()

        self.segmentation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect_glasses(self, frame):
        results = self.model(frame)[0]
        current_glasses = {}
        
        print(f"Number of detections: {len(results.boxes)}")  # Debug print
        
        for box in results.boxes:
            # Access box attributes properly
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Changed from .data
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            
            print(f"Detected class {cls} with confidence {conf}")  # Debug print
            
            if int(cls) in [39, 40, 41] and conf > 0.25:  # Cup class
                bbox = (int(x1), int(y1), int(x2), int(y2))
                try:
                    glass_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    is_full = self._check_glass_fullness(glass_region)
                    current_glasses[bbox] = is_full
                except Exception as e:
                    print(f"Error processing glass region: {e}")
                    current_glasses[bbox] = None  # Mark as unknown
        
        self.glasses = current_glasses

    def _check_glass_fullness(self, glass_region):
        if glass_region.size == 0:
            return False

        # YOLO-Modell auf das Glas-ROI anwenden (direkt auf das Bild)
        results = self.liquidmodel.predict(glass_region)

        # Alle erkannten Objekte durchgehen
        glass_box = None
        liquid_box = None

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Bounding Box-Koordinaten
                class_id = int(cls)

                if class_id == 0:  # Falls Klasse 0 = Glas
                    glass_box = (x1, y1, x2, y2)
                elif class_id == 1:  # Falls Klasse 1 = Flüssigkeit
                    liquid_box = (x1, y1, x2, y2)

        # Prüfen, ob beides erkannt wurde
        if glass_box is None or liquid_box is None:
            return False  # Kann nicht bestimmen, ob das Glas voll ist

        # Höhe von Glas & Flüssigkeit berechnen
        _, glass_y1, _, glass_y2 = glass_box
        _, liquid_y1, _, liquid_y2 = liquid_box

        glass_height = glass_y2 - glass_y1
        liquid_height = liquid_y2 - liquid_y1

        # Verhältnis berechnen (Flüssigkeit im Verhältnis zur Glasgröße)
        fill_ratio = liquid_height / glass_height

        # Definiere Schwellenwerte (z. B. mehr als 80% = voll, unter 20% = leer)
        if fill_ratio > 0.8:
            return "full"
        elif fill_ratio < 0.2:
            return "empty"
        else:
            return "half-full"


    def display_glasses(self, frame):
        print(f"Display called with {len(self.glasses)} glasses")
        if not isinstance(frame, np.ndarray):
            print("Error: frame is not a valid image")
            return frame

        for bbox, state in self.glasses.items():
            x1, y1, x2, y2 = bbox
            print(f"Drawing box at {bbox} with state {state}")

            if not all(isinstance(coord, (int, np.integer)) for coord in [x1, y1, x2, y2]):
                print(f"Error: Invalid bbox coordinates: {x1}, {y1}, {x2}, {y2}")
            continue

        # Map states to colors
        color_map = {
        "full": (0, 255, 0),     # Green
        "empty": (0, 0, 255),     # Red
        "half-full": (0, 255, 255),  # Yellow
        None: (0, 165, 255)       # Orange
        }
        
        color = color_map.get(state, (0, 165, 255))  # Default to orange if unknown state
        label = state.capitalize() if state else "Unknown"

        try:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"Error drawing rectangle: {e}")

        return frame

def main():
    drink_manager = DrinkManager()
    
    # Initialize camera with a webcam (device index 0)
    drink_manager.camera.add_camera("camera_1", 0)
    drink_manager.camera.start_camera_stream("camera_1")
    
    try:
        while True:
            frame = drink_manager.camera.get_frame("camera_1")
            if frame is None:
                continue
            
            drink_manager.detect_glasses(frame)
            display_frame = drink_manager.display_glasses(frame)
            
            cv2.imshow('Drink Management System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        drink_manager.camera.stop_camera_stream("camera_1")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()