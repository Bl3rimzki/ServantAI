import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch

class ZeroShotDetector:
    def __init__(self, prompt: str = "a glass", model_path: str = None):
        """
        Initialize the Zero-Shot Detector.
        
        Args:
            prompt (str): Textprompt für die Objekterkennung.
            model_path (str): Pfad zum Modell (optional). Wird hier nicht genutzt, 
                              stattdessen wird das vortrainierte OWL-ViT Modell von Hugging Face geladen.
        """
        self.prompt = prompt
        self.processor, self.model = self.load_model(model_path)

    def load_model(self, model_path: str = None):
        """
        Lädt den OWL-ViT Processor und das zugehörige Modell.
        Falls model_path nicht angegeben wird, wird das Standardmodell "google/owlvit-base-patch32" verwendet.
        
        Returns:
            Tuple[OwlViTProcessor, OwlViTForObjectDetection]: Den geladenen Processor und das Modell.
        """
        # Standardmäßig das vortrainierte Modell von Hugging Face verwenden
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        print("ZeroShotDetector: OWL-ViT Modell erfolgreich geladen.")
        return processor, model

    def detect(self, frame):
        """
        Führt die Zero-Shot-Erkennung auf dem gegebenen Frame aus.
        
        Args:
            frame (np.ndarray): Eingabebild im BGR-Format (cv2).
            
        Returns:
            List[Dict]: Liste der Detektionen, z. B. [{"class": "a glass", "confidence": 0.85, "bbox": (x1, y1, x2, y2)}].
        """
        # cv2-Frame von BGR in RGB konvertieren und in ein PIL-Image umwandeln
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        
        # Eingaben vorbereiten: Bild und Textprompt
        inputs = self.processor(text=[self.prompt], images=image, return_tensors="pt")
        
        # Inferenz durchführen (ohne Gradientenberechnung)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Postprocess: Bounding Boxes und Scores in Pixelkoordinaten umrechnen.
        target_sizes = [image.size]  # (width, height)
        results = self.processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # box: [x_min, y_min, x_max, y_max]
            bbox = tuple(int(coord) for coord in box.tolist())
            detections.append({
                "class": self.prompt,
                "confidence": score.item(),
                "bbox": bbox
            })
        return detections
