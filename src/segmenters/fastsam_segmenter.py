# segmenters/fastsam_segmenter.py
import cv2
import numpy as np
# Importiere FastSAM und den Predictor – passe ggf. den Import an deine Installation an
from FastSAM.fastsam import FastSAM, FastSAMPredictor

class FastSamSegmenter:
    def __init__(self, model_path: str = "path/to/fastsam_weights.pth"):
        """
        Initialisiert den FastSAM-Segmentierer.
        
        Args:
            model_path (str): Pfad zu den FastSAM-Gewichten.
        """
        # Lade das FastSAM-Modell und initialisiere den Predictor
        self.model = FastSAM()
        self.predictor = FastSAMPredictor(self.model)
        print("FastSamSegmenter: Modell erfolgreich geladen.")

    def segment_roi(self, image, roi):
        """
        Führt die Segmentierung auf dem gegebenen Region of Interest (ROI) aus.
        
        Args:
            image (np.ndarray): Das Originalbild (BGR).
            roi (tuple): Das ROI als (x1, y1, x2, y2).
        
        Returns:
            masks, scores, boxes: Die von FastSAM erzeugten Masken, Konfidenzen und Bounding Boxes.
        """
        x1, y1, x2, y2 = roi
        # Schneide das ROI aus dem Bild aus
        cropped = image[y1:y2, x1:x2]
        # Führe die Segmentierung durch
        masks, scores, boxes = self.predictor.predict(cropped)
        return masks, scores, boxes
    
    def compute_fill_ratio(masks, roi):
        """
        Berechnet den Füllstand als Verhältnis der Maskenfläche zur Fläche des ROI.
        
        Args:
            masks (List[np.ndarray]): Liste der binären Masken (dtype=bool oder 0/1).
            roi (tuple): (x1, y1, x2, y2) des ROI.
        
        Returns:
            float: fill_ratio im Bereich [0,1]
        """
        x1, y1, x2, y2 = roi
        roi_area = (x2 - x1) * (y2 - y1)
        if roi_area == 0:
            return 0.0
        best_mask_area = 0
        for mask in masks:
            # mask.sum() gibt die Anzahl der Pixel, die Teil der Maske sind
            area = mask.sum()
            if area > best_mask_area:
                best_mask_area = area
        fill_ratio = best_mask_area / roi_area
        return fill_ratio
