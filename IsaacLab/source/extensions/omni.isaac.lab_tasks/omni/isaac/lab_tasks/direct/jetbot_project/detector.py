# detector.py
import numpy as np
import cv2
from PIL import Image

COLOR_CENTRES = {
    'blue':   np.array([5,   5, 192]),
    'yellow': np.array([212, 212, 22]),
    'green':  np.array([22, 214, 21]),
    'red':    np.array([196, 15, 13]),
}

def grid_cell(cx, cy, W, H):
    col = "left"   if cx <  W/3 else "centre" if cx < 2*W/3 else "right"
    row = "top"    if cy <  H/3 else "middle" if cy < 2*H/3 else "bottom"
    return (f"{row}-{col}"
            .replace("middle-centre", "centre")
            .replace("middle-", "")
            .replace("-centre", ""))

class ObjectDetector:
    def __init__(self, threshold=30):
        self.threshold = threshold

    def detect(self, img, top_n=None):
        H, W = img.shape[:2]
        detections = []

        for color_name, center in COLOR_CENTRES.items():
            lo = np.clip(center - self.threshold, 0, 255)
            hi = np.clip(center + self.threshold, 0, 255)
            mask = cv2.inRange(img, lo, hi)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < 10:
                    continue
                cx, cy = x + w//2, y + h//2
                location = grid_cell(cx, cy, W, H)
                detections.append({
                    "bbox": (x, y, x+w, y+h),
                    "label": f"{location} {color_name} ball",
                    "confidence": 1.0  # Dummy confidence
                })

        if top_n is not None:
            return detections[:top_n]
        return detections