from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect(self, image_path, top_n=5):
        results = self.model(image_path)[0]
        detections = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls_id = box.tolist()
            label = results.names[int(cls_id)]
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": conf
            })

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections[:top_n]
