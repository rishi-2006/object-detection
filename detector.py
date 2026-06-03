import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 detector.
        Downloads the model weight file automatically if not present.
        """
        self.model_name = model_name
        self.model = YOLO(model_name)
        # Expose model names mapping (ID -> name)
        self.names = self.model.names

    def detect(self, frame, conf_threshold=0.25, iou_threshold=0.45, classes=None):
        """
        Runs YOLOv8 object detection on a single frame.
        
        Args:
            frame: OpenCV image (BGR numpy array).
            conf_threshold: Minimum confidence for detections.
            iou_threshold: IOU threshold for NMS.
            classes: List of class IDs to filter detections (None for all classes).
            
        Returns:
            detections: List of tuples in format ([x_min, y_min, w, h], confidence, class_name)
                        for DeepSORT ingestion.
            raw_boxes: Raw boxes, scores, and class IDs for drawing or reference.
        """
        # Run inference
        results = self.model(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            classes=classes, 
            verbose=False
        )
        
        deepsort_detections = []
        raw_boxes = []
        
        if not results:
            return deepsort_detections, raw_boxes
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Extract coordinates
            # xyxy returns [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Confidence score
            conf = float(box.conf[0].cpu().numpy())
            
            # Class ID and Name
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.names[class_id]
            
            # Convert xyxy [x1, y1, x2, y2] to xywh [x_min, y_min, w, h] as expected by DeepSORT
            w = x2 - x1
            h = y2 - y1
            
            # Filter out invalid or zero-area boxes
            if w <= 0 or h <= 0:
                continue
                
            deepsort_detections.append(([x1, y1, w, h], conf, class_name))
            raw_boxes.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": class_id,
                "class_name": class_name
            })
            
        return deepsort_detections, raw_boxes

    def get_class_id_from_name(self, name):
        """
        Helper to get class ID from class name.
        """
        for class_id, class_name in self.names.items():
            if class_name.lower() == name.lower():
                return class_id
        return None
