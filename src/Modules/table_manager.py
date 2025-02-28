from cameramanager.camera_manager import CameraManager
import cv2
import numpy as np
from ultralytics import YOLO

class TableManager:
    def __init__(self):
        self.camera = CameraManager()
        self.model = YOLO('yolov8n.pt')
        self.tables = {}  # {table_id: {'bbox': (x1,y1,x2,y2), 'state': str}}
        self.next_id = 1

    def identify_tables(self, frame):
        results = self.model(frame)[0]
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # assuming 0 is table class
                table_id = self._assign_table_id((x1, y1, x2, y2))
                self.tables[table_id] = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'state': 'Empty'
                }

    def update_table_states(self, frame):
        results = self.model(frame)[0]
        for table_id, table_info in self.tables.items():
            bbox = table_info['bbox']
            table_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            empty_items = self._detect_empty_items(table_region)
            self.tables[table_id]['state'] = 'Needs Attention' if empty_items else 'Served'

    def display_tables(self, frame):
        for table_id, table_info in self.tables.items():
            x1, y1, x2, y2 = table_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Table {table_id}: {table_info['state']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def _assign_table_id(self, new_bbox):
        for table_id, table_info in self.tables.items():
            if self._is_same_table(new_bbox, table_info['bbox']):
                return table_id
        table_id = self.next_id
        self.next_id += 1
        return table_id

    def _is_same_table(self, bbox1, bbox2, iou_threshold=0.5):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2-x1) * max(0, y2-y1)
        area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        return intersection / (area1 + area2 - intersection) > iou_threshold

    def _detect_empty_items(self, table_region):
        # Implement detection logic for empty glasses/plates
        # Return True if empty items detected
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > 200  # Simple threshold-based detection

if __name__ == "__main__":
    table_manager = TableManager()
    
    # Initialize camera once
    table_manager.camera.add_camera("camera_1", "192.168.1.12", 4747)
    table_manager.camera.start_camera_stream("camera_1")
    
    try:
        while True:
            frame = table_manager.camera.get_frame("camera_1")
            if frame is None:
                continue  # Changed from break to continue to keep trying

            table_manager.identify_tables(frame)
            table_manager.update_table_states(frame)
            display_frame = table_manager.display_tables(frame)
            
            cv2.imshow('Table Management System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        table_manager.camera.stop_camera_stream("camera_1")
        cv2.destroyAllWindows()