import cv2
from ultralytics import YOLO
import numpy as np
import threading
from typing import Dict, Any, Union

class CameraManager:
    def __init__(self):
        """Initialize the camera manager."""
        self.cameras = {}  # Dictionary to store camera instances by ID
        self.detector = YOLO('yolov8n.pt')  # Initialize YOLO detector once for efficiency
        
    def add_camera(self, camera_id: str, source: Union[str, int], port: int = None) -> None:
        """
        Add a new camera with the given source.
        
        Args:
            camera_id (str): Unique identifier for the camera.
            source (Union[str, int]): IP address for IP camera or device index for USB camera.
            port (int, optional): Port number for the IP camera stream. Not used for USB cameras.
        """
        if isinstance(source, str) and port is not None:
            # Construct the video URL for IP camera
            video_url = f"http://{source}:{port}/video"
        else:
            # Use the source directly for USB camera (device index)
            video_url = source
        
        self.cameras[camera_id] = {
            "stream": cv2.VideoCapture(video_url),
            "running": False,
            "thread": None,
            "frame": None
        }
        
    def start_camera_stream(self, camera_id: str) -> None:
        """
        Start the video stream for the specified camera.
        
        Args:
            camera_id (str): Unique identifier for the camera.
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if not camera["running"]:
                camera["running"] = True
                # Start a thread to capture frames continuously
                camera["thread"] = threading.Thread(target=self._capture_frames, args=(camera,))
                camera["thread"].start()
        
    def _capture_frames(self, camera: Dict[str, Any]) -> None:
        """
        Continuously capture frames from the camera stream.
        
        Args:
            camera (Dict): Dictionary containing camera state and stream information.
        """
        while camera["running"]:
            ret, frame = camera["stream"].read()
            if not ret:
                continue
            # Store the frame for processing
            camera["frame"] = frame
        
    def stop_camera_stream(self, camera_id: str) -> None:
        """
        Stop the video stream for the specified camera.
        
        Args:
            camera_id (str): Unique identifier for the camera.
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if camera["running"]:
                camera["running"] = False
                if camera["thread"] is not None:
                    camera["thread"].join()
        
    def process_camera_feed(self, camera_id: str):
            if camera_id not in self.cameras or self.cameras[camera_id]["frame"] is None:
                return {}

            frame = self.cameras[camera_id]["frame"]
            # Perform object detection
            results = self.detector(frame)

            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                confidence = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                detections.append({
                    "class": class_id,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

            return {"frame": frame, "detections": detections}
    
    def get_frame(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id]["frame"]
        return None
    
    
    def merge_camera_feeds(self) -> np.ndarray:
        """
        Merge video feeds from all cameras into a single display window.
        
        Returns:
            np.ndarray: Merged video feed as a numpy array.
        """
        merged_frames = []
        
        # Get the current frames from all cameras
        for camera_id, camera in self.cameras.items():
            if camera["frame"] is not None:
                frame = camera["frame"].copy()
                
                # Draw detections on the frame
                processed_frame = self._draw_detections(frame, camera_id)
                
                # Add a label showing the camera ID
                cv2.putText(processed_frame, f"Camera {camera_id}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                merged_frames.append(processed_frame)
        
        # If no cameras are active, return an empty frame
        if not merged_frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Arrange the frames in a grid (e.g., side by side horizontally)
        merged_feed = cv2.hconcat(merged_frames)
        
        return merged_feed
    
    def _draw_detections(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """
        Draw detection results on the specified frame.
        
        Args:
            frame (np.ndarray): Input frame to draw detections on.
            camera_id (str): Unique identifier for the camera.
            
        Returns:
            np.ndarray: Frame with detections drawn.
        """
        if camera_id not in self.cameras or self.cameras[camera_id]["frame"] is None:
            return frame
        
        processed_frame = self.process_camera_feed(camera_id)
        
        # Draw bounding boxes and labels on the frame
        for detection in processed_frame["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {detection['class']}: {confidence:.2f}",
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
        
def main():
    # Initialize the camera manager
    camera_manager = CameraManager()
    
    # Add multiple cameras
    camera_manager.add_camera("camera_1", "192.168.1.12", 4747)
    
    # Start all camera streams
    for camera_id in camera_manager.cameras:
        camera_manager.start_camera_stream(camera_id)
    
    try:
        while True:
            # Merge all camera feeds and display them
            merged_feed = camera_manager.merge_camera_feeds()
            
            # Display the merged feed
            cv2.imshow("Multi-Camera Feed", merged_feed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up resources
        for camera_id in camera_manager.cameras:
            camera_manager.stop_camera_stream(camera_id)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
