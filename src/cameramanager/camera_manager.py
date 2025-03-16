import cv2
import threading
import time
from typing import Dict, Any, Union


class CameraManager:
    def __init__(self):
        self.cameras = {}

    def add_camera(self, camera_id: str, source: Union[str, int], port: int = None,
                   width: int = None, height: int = None, max_fps: float = 0) -> None:
        """
        Kamera hinzufügen mit optionaler Auflösung und FPS-Limit.
        """
        if isinstance(source, str) and port is not None:
            video_url = f"http://{source}:{port}/video"
        else:
            video_url = source

        stream = cv2.VideoCapture(video_url)

        # Falls gewünscht: Auflösung setzen
        if width and height:
            stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Auflösung zur Sicherheit nach dem Setzen auslesen (nicht jede Kamera unterstützt jede Auflösung)
        actual_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Kamera {camera_id} läuft mit {actual_width}x{actual_height}")

        self.cameras[camera_id] = {
            "stream": stream,
            "running": False,
            "thread": None,
            "frame": None,
            "max_fps": max_fps,
            "last_frame_time": 0
        }

    def start_camera_stream(self, camera_id: str) -> None:
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if not camera["running"]:
                camera["running"] = True
                camera["thread"] = threading.Thread(target=self._capture_frames, args=(camera,))
                camera["thread"].start()

    def _capture_frames(self, camera: Dict[str, Any]) -> None:
        while camera["running"]:
            now = time.time()
            time_since_last_frame = now - camera["last_frame_time"]

            if camera["max_fps"] > 0 and time_since_last_frame < (1.0 / camera["max_fps"]):
                time.sleep((1.0 / camera["max_fps"]) - time_since_last_frame)
                continue

            ret, frame = camera["stream"].read()
            if not ret:
                continue

            camera["frame"] = frame
            camera["last_frame_time"] = time.time()

    def stop_camera_stream(self, camera_id: str) -> None:
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            camera["running"] = False
            if camera["thread"] is not None:
                camera["thread"].join()

    def get_frame(self, camera_id: str):
        if camera_id in self.cameras:
            return self.cameras[camera_id]["frame"]
        return None
