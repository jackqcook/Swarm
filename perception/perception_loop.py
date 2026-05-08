"""
perception/perception_loop.py
──────────────────────────────
Tier 2 perception: runs in a background thread at 10–30 Hz.

Pipeline per frame:
  1. Capture frame from camera (RealSense D435i or USB webcam)
  2. YOLOv8n obstacle detection  (~15–23 ms on Orin NX/Nano INT8)
  3. Monocular depth estimation   (MiDaS-Small or RT-MonoDepth)
  4. Fuse depth + bounding boxes → ObstacleMap
  5. Publish ObstacleMap to the planner via asyncio.Queue

The planner consumes the ObstacleMap and generates avoidance velocity setpoints.

Camera backends (set in config):
  - "realsense"  : Intel RealSense D435i (stereo + RGB, best depth quality)
  - "webcam"     : OpenCV VideoCapture (development only)
  - "mock"       : Synthetic frames for testing without hardware
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object / obstacle."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple          # (x1, y1, x2, y2) in pixels
    distance_m: Optional[float] = None   # From depth fusion; None if unavailable
    bearing_deg: Optional[float] = None  # Azimuth from drone heading


@dataclass
class ObstacleMap:
    """Published each perception cycle."""
    timestamp: float
    frame: Optional[np.ndarray]           # Latest RGB frame (H x W x 3)
    depth_map: Optional[np.ndarray]       # Metric depth (H x W), metres
    detections: List[Detection] = field(default_factory=list)
    nearest_obstacle_m: float = float("inf")
    nearest_obstacle_bearing_deg: float = 0.0


# ── Camera backends ────────────────────────────────────────────────────────────

class MockCamera:
    """Returns synthetic frames. No hardware required."""

    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height, self.fps = width, height, fps

    def read(self):
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        depth = np.random.uniform(1.0, 15.0, (self.height, self.width)).astype(np.float32)
        return True, frame, depth

    def release(self):
        pass


class WebcamCamera:
    """OpenCV VideoCapture — development on desktop."""

    def __init__(self, device_id: int = 0, width=640, height=480):
        import cv2
        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        import cv2
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = None  # No depth from webcam
        return ret, frame, depth

    def release(self):
        self._cap.release()


class RealSenseCamera:
    """Intel RealSense D435i — production hardware."""

    def __init__(self, width=640, height=480, fps=30):
        try:
            import pyrealsense2 as rs
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self._align = rs.align(rs.stream.color)
            self._pipeline.start(config)
            self._available = True
            logger.info("RealSense D435i initialized.")
        except ImportError:
            logger.warning("pyrealsense2 not installed. Falling back to mock camera.")
            self._available = False
            self._mock = MockCamera(width, height, fps)

    def read(self):
        if not self._available:
            return self._mock.read()

        import pyrealsense2 as rs
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth_frame = aligned.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_frame.get_units()
        return True, color, depth

    def release(self):
        if self._available:
            self._pipeline.stop()


# ── Depth estimator (monocular fallback) ──────────────────────────────────────

class MonocularDepthEstimator:
    """
    MiDaS-Small depth estimation for when stereo depth is unavailable
    (webcam mode or RealSense depth stream failure).

    On Jetson Orin NX: RT-MonoDepth-S runs at ~364 FPS.
    Here we use the HuggingFace MiDaS-Small for cross-platform compatibility.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._transform = transforms.small_transform
            self._model.to(self.device).eval()
            logger.info("MiDaS-Small depth estimator loaded.")
        except Exception as e:
            logger.warning("Could not load MiDaS: %s — depth will be None", e)

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns a metric-normalized depth map (H x W), or None on failure."""
        self._load()
        if self._model is None:
            return None
        try:
            import torch
            import cv2
            inp = self._transform(frame).to(self.device)
            with torch.no_grad():
                pred = self._model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = pred.cpu().numpy()
            # Invert + normalize to approximate metric scale (rough, not calibrated)
            depth = 1.0 / (depth + 1e-6)
            depth = (depth / depth.max()) * 20.0  # Scale to [0, 20m] range
            return depth.astype(np.float32)
        except Exception as e:
            logger.warning("Depth estimation error: %s", e)
            return None


# ── YOLO detector ─────────────────────────────────────────────────────────────

class ObstacleDetector:
    """
    YOLOv8n obstacle detector.
    On Jetson Orin NX INT8: ~15 ms/frame.
    On desktop CPU: ~100–200 ms/frame (use 'mock' backend for fast dev).
    """

    # Classes we care about as obstacles
    OBSTACLE_CLASSES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "bird", "cat", "dog", "horse", "cow", "elephant",
        "tree", "potted plant", "building",
    }

    def __init__(self, model_size: str = "n", device: str = "cpu", mock: bool = False):
        self.device = device
        self.mock = mock
        self._model = None
        self._model_size = model_size  # n=nano, s=small, m=medium

    def _load(self):
        if self._model is not None or self.mock:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"yolov8{self._model_size}.pt")
            self._model.to(self.device)
            logger.info("YOLOv8%s loaded.", self._model_size)
        except Exception as e:
            logger.warning("Could not load YOLOv8: %s — running mock detections", e)
            self.mock = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._load()

        if self.mock:
            # Return a synthetic far-away obstacle occasionally
            if np.random.random() < 0.05:
                return [Detection(0, "person", 0.85, (300, 200, 380, 400), distance_m=8.0, bearing_deg=5.0)]
            return []

        try:
            results = self._model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                ))
            return detections
        except Exception as e:
            logger.warning("YOLO detection error: %s", e)
            return []


def fuse_depth_with_detections(
    detections: List[Detection],
    depth_map: Optional[np.ndarray],
    frame_width: int,
    hfov_deg: float = 87.0,  # RealSense D435i horizontal FOV
) -> List[Detection]:
    """
    For each detection, sample the depth map within the bounding box
    to estimate distance and bearing.
    """
    if depth_map is None:
        return detections

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        roi = depth_map[y1:y2, x1:x2]
        if roi.size > 0:
            # Use 10th percentile (nearest point in box, ignoring noise)
            det.distance_m = float(np.percentile(roi[roi > 0.1], 10)) if (roi > 0.1).any() else None

        # Bearing from center of bbox relative to frame center
        cx = (x1 + x2) / 2.0
        bearing = (cx / frame_width - 0.5) * hfov_deg
        det.bearing_deg = bearing

    return detections


# ── Main perception loop ───────────────────────────────────────────────────────

class PerceptionLoop:
    """
    Runs camera capture + detection + depth in a background thread.
    Publishes ObstacleMap via an asyncio.Queue consumed by the planner.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        backend = cfg.get("camera_backend", "mock")
        device = cfg.get("device", "cpu")

        # Camera
        if backend == "realsense":
            self._camera = RealSenseCamera()
        elif backend == "webcam":
            self._camera = WebcamCamera(cfg.get("webcam_id", 0))
        else:
            self._camera = MockCamera()
            logger.info("Using mock camera.")

        # Models
        use_mock_detector = (backend == "mock") or cfg.get("mock_detections", False)
        self._detector = ObstacleDetector(
            model_size=cfg.get("yolo_size", "n"),
            device=device,
            mock=use_mock_detector,
        )
        self._depth_estimator = MonocularDepthEstimator(device=device)

        # State
        self._latest: Optional[ObstacleMap] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._target_fps = cfg.get("fps", 15)

    async def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("PerceptionLoop started.")

    async def stop(self):
        self._stop_event.set()
        self._camera.release()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("PerceptionLoop stopped.")

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        return self._latest.frame if self._latest else None

    @property
    def latest_obstacle_map(self) -> Optional[ObstacleMap]:
        return self._latest

    async def get_obstacle_map(self, timeout: float = 0.1) -> Optional[ObstacleMap]:
        """Non-blocking pull from the perception queue."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return self._latest

    def _run(self):
        """Background thread: capture → detect → publish."""
        loop = asyncio.new_event_loop()
        interval = 1.0 / self._target_fps

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            ret, frame, depth = self._camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Depth: prefer stereo; fall back to monocular
            if depth is None:
                depth = self._depth_estimator.estimate(frame)

            # Detect obstacles
            detections = self._detector.detect(frame)

            # Fuse depth
            if frame is not None:
                detections = fuse_depth_with_detections(
                    detections, depth, frame.shape[1]
                )

            # Find nearest obstacle
            distances = [d.distance_m for d in detections if d.distance_m is not None]
            nearest_dist = min(distances) if distances else float("inf")
            nearest_bearing = 0.0
            if distances:
                nearest_idx = distances.index(nearest_dist)
                nearest_bearing = detections[nearest_idx].bearing_deg or 0.0

            obstacle_map = ObstacleMap(
                timestamp=time.time(),
                frame=frame,
                depth_map=depth,
                detections=detections,
                nearest_obstacle_m=nearest_dist,
                nearest_obstacle_bearing_deg=nearest_bearing,
            )
            self._latest = obstacle_map

            # Non-blocking publish
            try:
                loop.run_until_complete(
                    asyncio.wait_for(self._queue.put(obstacle_map), timeout=0.05)
                )
            except Exception:
                pass  # Queue full; planner will use latest anyway

            # Maintain target FPS
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

        loop.close()
