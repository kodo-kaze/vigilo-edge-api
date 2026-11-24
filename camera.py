import json
import os
from pathlib import Path
import re
import threading
import time
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

try:
    from config import (
        DEFAULT_CONF,
        DEFAULT_IOU,
        DEFAULT_STRIDE,
        DEFAULT_SMOOTH,
        DEFAULT_ZONE,
        DEFAULT_HEATMAP,
        DEFAULT_CLUSTERS,
        REGISTRY_FILE,
        STATION_NAME,
        CANDIDATE_SAMPLE_VIDEOS,
    )
    from detector import CrowdDetector, draw_annotations, detect_spatial_clusters
    from stream import RTSPCameraStream, parse_camera_source
except ImportError:
    from backend.config import (
        DEFAULT_CONF,
        DEFAULT_IOU,
        DEFAULT_STRIDE,
        DEFAULT_SMOOTH,
        DEFAULT_ZONE,
        DEFAULT_HEATMAP,
        DEFAULT_CLUSTERS,
        REGISTRY_FILE,
        STATION_NAME,
        CANDIDATE_SAMPLE_VIDEOS,
    )
    from backend.detector import CrowdDetector, draw_annotations, detect_spatial_clusters
    from backend.stream import RTSPCameraStream, parse_camera_source


def generate_placeholder_frame(
    title: str = "CAMERA STANDBY",
    subtitle: str = "Power-saving standby mode active",
    status: str = "STANDBY",
    width: int = 640,
    height: int = 360,
    badge_color: Tuple[int, int, int] = (100, 100, 100),
) -> bytes:
    """
    Renders a futuristic dark CCTV placeholder frame for suspended or offline cameras.
    Prevents broken MJPEG client streams and informs operators of camera status.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (14, 18, 24)  # Deep Slate / Dark Glass fill

    # Subtle grid lines
    grid_color = (24, 30, 40)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)

    # Corner brackets (CCTV viewfinder styling)
    bracket_color = (60, 80, 100)
    c_len = 24
    pad = 16
    # Top-Left
    cv2.line(frame, (pad, pad), (pad + c_len, pad), bracket_color, 2)
    cv2.line(frame, (pad, pad), (pad, pad + c_len), bracket_color, 2)
    # Top-Right
    cv2.line(frame, (width - pad, pad), (width - pad - c_len, pad), bracket_color, 2)
    cv2.line(frame, (width - pad, pad), (width - pad, pad + c_len), bracket_color, 2)
    # Bottom-Left
    cv2.line(frame, (pad, height - pad), (pad + c_len, height - pad), bracket_color, 2)
    cv2.line(frame, (pad, height - pad), (pad, height - pad - c_len), bracket_color, 2)
    # Bottom-Right
    cv2.line(frame, (width - pad, height - pad), (width - pad - c_len, height - pad), bracket_color, 2)
    cv2.line(frame, (width - pad, height - pad), (width - pad, height - pad - c_len), bracket_color, 2)

    # Status Pill Badge
    (st_w, st_h), _ = cv2.getTextSize(status.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pill_x1 = width // 2 - (st_w + 24) // 2
    pill_y1 = height // 2 - 40
    pill_x2 = pill_x1 + st_w + 24
    pill_y2 = pill_y1 + st_h + 12
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (28, 35, 48), -1)
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), badge_color, 1)
    cv2.putText(
        frame,
        status.upper(),
        (pill_x1 + 12, pill_y2 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        badge_color,
        1,
        cv2.LINE_AA,
    )

    # Title & Subtitle
    (t_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(
        frame,
        title,
        (max(20, width // 2 - t_w // 2), height // 2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )

    (sub_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(
        frame,
        subtitle,
        (max(20, width // 2 - sub_w // 2), height // 2 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (120, 140, 160),
        1,
        cv2.LINE_AA,
    )

    # Station edge branding footer
    footer = "VIGILO // ON-PREM EDGE BRAIN • STATION CONTROL CENTER"
    cv2.putText(
        frame,
        footer,
        (pad + 4, height - pad - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (70, 90, 110),
        1,
        cv2.LINE_AA,
    )

    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes() if success else b""


class CameraNode:
    """
    Manages an individual CCTV camera pipeline within the On-Premise Edge Server:
    - Decoupled background RTSP/hardware capture
    - Real-time YOLO26m inference & 21-feature Conflux Random Forest analysis
    - Dynamic ground heatmap and spatial clustering rendering
    - Control room operations: Power standby, public visibility approval, physical location telemetry
    """

    def __init__(
        self,
        camera_id: str,
        name: str,
        source: Any,
        zone: str = DEFAULT_ZONE,
        coordinates: Optional[Dict[str, Any]] = None,
        detector: Optional[CrowdDetector] = None,
        active: bool = True,
        approved_for_public: bool = False,
        show_heatmap: bool = DEFAULT_HEATMAP,
        show_clusters: bool = DEFAULT_CLUSTERS,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        stride: int = DEFAULT_STRIDE,
        smooth: int = DEFAULT_SMOOTH,
        created_at: Optional[str] = None,
    ):
        self.id = str(camera_id).strip()
        self.name = str(name).strip()
        self.source = int(source) if str(source).isdigit() else source
        self.zone = str(zone).strip()
        self.coordinates = coordinates or {
            "floor": "Level 1 - Concourse",
            "x": 100.0,
            "y": 100.0,
            "z": 0.0,
            "latitude": None,
            "longitude": None,
        }
        self.active = bool(active)
        self.approved_for_public = bool(approved_for_public)
        self.show_heatmap = bool(show_heatmap)
        self.show_clusters = bool(show_clusters)
        self.conf = float(conf)
        self.iou = float(iou)
        self.stride = max(1, int(stride))
        self.smooth = max(1, int(smooth))
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.detector = detector
        self.source_meta = parse_camera_source(self.source)

        # Thread synchronization
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.running = True
        self.start_time = time.time()

        # Initialize decoupled camera stream (suspended if active == False)
        self.camera = RTSPCameraStream(self.source, suspended=not self.active)

        # Initial cached telemetry
        self.stats: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "zone": self.zone,
            "coordinates": self.coordinates,
            "source": self.source_meta.get("masked_source", str(self.source)),
            "raw_source": str(self.source),
            "camera_type": self.source_meta.get("type", "unknown"),
            "ip": self.source_meta.get("ip"),
            "port": self.source_meta.get("port"),
            "active": self.active,
            "is_suspended": not self.active,
            "approved_for_public": self.approved_for_public,
            "status": "INITIALIZING" if self.active else "STANDBY",
            "is_crowd": False,
            "crowd_probability": 0.0,
            "people_count": 0,
            "density_per_megapixel": 0.0,
            "nearest_neighbor_distance": 0.0,
            "dense_clusters_count": 0,
            "gatherings_count": 0,
            "isolated_count": 0,
            "heatmap_enabled": self.show_heatmap,
            "clusters_enabled": self.show_clusters,
            "fps": 0.0,
            "capture_fps": 0.0,
            "camera_connected": False,
            "resolution": "Offline" if self.active else "Standby (Idle)",
            "uptime_seconds": 0,
            "created_at": self.created_at,
        }

        # Standby placeholder cache
        self._cached_standby_frame = generate_placeholder_frame(
            title=f"{self.name}",
            subtitle=f"Zone: {self.zone} • {self.coordinates.get('floor', '')} (X:{self.coordinates.get('x', '-')}, Y:{self.coordinates.get('y', '-')})",
            status="STANDBY",
            badge_color=(120, 140, 160),
        )

        # Start dedicated inference & annotation thread
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

    def _inference_loop(self):
        prob_history = deque(maxlen=self.smooth)
        frame_counter = 0
        last_time = time.time()

        cached_boxes = np.empty((0, 4))
        cached_confs = np.empty(0)
        cached_features = {
            "person_count": 0,
            "people_per_megapixel": 0.0,
            "average_nearest_neighbor_distance": 0.0,
        }
        cached_prob = 0.0
        cached_is_crowd = False
        cached_dense_clusters = []
        cached_gatherings = []
        cached_isolated = []
        cached_dense_cnt = 0
        cached_gather_cnt = 0
        cached_isolated_cnt = 0

        while self.running:
            if not self.active:
                # Idle sleeping when suspended to consume zero CPU
                time.sleep(0.12)
                continue

            frame = self.camera.read()
            cam_info = self.camera.get_info()

            if frame is None:
                with self.lock:
                    self.stats["camera_connected"] = False
                    self.stats["status"] = "OFFLINE" if not cam_info.get("connected") else "CONNECTING..."
                    self.stats["resolution"] = cam_info.get("resolution", "Offline")
                    self.stats["fps"] = 0.0
                    self.stats["capture_fps"] = 0.0
                time.sleep(0.05)
                continue

            frame_counter += 1
            now = time.time()
            fps_instant = 1.0 / max(now - last_time, 1e-4)
            last_time = now

            # Neural network prediction (runs on every Nth frame, stride-controlled)
            if self.detector is not None and frame_counter % self.stride == 0:
                try:
                    boxes, confs, features, raw_prob = self.detector.predict(frame)
                    cached_boxes = boxes
                    cached_confs = confs
                    cached_features = features

                    prob_history.append(raw_prob)
                    smoothed_prob = float(np.mean(prob_history))
                    cached_prob = smoothed_prob
                    cached_is_crowd = smoothed_prob >= 0.50

                    dense_clusters, gatherings, isolated = detect_spatial_clusters(cached_boxes)
                    cached_dense_clusters = dense_clusters
                    cached_gatherings = gatherings
                    cached_isolated = isolated
                    cached_dense_cnt = len(dense_clusters)
                    cached_gather_cnt = len(gatherings)
                    cached_isolated_cnt = len(isolated)
                except Exception as e:
                    print(f"[{self.id} Inference Error]: {e}")

            with self.lock:
                cur_heatmap = self.show_heatmap
                cur_clusters = self.show_clusters

            # Render visual overlays (Ground thermal density heatmap & spatial cluster hulls)
            annotated_frame, dense_cnt, gather_cnt, isolated_cnt = draw_annotations(
                frame=frame,
                boxes=cached_boxes,
                confidences=cached_confs,
                is_crowd=cached_is_crowd,
                show_hud=False,
                show_heatmap=cur_heatmap,
                show_clusters=cur_clusters,
                dense_clusters=cached_dense_clusters,
                gatherings=cached_gatherings,
                isolated=cached_isolated,
            )

            # Compress JPEG for high-throughput MJPEG broadcaster
            success, jpeg_buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                jpeg_bytes = jpeg_buffer.tobytes()
                with self.lock:
                    self.latest_jpeg = jpeg_bytes

                    if cached_is_crowd or dense_cnt > 0:
                        status_str = "SURGE ALERT"
                    elif cam_info["connected"]:
                        status_str = "ONLINE"
                    else:
                        status_str = "OFFLINE"

                    self.stats.update({
                        "status": status_str,
                        "is_suspended": False,
                        "active": True,
                        "is_crowd": cached_is_crowd,
                        "crowd_probability": round(cached_prob, 4),
                        "people_count": cached_features.get("person_count", 0),
                        "density_per_megapixel": round(cached_features.get("people_per_megapixel", 0.0), 2),
                        "nearest_neighbor_distance": round(
                            cached_features.get("average_nearest_neighbor_distance", 0.0), 4
                        ),
                        "dense_clusters_count": dense_cnt,
                        "gatherings_count": gather_cnt,
                        "isolated_count": isolated_cnt,
                        "heatmap_enabled": cur_heatmap,
                        "clusters_enabled": cur_clusters,
                        "fps": round(fps_instant, 1),
                        "capture_fps": cam_info.get("capture_fps", 0.0),
                        "camera_connected": cam_info.get("connected", False),
                        "resolution": cam_info.get("resolution", "Online"),
                        "uptime_seconds": int(time.time() - self.start_time),
                    })

    def get_latest_frame(self) -> Optional[bytes]:
        """Returns the latest annotated JPEG frame or a standby placeholder if suspended."""
        if not self.active:
            return self._cached_standby_frame
        with self.lock:
            if self.latest_jpeg is not None:
                return self.latest_jpeg
            return self._cached_standby_frame

    def get_stats(self) -> Dict[str, Any]:
        """Returns complete thread-safe telemetry dictionary."""
        with self.lock:
            stats = dict(self.stats)
            stats["active"] = self.active
            stats["is_suspended"] = not self.active
            stats["approved_for_public"] = self.approved_for_public
            stats["heatmap_enabled"] = self.show_heatmap
            stats["clusters_enabled"] = self.show_clusters
            stats["coordinates"] = self.coordinates
            stats["zone"] = self.zone
            stats["name"] = self.name
            if not self.active:
                stats["status"] = "STANDBY"
                stats["fps"] = 0.0
                stats["resolution"] = "Standby (Idle)"
            return stats

    def suspend(self):
        """Suspends camera capture device and inference loop to conserve hardware resources."""
        with self.lock:
            self.active = False
            self.stats["active"] = False
            self.stats["is_suspended"] = True
            self.stats["status"] = "STANDBY"
            self.stats["camera_connected"] = False
            self.stats["fps"] = 0.0
            self.stats["capture_fps"] = 0.0
            self.stats["resolution"] = "Standby (Idle)"
        self.camera.suspend()
        print(f"[Camera {self.id}] Suspended (Standby Mode).")

    def resume(self):
        """Resumes camera capture and activates real-time pedestrian analysis."""
        with self.lock:
            self.active = True
            self.stats["active"] = True
            self.stats["is_suspended"] = False
            self.stats["status"] = "INITIALIZING"
        self.camera.resume()
        print(f"[Camera {self.id}] Resumed (Active Capture).")

    def toggle(self) -> bool:
        """Toggles active/standby state. Returns new active boolean."""
        if self.active:
            self.suspend()
            return False
        else:
            self.resume()
            return True

    def set_public_approval(self, approved: bool) -> bool:
        """Sets control room public visibility approval flag."""
        with self.lock:
            self.approved_for_public = bool(approved)
            self.stats["approved_for_public"] = self.approved_for_public
        print(f"[Camera {self.id}] Public Visibility set to: {self.approved_for_public}")
        return self.approved_for_public

    def set_display_options(self, heatmap: Optional[bool] = None, clusters: Optional[bool] = None):
        """Toggles real-time ground heatmap and spatial clustering overlays."""
        with self.lock:
            if heatmap is not None:
                self.show_heatmap = bool(heatmap)
                self.stats["heatmap_enabled"] = self.show_heatmap
            if clusters is not None:
                self.show_clusters = bool(clusters)
                self.stats["clusters_enabled"] = self.show_clusters

    def update_config(
        self,
        name: Optional[str] = None,
        source: Optional[Any] = None,
        zone: Optional[str] = None,
        coordinates: Optional[Dict[str, Any]] = None,
        active: Optional[bool] = None,
        approved_for_public: Optional[bool] = None,
        show_heatmap: Optional[bool] = None,
        show_clusters: Optional[bool] = None,
    ):
        """Updates camera parameters dynamically during runtime."""
        source_changed = False
        with self.lock:
            if name is not None:
                self.name = str(name).strip()
                self.stats["name"] = self.name
            if zone is not None:
                self.zone = str(zone).strip()
                self.stats["zone"] = self.zone
            if coordinates is not None:
                self.coordinates = coordinates
                self.stats["coordinates"] = self.coordinates
            if approved_for_public is not None:
                self.approved_for_public = bool(approved_for_public)
                self.stats["approved_for_public"] = self.approved_for_public
            if show_heatmap is not None:
                self.show_heatmap = bool(show_heatmap)
                self.stats["heatmap_enabled"] = self.show_heatmap
            if show_clusters is not None:
                self.show_clusters = bool(show_clusters)
                self.stats["clusters_enabled"] = self.show_clusters

            if source is not None and str(source).strip() != str(self.source).strip():
                self.source = int(source) if str(source).isdigit() else source
                self.source_meta = parse_camera_source(self.source)
                self.stats["source"] = self.source_meta.get("masked_source", str(self.source))
                self.stats["raw_source"] = str(self.source)
                self.stats["camera_type"] = self.source_meta.get("type", "unknown")
                self.stats["ip"] = self.source_meta.get("ip")
                self.stats["port"] = self.source_meta.get("port")
                source_changed = True

            self._cached_standby_frame = generate_placeholder_frame(
                title=f"{self.name}",
                subtitle=f"Zone: {self.zone} • {self.coordinates.get('floor', '')} (X:{self.coordinates.get('x', '-')}, Y:{self.coordinates.get('y', '-')})",
                status="STANDBY",
                badge_color=(120, 140, 160),
            )

        if source_changed:
            old_cam = self.camera
            self.camera = RTSPCameraStream(self.source, suspended=not self.active)
            if old_cam:
                old_cam.stop()

        if active is not None:
            if active and not self.active:
                self.resume()
            elif not active and self.active:
                self.suspend()

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable camera definition for persistence."""
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "source": self.source,
                "zone": self.zone,
                "coordinates": self.coordinates,
                "active": self.active,
                "approved_for_public": self.approved_for_public,
                "show_heatmap": self.show_heatmap,
                "show_clusters": self.show_clusters,
                "conf": self.conf,
                "iou": self.iou,
                "stride": self.stride,
                "smooth": self.smooth,
                "created_at": self.created_at,
            }

    def stop(self):
        """Cleanly releases all camera threads and hardware connections."""
        self.running = False
        self.camera.stop()


class CameraManager:
    """
    Central Local Edge Server Brain:
    - Maintains the multi-camera pipeline registry
    - Persists configurations to camera_registry.json
    - Computes aggregated Station-Wide crowd intelligence & surge warnings
    - Coordinates shared inference across all optical streams
    """

    def __init__(self, detector: Optional[CrowdDetector] = None, registry_file: Optional[Path] = None):
        self.detector = detector or CrowdDetector()
        self.registry_file = Path(registry_file or REGISTRY_FILE)
        self.cameras: Dict[str, CameraNode] = {}
        self.lock = threading.Lock()
        self.station_name = STATION_NAME

        # Load existing registry or seed initial station presets
        self.load_registry()

    def _resolve_sample_video(self, index: int = 0) -> Optional[str]:
        """Resolves local MP4 test video if present on the edge device."""
        existing = [v for v in CANDIDATE_SAMPLE_VIDEOS if os.path.exists(v)]
        if existing and index < len(existing):
            return existing[index]
        return None

    def _default_seed_cameras(self) -> List[Dict[str, Any]]:
        """Generates realistic initial station CCTV presets for immediate local testing."""
        v1 = self._resolve_sample_video(0) or "0"
        v2 = self._resolve_sample_video(1) or "rtsp://192.168.1.102:554/live"
        v3 = self._resolve_sample_video(2) or "rtsp://192.168.1.103:554/live"

        return [
            {
                "id": "cam-concourse-turnstiles",
                "name": "Concourse Main Turnstiles",
                "source": v1,
                "zone": "Terminal Concourse 04 • Gate South",
                "coordinates": {
                    "floor": "Level 1 - Concourse",
                    "x": 140.0,
                    "y": 260.0,
                    "z": 0.0,
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                },
                "active": True,
                "approved_for_public": False,
                "show_heatmap": True,
                "show_clusters": True,
                "stride": 1,
            },
            {
                "id": "cam-escalator-choke",
                "name": "Platform 1 Escalator Chokepoint",
                "source": v2,
                "zone": "Platform 1 • Vertical Escalator Flow",
                "coordinates": {
                    "floor": "Mezzanine Level",
                    "x": 320.0,
                    "y": 190.0,
                    "z": 1.0,
                    "latitude": 28.6142,
                    "longitude": 77.2095,
                },
                "active": True,
                "approved_for_public": True,
                "show_heatmap": True,
                "show_clusters": True,
                "stride": 2,
            },
            {
                "id": "cam-platform-boarding",
                "name": "Platform 2 Boarding Hall",
                "source": v3,
                "zone": "Platform Hub • Track Sector 3",
                "coordinates": {
                    "floor": "Platform Level",
                    "x": 480.0,
                    "y": 510.0,
                    "z": 0.0,
                    "latitude": 28.6146,
                    "longitude": 77.2102,
                },
                "active": True,
                "approved_for_public": True,
                "show_heatmap": True,
                "show_clusters": True,
                "stride": 2,
            },
            {
                "id": "cam-optical-local",
                "name": "Control Room Optical Sensor",
                "source": "0",
                "zone": "Operations Control Room • Hub Aux",
                "coordinates": {
                    "floor": "Control Center Level 2",
                    "x": 60.0,
                    "y": 45.0,
                    "z": 2.0,
                    "latitude": 28.6135,
                    "longitude": 77.2085,
                },
                "active": False,
                "approved_for_public": False,
                "show_heatmap": True,
                "show_clusters": True,
                "stride": 1,
            },
        ]

    def load_registry(self):
        """Loads registered cameras from JSON. Initializes with 0 cameras if none registered."""
        configs = []
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    configs = json.load(f)
                print(f"[CameraManager] Loaded {len(configs)} cameras from {self.registry_file}")
            except Exception as e:
                print(f"[CameraManager] Error loading {self.registry_file}: {e}")

        # If no cameras registered in registry, edge server remains completely idle (0 feeds, 0 processing)
        if not configs:
            print("[CameraManager] No cameras registered. Edge server running idle awaiting camera registration.")
            self._save_raw_configs([])
            self.cameras.clear()
            return

        for cfg in configs:
            cid = cfg.get("id")
            if not cid:
                continue
            try:
                node = CameraNode(
                    camera_id=cid,
                    name=cfg.get("name", f"Camera {cid}"),
                    source=cfg.get("source", "0"),
                    zone=cfg.get("zone", DEFAULT_ZONE),
                    coordinates=cfg.get("coordinates"),
                    detector=self.detector,
                    active=cfg.get("active", True),
                    approved_for_public=cfg.get("approved_for_public", False),
                    show_heatmap=cfg.get("show_heatmap", DEFAULT_HEATMAP),
                    show_clusters=cfg.get("show_clusters", DEFAULT_CLUSTERS),
                    conf=cfg.get("conf", DEFAULT_CONF),
                    iou=cfg.get("iou", DEFAULT_IOU),
                    stride=cfg.get("stride", DEFAULT_STRIDE),
                    smooth=cfg.get("smooth", DEFAULT_SMOOTH),
                    created_at=cfg.get("created_at"),
                )
                self.cameras[cid] = node
            except Exception as err:
                print(f"[CameraManager] Failed to instantiate camera {cid}: {err}")

    def _save_raw_configs(self, configs: List[Dict[str, Any]]):
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(configs, f, indent=2)
        except Exception as e:
            print(f"[CameraManager] Failed to write {self.registry_file}: {e}")

    def save_registry(self):
        """Saves current camera registry state to disk."""
        with self.lock:
            configs = [cam.to_dict() for cam in self.cameras.values()]
        self._save_raw_configs(configs)

    def register_camera(
        self,
        name: str,
        source: Any,
        zone: str = DEFAULT_ZONE,
        camera_id: Optional[str] = None,
        coordinates: Optional[Dict[str, Any]] = None,
        active: bool = True,
        approved_for_public: bool = False,
        show_heatmap: bool = DEFAULT_HEATMAP,
        show_clusters: bool = DEFAULT_CLUSTERS,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        stride: int = DEFAULT_STRIDE,
    ) -> CameraNode:
        """Registers and immediately spins up a new camera ingestion node."""
        if not camera_id:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
            camera_id = f"cam-{slug}" if slug else f"cam-{int(time.time())}"

        # Ensure uniqueness
        base_id = camera_id
        counter = 1
        with self.lock:
            while camera_id in self.cameras:
                camera_id = f"{base_id}-{counter}"
                counter += 1

            node = CameraNode(
                camera_id=camera_id,
                name=name,
                source=source,
                zone=zone,
                coordinates=coordinates,
                detector=self.detector,
                active=active,
                approved_for_public=approved_for_public,
                show_heatmap=show_heatmap,
                show_clusters=show_clusters,
                conf=conf,
                iou=iou,
                stride=stride,
            )
            self.cameras[camera_id] = node

        self.save_registry()
        print(f"[CameraManager] Successfully registered new camera: {camera_id} ({name})")
        return node

    def update_camera(self, camera_id: str, **kwargs) -> Optional[CameraNode]:
        """Updates parameters for an existing camera node."""
        node = self.cameras.get(camera_id)
        if not node:
            return None
        node.update_config(**kwargs)
        self.save_registry()
        return node

    def delete_camera(self, camera_id: str) -> bool:
        """Safely stops and deregisters a camera node."""
        with self.lock:
            node = self.cameras.pop(camera_id, None)
        if node:
            node.stop()
            self.save_registry()
            print(f"[CameraManager] Deregistered camera {camera_id}")
            return True
        return False

    def get_camera(self, camera_id: str) -> Optional[CameraNode]:
        return self.cameras.get(camera_id)

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Returns live status and telemetry for all registered cameras."""
        with self.lock:
            nodes = list(self.cameras.values())
        return [node.get_stats() for node in nodes]

    def get_overview_stats(self) -> Dict[str, Any]:
        """
        Computes station-wide aggregated intelligence:
        - Total pedestrians detected across all camera feeds
        - Multi-camera surge warnings and bottleneck zones
        - Active vs Standby camera counts
        - Public broadcast feed approvals
        """
        cam_stats = self.list_cameras()
        total_cams = len(cam_stats)
        active_cams = sum(1 for c in cam_stats if c["active"])
        online_cams = sum(1 for c in cam_stats if c["active"] and c["camera_connected"])
        standby_cams = total_cams - active_cams
        approved_public = sum(1 for c in cam_stats if c["approved_for_public"])

        total_people = sum(c["people_count"] for c in cam_stats if c["active"])
        total_dense_clusters = sum(c["dense_clusters_count"] for c in cam_stats if c["active"])
        total_gatherings = sum(c["gatherings_count"] for c in cam_stats if c["active"])

        surge_alerts = []
        highest_density_zone = "Idle (No Cameras Registered)" if total_cams == 0 else "Normal"
        max_density = 0.0

        for c in cam_stats:
            if not c["active"]:
                continue
            density = c["density_per_megapixel"]
            if density > max_density:
                max_density = density
                highest_density_zone = f"{c['zone']} ({c['name']})"

            # Surge warning trigger: Conflux crowd classification or high-density cluster detected
            if c["is_crowd"] or c["dense_clusters_count"] > 0:
                surge_alerts.append({
                    "camera_id": c["id"],
                    "name": c["name"],
                    "zone": c["zone"],
                    "coordinates": c["coordinates"],
                    "people_count": c["people_count"],
                    "density_per_megapixel": density,
                    "dense_clusters_count": c["dense_clusters_count"],
                    "gatherings_count": c["gatherings_count"],
                    "crowd_probability": c["crowd_probability"],
                    "level": "CRITICAL" if c["dense_clusters_count"] >= 2 else ("HIGH" if c["is_crowd"] else "ELEVATED"),
                    "message": f"Spatial crowd surge alert detected at {c['zone']} ({c['people_count']} people)",
                })

        return {
            "station_name": self.station_name,
            "status": "OPERATIONAL",
            "network_mode": "LOCAL ON-PREM",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_cameras": total_cams,
            "active_cameras": active_cams,
            "online_cameras": online_cams,
            "standby_cameras": standby_cams,
            "approved_public_cameras": approved_public,
            "total_people_detected": total_people,
            "total_dense_clusters": total_dense_clusters,
            "total_gatherings": total_gatherings,
            "active_surge_alerts_count": len(surge_alerts),
            "highest_density_zone": highest_density_zone,
            "max_density_per_megapixel": round(max_density, 2),
            "surge_alerts": surge_alerts,
        }

    def get_primary_camera(self) -> Optional[CameraNode]:
        """Finds primary active camera (used for single-feed backward compatibility)."""
        with self.lock:
            for cam in self.cameras.values():
                if cam.active:
                    return cam
            if self.cameras:
                return next(iter(self.cameras.values()))
        return None

    def stop_all(self):
        """Clean shutdown of all cameras and ingestion pipelines."""
        print("[CameraManager] Halting all camera feeds and ingestion threads...")
        with self.lock:
            cams = list(self.cameras.values())
        for c in cams:
            c.stop()
