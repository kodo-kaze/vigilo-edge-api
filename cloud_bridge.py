import asyncio
import base64
import json
import time
from typing import Any, Dict, Optional, Set

import websockets

try:
    from config import (
        EDGE_API_KEY,
        EDGE_NODE_ID,
        GEO_SERVER_WS_URL,
        STATION_NAME,
        TELEMETRY_PUSH_INTERVAL,
    )
    from camera import CameraManager
    from stream import CloudStreamRelay
except ImportError:
    from backend.config import (
        EDGE_API_KEY,
        EDGE_NODE_ID,
        GEO_SERVER_WS_URL,
        STATION_NAME,
        TELEMETRY_PUSH_INTERVAL,
    )
    from backend.camera import CameraManager
    from backend.stream import CloudStreamRelay


class EdgeCloudBridge:
    """
    Persistent reverse-outbound WebSocket bridge from vigilo-edge-api to vigilo-geo-server.

    - Initiates outbound ws:// / wss:// connection (no inbound station ports exposed).
    - Pushes initial camera registry (`register_node`), periodic telemetry (`telemetry_batch`),
      and immediate CRUD/toggle events (`camera_updated` / `camera_removed`).
    - Listens for Geo Server `start_stream` / `stop_stream` commands and drives CloudStreamRelay.
    - Reconnects automatically with exponential backoff (1s -> 2s -> 4s -> max 30s).
    """

    def __init__(
        self,
        manager: CameraManager,
        relay: CloudStreamRelay,
        ws_url: str = GEO_SERVER_WS_URL,
        edge_node_id: str = EDGE_NODE_ID,
        api_key: str = EDGE_API_KEY,
        telemetry_interval: float = TELEMETRY_PUSH_INTERVAL,
    ):
        self.manager = manager
        self.relay = relay
        self.ws_url = (ws_url or "").strip()
        self.edge_node_id = edge_node_id
        self.api_key = api_key
        self.telemetry_interval = max(0.2, float(telemetry_interval))

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._event_queue: Optional[asyncio.Queue] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._active_ws: Any = None
        self._active_stream_cams: Set[str] = set()

        self._status: str = "Disabled" if not self.ws_url else "Reconnecting"
        self._connected: bool = False
        self._last_connected_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._reconnect_attempt: int = 0

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[asyncio.Task]:
        """Starts the outbound bridge task on the active asyncio event loop."""
        if not self.ws_url:
            self._status = "Disabled"
            print("[EdgeCloudBridge] GEO_SERVER_WS_URL is empty; cloud bridge disabled.")
            return None

        self._loop = loop or asyncio.get_running_loop()
        self._event_queue = asyncio.Queue(maxsize=256)
        self._stop_event = asyncio.Event()
        self._task = self._loop.create_task(self._run_loop(), name="edge-cloud-bridge")
        print(f"[EdgeCloudBridge] Started outbound bridge -> {self.ws_url} (node={self.edge_node_id})")
        return self._task

    async def stop(self):
        """Cleanly shuts down the WebSocket bridge and stops all outbound media relays."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._active_ws is not None:
            try:
                await self._active_ws.close()
            except Exception:
                pass
            self._active_ws = None

        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        await asyncio.to_thread(self.relay.stop_all)
        self._connected = False
        self._status = "Disconnected"
        print("[EdgeCloudBridge] Stopped cleanly.")

    def get_status(self) -> Dict[str, Any]:
        """Returns current Geo Server link and outbound relay status for admin UI."""
        return {
            "status": self._status,
            "connected": self._connected,
            "edge_node_id": self.edge_node_id,
            "geo_server_ws_url": self.ws_url,
            "active_cloud_streams": self.relay.active_streams(),
            "last_connected_at": self._last_connected_at,
            "last_error": self._last_error,
            "reconnect_attempt": self._reconnect_attempt,
        }

    def _enqueue_event(self, payload: Dict[str, Any]):
        """Thread-safe enqueue of immediate state sync events."""
        if self._loop is None or self._event_queue is None or not self._connected:
            return

        def _put():
            if self._event_queue is None:
                return
            if self._event_queue.full():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._event_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        try:
            self._loop.call_soon_threadsafe(_put)
        except RuntimeError:
            pass

    def notify_camera_updated(self, cam_id: str):
        """Immediately syncs camera metadata/visibility changes to Geo Server and enforces kill-switch."""
        node = self.manager.get_camera(cam_id)
        if node is None:
            return
        if not node.is_public or not node.active:
            self.relay.stop(cam_id)
        self._enqueue_event(
            {
                "type": "camera_updated",
                "edge_node_id": self.edge_node_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "camera": node.get_cloud_registration(),
            }
        )

    def notify_camera_removed(self, cam_id: str):
        """Immediately stops any cloud stream and notifies Geo Server of camera deletion."""
        self.relay.stop(cam_id)
        self._enqueue_event(
            {
                "type": "camera_removed",
                "edge_node_id": self.edge_node_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cam_id": cam_id,
            }
        )

    def _drain_event_queue(self):
        if self._event_queue is None:
            return
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _run_loop(self):
        backoff = 1.0
        max_backoff = 30.0
        assert self._stop_event is not None

        while not self._stop_event.is_set():
            self._status = "Reconnecting"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            try:
                async with websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                    ping_interval=15,
                    ping_timeout=15,
                    close_timeout=5,
                ) as ws:
                    self._active_ws = ws
                    self._connected = True
                    self._status = "Connected"
                    self._last_connected_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    self._last_error = None
                    self._reconnect_attempt = 0
                    backoff = 1.0
                    self._drain_event_queue()

                    print(f"[EdgeCloudBridge] Connected to Geo Server ({self.ws_url})")

                    # Initial register_node handshake
                    register_payload = {
                        "type": "register_node",
                        "edge_node_id": self.edge_node_id,
                        "station_name": STATION_NAME,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "cameras": self.manager.get_cloud_registrations(),
                    }
                    await ws.send(json.dumps(register_payload))

                    telemetry_task = asyncio.create_task(self._telemetry_loop(ws))
                    events_task = asyncio.create_task(self._immediate_events_loop(ws))
                    video_task = asyncio.create_task(self._video_stream_loop(ws))
                    recv_task = asyncio.create_task(self._command_listener_loop(ws))

                    done, pending = await asyncio.wait(
                        {telemetry_task, events_task, video_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                    for task in done:
                        exc = task.exception()
                        if exc and not isinstance(exc, asyncio.CancelledError):
                            raise exc

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._last_error = str(exc)
                self._reconnect_attempt += 1
                print(
                    f"[EdgeCloudBridge] Link lost ({exc}); "
                    f"reconnecting in {backoff:.0f}s (attempt {self._reconnect_attempt})"
                )
            finally:
                self._active_ws = None
                self._connected = False
                await asyncio.to_thread(self.relay.stop_all)
                if not self._stop_event.is_set():
                    self._status = "Reconnecting"

            if self._stop_event.is_set():
                break

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                break
            except asyncio.TimeoutError:
                backoff = min(max_backoff, backoff * 2.0)

    async def _telemetry_loop(self, ws: Any):
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            payload = {
                "type": "telemetry_batch",
                "edge_node_id": self.edge_node_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cameras": self.manager.get_cloud_telemetry_batch(),
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(self.telemetry_interval)

    async def _immediate_events_loop(self, ws: Any):
        assert self._event_queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            event = await self._event_queue.get()
            await ws.send(json.dumps(event))

    async def _video_stream_loop(self, ws: Any):
        """
        Pipes annotated JPEG frames from actively watched public cameras directly
        through the existing reverse WebSocket connection to the Geo Server.
        Zero inbound port opening or tunnels needed!
        """
        assert self._stop_event is not None
        frame_interval = 0.08  # ~12 FPS
        while not self._stop_event.is_set():
            if not self._active_stream_cams:
                await asyncio.sleep(0.2)
                continue

            for cam_id in list(self._active_stream_cams):
                node = self.manager.get_camera(cam_id)
                if node is None or not node.active or not node.is_public:
                    self._active_stream_cams.discard(cam_id)
                    continue

                frame_bytes = node.get_latest_frame()
                if frame_bytes:
                    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
                    msg = {
                        "type": "video_frame",
                        "edge_node_id": self.edge_node_id,
                        "cam_id": cam_id,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "frame_base64": frame_b64,
                    }
                    try:
                        await ws.send(json.dumps(msg))
                    except Exception:
                        break

            await asyncio.sleep(frame_interval)

    async def _command_listener_loop(self, ws: Any):
        async for raw_message in ws:
            try:
                cmd = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            action = cmd.get("action")
            cam_id = cmd.get("cam_id")
            if not action or not cam_id:
                continue

            if action == "start_stream":
                node = self.manager.get_camera(cam_id)
                if node is not None and node.is_public and node.active:
                    self._active_stream_cams.add(cam_id)
                    node.set_streaming_to_cloud(True)
                    self.relay.start(node)
                    self.notify_camera_updated(cam_id)
                    print(f"[EdgeCloudBridge] Streaming camera {cam_id} securely to Geo Server over reverse WebSocket")
                else:
                    print(
                        f"[EdgeCloudBridge] Ignoring start_stream for {cam_id} "
                        f"(exists={node is not None}, is_public={getattr(node, 'is_public', False)}, "
                        f"active={getattr(node, 'active', False)})"
                    )
                    if node is not None:
                        self.notify_camera_updated(cam_id)
            elif action == "stop_stream":
                self._active_stream_cams.discard(cam_id)
                await asyncio.to_thread(self.relay.stop, cam_id)
                node = self.manager.get_camera(cam_id)
                if node is not None:
                    node.set_streaming_to_cloud(False)
                    self.notify_camera_updated(cam_id)
                print(f"[EdgeCloudBridge] Stopped streaming camera {cam_id}")
