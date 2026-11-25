import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

import httpx

EDGE_DIR = Path(__file__).resolve().parent.parent
GEO_DIR = EDGE_DIR.parent / "vigilo-geo-server"

if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from camera import CameraManager  # noqa: E402
from stream import CloudStreamRelay  # noqa: E402
from cloud_bridge import EdgeCloudBridge  # noqa: E402


class DummyDetector:
    def predict(self, frame):
        import numpy as np
        return np.empty((0, 4)), np.empty(0), {
            "person_count": 0,
            "people_per_megapixel": 0.0,
            "average_nearest_neighbor_distance": 0.0,
        }, 0.0


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class EdgeBridgeIntegrationTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_camera_fields_and_live_bridge_to_geo_server(self):
        port = _free_port()
        env = os.environ.copy()
        env["REDIS_URL"] = ""
        env["EDGE_API_KEY"] = "vigilo-dev-edge-key"

        geo_proc = subprocess.Popen(
            [sys.executable, str(GEO_DIR / "main.py"), "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(GEO_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait until Geo Server /health responds
        async with httpx.AsyncClient() as client:
            ready = False
            for _ in range(50):
                if geo_proc.poll() is not None:
                    break
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
            if not ready:
                geo_proc.terminate()
                out, _ = geo_proc.communicate(timeout=3.0)
                self.fail(f"Geo Server subprocess failed to become ready:\n{out.decode('utf-8', errors='replace')}")

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "camera_registry.json"
            manager = CameraManager(detector=DummyDetector(), registry_file=reg_path)
            relay = CloudStreamRelay(
                media_push_url="rtsp://127.0.0.1:8554",
                edge_node_id="test-hub-01",
            )
            bridge = EdgeCloudBridge(
                manager=manager,
                relay=relay,
                ws_url=f"ws://127.0.0.1:{port}/ws/edge",
                edge_node_id="test-hub-01",
                api_key="vigilo-dev-edge-key",
                telemetry_interval=0.2,
            )

            try:
                node = manager.register_camera(
                    name="Test Concourse Cam",
                    source="nonexistent_test_source.mp4",
                    camera_id="cam-test-1",
                    zone="Concourse A",
                    lat=22.5675,
                    lon=88.3705,
                    active=True,
                )

                # Verify defaults & persisted fields
                self.assertEqual(node.cam_id, "cam-test-1")
                self.assertEqual(node.lat, 22.5675)
                self.assertEqual(node.lon, 88.3705)
                self.assertTrue(node.is_public)
                self.assertFalse(node.is_streaming_to_cloud)

                persisted = json.loads(reg_path.read_text(encoding="utf-8"))
                self.assertEqual(len(persisted), 1)
                self.assertEqual(persisted[0]["cam_id"], "cam-test-1")
                self.assertEqual(persisted[0]["lat"], 22.5675)
                self.assertEqual(persisted[0]["lon"], 88.3705)
                self.assertTrue(persisted[0]["is_public"])

                # Start EdgeCloudBridge and wait for Connected status
                bridge.start(asyncio.get_running_loop())
                for _ in range(40):
                    if bridge.get_status()["connected"]:
                        break
                    await asyncio.sleep(0.05)
                self.assertTrue(bridge.get_status()["connected"])

                await asyncio.sleep(0.3)

                async with httpx.AsyncClient() as client:
                    res = await client.get(
                        f"http://127.0.0.1:{port}/api/nearest-camera",
                        params={"lat": 22.5675, "lon": 88.3705},
                    )
                    self.assertEqual(res.status_code, 200)
                    body = res.json()
                    self.assertEqual(body["cam_id"], "cam-test-1")
                    self.assertEqual(body["edge_node_id"], "test-hub-01")

                # Wait for start_stream command to flip is_streaming_to_cloud
                for _ in range(20):
                    if node.is_streaming_to_cloud:
                        break
                    await asyncio.sleep(0.05)
                self.assertTrue(node.is_streaming_to_cloud)

                # Flip is_public to False -> must immediately stop relay & sync to Geo Server
                node.set_public_approval(False)
                bridge.notify_camera_updated("cam-test-1")
                for _ in range(20):
                    if not node.is_streaming_to_cloud:
                        break
                    await asyncio.sleep(0.05)
                self.assertFalse(node.is_streaming_to_cloud)

                await asyncio.sleep(0.2)
                async with httpx.AsyncClient() as client:
                    res_after = await client.get(
                        f"http://127.0.0.1:{port}/api/nearest-camera",
                        params={"lat": 22.5675, "lon": 88.3705},
                    )
                    self.assertEqual(res_after.status_code, 404)
            finally:
                await bridge.stop()
                relay.stop_all()
                manager.stop_all()
                geo_proc.terminate()
                try:
                    geo_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    geo_proc.kill()
                    geo_proc.wait()
                if geo_proc.stdout:
                    geo_proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
