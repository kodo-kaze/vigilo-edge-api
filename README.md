# Vigilo On-Premise Multi-Camera Edge Server (`vigilo-edge-api`)

Autonomous On-Premise Edge Brain deployed inside the station's local control center. Ingests live video streams from multiple CCTV cameras across the premises simultaneously, analyzes crowd density and spatial compression on every feed in real time, overlays live visual alerts and ground thermal heatmaps onto each stream, and provides a lightweight local control room admin panel.

Part of the **Vigilo** intelligent crowd flow architecture. Pairs with [`vigilo-user-interface`](https://github.com/kodo-kaze/vigilo-user-interface).

---

## 🌟 Key Capabilities

- **Concurrent Multi-Camera Ingestion**: Ingests multiple RTSP streams, IP cameras, hardware webcams, and local video feeds simultaneously without bottlenecking.
- **Shared High-Performance AI Engine**: Thread-safe singleton `CrowdDetector` running YOLO26m (`model/best.pt`) and 21-feature Conflux Random Forest (`model/crowd_random_forest_v1.pkl`), preventing duplicate memory footprints in RAM/VRAM.
- **2D Floor-Level Ground Thermal Density Heatmap**: Computes pedestrian floor contact points and projects perspective-scaled Gaussian accumulator kernels (`COLORMAP_TURBO`, $<1.5\text{ms}$ latency).
- **Perspective-Adaptive Spatial Clustering**: Fast BFS graph clustering segregating pedestrians into:
  - *High-Density Zones* ($\ge 6$ people): Crimson convex hull polygons with perimeter alerts.
  - *Abnormal Gatherings* ($3-5$ people): Royal blue / amber bounding zones.
  - *Isolated Individuals*: Cyber corner brackets with confidence tags.
- **Local Control-Room Admin Panel**: Zero-dependency, responsive dark-glass HTML5/CSS/vanilla JS dashboard served directly from the edge server at `http://localhost:8000/`:
  - Multi-camera matrix grid (1x1, 2x2, 3x3, auto responsive).
  - Register new CCTV cameras with physical location coordinates (Floor, X, Y, Lat/Long).
  - Real-time surge warning ticker and emergency chokepoint alerts.
  - Granular control over feeds: Active/Standby power toggle, Public Screen Approval toggle, Heatmap toggle, Cluster hulls toggle.
  - Camera reachability socket ping tester.
- **Persistent Storage**: Camera registry state persists to `camera_registry.json`.
- **100% Local-First / Air-Gapped**: Runs entirely inside the station's local LAN before any external cloud or passenger broadcast integration.

---

## 🔌 API Reference

### Local Control Room UI
- `GET /` or `GET /admin`: Interactive local admin control panel.

### Station Overview & Telemetry
- `GET /api/overview`: Station-wide aggregated telemetry (total pedestrians across all cameras, active surge warnings, online/standby counts, peak chokepoint zone, public visibility count).
- `GET /health`: Server health check and camera statistics.

### Multi-Camera Management
- `GET /api/cameras`: Lists all registered cameras with real-time status and coordinates.
- `POST /api/cameras`: Registers a new camera (`name`, `source`, `zone`, `coordinates`, `active`, `approved_for_public`, `show_heatmap`, `show_clusters`).
- `GET /api/cameras/{camera_id}`: Single camera telemetry and configuration.
- `PUT /api/cameras/{camera_id}` or `PATCH`: Updates camera configuration (name, source, coordinates, active state).
- `DELETE /api/cameras/{camera_id}`: Deregisters and stops camera pipeline.

### Per-Camera Stream & Controls
- `GET /api/cameras/{camera_id}/stream`: High-throughput Multipart MJPEG live stream with real-time HUD overlays.
- `GET /api/cameras/{camera_id}/stats`: Live JSON telemetry for the specified camera.
- `POST /api/cameras/{camera_id}/toggle`: Toggles power-saving standby mode on/off.
- `POST /api/cameras/{camera_id}/public`: Toggles or sets public visibility approval (`approved_for_public`).
- `POST /api/cameras/{camera_id}/options`: Toggles ground heatmap and spatial cluster overlays.
- `POST /api/cameras/test` or `POST /camera/test`: TCP socket reachability test for target IP/port.

### Backward-Compatible Endpoints
- `GET /video_feed`: Streams primary active camera (or `?camera_id=xxx`).
- `GET /stats`: Primary camera telemetry with embedded `station_overview`.
- `GET /camera/info`, `GET /camera/sources`, `POST /camera/suspend`, `POST /camera/resume`, `POST /camera/toggle`, `POST /camera/options`.

---

## 🚀 Running the Edge Server Locally

```bash
# 1. Activate Python virtual environment
cd /home/bimbok/shared/code/working_on/vigilo/vigilo-edge-api
# or: cd /home/bimbok/shared/code/working_on/vigilo-edge-api

source ../model/venv/bin/activate
# or: source .venv/bin/activate

# 2. Start the Edge Server
bash start.sh
# or directly:
python main.py --host 0.0.0.0 --port 8000
```

Once running:
- Open **`http://localhost:8000`** in any browser on the local station network to access the **Control Room Admin Panel**.
