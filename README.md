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

---

## 🖥️ System Requirements & Hardware Specifications

The Vigilo Edge Server runs **YOLO26m** (21.8M parameters / 75.0 GFLOPs per frame), a 21-feature **Conflux Random Forest** crowd density classifier, real-time 2D floor thermal heatmaps, perspective-adaptive spatial cluster hulls, and reverse-WebSocket cloud video streaming.

### ⚡ Hardware Specifications Matrix

| Component | Minimum Specification (1 Camera @ 12–15 FPS) | Recommended Specification (1–4 Cameras @ 25–30 FPS) | Production / High-Load Rig (4+ Cameras) |
| :--- | :--- | :--- | :--- |
| **GPU** | None *(CPU fallback)* | **NVIDIA GPU with 4GB+ VRAM**<br>*(GTX 1650, RTX 3050/3060, RTX 4060)* | **NVIDIA Jetson Orin Nano (8GB) / Orin NX (16GB)** or **RTX 4070+ (8GB+ VRAM)** |
| **CPU** | Quad-Core CPU *(Intel i5 10th Gen+, AMD Ryzen 3/5)* | **6 Cores / 12 Threads** *(Intel Core i5-11400 / i5-12400, AMD Ryzen 5 5600+)* | **8+ Cores / 16 Threads** *(Intel Core i7-12700K+, AMD Ryzen 7 5800X+)* |
| **RAM** | **8 GB DDR4** | **16 GB DDR4 / DDR5** | **16 GB – 32 GB DDR5** |
| **Storage** | 10 GB SSD | 20 GB NVMe SSD | 50 GB+ High-Endurance NVMe SSD |
| **Network** | 10 Mbps Outbound Upload | 25–50 Mbps Outbound Upload *(Gigabit LAN / Wi-Fi 6)* | 100 Mbps+ Dedicated Low-Latency Station Fiber / Ethernet |
| **Operating System** | Linux (Ubuntu 22.04 / 24.04, Arch, Debian) or macOS / Windows 10/11 with WSL2 | Linux (Ubuntu 22.04 LTS recommended) | Linux (Ubuntu 22.04 / 24.04 Server with CUDA 12.x) |

---

### ⚙️ Performance Tuning & Optimization Guidelines

1. **GPU Acceleration (CUDA / TensorRT)**:
   - On NVIDIA hardware with CUDA installed, PyTorch runs YOLO inference in **8–15 ms** (~60–100 FPS capacity), freeing the CPU entirely for video ingestion and streaming.
2. **Camera Stride Tuning (`stride: 2` or `stride: 3`)**:
   - In `camera_registry.json`, setting `"stride": 2` executes YOLO inference on every alternate frame and interpolates, reducing compute load by **50%** while maintaining smooth visual tracking.
3. **Stream Resolution**:
   - Ingest CCTV feeds at **720p (1280×720)** rather than 1080p or 4K. 720p provides the ideal balance between detection accuracy and inference throughput.
4. **Intel OpenVINO / ONNX Acceleration (for CPU/iGPU)**:
   - For Intel CPUs or integrated Iris Xe graphics, exporting the model via `yolo export format=openvino` accelerates inference by **2x to 3x** on the integrated GPU.

