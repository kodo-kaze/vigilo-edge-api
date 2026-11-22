# Vigilo Edge API (`vigilo-edge-api`)

Real-time Computer Vision & Edge AI Ingestion Server for Pedestrian Crowd Density Intelligence.

Part of the **Vigilo** intelligent crowd flow architecture. Pairs with [`vigilo-user-interface`](https://github.com/kodo-kaze/vigilo-user-interface).

## Features
- **YOLO Person Detection**: High-performance bounding box tracking (`model/best.pt`).
- **Density-Aware Spatial Clustering**: Connected-component graph clustering detecting high-density crowd hulls ($\ge 6$), gatherings ($3–5$), and isolated pedestrians.
- **Continuous 2D Ground Heatmap**: Floor-level Gaussian thermal field with `COLORMAP_TURBO` blended onto walking corridors.
- **Conflux Random Forest**: 21 scene-level spatial features with temporal probability smoothing.
- **Zero-Latency Ingestion**: Bufferless RTSP / Webcam / File streaming.
- **Power Standby Controls**: Suspends optical sensor and neural pipeline on demand (0% compute when idle).
- **FastAPI Endpoints**:
  - `GET /video_feed`: Multipart MJPEG live stream.
  - `GET /stats`: Live JSON telemetry state (people count, density/MP, NN proximity, FPS, cluster counts).
  - `POST /camera/switch`: In-memory runtime camera switcher (IP, port, RTSP, hardware webcam, video file).
  - `POST /camera/options` & `POST /toggle`: Live toggle for ground heatmap and spatial cluster polygons.
  - `POST /camera/test`: TCP socket reachability test.
  - `POST /camera/suspend`, `resume`, `toggle`: Power-saving idle standby controls.
  - `GET /health` & `GET /camera/info`.

## Quick Start (Local)

```bash
# 1. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run server
bash start.sh
# Or directly:
python main.py --host 0.0.0.0 --port 8000
```

## Quick Start (Docker)

```bash
docker compose up -d
# Or:
docker build -t vigilo-edge-api:latest .
docker run -d -p 8000:8000 --name vigilo-edge-api vigilo-edge-api:latest
```
