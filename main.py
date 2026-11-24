#!/usr/bin/env python3
import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import socket
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

try:
    from config import (
        DEFAULT_SOURCE,
        DEFAULT_HOST,
        DEFAULT_PORT,
        DEFAULT_CONF,
        DEFAULT_IOU,
        DEFAULT_STRIDE,
        DEFAULT_SMOOTH,
        DEFAULT_ZONE,
        DEFAULT_HEATMAP,
        DEFAULT_CLUSTERS,
        STATIC_DIR,
        STATION_NAME,
    )
    from detector import CrowdDetector
    from camera import CameraManager, CameraNode, generate_placeholder_frame
except ImportError:
    from backend.config import (
        DEFAULT_SOURCE,
        DEFAULT_HOST,
        DEFAULT_PORT,
        DEFAULT_CONF,
        DEFAULT_IOU,
        DEFAULT_STRIDE,
        DEFAULT_SMOOTH,
        DEFAULT_ZONE,
        DEFAULT_HEATMAP,
        DEFAULT_CLUSTERS,
        STATIC_DIR,
        STATION_NAME,
    )
    from backend.detector import CrowdDetector
    from backend.camera import CameraManager, CameraNode

# Global Central Multi-Camera Manager instance
manager: Optional[CameraManager] = None


class CleanStreamingResponse(StreamingResponse):
    """
    Subclass of Starlette StreamingResponse that cleanly suppresses
    asyncio.CancelledError during server shutdown or client disconnection,
    preventing noisy ASGI application exception tracebacks in the terminal.
    """
    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except (asyncio.CancelledError, GeneratorExit):
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    if manager is None:
        detector = CrowdDetector(conf=DEFAULT_CONF, iou=DEFAULT_IOU)
        manager = CameraManager(detector=detector)
    yield
    # Shutdown: Cleanly release all camera pipelines and worker threads
    if manager:
        print("[Shutdown] Releasing camera hardware and worker threads...")
        manager.stop_all()


app = FastAPI(
    title="Vigilo On-Premise Multi-Camera Edge Server",
    description="Central Station Edge Brain — Ingests and analyzes multiple CCTV feeds simultaneously with local admin controls",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for Next.js web application and external station tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# ADMIN PANEL & STATIC ROUTES
# =============================================================================

@app.get("/")
def serve_index():
    """
    Serves the lightweight local control room admin panel for testing and operations.
    """
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "Vigilo On-Premise Multi-Camera Edge Server",
        "station": STATION_NAME,
        "status": "OPERATIONAL",
        "admin_ui": "/admin",
        "api_docs": "/docs",
    }


@app.get("/admin")
def serve_admin():
    """Direct route for the control center admin panel."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"error": "Admin template not found"})


# =============================================================================
# SYSTEM HEALTH & STATION OVERVIEW ENDPOINTS
# =============================================================================

@app.get("/health")
def healthcheck():
    """System health check endpoint."""
    if manager is None:
        return {"status": "starting", "manager_running": False}
    stats = manager.get_overview_stats()
    return {
        "status": "healthy",
        "manager_running": True,
        "station_name": stats["station_name"],
        "total_cameras": stats["total_cameras"],
        "active_cameras": stats["active_cameras"],
        "online_cameras": stats["online_cameras"],
        "active_surge_alerts": stats["active_surge_alerts_count"],
    }


@app.get("/api/overview")
def get_station_overview():
    """
    Station-Wide Aggregated Telemetry:
    - Total pedestrians detected across all concurrent CCTV feeds
    - Multi-camera surge warnings and bottleneck chokepoints
    - Active vs Standby camera counts
    - Public broadcast visibility approvals
    """
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    return manager.get_overview_stats()


# =============================================================================
# MULTI-CAMERA MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/cameras")
def list_cameras():
    """Lists all registered cameras with real-time operational status and metrics."""
    if manager is None:
        return []
    return manager.list_cameras()


class CameraRegisterRequest(BaseModel):
    name: str
    source: Any
    id: Optional[str] = None
    zone: Optional[str] = DEFAULT_ZONE
    coordinates: Optional[Dict[str, Any]] = None
    active: Optional[bool] = True
    approved_for_public: Optional[bool] = False
    show_heatmap: Optional[bool] = DEFAULT_HEATMAP
    show_clusters: Optional[bool] = DEFAULT_CLUSTERS
    stride: Optional[int] = DEFAULT_STRIDE


@app.post("/api/cameras")
def register_camera(req: CameraRegisterRequest):
    """
    Registers a new CCTV camera with its physical location coordinates and ingestion source.
    Immediately spins up concurrent zero-latency ingestion and AI inference.
    """
    global manager
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})

    node = manager.register_camera(
        name=req.name,
        source=req.source,
        camera_id=req.id,
        zone=req.zone or DEFAULT_ZONE,
        coordinates=req.coordinates,
        active=req.active if req.active is not None else True,
        approved_for_public=bool(req.approved_for_public),
        show_heatmap=req.show_heatmap if req.show_heatmap is not None else True,
        show_clusters=req.show_clusters if req.show_clusters is not None else True,
        stride=req.stride or DEFAULT_STRIDE,
    )
    return node.get_stats()


@app.get("/api/cameras/{camera_id}")
def get_camera_details(camera_id: str):
    """Returns real-time telemetry and metadata for a specific camera node."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})
    return node.get_stats()


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    source: Optional[Any] = None
    zone: Optional[str] = None
    coordinates: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None
    approved_for_public: Optional[bool] = None
    show_heatmap: Optional[bool] = None
    show_clusters: Optional[bool] = None


@app.put("/api/cameras/{camera_id}")
@app.patch("/api/cameras/{camera_id}")
def update_camera(camera_id: str, req: CameraUpdateRequest):
    """Updates camera configuration (coordinates, zone, name, stream source, active state)."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})

    node = manager.update_camera(
        camera_id=camera_id,
        name=req.name,
        source=req.source,
        zone=req.zone,
        coordinates=req.coordinates,
        active=req.active,
        approved_for_public=req.approved_for_public,
        show_heatmap=req.show_heatmap,
        show_clusters=req.show_clusters,
    )
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})
    return node.get_stats()


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    """Safely halts and deregisters a CCTV camera feed."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    success = manager.delete_camera(camera_id)
    if not success:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})
    return {"status": "success", "message": f"Camera {camera_id} deregistered"}


# =============================================================================
# PER-CAMERA STREAM & CONTROL ENDPOINTS
# =============================================================================

@app.get("/api/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str, request: Request):
    """
    Continuous Multipart MJPEG video stream for a specific camera feed with
    overlaid 2D ground density heatmaps and spatial clustering polygons.
    """
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})

    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})

    async def frame_generator():
        try:
            while node.running:
                if await request.is_disconnected():
                    break

                frame_bytes = node.get_latest_frame()
                if frame_bytes is None:
                    await asyncio.sleep(0.04)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                # Rate pacing ~30-40 fps cap per client connection
                await asyncio.sleep(0.025)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return CleanStreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cameras/{camera_id}/stats")
def get_camera_stats(camera_id: str):
    """Returns live JSON telemetry for a specific camera."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})
    return node.get_stats()


@app.post("/api/cameras/{camera_id}/toggle")
def toggle_camera_power(camera_id: str):
    """Toggles power-saving standby mode on/off for a specific camera."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})
    is_active = node.toggle()
    manager.save_registry()
    return {
        "status": "success",
        "camera_id": camera_id,
        "active": is_active,
        "is_suspended": not is_active,
        "message": f"Camera {camera_id} is now {'ACTIVE' if is_active else 'STANDBY'}"
    }


class PublicApprovalRequest(BaseModel):
    approved: Optional[bool] = None


@app.post("/api/cameras/{camera_id}/public")
def set_camera_public_approval(camera_id: str, req: Optional[PublicApprovalRequest] = None):
    """
    Sets or toggles control room approval for public station visibility.
    """
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})

    if req and req.approved is not None:
        new_val = node.set_public_approval(req.approved)
    else:
        new_val = node.set_public_approval(not node.approved_for_public)

    manager.save_registry()
    return {
        "status": "success",
        "camera_id": camera_id,
        "approved_for_public": new_val,
        "message": f"Camera {camera_id} public visibility set to: {new_val}"
    }


class DisplayOptionsRequest(BaseModel):
    heatmap: Optional[bool] = None
    clusters: Optional[bool] = None


@app.post("/api/cameras/{camera_id}/options")
def set_camera_options(
    camera_id: str,
    heatmap: Optional[bool] = None,
    clusters: Optional[bool] = None,
    req: Optional[DisplayOptionsRequest] = None
):
    """Toggles 2D ground heatmap and spatial clustering overlays for a specific camera."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})
    node = manager.get_camera(camera_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": f"Camera {camera_id} not found"})

    h_val = req.heatmap if (req and req.heatmap is not None) else heatmap
    c_val = req.clusters if (req and req.clusters is not None) else clusters
    node.set_display_options(heatmap=h_val, clusters=c_val)
    manager.save_registry()

    return {
        "status": "ok",
        "camera_id": camera_id,
        "heatmap_enabled": node.show_heatmap,
        "clusters_enabled": node.show_clusters
    }


class TestConnectionRequest(BaseModel):
    ip: str
    port: int
    timeout: Optional[float] = 2.0


@app.post("/api/cameras/test")
@app.post("/camera/test")
def test_camera_connection(req: TestConnectionRequest):
    """
    Rapidly tests TCP socket reachability for an IP camera host and port.
    Prevents locking or configuring non-existent CCTV nodes.
    """
    t0 = time.time()
    try:
        sock = socket.create_connection((req.ip.strip(), req.port), timeout=req.timeout or 2.0)
        sock.close()
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {
            "reachable": True,
            "ip": req.ip.strip(),
            "port": req.port,
            "latency_ms": latency_ms,
            "message": f"Online: successfully reached {req.ip}:{req.port} in {latency_ms}ms"
        }
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return {
            "reachable": False,
            "ip": req.ip.strip(),
            "port": req.port,
            "latency_ms": latency_ms,
            "error": str(e),
            "message": f"Unreachable: failed to reach {req.ip}:{req.port} ({e})"
        }


# =============================================================================
# BACKWARD COMPATIBILITY ENDPOINTS (Single-camera legacy contracts)
# =============================================================================

@app.get("/video_feed")
async def legacy_video_feed(request: Request, camera_id: Optional[str] = None):
    """
    Backward-compatible MJPEG video feed. Defaults to the primary active camera
    or accepts an optional ?camera_id=xxx query parameter.
    If no cameras are registered, yields an idle standby placeholder.
    """
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Edge Brain Manager not initialized"})

    target_node = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target_node:
        idle_frame = generate_placeholder_frame(
            title="EDGE BRAIN IDLE",
            subtitle="No cameras registered. Server in idle standby.",
            status="IDLE",
            badge_color=(100, 116, 139)
        )
        async def idle_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + idle_frame + b"\r\n"
                    )
                    await asyncio.sleep(0.5)
            except (asyncio.CancelledError, GeneratorExit):
                pass

        return CleanStreamingResponse(
            idle_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    async def frame_generator():
        try:
            while target_node.running:
                if await request.is_disconnected():
                    break

                frame_bytes = target_node.get_latest_frame()
                if frame_bytes is None:
                    await asyncio.sleep(0.04)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                await asyncio.sleep(0.025)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return CleanStreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
def legacy_get_stats(camera_id: Optional[str] = None):
    """
    Backward-compatible telemetry endpoint polled by existing frontend dashboards.
    """
    if manager is None:
        return {
            "status": "INITIALIZING",
            "is_suspended": False,
            "is_crowd": False,
            "crowd_probability": 0.0,
            "people_count": 0,
            "density_per_megapixel": 0.0,
            "nearest_neighbor_distance": 0.0,
            "dense_clusters_count": 0,
            "gatherings_count": 0,
            "isolated_count": 0,
            "heatmap_enabled": True,
            "clusters_enabled": True,
            "fps": 0.0,
            "capture_fps": 0.0,
            "camera_connected": False,
            "resolution": "Offline",
            "camera_source": "Standby",
            "zone": DEFAULT_ZONE,
            "uptime_seconds": 0
        }

    target_node = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target_node:
        return {"status": "NO_CAMERAS", "active": False}

    stats = target_node.get_stats()
    # Merge station overview for enhanced frontend insight
    overview = manager.get_overview_stats()
    stats["station_overview"] = overview
    return stats


@app.get("/camera/info")
def legacy_get_camera_info(camera_id: Optional[str] = None):
    """Returns camera hardware and zone telemetry for primary camera."""
    stats = legacy_get_stats(camera_id)
    return {
        "source": stats.get("source", "Unknown"),
        "raw_source": stats.get("raw_source", "Unknown"),
        "type": stats.get("camera_type", "unknown"),
        "ip": stats.get("ip"),
        "port": stats.get("port"),
        "resolution": stats.get("resolution", "Standby"),
        "zone": stats.get("zone", DEFAULT_ZONE),
        "capture_fps": stats.get("capture_fps", 0.0),
        "connected": stats.get("camera_connected", False),
        "is_suspended": stats.get("is_suspended", False),
        "protocol": "Multi-Camera Zero-Latency Ingestion",
        "mesh_latency_ms": 12,
        "ai_pipeline": "YOLO26m + 21-Feature Random Forest + Spatial Clustering",
        "uptime_seconds": stats.get("uptime_seconds", 0)
    }


@app.get("/camera/sources")
def legacy_get_camera_sources():
    """Returns list of registered cameras and preset CCTV nodes."""
    if manager is None:
        return {"current": {}, "cameras": []}
    primary = manager.get_primary_camera()
    return {
        "current": primary.get_stats() if primary else {},
        "cameras": manager.list_cameras()
    }


@app.post("/camera/suspend")
def legacy_suspend_camera(camera_id: Optional[str] = None):
    """Suspends primary or specified camera capture device."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Manager not initialized"})
    target = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target:
        return JSONResponse(status_code=404, content={"error": "No camera available"})
    target.suspend()
    manager.save_registry()
    return {"status": "success", "suspended": True}


@app.post("/camera/resume")
def legacy_resume_camera(camera_id: Optional[str] = None):
    """Resumes primary or specified camera capture device."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Manager not initialized"})
    target = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target:
        return JSONResponse(status_code=404, content={"error": "No camera available"})
    target.resume()
    manager.save_registry()
    return {"status": "success", "suspended": False}


@app.post("/camera/toggle")
def legacy_toggle_camera(camera_id: Optional[str] = None):
    """Toggles suspended and active states on primary or specified camera."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Manager not initialized"})
    target = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target:
        return JSONResponse(status_code=404, content={"error": "No camera available"})
    is_active = target.toggle()
    manager.save_registry()
    return {"status": "success", "suspended": not is_active, "active": is_active}


@app.post("/camera/options")
@app.get("/camera/options")
@app.post("/toggle")
@app.get("/toggle")
def legacy_toggle_display(
    heatmap: Optional[bool] = None,
    clusters: Optional[bool] = None,
    req: Optional[DisplayOptionsRequest] = None,
    camera_id: Optional[str] = None
):
    """Dynamically enables or disables ground density heatmap and spatial cluster polygons."""
    if manager is None:
        return JSONResponse(status_code=503, content={"error": "Manager not initialized"})
    target = manager.get_camera(camera_id) if camera_id else manager.get_primary_camera()
    if not target:
        return JSONResponse(status_code=404, content={"error": "No camera available"})

    h_val = req.heatmap if (req and req.heatmap is not None) else heatmap
    c_val = req.clusters if (req and req.clusters is not None) else clusters
    target.set_display_options(heatmap=h_val, clusters=c_val)
    manager.save_registry()
    return {
        "status": "ok",
        "heatmap_enabled": target.show_heatmap,
        "clusters_enabled": target.show_clusters
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vigilo On-Premise Multi-Camera Edge Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="YOLO IoU threshold")
    return parser.parse_args()


def main():
    global manager
    args = parse_args()

    print()
    print("=" * 80)
    print(" VIGILO — ON-PREMISE MULTI-CAMERA EDGE SERVER & CONTROL ROOM BRAIN")
    print("=" * 80)
    print(f" Station Hub      : {STATION_NAME}")
    print(f" Network Scope    : AIR-GAPPED / LOCAL ON-PREM STATION CONTROL CENTER")
    print(f" Control Room UI  : http://localhost:{args.port}/ or http://localhost:{args.port}/admin")
    print(f" Station Overview : http://localhost:{args.port}/api/overview")
    print(f" Multi-Cam API    : http://localhost:{args.port}/api/cameras")
    print(f" Legacy Stream    : http://localhost:{args.port}/video_feed")
    print(f" Legacy Stats     : http://localhost:{args.port}/stats")
    print("=" * 80)
    print()

    # Initialize shared detector and camera manager
    detector = CrowdDetector(conf=args.conf, iou=args.iou)
    manager = CameraManager(detector=detector)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)

    original_handle_exit = server.handle_exit

    def handle_exit(sig: int, frame) -> None:
        if manager:
            manager.stop_all()
        original_handle_exit(sig, frame)

    server.handle_exit = handle_exit
    server.run()


if __name__ == "__main__":
    main()
