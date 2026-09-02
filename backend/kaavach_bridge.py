#!/usr/bin/env python3
"""
kaavach_bridge.py — Simple 5 FPS inference + stream loop.
Each polar cell is sent as a wedge polygon (list of [x,y] vertices)
so DeckGL PolygonLayer renders proper radar-sector shapes.
"""

import asyncio
import json
import math
import time
import numpy as np
import torch
import websockets
from pathlib import Path
from collections import Counter, deque

from cylinder3d_net import CylinderVoxelizer, GRID_SIZE, build_model

# ── Config ─────────────────────────────────────────────────────────────────────
WEIGHTS  = "weights/model_load.pth"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCAN_DIR = Path("data/sequences/00/velodyne")

FPS              = 5
FRAME_INTERVAL   = 1.0 / FPS   # 200 ms

HOST  = "localhost"
PORT  = 8000
ROUTE = "/ws/stream_map"

# Remap 20 SemanticKITTI classes → 3 Kavach classes
REMAP = torch.tensor(
    [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    dtype=torch.uint8,
)

# Cartesian grid config
GRID_X_MIN, GRID_X_MAX = -40.0, 40.0
GRID_Y_MIN, GRID_Y_MAX = -20.0, 20.0
CELL_SIZE = 2.0  # meters per cell edge
N_COLS = int((GRID_X_MAX - GRID_X_MIN) / CELL_SIZE)
N_ROWS = int((GRID_Y_MAX - GRID_Y_MIN) / CELL_SIZE)

# Temporal label smoother — per cell, vote over last N frames
HISTORY_LEN = 10
cell_history: dict[tuple, deque] = {}


# ─────────────────────────────────────────────────────────────────────────────
def box_polygon(x0: float, y0: float, x1: float, y1: float) -> list:
    """Return [x,y] polygon vertices for a rectangular cell."""
    return [
        [round(x0, 3), round(y0, 3)],
        [round(x1, 3), round(y0, 3)],
        [round(x1, 3), round(y1, 3)],
        [round(x0, 3), round(y1, 3)]
    ]


def load_scan_list() -> list[Path]:
    scans = sorted(SCAN_DIR.glob("*.bin"))
    if not scans:
        raise FileNotFoundError(f"No .bin files found in {SCAN_DIR.resolve()}")
    print(f"[ScanLoader] {len(scans):,} scans ready from {SCAN_DIR}")
    return scans


def read_bin(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


@torch.inference_mode()
def run_inference(model, voxelizer, points: np.ndarray) -> np.ndarray:
    pt_fea, grid_ind = voxelizer(points)
    logits     = model([pt_fea.to(DEVICE)], [grid_ind.to(DEVICE)], batch_size=1)
    vox_labels = logits[0].argmax(dim=0)
    gx = grid_ind[:, 0].long().clamp(0, GRID_SIZE[0] - 1).to(DEVICE)
    gy = grid_ind[:, 1].long().clamp(0, GRID_SIZE[1] - 1).to(DEVICE)
    gz = grid_ind[:, 2].long().clamp(0, GRID_SIZE[2] - 1).to(DEVICE)
    pt_sem    = vox_labels[gx, gy, gz]
    pt_labels = REMAP.to(DEVICE)[pt_sem]
    return pt_labels.cpu().numpy()


def build_grid(points: np.ndarray, labels: np.ndarray) -> list:
    """
    Project points into N_RINGS × N_SECTORS polar grid.
    Fuses Cylinder3D AI model labels with geometry heuristics for stability.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Remove glass/sky reflections
    good = (z > -3.5) & (z < 8.0) & (np.sqrt(x**2 + y**2) > 0.5)
    x, y, z, labels = x[good], y[good], z[good], labels[good]

    # Cartesian index
    c_idx = np.floor((x - GRID_X_MIN) / CELL_SIZE).astype(int)
    r_idx = np.floor((y - GRID_Y_MIN) / CELL_SIZE).astype(int)

    valid = (c_idx >= 0) & (c_idx < N_COLS) & (r_idx >= 0) & (r_idx < N_ROWS)
    if not np.any(valid):
        return []

    flat_idx = r_idx[valid] * N_COLS + c_idx[valid]
    z_v      = z[valid]
    lb_v     = labels[valid]

    cells = []
    for fid in np.unique(flat_idx):
        mask    = flat_idx == fid
        cell_z  = z_v[mask]
        cell_lb = lb_v[mask]
        if len(cell_z) < 5:       # too sparse, skip
            continue

        ri = int(fid) // N_COLS
        ci = int(fid) % N_COLS

        z_min   = float(cell_z.min())
        z_max   = float(cell_z.max())
        delta_z = z_max - z_min

        # The AI's dominant vote for this cell (0: road/unlabeled, 1: static, 2: dynamic)
        ml_vote = Counter(cell_lb.tolist()).most_common(1)[0][0]

        # ── AI + Geometry Fusion ──────────────────────────────────────────
        if ml_vote == 2:
            raw_label = "dynamic_target"   # AI detected car/person/cyclist
        elif ml_vote == 1 and delta_z > 1.2:
            raw_label = "static_obstacle"  # AI detected building/tree
        elif delta_z > 2.0:
            raw_label = "static_obstacle"  # Fallback for tall unregistered objects
        elif delta_z > 1.0 and z_max > -0.5:
            raw_label = "dynamic_target"   # Fallback for car-sized objects
        else:
            raw_label = "road"             # Default to road for flat ground


        # ── Temporal smoothing: 10-frame majority vote ────────────────────
        key = (ri, ci)
        if key not in cell_history:
            cell_history[key] = deque(maxlen=HISTORY_LEN)
        cell_history[key].append(raw_label)
        label = Counter(cell_history[key]).most_common(1)[0][0]

        # Elevation: road → thin slab, obstacles → proportional height
        if label == "road":
            elev = 0.2
        else:
            elev = round(max(0.4, min(delta_z, 5.0)), 3)

        # Rectangular polygon
        x0 = GRID_X_MIN + ci * CELL_SIZE
        y0 = GRID_Y_MIN + ri * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE
        polygon = box_polygon(x0, y0, x1, y1)

        cx = x0 + CELL_SIZE / 2
        cy = y0 + CELL_SIZE / 2

        cells.append({
            "polygon": polygon,
            "elev":    elev,
            "delta_z": round(delta_z, 3),
            "label":   label,
            "cx":      round(cx, 2),
            "cy":      round(cy, 2),
        })

    return cells


# ─────────────────────────────────────────────────────────────────────────────
CLIENTS: set = set()


async def stream_handler(websocket):
    if websocket.request.path != ROUTE:
        await websocket.close(code=4004, reason="Unknown route")
        return
    CLIENTS.add(websocket)
    print(f"[+] {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"[-] {websocket.remote_address}")


async def inference_loop(model, voxelizer, scans):
    frame = 1350  # Start deeper into Sequence 00 to immediately see busy intersections
    while True:
        t0        = time.perf_counter()
        path      = scans[frame % len(scans)]
        frame    += 1
        points    = read_bin(path)
        labels    = run_inference(model, voxelizer, points)  # Active AI Inference
        cells     = build_grid(points, labels)
        inf_ms    = (time.perf_counter() - t0) * 1000

        memory_saved = 100.0 * (1.0 - (len(cells) / max(1, len(points))))

        payload = {
            "header": {
                "frame_id":      frame,
                "timestamp":     time.time(),
                "active_engine": "CUDA_TIER_1",
            },
            "telemetry": {
                "fps":                    FPS,
                "latency_ms":             round(inf_ms, 1),
                "raw_points_count":       len(points),
                "compressed_cells_count": len(cells),
                "memory_saved_percent":   round(memory_saved, 1),
            },
            "grid_data": cells,
            "threats":   [],
        }

        msg = json.dumps(payload)
        if CLIENTS:
            await asyncio.gather(
                *[ws.send(msg) for ws in list(CLIENTS)],
                return_exceptions=True,
            )

        print(
            f"  frame {frame:04d}  pts={len(points):,}  cells={len(cells)}"
            f"  {inf_ms:.0f}ms",
            end="\r",
        )

        # Pace to target FPS (subtract inference time already spent)
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


async def main():
    print("=" * 60)
    print(f"  Kavach x Cylinder3D  -  {FPS} FPS simple loop")
    print(f"  ws://{HOST}:{PORT}{ROUTE}")
    print("=" * 60)

    print("[1/3] Loading Cylinder3D model...")
    model     = build_model(WEIGHTS, DEVICE)
    voxelizer = CylinderVoxelizer()

    print("[2/3] Indexing scans...")
    scans = load_scan_list()

    print("[3/3] Starting server...")
    async with websockets.serve(stream_handler, HOST, PORT):
        await inference_loop(model, voxelizer, scans)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[~] Bridge stopped.")
