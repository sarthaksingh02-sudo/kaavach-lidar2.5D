#!/usr/bin/env python3
"""
mock_server.py — Kavach 2.5D LiDAR Mock WebSocket Server
=========================================================
Streams synthetic 2.5D polar-grid JSON data at ~30 FPS to
  ws://localhost:8000/ws/stream_map

Usage:
    pip install websockets
    python mock_server.py

The payload schema per frame:
{
  "cells": [[x, y, zMax, deltaZ, label, radius], ...],
  "telemetry": {
    "engine": "CUDA GPU TIER 1" | "CPU OPENMP FALLBACK",
    "fps": float,
    "latencyMs": float,
    "memorySavedPct": float
  }
}
"""

import asyncio
import json
import math
import random
import time
import sys
from typing import Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("[ERROR] websockets not installed. Run:  pip install websockets")
    sys.exit(1)

HOST = "localhost"
PORT = 8000
ROUTE = "/ws/stream_map"
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

# ─── Label pool ────────────────────────────────────────────────────────────────
LABELS = ["road", "static_obstacle", "dynamic_target", "pothole", "unknown"]
LABEL_WEIGHTS = [0.50, 0.20, 0.15, 0.10, 0.05]

# ─── Grid parameters ───────────────────────────────────────────────────────────
N_RINGS = 12          # radial rings
N_SECTORS = 24        # angular sectors per ring
BASE_RING_RADIUS = 3.0  # metres per ring step


def _generate_cells(tick: int) -> list[list[Any]]:
    """Build one frame of synthetic 2.5D polar-grid cells."""
    cells: list[list[Any]] = []
    phase = tick * 0.06  # slow wave animation

    for ring in range(1, N_RINGS + 1):
        for sector in range(N_SECTORS):
            angle_deg = (sector / N_SECTORS) * 360.0
            angle_rad = math.radians(angle_deg)

            # Polar → Cartesian
            base_r = ring * BASE_RING_RADIUS
            x = base_r * math.cos(angle_rad)
            y = base_r * math.sin(angle_rad)

            # Randomized elevation with wave ripple
            z_base = random.uniform(0.05, 0.3)
            wave = 0.4 * math.sin(phase + ring * 0.5 + sector * 0.25)
            z_noise = random.gauss(0, 0.05)
            z_max = max(0.05, z_base + wave + z_noise)

            # Delta Z: mostly small, rare spikes
            if random.random() < 0.04:
                delta_z = random.uniform(0.6, 2.5)   # spike → triggers threat
            elif random.random() < 0.03:
                delta_z = random.uniform(-1.5, -0.6)
            else:
                delta_z = random.uniform(-0.3, 0.3)

            # Label
            label = random.choices(LABELS, weights=LABEL_WEIGHTS, k=1)[0]

            # Assign close radius to dynamic_target occasionally
            if label == "dynamic_target" and random.random() < 0.25:
                radius = random.uniform(1.0, 4.5)  # close range → triggers alert
            else:
                radius = base_r * random.uniform(0.9, 1.1)

            cells.append([
                round(x, 3),
                round(y, 3),
                round(z_max, 3),
                round(delta_z, 3),
                label,
                round(radius, 3),
            ])

    return cells


def _generate_telemetry(fps_actual: float) -> dict[str, Any]:
    """Build a synthetic telemetry payload."""
    # Randomly flip engine tier every ~10 s
    engine = (
        "CUDA GPU TIER 1"
        if (int(time.time()) // 10) % 3 != 0
        else "CPU OPENMP FALLBACK"
    )
    return {
        "engine": engine,
        "fps": round(fps_actual + random.gauss(0, 0.5), 1),
        "latencyMs": round(random.uniform(18.0, 65.0), 2),
        "memorySavedPct": round(random.uniform(55.0, 78.0), 1),
    }


async def stream_handler(websocket: WebSocketServerProtocol, path: str) -> None:
    if path != ROUTE:
        await websocket.close(code=4004, reason="Unknown route")
        return

    client = websocket.remote_address
    print(f"[+] Client connected — {client[0]}:{client[1]}")

    tick = 0
    last_report = time.perf_counter()
    frame_count = 0

    try:
        while True:
            t_start = time.perf_counter()

            cells = _generate_cells(tick)
            fps_actual = (
                frame_count / max(1e-9, t_start - last_report)
                if frame_count > 0
                else TARGET_FPS
            )

            payload = {
                "cells": cells,
                "telemetry": _generate_telemetry(fps_actual),
            }

            await websocket.send(json.dumps(payload, separators=(",", ":")))

            tick += 1
            frame_count += 1

            # Log every second
            if t_start - last_report >= 1.0:
                print(
                    f"  [stream] tick={tick:05d}  fps={fps_actual:.1f}"
                    f"  cells={len(cells)}  client={client[0]}"
                )
                last_report = t_start
                frame_count = 0

            # Pace to target FPS
            elapsed = time.perf_counter() - t_start
            sleep_for = FRAME_INTERVAL - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    except websockets.exceptions.ConnectionClosed as exc:
        print(f"[-] Client disconnected — {client[0]}:{client[1]} — {exc}")
    except Exception as exc:
        print(f"[!] Unexpected error: {exc}")
        raise


async def main() -> None:
    print("=" * 60)
    print("  KAVACH 2.5D Mock LiDAR WebSocket Server")
    print(f"  Listening on  ws://{HOST}:{PORT}{ROUTE}")
    print(f"  Target FPS    {TARGET_FPS}")
    print(f"  Cells/frame   {N_RINGS * N_SECTORS}")
    print("=" * 60)

    async with websockets.serve(stream_handler, HOST, PORT, max_size=2**22):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[~] Server stopped.")
