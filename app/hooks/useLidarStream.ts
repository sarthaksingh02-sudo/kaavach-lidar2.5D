// app/hooks/useLidarStream.ts
"use client";

import { useEffect, useRef, useCallback } from "react";

// ─── Interface 3 payload schema from Saksham's backend ──────────────────────
// grid_data row: [x, y, zMax, deltaZ, label_id (0-3), radius]
export type CellTuple = [number, number, number, number, string, number];

export interface TelemetryMetrics {
    engine: "CUDA GPU TIER 1" | "CPU OPENMP FALLBACK";
    fps: number;
    latencyMs: number;
    memorySavedPct: number;
}

export interface ThreatObject {
    type: string;          // "NEGATIVE_OBSTACLE" | "CLOSE_RANGE" | etc.
    distance_m: number;
    coordinates: [number, number];
    depth_m?: number;
}

// ─── Raw backend payload (Interface 3) ──────────────────────────────────────
interface BackendPayload {
    header?: {
        frame_id: number;
        timestamp: number;
        active_engine: string; // "CUDA_TIER_1" | "NUMBA_TIER_2" | "PYTHON_FALLBACK"
    };
    telemetry?: {
        fps: number;
        latency_ms: number;
        raw_points_count: number;
        compressed_cells_count: number;
        memory_saved_percent: number;
    };
    grid_data?: Array<[number, number, number, number, number | string, number]>;
    threats?: ThreatObject[];
    // Legacy mock-server fields (kept for backward compat)
    cells?: Array<[number, number, number, number, string, number]>;
}

// ─── Label ID → semantic string map ─────────────────────────────────────────
const LABEL_ID_MAP: Record<number, string> = {
    0: "road",
    1: "static_obstacle",
    2: "dynamic_target",
    3: "pothole",
};

function resolveLabel(raw: number | string): string {
    if (typeof raw === "string") return raw;
    return LABEL_ID_MAP[raw] ?? "unknown";
}

function normalizeEngine(
    tier: string | undefined
): TelemetryMetrics["engine"] {
    if (!tier) return "CPU OPENMP FALLBACK";
    const t = tier.toUpperCase();
    if (t.includes("CUDA") || t.includes("TIER_1") || t.includes("TIER 1"))
        return "CUDA GPU TIER 1";
    return "CPU OPENMP FALLBACK";
}

/** Transform raw backend payload → normalized frontend types */
function adaptPayload(raw: BackendPayload): {
    cells: CellTuple[];
    telemetry: TelemetryMetrics;
    threats: ThreatObject[];
} | null {
    // Support both real-backend schema and legacy mock-server schema
    let cells: CellTuple[] = [];

    if (raw.grid_data && raw.grid_data.length > 0) {
        cells = raw.grid_data.map(([x, y, zMax, deltaZ, label, radius]) => [
            x,
            y,
            zMax,
            deltaZ,
            resolveLabel(label as number | string),
            radius,
        ]);
    } else if (raw.cells && raw.cells.length > 0) {
        // Legacy mock_server.py schema
        cells = raw.cells;
    }

    const t = raw.telemetry;
    const header = raw.header;

    const telemetry: TelemetryMetrics = {
        engine: normalizeEngine(header?.active_engine),
        fps: t?.fps ?? 0,
        latencyMs: t?.latency_ms ?? 0,
        memorySavedPct: t?.memory_saved_percent ?? 0,
    };

    return { cells, telemetry, threats: raw.threats ?? [] };
}

// ─── Hook options ────────────────────────────────────────────────────────────
export interface UseLidarStreamOptions {
    url?: string;
    onCells?: (cells: CellTuple[]) => void;
    onTelemetry?: (metrics: TelemetryMetrics) => void;
    onThreats?: (threats: ThreatObject[]) => void;
    reconnectDelayMs?: number;
}

/**
 * Connects to the Kavach WebSocket backend and streams normalized 2.5D data.
 *
 * Supports both:
 *   - Real backend: ws://localhost:8000/ws/stream_map (FastAPI)
 *   - Legacy mock:  mock_server.py
 *
 * Uses refs internally for 30+ FPS streaming without React re-render lag.
 */
export function useLidarStream({
    url = "ws://localhost:8000/ws/stream_map",
    onCells,
    onTelemetry,
    onThreats,
    reconnectDelayMs = 2000,
}: UseLidarStreamOptions = {}) {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isMounted = useRef(true);

    // Stable callback refs — never re-subscribe on callback change
    const onCellsRef = useRef(onCells);
    const onTelemetryRef = useRef(onTelemetry);
    const onThreatsRef = useRef(onThreats);
    useEffect(() => {
        onCellsRef.current = onCells;
        onTelemetryRef.current = onTelemetry;
        onThreatsRef.current = onThreats;
    }, [onCells, onTelemetry, onThreats]);

    const connect = useCallback(() => {
        if (!isMounted.current) return;

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onmessage = (event: MessageEvent) => {
                try {
                    const raw: BackendPayload =
                        typeof event.data === "string"
                            ? JSON.parse(event.data)
                            : JSON.parse(new TextDecoder().decode(event.data as ArrayBuffer));

                    const adapted = adaptPayload(raw);
                    if (!adapted) return;

                    if (adapted.cells.length && onCellsRef.current) {
                        onCellsRef.current(adapted.cells);
                    }
                    if (onTelemetryRef.current) {
                        onTelemetryRef.current(adapted.telemetry);
                    }
                    if (adapted.threats.length && onThreatsRef.current) {
                        onThreatsRef.current(adapted.threats);
                    }
                } catch {
                    // Silently drop malformed frames at high FPS
                }
            };

            ws.onclose = () => {
                if (!isMounted.current) return;
                reconnectTimer.current = setTimeout(connect, reconnectDelayMs);
            };

            ws.onerror = () => {
                ws.close();
            };
        } catch {
            if (isMounted.current) {
                reconnectTimer.current = setTimeout(connect, reconnectDelayMs);
            }
        }
    }, [url, reconnectDelayMs]);

    useEffect(() => {
        isMounted.current = true;
        connect();

        return () => {
            isMounted.current = false;
            if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
            wsRef.current?.close();
        };
    }, [connect]);

    return {
        reconnect: () => wsRef.current?.close(),
    };
}
