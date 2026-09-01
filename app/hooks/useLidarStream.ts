// app/hooks/useLidarStream.ts
"use client";

import { useEffect, useRef, useCallback } from "react";

// [x, y, zMax, deltaZ, label, radius]
export type CellTuple = [number, number, number, number, string, number];

export interface TelemetryMetrics {
    engine: "CUDA GPU TIER 1" | "CPU OPENMP FALLBACK";
    fps: number;
    latencyMs: number;
    memorySavedPct: number;
}

export interface LidarPayload {
    cells: CellTuple[];
    telemetry: TelemetryMetrics;
}

export interface UseLidarStreamOptions {
    url?: string;
    onCells?: (cells: CellTuple[]) => void;
    onTelemetry?: (metrics: TelemetryMetrics) => void;
    reconnectDelayMs?: number;
}

/**
 * Connects to a WebSocket LiDAR stream and calls callbacks with parsed data.
 * Uses refs internally so callbacks can read the latest data without
 * triggering React re-renders on every incoming frame — safe at 30+ FPS.
 */
export function useLidarStream({
    url = "ws://localhost:8000/ws/stream_map",
    onCells,
    onTelemetry,
    reconnectDelayMs = 2000,
}: UseLidarStreamOptions = {}) {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isMounted = useRef(true);

    // Keep callbacks stable in refs so the WebSocket handler never stales
    const onCellsRef = useRef(onCells);
    const onTelemetryRef = useRef(onTelemetry);
    useEffect(() => {
        onCellsRef.current = onCells;
        onTelemetryRef.current = onTelemetry;
    }, [onCells, onTelemetry]);

    const connect = useCallback(() => {
        if (!isMounted.current) return;

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onmessage = (event: MessageEvent) => {
                try {
                    const payload: LidarPayload = JSON.parse(event.data as string);
                    if (payload.cells && onCellsRef.current) {
                        onCellsRef.current(payload.cells);
                    }
                    if (payload.telemetry && onTelemetryRef.current) {
                        onTelemetryRef.current(payload.telemetry);
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
        /** Manually trigger a disconnect + reconnect */
        reconnect: () => {
            wsRef.current?.close();
        },
    };
}
