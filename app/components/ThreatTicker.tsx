// app/components/ThreatTicker.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, ShieldAlert, ChevronRight, Zap } from "lucide-react";
import type { CellTuple, ThreatObject } from "../hooks/useLidarStream";

// ─── Internal event shape ────────────────────────────────────────────────────
interface ThreatEvent {
    id: string;
    timestamp: string;
    type: "DROP_OFF" | "CLOSE_RANGE" | "NEGATIVE_OBSTACLE" | "BACKEND";
    label: string;
    severity: "HIGH" | "CRITICAL";
    detail: string;
    source: "backend" | "client";
}

interface Props {
    cells: CellTuple[];
    /** Pre-detected threats from the backend's Interface 3 `threats` array */
    backendThreats?: ThreatObject[];
    maxEvents?: number;
}

let eventCounter = 0;

function buildTimestamp(): string {
    return new Date().toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 2,
    } as Intl.DateTimeFormatOptions);
}

function makeId(): string {
    return (++eventCounter).toString(16).padStart(4, "0").toUpperCase();
}

// ── Convert backend ThreatObject → ThreatEvent ──────────────────────────────
function adaptBackendThreats(threats: ThreatObject[]): ThreatEvent[] {
    const ts = buildTimestamp();
    return threats.map((t) => {
        const depth = t.depth_m ?? 0;
        const isCritical = depth > 0.8 || t.distance_m < 2;
        const [bx, by] = t.coordinates;
        return {
            id: makeId(),
            timestamp: ts,
            type: t.type === "NEGATIVE_OBSTACLE" ? "NEGATIVE_OBSTACLE" : "BACKEND",
            label: t.type.replace(/_/g, " "),
            severity: isCritical ? "CRITICAL" : "HIGH",
            detail:
                t.type === "NEGATIVE_OBSTACLE"
                    ? `depth=${depth.toFixed(2)}m  dist=${t.distance_m.toFixed(1)}m @ (${bx.toFixed(1)},${by.toFixed(1)})`
                    : `dist=${t.distance_m.toFixed(1)}m @ (${bx.toFixed(1)},${by.toFixed(1)})`,
            source: "backend",
        };
    });
}

// ── Client-side detection from grid cells (fallback / supplemental) ──────────
function detectFromCells(cells: CellTuple[]): ThreatEvent[] {
    const events: ThreatEvent[] = [];
    const ts = buildTimestamp();

    const sample = cells.length > 500
        ? cells.filter((_, i) => i % Math.ceil(cells.length / 500) === 0)
        : cells;

    for (const cell of sample) {
        // New format: { polygon, elev, delta_z, label, cx, cy }
        // Old format: [x, y, zMax, deltaZ, label, radius]
        let x: number, y: number, deltaZ: number, label: string, dist: number;

        if (Array.isArray(cell)) {
            [x, y, , deltaZ, label] = cell as [number, number, number, number, string, number];
            dist = Math.sqrt(x * x + y * y);
        } else {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const c = cell as any;
            x = c.cx ?? 0; y = c.cy ?? 0;
            deltaZ = c.delta_z ?? 0;
            label = c.label ?? "unknown";
            dist = Math.sqrt(x * x + y * y);
        }

        if (deltaZ > 0.5) {
            events.push({
                id: makeId(), timestamp: ts, type: "DROP_OFF",
                label: String(label),
                severity: deltaZ > 1.2 ? "CRITICAL" : "HIGH",
                detail: `ΔZ=+${deltaZ.toFixed(2)}m @ (${x.toFixed(1)},${y.toFixed(1)})`,
                source: "client",
            });
        }
        if (dist < 5 && String(label).toLowerCase().includes("dynamic")) {
            events.push({
                id: makeId(), timestamp: ts, type: "CLOSE_RANGE",
                label: String(label),
                severity: dist < 2 ? "CRITICAL" : "HIGH",
                detail: `r=${dist.toFixed(1)}m @ (${x.toFixed(1)},${y.toFixed(1)})`,
                source: "client",
            });
        }
    }

    return events;
}

// ── Label for event type ─────────────────────────────────────────────────────
function typeLabel(type: ThreatEvent["type"]): string {
    switch (type) {
        case "DROP_OFF": return "OBSTACLE DROP-OFF";
        case "CLOSE_RANGE": return "CLOSE DYNAMIC TARGET";
        case "NEGATIVE_OBSTACLE": return "NEGATIVE OBSTACLE";
        default: return "THREAT DETECTED";
    }
}

// ── Component ────────────────────────────────────────────────────────────────
export default function ThreatTicker({
    cells,
    backendThreats,
    maxEvents = 60,
}: Props) {
    const [events, setEvents] = useState<ThreatEvent[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Backend threats take priority — ingest them whenever the array changes
    useEffect(() => {
        if (!backendThreats?.length) return;
        const incoming = adaptBackendThreats(backendThreats);
        setEvents((prev) => [...incoming, ...prev].slice(0, maxEvents));
    }, [backendThreats, maxEvents]);

    // Client-side cell scanning — only runs when backend sends NO threats
    useEffect(() => {
        if (!cells?.length) return;
        if (backendThreats && backendThreats.length > 0) return; // backend already handles it
        const incoming = detectFromCells(cells);
        if (!incoming.length) return;
        setEvents((prev) => [...incoming, ...prev].slice(0, maxEvents));
    }, [cells, backendThreats, maxEvents]);

    // Auto-scroll to top on new event
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = 0;
    }, [events.length]);

    const critCount = events.filter((e) => e.severity === "CRITICAL").length;

    return (
        <div className="flex flex-col h-full">
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-1 mb-2">
                <div className="flex items-center gap-2">
                    <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                    <h2 className="text-xs font-mono uppercase tracking-[0.25em] text-slate-400">
                        Threat Ticker
                    </h2>
                    {critCount > 0 && (
                        <span className="flex items-center gap-0.5 text-[8px] font-mono font-bold text-red-400 animate-pulse">
                            <Zap className="w-2.5 h-2.5" />
                            {critCount} CRIT
                        </span>
                    )}
                </div>
                <span className="text-[9px] font-mono bg-red-500/20 text-red-400 border border-red-500/30 rounded px-1.5 py-0.5">
                    {events.length} EVENTS
                </span>
            </div>

            {/* ── Scrolling log ── */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto space-y-1.5 pr-1"
                style={{
                    scrollbarWidth: "thin",
                    scrollbarColor: "rgba(255,255,255,0.08) transparent",
                }}
            >
                {events.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
                        <ShieldAlert className="w-8 h-8 opacity-20" />
                        <p className="text-xs font-mono">No threats detected</p>
                        <p className="text-[9px] text-slate-700 font-mono">
                            Monitoring grid for anomalies…
                        </p>
                    </div>
                ) : (
                    events.map((ev, idx) => (
                        <div
                            key={ev.id}
                            className={`relative overflow-hidden flex items-start gap-2.5 rounded-lg border px-3 py-2 text-xs font-mono ${ev.severity === "CRITICAL"
                                ? "border-red-500/50 bg-red-950/25"
                                : "border-amber-500/30 bg-amber-950/15"
                                } ${idx === 0
                                    ? "ring-1 ring-inset " +
                                    (ev.severity === "CRITICAL"
                                        ? "ring-red-500/25"
                                        : "ring-amber-500/15")
                                    : ""
                                }`}
                            style={{
                                animation: idx === 0 ? "slideIn 0.22s ease-out" : undefined,
                            }}
                        >
                            {/* Left severity bar */}
                            <div
                                className={`absolute left-0 top-0 w-0.5 h-full ${ev.severity === "CRITICAL" ? "bg-red-500" : "bg-amber-400"
                                    }`}
                            />

                            <AlertTriangle
                                className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${ev.severity === "CRITICAL" ? "text-red-400" : "text-amber-400"
                                    }`}
                            />

                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-2 flex-wrap">
                                    <span
                                        className={`font-bold uppercase tracking-wider text-[10px] ${ev.severity === "CRITICAL"
                                            ? "text-red-300"
                                            : "text-amber-300"
                                            }`}
                                    >
                                        {typeLabel(ev.type)}
                                    </span>
                                    <span className="text-slate-600 text-[9px] flex-shrink-0 tabular-nums">
                                        {ev.timestamp}
                                    </span>
                                </div>

                                <div className="flex items-center gap-1 text-slate-500 mt-0.5">
                                    <ChevronRight className="w-2.5 h-2.5 flex-shrink-0" />
                                    <span className="text-slate-400 truncate">{ev.label}</span>
                                    <span className="text-slate-600">·</span>
                                    <span className="text-slate-500 truncate">{ev.detail}</span>
                                </div>

                                <div className="flex items-center gap-1.5 mt-0.5">
                                    <span
                                        className={`text-[7px] px-1 py-0.5 rounded uppercase font-bold tracking-widest ${ev.severity === "CRITICAL"
                                            ? "bg-red-500/20 text-red-400"
                                            : "bg-amber-500/20 text-amber-400"
                                            }`}
                                    >
                                        {ev.severity}
                                    </span>
                                    <span
                                        className={`text-[7px] px-1 py-0.5 rounded uppercase font-bold tracking-widest ${ev.source === "backend"
                                            ? "bg-cyan-500/10 text-cyan-600"
                                            : "bg-slate-500/10 text-slate-600"
                                            }`}
                                    >
                                        {ev.source === "backend" ? "ENGINE" : "CLIENT"}
                                    </span>
                                    <span className="text-[8px] text-slate-700 font-mono">
                                        #{ev.id}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
        </div>
    );
}
