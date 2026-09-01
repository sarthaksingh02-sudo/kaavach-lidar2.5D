// app/components/ThreatTicker.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { AlertTriangle, ShieldAlert, ChevronRight } from "lucide-react";
import type { CellTuple } from "../hooks/useLidarStream";

interface ThreatEvent {
    id: string;
    timestamp: string;
    type: "DROP_OFF" | "CLOSE_RANGE";
    label: string;
    severity: "HIGH" | "CRITICAL";
    detail: string;
}

interface Props {
    cells: CellTuple[];
    maxEvents?: number;
}

let eventCounter = 0;

function buildTimestamp() {
    return new Date().toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        fractionalSecondDigits: 2,
    });
}

function hexId() {
    return (++eventCounter).toString(16).padStart(4, "0").toUpperCase();
}

function detectThreats(cells: CellTuple[]): ThreatEvent[] {
    const events: ThreatEvent[] = [];
    const ts = buildTimestamp();

    for (const [x, y, , deltaZ, label, radius] of cells) {
        // Negative obstacle drop-off: |ΔZ| > 0.5 m and ΔZ > 0 (rising spike)
        if (deltaZ > 0.5) {
            events.push({
                id: hexId(),
                timestamp: ts,
                type: "DROP_OFF",
                label,
                severity: deltaZ > 1.2 ? "CRITICAL" : "HIGH",
                detail: `ΔZ=+${deltaZ.toFixed(2)}m @ (${x.toFixed(1)},${y.toFixed(1)})`,
            });
        }

        // Close-range dynamic obstacle: radius < 5 m and dynamic label
        if (
            radius < 5 &&
            (label.toLowerCase().includes("dynamic") ||
                label.toLowerCase().includes("target"))
        ) {
            events.push({
                id: hexId(),
                timestamp: ts,
                type: "CLOSE_RANGE",
                label,
                severity: radius < 2 ? "CRITICAL" : "HIGH",
                detail: `r=${radius.toFixed(1)}m @ (${x.toFixed(1)},${y.toFixed(1)})`,
            });
        }
    }

    return events;
}

export default function ThreatTicker({ cells, maxEvents = 60 }: Props) {
    const [events, setEvents] = useState<ThreatEvent[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!cells?.length) return;
        const newThreats = detectThreats(cells);
        if (!newThreats.length) return;

        setEvents((prev) => {
            const combined = [...newThreats, ...prev];
            return combined.slice(0, maxEvents);
        });
    }, [cells, maxEvents]);

    // Auto-scroll to top on new event
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = 0;
        }
    }, [events]);

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between px-1 mb-2">
                <div className="flex items-center gap-2">
                    <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                    <h2 className="text-xs font-mono uppercase tracking-[0.25em] text-slate-400">
                        Threat Ticker
                    </h2>
                </div>
                <span className="text-[9px] font-mono bg-red-500/20 text-red-400 border border-red-500/30 rounded px-1.5 py-0.5">
                    {events.length} EVENTS
                </span>
            </div>

            {/* Scrolling log */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto space-y-1.5 pr-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10"
                style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(255,255,255,0.1) transparent" }}
            >
                {events.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
                        <ShieldAlert className="w-8 h-8 opacity-30" />
                        <p className="text-xs font-mono">No threats detected</p>
                    </div>
                ) : (
                    events.map((ev, idx) => (
                        <div
                            key={ev.id}
                            className={`relative overflow-hidden flex items-start gap-3 rounded-lg border px-3 py-2 text-xs font-mono transition-all duration-300 ${ev.severity === "CRITICAL"
                                    ? "border-red-500/50 bg-red-950/30"
                                    : "border-amber-500/30 bg-amber-950/20"
                                } ${idx === 0 ? "ring-1 ring-inset " + (ev.severity === "CRITICAL" ? "ring-red-500/30" : "ring-amber-500/20") : ""}`}
                            style={{
                                animation: idx === 0 ? "slideIn 0.25s ease-out" : undefined,
                            }}
                        >
                            {/* Severity glow bar */}
                            <div
                                className={`absolute left-0 top-0 w-0.5 h-full ${ev.severity === "CRITICAL" ? "bg-red-500" : "bg-amber-400"
                                    }`}
                            />

                            <AlertTriangle
                                className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${ev.severity === "CRITICAL" ? "text-red-400" : "text-amber-400"
                                    }`}
                            />

                            <div className="flex-1 min-w-0">
                                <div className="flex items-center justify-between gap-2">
                                    <span
                                        className={`font-bold uppercase tracking-wider ${ev.severity === "CRITICAL"
                                                ? "text-red-300"
                                                : "text-amber-300"
                                            }`}
                                    >
                                        {ev.type === "DROP_OFF" ? "OBSTACLE DROP-OFF" : "CLOSE DYNAMIC TARGET"}
                                    </span>
                                    <span className="text-slate-600 text-[9px] flex-shrink-0">
                                        {ev.timestamp}
                                    </span>
                                </div>
                                <div className="flex items-center gap-1 text-slate-500 mt-0.5">
                                    <ChevronRight className="w-2.5 h-2.5" />
                                    <span className="text-slate-400">{ev.label}</span>
                                    <span>·</span>
                                    <span className="text-slate-500 truncate">{ev.detail}</span>
                                </div>
                                <div className="flex items-center gap-1 mt-0.5">
                                    <span className={`text-[8px] px-1 rounded uppercase font-bold tracking-widest ${ev.severity === "CRITICAL" ? "bg-red-500/20 text-red-400" : "bg-amber-500/20 text-amber-400"
                                        }`}>
                                        {ev.severity}
                                    </span>
                                    <span className="text-[8px] text-slate-600 font-mono">
                                        EVT #{ev.id}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
        </div>
    );
}
