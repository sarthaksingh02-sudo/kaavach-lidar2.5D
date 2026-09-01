// app/components/DeckCanvas2D.tsx
"use client";

import React, { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer } from "@deck.gl/layers";
import { OrbitView } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";

// ─── Label → colour (RGBA) ────────────────────────────────────────────────────
const LABEL_COLORS: Record<string, [number, number, number, number]> = {
    road: [34, 197, 94, 200],
    static_obstacle: [239, 68, 68, 230],
    dynamic_target: [251, 191, 36, 230],
    pothole: [168, 85, 247, 220],
    threat: [168, 85, 247, 220],
    unknown: [148, 163, 184, 160],
};

function getColor(label: string): [number, number, number, number] {
    const key = label.toLowerCase().replace(/\s+/g, "_");
    return (
        LABEL_COLORS[key] ??
        LABEL_COLORS[Object.keys(LABEL_COLORS).find((k) => key.includes(k)) ?? ""] ??
        LABEL_COLORS.unknown
    );
}

// Bridge now sends wedge objects; CellTuple import no longer needed
interface WedgeCell {
    polygon: [number, number][];
    elev: number;
    delta_z: number;
    label: string;
    cx: number;
    cy: number;
    color: [number, number, number, number];
}

interface TooltipInfo {
    x: number;
    y: number;
    cell: WedgeCell;
}

interface Props {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cells: any[];   // accept both old CellTuple[] and new wedge dict[]
}

// ─── Camera ───────────────────────────────────────────────────────────────────
const INITIAL_VIEW_STATE = {
    target: [0, 0, 0] as [number, number, number],
    rotationX: 40,       // ~40° tilt — shows ring structure + column height
    rotationOrbit: 0,
    zoom: 2.0,
};

const VIEWS = [new OrbitView({ id: "orbit", fovy: 45 })];

export default function DeckCanvas2D({ cells }: Props) {
    const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);

    // Normalise payload — bridge sends {polygon, elev, delta_z, label, cx, cy}
    const data = useMemo<WedgeCell[]>(() => {
        if (!cells?.length) return [];

        // New format: array of wedge dicts from bridge
        if (typeof cells[0] === "object" && !Array.isArray(cells[0])) {
            return (cells as WedgeCell[]).map((c) => ({
                ...c,
                color: getColor(c.label),
            }));
        }

        // Fallback for old CellTuple format [x, y, zMax, deltaZ, label, radius]
        // (should not happen after bridge update, but keeps things safe)
        return [];
    }, [cells]);

    const layers = useMemo(
        () => [
            new PolygonLayer<WedgeCell>({
                id: "lidar-wedges",
                data,
                getPolygon: (d) => d.polygon,
                getElevation: (d) => d.elev,
                getFillColor: (d) => d.color,
                getLineColor: [0, 0, 0, 120] as [number, number, number, number],
                lineWidthMinPixels: 1,
                pickable: true,
                extruded: true,
                elevationScale: 1,
                updateTriggers: { getElevation: data, getFillColor: data },
                transitions: {
                    getElevation: { duration: 200 },
                },
                onHover: (info: PickingInfo<WedgeCell>) => {
                    if (info.object) {
                        setTooltip({ x: info.x, y: info.y, cell: info.object });
                    } else {
                        setTooltip(null);
                    }
                },
            }),
        ],
        [data]
    );

    return (
        <div className="relative w-full h-full bg-[#030712]" style={{ height: "100%" }}>
            {/* Tactical grid overlay */}
            <div
                className="absolute inset-0 pointer-events-none z-0"
                style={{
                    backgroundImage:
                        "linear-gradient(rgba(34,211,238,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.04) 1px, transparent 1px)",
                    backgroundSize: "40px 40px",
                }}
            />

            <DeckGL
                views={VIEWS}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                initialViewState={INITIAL_VIEW_STATE as any}
                controller={true}
                layers={layers}
                style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
            />

            {/* Hover Tooltip */}
            {tooltip && (
                <div
                    className="absolute z-50 pointer-events-none"
                    style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
                >
                    <div className="bg-[#0f172a]/95 border border-cyan-500/40 rounded-lg px-3 py-2 text-xs font-mono shadow-2xl backdrop-blur-md">
                        <p className="text-cyan-400 font-bold uppercase tracking-widest mb-1">
                            {tooltip.cell.label}
                        </p>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-slate-300">
                            <span className="text-slate-500">X</span>
                            <span>{tooltip.cell.cx.toFixed(2)} m</span>
                            <span className="text-slate-500">Y</span>
                            <span>{tooltip.cell.cy.toFixed(2)} m</span>
                            <span className="text-slate-500">ΔZ</span>
                            <span className={Math.abs(tooltip.cell.delta_z) > 0.5 ? "text-red-400 font-semibold" : ""}>
                                {tooltip.cell.delta_z > 0 ? "+" : ""}{tooltip.cell.delta_z.toFixed(2)} m
                            </span>
                            <span className="text-slate-500">Height</span>
                            <span>{tooltip.cell.elev.toFixed(2)} m</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="absolute bottom-4 left-4 flex flex-col gap-1.5 z-10">
                {[
                    { label: "Road", color: "bg-green-500" },
                    { label: "Static Obstacle", color: "bg-red-500" },
                    { label: "Dynamic Target", color: "bg-amber-400" },
                ].map(({ label, color }) => (
                    <div key={label} className="flex items-center gap-2">
                        <div className={`w-2.5 h-2.5 rounded-sm ${color} opacity-80`} />
                        <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">
                            {label}
                        </span>
                    </div>
                ))}
            </div>

            {/* Scanline */}
            <div className="absolute inset-0 pointer-events-none overflow-hidden z-10">
                <div
                    className="absolute w-full h-px bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent"
                    style={{ animation: "scanline 4s linear infinite" }}
                />
            </div>

            <style>{`
        @keyframes scanline {
          0%   { top: 0%; }
          100% { top: 100%; }
        }
      `}</style>
        </div>
    );
}
