// app/page.tsx
"use client";

import React, { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { Shield, Radio, Crosshair } from "lucide-react";
import TelemetryPanel from "./components/TelemetryPanel";
import ThreatTicker from "./components/ThreatTicker";
import {
  useLidarStream,
  type CellTuple,
  type TelemetryMetrics,
  type ThreatObject,
} from "./hooks/useLidarStream";

// Dynamically import DeckGL canvas — SSR incompatible (WebGL)
const DeckCanvas2D = dynamic(() => import("./components/DeckCanvas2D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-[#030712]">
      <div className="flex flex-col items-center gap-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-2 border-cyan-500/30 animate-ping" />
          <div
            className="absolute inset-2 rounded-full border-2 border-cyan-400/60 animate-spin"
            style={{ animationDuration: "1s" }}
          />
          <Crosshair className="absolute inset-0 m-auto w-6 h-6 text-cyan-400" />
        </div>
        <p className="text-cyan-500/60 text-xs font-mono uppercase tracking-widest">
          Loading Perception Engine…
        </p>
      </div>
    </div>
  ),
});

export default function KavachDashboard() {
  const [cells, setCells] = useState<CellTuple[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMetrics | null>(null);
  const [threats, setThreats] = useState<ThreatObject[]>([]);
  const [connected, setConnected] = useState(false);

  const handleCells = useCallback(
    (incoming: CellTuple[]) => {
      setCells(incoming);
      if (!connected) setConnected(true);
    },
    [connected]
  );

  const handleTelemetry = useCallback((metrics: TelemetryMetrics) => {
    setTelemetry(metrics);
  }, []);

  const handleThreats = useCallback((incoming: ThreatObject[]) => {
    setThreats(incoming);
  }, []);

  useLidarStream({
    onCells: handleCells,
    onTelemetry: handleTelemetry,
    onThreats: handleThreats,
    reconnectDelayMs: 2000,
  });

  return (
    <div className="h-screen w-screen bg-[#030712] text-white flex flex-col overflow-hidden font-mono">
      {/* ── Top Navigation Bar ─────────────────────────────────────────── */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 border-b border-white/5 bg-[#050d1a]/80 backdrop-blur-md z-20">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Shield className="w-7 h-7 text-cyan-400" />
            <div className="absolute inset-0 bg-cyan-400/20 blur-md rounded-full animate-pulse" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-white">
              KAVACH <span className="text-cyan-400">2.5D</span>
            </h1>
            <p className="text-[9px] text-slate-500 tracking-widest uppercase">
              Tactical Perception Dashboard · ETH-DiM
            </p>
          </div>
        </div>

        {/* Status indicators */}
        <div className="flex items-center gap-6">
          {/* WebSocket connection pill */}
          <div className="flex items-center gap-2 text-xs">
            <Radio className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-500">WS</span>
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-2 w-2 rounded-full opacity-75 ${connected ? "bg-green-400" : "bg-red-500"
                  }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${connected ? "bg-green-400" : "bg-red-500"
                  }`}
              />
            </span>
            <span
              className={`text-[10px] font-mono ${connected ? "text-green-400" : "text-red-400"
                }`}
            >
              {connected ? "STREAMING" : "CONNECTING…"}
            </span>
          </div>

          {/* Live FPS */}
          <div className="text-xs text-slate-500">
            <span className="text-cyan-300 font-bold tabular-nums">
              {telemetry ? `${telemetry.fps.toFixed(0)} FPS` : "—"}
            </span>
          </div>

          {/* Threat count badge */}
          {threats.length > 0 && (
            <div className="flex items-center gap-1 bg-red-500/15 border border-red-500/30 rounded px-2 py-0.5">
              <span className="text-[9px] font-mono font-bold text-red-400 animate-pulse">
                ⚠ {threats.length} THREAT{threats.length > 1 ? "S" : ""}
              </span>
            </div>
          )}

          {/* Engine tier badge */}
          {telemetry && (
            <div
              className={`text-[9px] font-mono font-bold tracking-widest px-2 py-0.5 rounded border ${telemetry.engine === "CUDA GPU TIER 1"
                  ? "text-cyan-400 border-cyan-500/30 bg-cyan-500/10"
                  : "text-amber-400 border-amber-500/30 bg-amber-500/10"
                }`}
            >
              {telemetry.engine === "CUDA GPU TIER 1" ? "GPU" : "CPU"}
            </div>
          )}

          {/* Timestamp */}
          <div className="text-[10px] text-slate-600 tabular-nums">
            {new Date().toLocaleTimeString("en-GB", {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </div>
        </div>
      </header>

      {/* ── Main Content ───────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* ── Left: Deck.gl Canvas (70%) ──────────────────────────────── */}
        <div className="relative flex-[7] min-w-0 overflow-hidden border-r border-white/5">
          {/* Corner brackets */}
          {(["top-2 left-2", "top-2 right-2", "bottom-2 left-2", "bottom-2 right-2"] as const).map(
            (pos, i) => (
              <div
                key={i}
                className={`absolute ${pos} w-5 h-5 border-cyan-500/40 pointer-events-none z-10`}
                style={{
                  borderTopWidth: i < 2 ? 1 : 0,
                  borderBottomWidth: i >= 2 ? 1 : 0,
                  borderLeftWidth: i % 2 === 0 ? 1 : 0,
                  borderRightWidth: i % 2 === 1 ? 1 : 0,
                }}
              />
            )
          )}

          {/* Canvas label */}
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10">
            <span className="text-[9px] font-mono uppercase tracking-[0.35em] text-slate-600 bg-[#030712]/80 px-3 py-1 rounded-full border border-white/5">
              LiDAR · 2.5D Polar Grid · OrbitView
            </span>
          </div>

          {/* Frame counter */}
          {telemetry && (
            <div className="absolute top-3 right-4 z-10 text-[9px] font-mono text-slate-700">
              {cells.length.toLocaleString()} cells ·{" "}
              {telemetry.latencyMs.toFixed(1)} ms
            </div>
          )}

          <DeckCanvas2D cells={cells} />
        </div>

        {/* ── Right: Sidebar (30%) ────────────────────────────────────── */}
        <div className="flex-[3] min-w-0 flex flex-col gap-4 overflow-hidden p-4 bg-[#050d1a]/60">
          {/* Telemetry Panel */}
          <div className="flex-[4] min-h-0 overflow-hidden">
            <TelemetryPanel metrics={telemetry} />
          </div>

          {/* Divider */}
          <div className="flex-shrink-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

          {/* Threat Ticker — wired to both backend threats + cell scan */}
          <div className="flex-[5] min-h-0 overflow-hidden">
            <ThreatTicker
              cells={cells}
              backendThreats={threats}
              maxEvents={50}
            />
          </div>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="flex-shrink-0 flex items-center justify-between px-6 py-1.5 border-t border-white/5 bg-[#050d1a]/80">
        <span className="text-[9px] text-slate-700 uppercase tracking-widest">
          KAVACH PERCEPTION v1.0 · ETH-DiM · 2.5D LiDAR Threat Intelligence
        </span>
        <span className="text-[9px] text-slate-700 uppercase tracking-widest">
          ws://localhost:8000/ws/stream_map
        </span>
      </footer>
    </div>
  );
}
