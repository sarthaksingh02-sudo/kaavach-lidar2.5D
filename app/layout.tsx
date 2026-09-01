import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KAVACH 2.5D — Tactical Perception Dashboard",
  description:
    "Real-time 2.5D LiDAR perception dashboard for autonomous threat intelligence · ETH-DiM · CUDA accelerated.",
  keywords: ["LiDAR", "2.5D", "perception", "tactical", "autonomous", "KAVACH"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full bg-[#030712]">
      <body className="h-full overflow-hidden antialiased">{children}</body>
    </html>
  );
}
