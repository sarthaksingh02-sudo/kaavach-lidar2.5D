import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Silence the turbopack/webpack conflict warning:
  // Deck.gl is loaded client-side only via dynamic() so no special
  // server-side externals config is needed.
  turbopack: {},
};

export default nextConfig;
