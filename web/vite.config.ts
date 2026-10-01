import { defineConfig } from "vite";

export default defineConfig({
  server: { proxy: { "/api": "http://localhost:8000", "/metrics": "http://localhost:8000" } },
  build: { outDir: "dist", sourcemap: true, chunkSizeWarningLimit: 1500 },
});
