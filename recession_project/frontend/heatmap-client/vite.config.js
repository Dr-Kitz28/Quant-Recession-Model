import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API endpoints to the FastAPI server during development
      '/meta': 'http://localhost:8001',
      '/frame': 'http://localhost:8001',
      '/frames': 'http://localhost:8001',
      '/tiles': 'http://localhost:8001',
    },
  },
  build: {
    chunkSizeWarningLimit: 5000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("plotly")) {
            return "plotly"
          }
          return undefined
        },
      },
    },
  },
})
