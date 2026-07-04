import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/patients": "http://localhost:8000",
      "/queue": "http://localhost:8000",
      "/prioritize": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/symptoms": "http://localhost:8000",
      "/decisions": "http://localhost:8000",
      "/model-info": "http://localhost:8000",
      "/demo": "http://localhost:8000",
    },
  },
});
