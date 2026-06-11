import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // honor the port assigned by the preview/launch system (autoPort), else default
    port: Number(process.env.PORT) || 5173,
  },
})
