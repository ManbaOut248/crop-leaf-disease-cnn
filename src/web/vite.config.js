// vite 配置：开发服务器 + 反向代理到 FastAPI 后端(127.0.0.1:8000)
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // 前端请求 /api/* 自动转发到后端，避免跨域
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
