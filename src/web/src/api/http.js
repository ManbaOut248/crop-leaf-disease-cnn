// axios 封装：统一 baseURL、token 注入、401 自动登出
import axios from 'axios'
import { getToken, clearAuth } from '../store/auth'

// 脚手架阶段：后端/模型未完全就绪时置 true，所有接口走本地 mock
// 训练完成、后端启动后改为 false 即可切换为真实联调
export const USE_MOCK = false

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

http.interceptors.request.use((config) => {
  const t = getToken()
  if (t) config.headers.Authorization = `Bearer ${t}`
  return config
})

http.interceptors.response.use(
  (res) => res.data,
  (err) => {
    if (err.response && err.response.status === 401) {
      clearAuth()
    }
    return Promise.reject(err)
  },
)

export default http
