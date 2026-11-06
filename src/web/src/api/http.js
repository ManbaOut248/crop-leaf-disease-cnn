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
      // 会话过期必须能把用户带到登录页（2026-10-29 修复「有时无法查看」）。
      // 旧实现只 clearAuth() 不跳转：token 过期后用户仍停留在当前页，
      // ProtectedRoute 只在**首次进入**时检查 token 是否存在、不验有效性，
      // 于是所有请求 401 → 列表空白 + 一句「加载失败」，看起来就是
      // 「历史打不开了」，必须手动刷新才会被踢回登录页。
      const url = (err.config && err.config.url) || ''
      // 登录/注册接口的 401 是「账号或密码错」，不是会话过期，跳转会打断用户输入
      const isAuthCall = url.includes('/auth/login') || url.includes('/auth/register')
      if (!isAuthCall && !window.location.pathname.startsWith('/login')) {
        // 带 expired=1：登录页据此提示「登录已过期」，否则用户会以为是系统坏了
        window.location.assign('/login?expired=1')
      }
    }
    return Promise.reject(err)
  },
)

export default http
