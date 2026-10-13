// 前端本地存储：token 与用户信息（localStorage）
const TOKEN_KEY = 'crop_token'
const USER_KEY = 'crop_user'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || ''
}

export function setToken(t) {
  localStorage.setItem(TOKEN_KEY, t)
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export function getUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || 'null')
  } catch {
    return null
  }
}

export function setUser(u) {
  localStorage.setItem(USER_KEY, JSON.stringify(u))
}
