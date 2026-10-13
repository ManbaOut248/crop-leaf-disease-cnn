// 路由守卫：未登录跳转 /login
import React from 'react'
import { Navigate } from 'react-router-dom'
import { getToken } from '../store/auth'

export default function ProtectedRoute({ children }) {
  if (!getToken()) {
    return <Navigate to="/login" replace />
  }
  return children
}
