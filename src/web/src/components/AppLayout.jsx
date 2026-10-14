// 整体布局：左侧菜单 + 顶部用户信息 + 退出
import React from 'react'
import { Layout, Menu, Avatar, Dropdown, Typography } from 'antd'
import {
  ExperimentOutlined,
  HistoryOutlined,
  BookOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { getUser, clearAuth } from '../store/auth'

const { Header, Sider, Content } = Layout

const items = [
  { key: '/', icon: <ExperimentOutlined />, label: '病害识别' },
  { key: '/history', icon: <HistoryOutlined />, label: '识别历史' },
  { key: '/knowledge', icon: <BookOutlined />, label: '病害知识库' },
  { key: '/profile', icon: <UserOutlined />, label: '个人中心' },
]

export default function AppLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const user = getUser() || { username: 'demo', nickname: '演示用户' }

  const userMenu = {
    items: [{ key: 'logout', icon: <LogoutOutlined />, label: '退出登录' }],
    onClick: () => {
      clearAuth()
      navigate('/login')
    },
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="light" breakpoint="lg" collapsedWidth={0}>
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            color: '#52c41a',
            fontWeight: 700,
            fontSize: 16,
          }}
        >
          <span style={{ fontSize: 22, lineHeight: 1 }}>🌿</span>
          <span>病害识别</span>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            background: '#fff',
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            paddingRight: 24,
            boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          }}
        >
          <Dropdown menu={userMenu}>
            <span style={{ cursor: 'pointer' }}>
              <Avatar
                icon={<UserOutlined />}
                style={{ backgroundColor: '#52c41a', marginRight: 8 }}
              />
              {user.nickname || user.username}
            </span>
          </Dropdown>
        </Header>
        <Content
          style={{
            margin: 24,
            padding: 24,
            background: '#fff',
            borderRadius: 12,
            minHeight: 280,
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
