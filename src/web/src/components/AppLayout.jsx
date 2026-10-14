// 整体布局：左侧菜单 + 顶部用户信息 + 退出
import React from 'react'
import { Layout, Menu, Avatar, Dropdown } from 'antd'
import {
  ExperimentOutlined,
  HistoryOutlined,
  BookOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { getUser, clearAuth } from '../store/auth'
import LeafMark from './LeafMark'

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
        {/* 品牌区：高度与顶栏对齐（64px）；标识用自绘 SVG */}
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            color: 'var(--c-text-title)',
            fontWeight: 600,
            fontSize: 16,
          }}
        >
          <LeafMark size={22} color="var(--c-primary-500)" />
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
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            paddingRight: 24,
            boxShadow: 'var(--sh-sm)',
          }}
        >
          <Dropdown menu={userMenu}>
            <span style={{ cursor: 'pointer' }}>
              <Avatar
                icon={<UserOutlined />}
                style={{ backgroundColor: 'var(--c-primary-500)', marginRight: 8 }}
              />
              {user.nickname || user.username}
            </span>
          </Dropdown>
        </Header>
        {/* 内容区不再自带白底与圆角 —— 层次交给页面内的 Card，
            底色跟随 Layout（#F7FAF8），避免「白卡套白卡」 */}
        <Content style={{ margin: 24, padding: 0, minHeight: 280 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
