// 整体布局：左侧菜单 + 顶部用户信息 + 退出
import React, { useEffect, useRef } from 'react'
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

  // 滚动容器从 window 变成内容区后，切页要显式回到顶部：
  // 虽然内容高度变化通常会让浏览器把 scrollTop 钳回 0，但那是「碰巧」——
  // 新旧页面高度接近时位置会残留，表现为「点进知识库却在半截」。
  const contentRef = useRef(null)
  useEffect(() => {
    if (contentRef.current) contentRef.current.scrollTop = 0
  }, [location.pathname])

  const userMenu = {
    items: [{ key: 'logout', icon: <LogoutOutlined />, label: '退出登录' }],
    onClick: () => {
      clearAuth()
      navigate('/login')
    },
  }

  return (
    /* 外壳锁成 100vh 且不滚动：滚动只发生在右侧 Content 内部，
       这样左侧菜单与顶栏在物理上就不可能被内容顶走。
       另一种常见做法是给 Sider 加 position: fixed，但要配 margin-left 和
       折叠态的位移补偿，容易在断点切换时错位；用容器限高更稳。 */
    <Layout style={{ height: '100vh', overflow: 'hidden' }}>
      <Sider
        theme="light"
        breakpoint="lg"
        collapsedWidth={0}
        // 菜单项将来变多时在侧栏内部滚动，而不是把侧栏撑出视口
        style={{ height: '100vh', overflowY: 'auto' }}
      >
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
      <Layout style={{ height: '100vh', overflow: 'hidden' }}>
        <Header
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            paddingRight: 24,
            boxShadow: 'var(--sh-sm)',
            // 纵向 flex 子项默认 flex-shrink:1，内容过高时顶栏会被压扁
            flexShrink: 0,
            zIndex: 2,
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
            底色跟随 Layout（#F7FAF8），避免「白卡套白卡」。
            margin 换成 padding：滚动条因此贴在视口右缘，不会出现
            「距右边缘 24px 处悬着一根滚动条」的割裂感。
            minHeight:0 是 flex 滚动容器的必要条件 —— 缺了它，子元素会把
            容器撑高（flex 默认 min-height:auto），overflow 永远不触发。 */}
        <Content
          ref={contentRef}
          className="app-content-scroll"
          style={{
            padding: 24,
            overflowY: 'auto',
            overflowX: 'hidden',
            minHeight: 0,
            flex: '1 1 auto',
            // 预留滚动条槽位：否则「长页面有滚动条、短页面没有」时
            // 内容宽度差 10px，切页会看到卡片轻微横向抖动。
            scrollbarGutter: 'stable',
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
