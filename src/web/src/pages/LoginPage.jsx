// 登录 / 注册页（对接 /auth/login、/auth/register；mock 模式降级）
import React, { useState } from 'react'
import { Form, Input, Button, Tabs, Card, Typography, App } from 'antd'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import http, { USE_MOCK } from '../api/http'
import { setToken, setUser } from '../store/auth'
import LeafMark from '../components/LeafMark'

export default function LoginPage() {
  // 必须用 App.useApp() 取 message：
  // 静态的 `message.xxx` 走的是 antd 内部独立实例，拿不到 ConfigProvider
  // 注入的动态主题（会打印 "Static function can not consume context" 警告，
  // 且弹出提示也不带本项目的「田野翠绿」配色）。
  const { message } = App.useApp()
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const doLogin = async (vals) => {
    setLoading(true)
    try {
      if (USE_MOCK) {
        setToken('mock-token-000')
        setUser({ id: 1, username: vals.username, nickname: vals.username })
        message.success('登录成功（演示模式）')
        navigate('/')
        return
      }
      const res = await http.post('/auth/login', vals)
      setToken(res.data.access_token)
      setUser(res.data.user)
      message.success('登录成功')
      navigate('/')
    } catch (e) {
      message.error((e.response && e.response.data && e.response.data.msg) || '登录失败')
    } finally {
      setLoading(false)
    }
  }

  const doRegister = async (vals) => {
    setLoading(true)
    try {
      if (USE_MOCK) {
        message.success('注册成功，请登录（演示模式）')
        return
      }
      await http.post('/auth/register', vals)
      message.success('注册成功，请登录')
    } catch (e) {
      message.error((e.response && e.response.data && e.response.data.msg) || '注册失败')
    } finally {
      setLoading(false)
    }
  }

  // 两个表单抽成变量，配合 Tabs 的 items 写法（TabPane 已废弃）
  const loginForm = (
    <Form onFinish={doLogin} size="large">
      <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
        <Input prefix={<UserOutlined />} placeholder="用户名" />
      </Form.Item>
      <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
        <Input.Password prefix={<LockOutlined />} placeholder="密码" />
      </Form.Item>
      <Button type="primary" htmlType="submit" block loading={loading}>
        登录
      </Button>
    </Form>
  )

  const registerForm = (
    <Form onFinish={doRegister} size="large">
      <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
        <Input prefix={<UserOutlined />} placeholder="用户名" />
      </Form.Item>
      <Form.Item name="nickname">
        <Input placeholder="昵称（可选）" />
      </Form.Item>
      <Form.Item
        name="password"
        rules={[{ required: true, min: 6, message: '密码至少 6 位' }]}
      >
        <Input.Password prefix={<LockOutlined />} placeholder="密码（至少 6 位）" />
      </Form.Item>
      <Button type="primary" htmlType="submit" block loading={loading}>
        注册
      </Button>
    </Form>
  )

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--c-bg-page)',
      }}
    >
      <Card style={{ width: 380, boxShadow: 'var(--sh-lg)' }}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 10,
            marginBottom: 24,
            textAlign: 'center',
          }}
        >
          <LeafMark size={40} color="var(--c-primary-500)" />
          <Typography.Title level={3} style={{ margin: 0 }}>
            农作物叶片病害识别
          </Typography.Title>
          <Typography.Text type="secondary">上传叶片图片，智能识别病害</Typography.Text>
        </div>

        <Tabs
          defaultActiveKey="login"
          centered
          items={[
            { key: 'login', label: '登录', children: loginForm },
            { key: 'register', label: '注册', children: registerForm },
          ]}
        />

        {USE_MOCK && (
          <Typography.Text
            type="secondary"
            style={{ display: 'block', marginTop: 12, textAlign: 'center', fontSize: 12 }}
          >
            当前为演示模式（USE_MOCK=true），任意账号即可登录
          </Typography.Text>
        )}
      </Card>
    </div>
  )
}
