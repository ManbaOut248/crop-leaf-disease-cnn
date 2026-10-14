// 登录 / 注册页（对接 /auth/login、/auth/register；mock 模式降级）
import React, { useState } from 'react'
import { Form, Input, Button, Tabs, Card, message, Typography } from 'antd'
import { UserOutlined, LockOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import http, { USE_MOCK } from '../api/http'
import { setToken, setUser } from '../store/auth'
import LeafMark from '../components/LeafMark'

export default function LoginPage() {
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

        <Tabs defaultActiveKey="login" centered>
          <Tabs.TabPane tab="登录" key="login">
            <Form onFinish={doLogin} size="large">
              <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
                <Input prefix={<UserOutlined />} placeholder="用户名" />
              </Form.Item>
              <Form.Item
                name="password"
                rules={[{ required: true, message: '请输入密码' }]}
              >
                <Input.Password prefix={<LockOutlined />} placeholder="密码" />
              </Form.Item>
              <Button type="primary" htmlType="submit" block loading={loading}>
                登录
              </Button>
            </Form>
          </Tabs.TabPane>

          <Tabs.TabPane tab="注册" key="register">
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
          </Tabs.TabPane>
        </Tabs>

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
