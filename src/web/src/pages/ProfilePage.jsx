// 个人中心页：展示用户信息 + 修改密码（对接 /auth/me）
import React, { useState } from 'react'
import { Card, Form, Input, Button, Typography, Avatar, App } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { getUser } from '../store/auth'

export default function ProfilePage() {
  const { message } = App.useApp()
  const [user] = useState(getUser() || { username: 'demo', nickname: '演示用户' })
  const [form] = Form.useForm()

  const onChangePwd = async (vals) => {
    if (USE_MOCK) {
      message.success('密码已更新（演示模式）')
      return
    }
    try {
      await http.put('/auth/me', { password: vals.password })
      message.success('密码已更新')
      form.resetFields()
    } catch (e) {
      message.error('更新失败')
    }
  }

  return (
    <div>
      <Typography.Title level={3}>个人中心</Typography.Title>
      <Card style={{ maxWidth: 520 }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
          <Avatar size={64} icon={<UserOutlined />} style={{ backgroundColor: '#52c41a' }} />
          <div style={{ marginLeft: 16 }}>
            <Typography.Title level={4} style={{ margin: 0 }}>
              {user.nickname || user.username}
            </Typography.Title>
            <Typography.Text type="secondary">@{user.username}</Typography.Text>
          </div>
        </div>

        <Typography.Paragraph strong>修改密码</Typography.Paragraph>
        <Form form={form} onFinish={onChangePwd} layout="vertical">
          <Form.Item
            name="password"
            label="新密码"
            rules={[{ required: true, min: 6, message: '密码至少 6 位' }]}
          >
            <Input.Password placeholder="请输入新密码（至少 6 位）" />
          </Form.Item>
          <Button type="primary" htmlType="submit">
            保存
          </Button>
        </Form>
      </Card>
    </div>
  )
}
