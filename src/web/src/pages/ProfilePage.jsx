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
      // 字段名必须对齐后端 schemas.UpdateMeRequest：old_password / new_password。
      // 修复（审计 P1）：此前误用 { password }，后端读不到 new_password →
      // 落入「只更新昵称」的分支，结果是「密码根本没改、昵称反被置为 NULL」，
      // 用户误以为已改密（安全误导）。
      await http.put('/auth/me', {
        old_password: vals.old_password,
        new_password: vals.password,
      })
      message.success('密码已更新')
      form.resetFields()
    } catch (e) {
      message.error((e.response && e.response.data && e.response.data.msg) || '更新失败')
    }
  }

  return (
    <div>
      <Typography.Title level={3}>个人中心</Typography.Title>
      <Card style={{ maxWidth: 520 }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
          <Avatar size={64} icon={<UserOutlined />} style={{ backgroundColor: 'var(--c-primary-500)' }} />
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
            name="old_password"
            label="当前密码"
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password placeholder="请输入当前密码" autoComplete="current-password" />
          </Form.Item>
          <Form.Item
            name="password"
            label="新密码"
            rules={[{ required: true, min: 6, message: '密码至少 6 位' }]}
          >
            <Input.Password placeholder="请输入新密码（至少 6 位）" autoComplete="new-password" />
          </Form.Item>
          <Button type="primary" htmlType="submit">
            保存
          </Button>
        </Form>
      </Card>
    </div>
  )
}
