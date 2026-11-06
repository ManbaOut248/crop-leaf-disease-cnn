// 个人中心页：用户信息 + 修改密码（/auth/me）+ 界面偏好（本地图表可见性）
import React, { useState } from 'react'
import { Card, Form, Input, Button, Typography, Avatar, App, Switch, Space, Divider } from 'antd'
import { UserOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { getUser } from '../store/auth'
import { usePrefs, setPref, resetPrefs, PREF_META } from '../utils/prefs'

export default function ProfilePage() {
  const { message } = App.useApp()
  const [user] = useState(getUser() || { username: 'demo', nickname: '演示用户' })
  const [form] = Form.useForm()
  const prefs = usePrefs()

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

      <Card style={{ maxWidth: 520, marginBottom: 16 }}>
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

      {/* ---------- 界面偏好 ----------
          对应识别历史页每张图表右上角的「隐藏」按钮：那是入口，这里是恢复入口。
          只存在浏览器本地（localStorage），换设备/换浏览器会回到默认全显示 ——
          纯展示偏好，不值得为它加表与同步逻辑。 */}
      <Card style={{ maxWidth: 520 }} title="界面偏好">
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          控制「识别历史」页要显示哪些统计图表。隐藏只影响你自己的这个浏览器，
          数据本身不会被删除。
        </Typography.Text>
        <Divider style={{ margin: '12px 0' }} />
        <Space direction="vertical" size={14} style={{ width: '100%' }}>
          {PREF_META.map((m) => (
            <div
              key={m.key}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 14 }}>{m.label}</div>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {m.desc}
                </Typography.Text>
              </div>
              <Switch
                checked={prefs[m.key]}
                onChange={(v) => {
                  setPref(m.key, v)
                  message.success(`${v ? '已显示' : '已隐藏'}「${m.label}」`)
                }}
              />
            </div>
          ))}
        </Space>
        <Divider style={{ margin: '12px 0' }} />
        <Button
          size="small"
          onClick={() => {
            resetPrefs()
            message.success('已恢复默认（全部显示）')
          }}
        >
          恢复默认
        </Button>
      </Card>
    </div>
  )
}
