// 识别历史页：列表 + 详情抽屉 + CSV 导出（对接 /records）
import React, { useEffect, useState } from 'react'
import { Table, Button, Tag, Typography, Drawer, Descriptions, App } from 'antd'
import { DownloadOutlined, PictureOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockRecords } from '../api/mock'
import { formatPercent } from '../utils/format'

export default function HistoryPage() {
  const { message } = App.useApp()
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)
  const [cur, setCur] = useState(null)
  const [open, setOpen] = useState(false)

  const load = () => {
    setLoading(true)
    if (USE_MOCK) {
      setData(mockRecords().items)
      setLoading(false)
      return
    }
    http
      .get('/records')
      .then((r) => setData((r.data && r.data.items) || []))
      .catch(() => message.error('加载失败'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const onExport = () => {
    const header = 'id,image,result,confidence,time\n'
    const body = data
      .map(
        (r) =>
          // 导出是数据用途，保留两位小数，不套用界面上的极端值守卫
          `${r.id},${r.image_name},${r.disease_cn},${(r.confidence * 100).toFixed(2)}%,${r.created_at}`,
      )
      .join('\n')
    const blob = new Blob([header + body], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = '识别历史.csv'
    a.click()
    message.success('已导出 CSV')
  }

  const columns = [
    {
      // 缩略图由后端以 base64 内联返回（见 src/api/thumbs.py），
      // 因此这里无需鉴权头，也不存在 objectURL 回收问题。
      title: '图片',
      dataIndex: 'thumb',
      width: 76,
      render: (src, r) =>
        src ? (
          <img
            src={src}
            alt={`${r.disease_cn || '叶片'}缩略图`}
            style={{
              width: 48,
              height: 48,
              objectFit: 'cover',
              borderRadius: 6,
              border: '1px solid var(--c-border)',
              display: 'block',
            }}
          />
        ) : (
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 6,
              background: 'var(--c-bg-subtle)',
              border: '1px solid var(--c-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <PictureOutlined style={{ color: 'var(--c-text-placeholder)', fontSize: 18 }} />
          </div>
        ),
    },
    { title: '#', dataIndex: 'id', width: 60 },
    { title: '图片名', dataIndex: 'image_name' },
    {
      title: '识别结果',
      dataIndex: 'disease_cn',
      render: (t) => <Tag color={t && t.includes('健康') ? 'green' : 'volcano'}>{t}</Tag>,
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      render: (v) => formatPercent(v),
    },
    { title: '时间', dataIndex: 'created_at' },
    {
      title: '操作',
      render: (_, r) => (
        <Button size="small" onClick={() => { setCur(r); setOpen(true) }}>
          详情
        </Button>
      ),
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={3}>识别历史</Typography.Title>
        <Button icon={<DownloadOutlined />} onClick={onExport}>
          导出 CSV
        </Button>
      </div>
      <Table
        rowKey="id"
        loading={loading}
        columns={columns}
        dataSource={data}
        pagination={{ pageSize: 8 }}
      />
      <Drawer title="识别详情" open={open} onClose={() => setOpen(false)}>
        {cur && (
          <>
            {cur.thumb && (
              <img
                src={cur.thumb}
                alt={`${cur.disease_cn || '叶片'}缩略图`}
                style={{
                  width: 96,
                  height: 96,
                  objectFit: 'cover',
                  borderRadius: 8,
                  border: '1px solid var(--c-border)',
                  marginBottom: 16,
                  display: 'block',
                }}
              />
            )}
            <Descriptions column={1} bordered>
              <Descriptions.Item label="图片名">{cur.image_name}</Descriptions.Item>
              <Descriptions.Item label="识别结果">{cur.disease_cn}</Descriptions.Item>
              <Descriptions.Item label="置信度">{formatPercent(cur.confidence)}</Descriptions.Item>
              <Descriptions.Item label="时间">{cur.created_at}</Descriptions.Item>
            </Descriptions>
          </>
        )}
      </Drawer>
    </div>
  )
}
