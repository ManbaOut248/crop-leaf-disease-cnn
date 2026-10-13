// 病害知识库页：按作物分组展示 38 类病害中文症状与防治建议（mock 模式展示全部）
import React, { useMemo, useState } from 'react'
import { Card, Tag, Input, Typography, Empty, Row, Col } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { mockDiseases } from '../api/mock'

export default function KnowledgePage() {
  const [kw, setKw] = useState('')

  const list = useMemo(() => {
    const k = kw.trim()
    if (!k) return mockDiseases
    return mockDiseases.filter(
      (d) => d.name.includes(k) || d.crop.includes(k) || d.disease.includes(k),
    )
  }, [kw])

  const groups = useMemo(() => {
    const m = {}
    list.forEach((d) => {
      ;(m[d.crop] = m[d.crop] || []).push(d)
    })
    return Object.entries(m)
  }, [list])

  return (
    <div>
      <Typography.Title level={3}>病害知识库</Typography.Title>
      <Input
        prefix={<SearchOutlined />}
        placeholder="搜索作物或病害，如：番茄 / 晚疫病"
        value={kw}
        onChange={(e) => setKw(e.target.value)}
        style={{ maxWidth: 360, marginBottom: 16 }}
        allowClear
      />

      {groups.length === 0 && <Empty description="未找到匹配的病害" />}

      {groups.map(([crop, items]) => (
        <div key={crop} style={{ marginBottom: 24 }}>
          <Typography.Title level={5}>
            <Tag color="green">{crop}</Tag> 共 {items.length} 类
          </Typography.Title>
          <Row gutter={[16, 16]}>
            {items.map((d) => (
              <Col xs={24} sm={12} md={8} lg={6} key={d.id}>
                <Card
                  size="small"
                  title={d.name}
                  extra={
                    <Tag color={d.isHealthy ? 'green' : 'volcano'}>
                      {d.isHealthy ? '健康' : '病害'}
                    </Tag>
                  }
                >
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    症状：{d.symptom}
                  </Typography.Text>
                  <br />
                  <br />
                  <Typography.Text style={{ fontSize: 12 }}>
                    防治：{d.prevention}
                  </Typography.Text>
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      ))}
    </div>
  )
}
