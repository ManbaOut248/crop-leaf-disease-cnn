// 病害识别页：上传图片 → 调 /predict → Top-3 概率 + 病害说明 + 防治建议
// 响应契约对齐 docs/04_API接口文档.md：
//   { pred_class_id, crop_cn, disease_cn, is_healthy, confidence,
//     top3:[{class_id, disease_cn, prob}], need_review, knowledge:{symptom,cause,prevention,treatment} }
import React, { useState } from 'react'
import {
  Upload,
  Button,
  Card,
  Progress,
  Tag,
  Typography,
  Descriptions,
  Alert,
  App,
  Divider,
} from 'antd'
import { UploadOutlined, ExperimentOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockPredict } from '../api/mock'
import { formatPercent } from '../utils/format'

const LOW_CONF = 0.5 // 置信度低于此值提示人工复核（对齐 API 文档）

export default function DetectPage() {
  const { message } = App.useApp()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const beforeUpload = (f) => {
    const okType = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'].includes(f.type)
    if (!okType) {
      message.error('仅支持 JPG / PNG / BMP / WEBP 格式')
      return Upload.LIST_IGNORE
    }
    if (f.size > 10 * 1024 * 1024) {
      message.error('图片大小不能超过 10MB')
      return Upload.LIST_IGNORE
    }
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
    setFile(f)
    setResult(null)
    return false // 阻止自动上传，改为手动“开始识别”
  }

  const onDetect = () => {
    if (!file) {
      message.warning('请先选择一张叶片图片')
      return
    }
    setLoading(true)
    if (USE_MOCK) {
      setTimeout(() => {
        setResult(mockPredict())
        setLoading(false)
      }, 900)
      return
    }
    const fd = new FormData()
    fd.append('file', file)
    http
      .post('/predict', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      .then((res) => setResult(res.data))
      .catch((e) => message.error((e.response && e.response.data && e.response.data.msg) || '识别失败，请重试'))
      .finally(() => setLoading(false))
  }

  // 把后端契约字段归一化，便于展示
  const top1 = result && {
    name: result.disease_cn,
    confidence: result.confidence || 0,
    is_healthy: !!(result.is_healthy === true || result.is_healthy === 1),
    crop: result.crop_cn,
    symptom: result.knowledge && result.knowledge.symptom,
    prevention:
      (result.knowledge && (result.knowledge.prevention || result.knowledge.treatment)) || '',
  }
  const top3 = (result && result.top3) || []
  const needReview = result && (result.need_review || (top1 && top1.confidence < LOW_CONF))

  return (
    <div>
      <Typography.Title level={3}>病害识别</Typography.Title>
      <Typography.Paragraph type="secondary">
        上传一张农作物叶片照片，系统将自动识别病害类别并给出防治建议。
      </Typography.Paragraph>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {/* 左：上传 + 预览 */}
        <Card title="上传叶片图片" style={{ width: 360 }}>
          <div className="drop-zone" style={{ textAlign: 'center' }}>
            <Upload.Dragger beforeUpload={beforeUpload} showUploadList={false} accept="image/*">
              {preview ? (
                <img src={preview} alt="预览" style={{ maxWidth: '100%', maxHeight: 220, borderRadius: 8 }} />
              ) : (
                <div>
                  <p className="ant-upload-drag-icon">
                    <UploadOutlined />
                  </p>
                  <p className="ant-upload-text">点击或拖拽图片到此</p>
                  <p className="ant-upload-hint">支持 JPG / PNG / BMP / WEBP</p>
                </div>
              )}
            </Upload.Dragger>
          </div>
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            block
            loading={loading}
            onClick={onDetect}
            style={{ marginTop: 16 }}
          >
            开始识别
          </Button>
          {result && result.is_mock && (
            <Alert
              style={{ marginTop: 12 }}
              type="info"
              showIcon
              message="演示数据"
              description="模型尚未加载，以下为示例结果。"
            />
          )}
        </Card>

        {/* 右：结果 */}
        <Card title="识别结果" style={{ flex: 1, minWidth: 360 }}>
          {!result && (
            <Typography.Text type="secondary">
              尚未识别，请先上传图片并点击「开始识别」。
            </Typography.Text>
          )}

          {result && top1 && (
            <>
              {needReview && (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="置信度较低"
                  description="识别置信度低于 50%，建议结合人工经验复核或重新拍摄清晰图片。"
                />
              )}

              {/* 结论区：左「Tag + 病害名」，右「大号置信度 + 标签」，左右分列，
                  与高保真原型（docs/prototype）结论卡一致。
                  原实现的问题：<Space> 只负责其内部子元素之间的 gap，
                  管不到紧跟在它后面的那个 Text 兄弟节点；而 Title 又被 margin:0
                  抹掉了默认外边距 —— 两者因此零间距，文字直接粘在一起。 */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'space-between',
                  gap: 16,
                  marginBottom: 20,
                  paddingBottom: 20,
                  borderBottom: '1px solid var(--c-border-split)',
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <Tag
                    color={top1.is_healthy ? 'green' : 'volcano'}
                    style={{ fontSize: 14, padding: '4px 10px' }}
                  >
                    {top1.is_healthy ? '健康' : '病害'}
                  </Tag>
                  <Typography.Title level={4} style={{ margin: '10px 0 0' }}>
                    {top1.name}
                  </Typography.Title>
                </div>
                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                  <div
                    style={{
                      fontSize: 28,
                      fontWeight: 600,
                      lineHeight: 1.2,
                      color: 'var(--c-primary-600)',
                    }}
                  >
                    {formatPercent(top1.confidence)}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--c-text-sub)', marginTop: 2 }}>
                    Top-1 置信度
                  </div>
                </div>
              </div>

              <Typography.Paragraph strong>Top-3 可能性</Typography.Paragraph>
              {top3.map((t, i) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>
                      {i === 0 ? '最可能' : `第 ${i + 1} 可能`}：{t.disease_cn}
                    </span>
                    <span>{formatPercent(t.prob)}</span>
                  </div>
                  <Progress
                    percent={Math.round((t.prob || 0) * 100)}
                    showInfo={false}
                    strokeColor={i === 0 ? 'var(--c-primary-500)' : 'var(--c-primary-300)'}
                  />
                </div>
              ))}

              <Divider />
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="作物">{top1.crop}</Descriptions.Item>
                <Descriptions.Item label="症状特征">{top1.symptom}</Descriptions.Item>
                <Descriptions.Item label="防治建议">{top1.prevention}</Descriptions.Item>
              </Descriptions>
            </>
          )}
        </Card>
      </div>
    </div>
  )
}
