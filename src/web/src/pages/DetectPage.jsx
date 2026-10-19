// 病害识别页：上传图片 → 调 /predict → Top-3 概率 + 病害说明 + 防治建议
// 响应契约对齐 docs/04_API接口文档.md：
//   { pred_class_id, crop_cn, disease_cn, is_healthy, confidence,
//     top3:[{class_id, disease_cn, prob}], need_review, knowledge:{symptom,cause,prevention,treatment},
//     likely_not_leaf, leaf_prob, crop_applied, crop_candidates }
//
// 作物先验（2026-10-19）：用户可选作物，后端把 softmax 限制在该作物的候选类内。
// 实测 120 张 PlantDoc test 真实照片：多类别作物 Top-1 15.8% → 43.6%（+27.7pp）。
// 根因是模型「只看斑点纹理、不看作物」——苹果锈病会被判成玉米灰斑病且置信度 0.96+。
import React, { useState, useEffect } from 'react'
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
  Select,
  Space,
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
  // 作物先验：'' 表示不限制（走全 38 类，准确率较低但不需要用户输入）
  const [crop, setCrop] = useState('')
  const [cropOptions, setCropOptions] = useState([])
  const [singleCandidateCrops, setSingleCandidateCrops] = useState([])

  // 作物清单从后端取，不在前端硬编码 —— 类别体系以后端为唯一来源，
  // 否则类别调整后前端选项会静默失真（知识库页就是反例）。
  useEffect(() => {
    if (USE_MOCK) return
    http
      .get('/crops')
      .then((res) => {
        const d = res.data || {}
        const items = d.items || []
        setCropOptions(
          items.map((x) => ({
            value: x.crop,
            label: `${x.crop}（${x.n_classes} 个候选）`,
          })),
        )
        setSingleCandidateCrops(d.single_candidate || [])
      })
      .catch(() => {
        // 取不到就退化成「不提供选择」，不影响识别主流程
      })
  }, [])

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
      .post('/predict', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        // 只在选了作物时带参数，避免空串被后端当成「未知作物」而报 400
        params: crop ? { crop } : undefined,
      })
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
  // 「大概率不是叶片」时只提示这一件事，不再叠加"置信度低"的橙色提示，避免两条信息打架
  const needReview =
    result &&
    !result.likely_not_leaf &&
    (result.need_review || (top1 && top1.confidence < LOW_CONF))

  return (
    <div>
      <Typography.Title level={3}>病害识别</Typography.Title>
      <Typography.Paragraph type="secondary">
        上传一张农作物叶片照片，系统将自动识别病害类别并给出防治建议。
      </Typography.Paragraph>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {/* 左：上传 + 预览 */}
        <Card title="上传叶片图片" style={{ width: 360 }}>
          {/* 作物先验：指定后在候选类内归一化，实测多类别作物准确率翻近三倍。
              默认「不确定」——不强制用户输入，代价是准确率低一档。 */}
          <div style={{ marginBottom: 14 }}>
            <Typography.Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>
              作物类型（选填，指定后更准确）
            </Typography.Text>
            <Select
              allowClear
              value={crop || undefined}
              onChange={(v) => setCrop(v || '')}
              placeholder="不确定（全 38 类中判断）"
              options={cropOptions}
              style={{ width: '100%' }}
              disabled={cropOptions.length === 0}
            />
            {crop && singleCandidateCrops.includes(crop) && (
              <Typography.Text type="warning" style={{ fontSize: 12 }}>
                该作物在模型中只有一个候选类别，结果等同直接给出结论，参考价值有限。
              </Typography.Text>
            )}
          </div>
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
              {/* 非叶片输入提示：判定只看主判据（详见 src/model/predict.py）——
                    · 叶片检测头（主）→ 真实叶 vs 非叶 AUROC 0.97~0.99，跨来源留一 0.77~0.99，
                      故可明确说「可能不是叶片照片」；
                    · MSP（回退，仅检测头不可用时）→ 跨域不可靠（真实田间 AUROC 仅 0.553，
                      曾把 13 张真叶片中的 7 张误判），故只说「结果可能不可靠」。
                    · 平均色度**不再参与判定**：它隐含「叶片必为彩色」的假设，会把灰阶真实叶片
                      100% 误杀；现仅用于选择措辞，并对「灰阶但放行」的情况给一条温和提醒。
                  实测（灰度增强后）：真实田间叶片误杀 4.6%，非叶片拦截 95.9%，灰度叶通过 93%。 */}
              {result.likely_not_leaf && (
                <Alert
                  type="error"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message={
                    result.leaf_prob_source === 'head'
                      ? '这张图片可能不是叶片照片'
                      : '识别结果可能不可靠'
                  }
                  description={
                    result.leaf_prob_source === 'head' ? (
                      <>
                        系统判断该图为「叶片」的把握仅 <b>{formatPercent(result.leaf_prob)}</b>
                        {result.grayscale ? (
                          <>
                            ，且这是一张<b>黑白 / 灰度图片</b>（平均色度 {result.mean_chroma}）
                          </>
                        ) : null}
                        。请上传<b>单一叶片</b>的清晰特写、背景尽量干净，
                        下方结果<b>不可信</b>，请勿据此用药。
                      </>
                    ) : (
                      <>
                        这张照片与模型训练时的样本差异较大（常见于背景杂乱、光照特殊，
                        或确实不是叶片特写）。模型对「叶片」的把握仅{' '}
                        <b>{formatPercent(result.leaf_prob)}</b>，下方结果<b>仅供参考</b>，
                        请结合人工经验复核。
                        <br />
                        提示：上传<b>单一叶片</b>的清晰特写、背景尽量干净，识别会更可靠。
                      </>
                    )
                  }
                />
              )}

              {/* 灰阶但判定为叶片：不拦截，只提醒 —— 模型主要在彩色图上训练，
                  灰度化会损失病斑颜色信息（黄化、褐变等关键症状），准确率可能下降。 */}
              {!result.likely_not_leaf && result.grayscale && (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="检测到黑白 / 灰度图片"
                  description={
                    <>
                      系统仍判断这是一张叶片照片，但病害识别主要依赖<b>颜色信息</b>
                      （黄化、褐变、霉层等），灰度图可能影响判断准确率。
                      建议改用<b>彩色</b>照片；下方结果请结合人工经验复核。
                    </>
                  }
                />
              )}

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
                  {/* 作物先验是否生效：不标出来用户无从知道概率是在几个候选里算的 */}
                  {result.crop_applied && (
                    <Tag style={{ fontSize: 12, marginLeft: 6 }}>
                      已限定作物 {result.crop_candidates} 个候选
                    </Tag>
                  )}
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
