// 识别记录详情弹窗（2026-10-29 新增，替换原来的右侧 Drawer）
//
// 为什么从 Drawer 改成居中 Modal：
//   Drawer 贴着屏幕右边、宽度受限，叶片图只能挤成 96px 小图，病害细节根本看不清；
//   而这张图的用途恰恰是「对照病斑判断识别得对不对」。居中弹窗有横向空间做
//   左图右文，图能放到 300px 以上并点击放大。
//
// 为什么详情要单独请求 /records/{rid} 而不是复用列表行：
//   列表为了控制体积只内联 96px 缩略图，且不解析 Top-3。放大看图必须要有
//   720px 预览图与候选列表，这两样都在详情接口里。
import React, { useEffect, useState } from 'react'
import {
  Modal, Row, Col, Image, Typography, Tag, Space, Input, Button, Segmented,
  Progress, Descriptions, App, Skeleton, Empty, Popconfirm, Alert, Select,
} from 'antd'
import { DeleteOutlined, SaveOutlined, PictureOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockRecords } from '../api/mock'
import { formatPercent } from '../utils/format'

// 与识别页同一套参考图约定（见 pages/DetectPage.jsx 顶部说明）：
// `/disease-ref/<class_id>-<1..3>.webp`，由 scripts/build_disease_refs.py 生成。
const REF_PER_CLASS = 3

function diseaseRefs(classId) {
  if (classId === null || classId === undefined || Number.isNaN(Number(classId))) return []
  return Array.from({ length: REF_PER_CLASS }, (_, i) => `/disease-ref/${classId}-${i + 1}.webp`)
}

// 人工判断的三态：值要和后端 db.VERDICT_VALUES 对齐（correct / wrong / null）。
// 「未核对」不是第三种状态值，而是把 verdict 置 null —— 这样统计口径只有两档。
//
// 文案是「识别正确 / 识别错误」而非「判断正确 / 判断错误」（2026-10-29 修正）：
// 用户**已经知道真实病害**，来这里只是给 AI 判卷。写成「判断错误」会被读成
// 「我自己判断失误」，语义正好反了（用户原话：「不是我的答案相对识别的选错了」）。
const VERDICT_OPTIONS = [
  { label: '识别正确', value: 'correct' },
  { label: '识别错误', value: 'wrong' },
  { label: '未核对', value: 'unset' },
]

const VERDICT_TAG = {
  correct: { color: 'green', text: '识别正确' },
  wrong: { color: 'red', text: '识别错误' },
}

// 38 类病害选项（用于「实际病害」下拉）。
// 模块级缓存：38 条数据很小，但弹窗一开一合就重拉一次没必要；
// 类别体系不会在运行时变化，进程内缓存没有失效风险。
let diseaseOptionCache = null

async function loadDiseaseOptions() {
  if (diseaseOptionCache) return diseaseOptionCache
  const r = await http.get('/diseases')
  diseaseOptionCache = (r.data || []).map((d) => ({
    // class_id 由后端显式回传（知识库主键即类别 id），不要让前端猜
    value: d.class_id ?? d.id,
    label: `${d.crop_cn} · ${d.disease_cn}`,
  }))
  return diseaseOptionCache
}

export default function RecordDetailModal({ open, recordId, onClose, onChanged, onDeleted }) {
  const { message } = App.useApp()
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(false)
  const [savingName, setSavingName] = useState(false)
  const [savingVerdict, setSavingVerdict] = useState(false)
  const [savingGt, setSavingGt] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [diseaseOpts, setDiseaseOpts] = useState([])

  // 每次打开都重新拉：用户可能刚在别的标签页改过，也可能是删过的记录
  useEffect(() => {
    if (!open || !recordId) return
    let alive = true
    setLoading(true)
    setDetail(null)
    if (USE_MOCK) {
      // 演示模式没有后端：用列表里的 mock 行兜底，至少不会显示「记录不存在」
      const found = mockRecords().items.find((x) => x.id === recordId)
      setDetail(found ? { ...found, top3: [] } : null)
      setLoading(false)
      return
    }
    http
      .get(`/records/${recordId}`)
      .then((r) => {
        if (!alive) return
        const d = r.data || {}
        setDetail(d)
        setNameDraft(d.custom_name || '')
      })
      .catch((e) => {
        if (!alive) return
        const code = e?.response?.status
        message.error(code === 404 ? '该记录已不存在（可能已被删除）' : '详情加载失败，请稍后重试')
        onClose?.()
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, recordId])

  // 统一的 PATCH 出口：改名与判定走同一接口，错误提示也统一
  const patch = async (body, { onOk, setBusy, okMsg }) => {
    setBusy(true)
    try {
      const r = await http.patch(`/records/${recordId}`, body)
      const d = r.data || {}
      setDetail((prev) => ({ ...prev, ...d }))
      onOk?.(d)
      if (okMsg) message.success(okMsg)
      onChanged?.()
      return true
    } catch (e) {
      const msg = e?.response?.data?.msg || '保存失败，请稍后重试'
      message.error(msg)
      return false
    } finally {
      setBusy(false)
    }
  }

  const saveName = async () => {
    const next = (nameDraft || '').trim()
    // 没改就不发请求，避免每次点保存都刷一次「已保存」把列表无谓重拉
    if (next === (detail?.custom_name || '')) {
      message.info('名称未变更')
      return
    }
    await patch(
      { custom_name: next || null },
      { setBusy: setSavingName, okMsg: next ? '已重命名' : '已恢复为原始文件名' },
    )
  }

  const changeVerdict = async (v) => {
    const next = v === 'unset' ? null : v
    if (next === (detail?.verdict ?? null)) return
    await patch({ verdict: next }, { setBusy: setSavingVerdict })
  }

  // 「实际病害」下拉懒加载：只有判了「识别错误」才有意义，
  // 没判错就不该为 38 条字典数据多发一次请求。
  useEffect(() => {
    if (!open || detail?.verdict !== 'wrong' || diseaseOpts.length) return
    let alive = true
    loadDiseaseOptions()
      .then((opts) => {
        if (alive) setDiseaseOpts(opts)
      })
      .catch(() => {
        /* 字典拉不到就只保留「识别错误」结论，不阻断判定主流程 */
      })
    return () => {
      alive = false
    }
  }, [open, detail?.verdict, diseaseOpts.length])

  // 选了实际病害 = 把这条错误记录升级成带真值的标注样本；
  // 留空也允许 —— 挑错图这件事本身有价值，不必强填正确答案。
  const changeGroundTruth = async (v) => {
    const next = v ?? null
    if ((detail?.ground_truth_class_id ?? null) === next) return
    await patch(
      { ground_truth_class_id: next },
      { setBusy: setSavingGt, okMsg: next == null ? '已清除实际病害' : '已记录实际病害' },
    )
  }

  const doDelete = async () => {
    try {
      await http.delete(`/records/${recordId}`)
      message.success('已删除该记录')
      onDeleted?.(recordId)
      onClose?.()
    } catch {
      message.error('删除失败，请稍后重试')
    }
  }

  const top3 = detail?.top3 || []
  const verdictTag = VERDICT_TAG[detail?.verdict]

  return (
    <Modal
      open={open}
      onCancel={onClose}
      width={900}
      centered
      destroyOnHidden
      title={
        <Space size={8} wrap>
          <span>识别详情</span>
          {detail?.image_name && (
            <Typography.Text type="secondary" style={{ fontSize: 13, fontWeight: 400 }}>
              {detail.image_name}
            </Typography.Text>
          )}
          {verdictTag && <Tag color={verdictTag.color} style={{ marginInlineEnd: 0 }}>{verdictTag.text}</Tag>}
        </Space>
      }
      footer={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Popconfirm
            title="删除这条记录？"
            description="记录与已上传的原图会被一并删除，无法恢复。"
            okText="删除"
            okButtonProps={{ danger: true }}
            cancelText="取消"
            onConfirm={doDelete}
          >
            <Button danger icon={<DeleteOutlined />}>
              删除记录
            </Button>
          </Popconfirm>
          <Button onClick={onClose}>关闭</Button>
        </div>
      }
    >
      {loading ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : !detail ? (
        <Empty description="记录不存在" />
      ) : (
        <Row gutter={24}>
          {/* ---------- 左：原图（可点击放大 / 缩放 / 旋转） ---------- */}
          <Col xs={24} md={10}>
            {detail.preview ? (
              <Image
                src={detail.preview}
                alt={`${detail.disease_cn || '叶片'}原图`}
                // antd Image 的预览工具条自带放大、缩小、旋转、重置；
                // 这里只改蒙层文案，让「可以点」这件事显式说出来。
                preview={{ mask: <span>点击放大查看</span> }}
                style={{
                  width: '100%',
                  aspectRatio: '1 / 1',
                  objectFit: 'contain',
                  background: '#0B1F16',
                  borderRadius: 8,
                  border: '1px solid var(--c-border)',
                }}
              />
            ) : (
              <div
                style={{
                  width: '100%',
                  aspectRatio: '1 / 1',
                  borderRadius: 8,
                  border: '1px dashed var(--c-border)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--c-text-placeholder)',
                  background: 'var(--c-bg-subtle)',
                }}
              >
                <PictureOutlined style={{ fontSize: 28 }} />
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  原图已不在服务器上
                </Typography.Text>
              </div>
            )}
            <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
              图片可放大、缩放、旋转，用来对照病斑判断识别是否准确。
            </Typography.Text>
          </Col>

          {/* ---------- 右：名称 / 结果 / 人工判断 / 候选 ---------- */}
          <Col xs={24} md={14}>
            {/* 自定义命名 */}
            <div style={{ marginBottom: 16 }}>
              <Typography.Text strong style={{ fontSize: 13 }}>记录名称</Typography.Text>
              <Space.Compact style={{ width: '100%', marginTop: 6 }}>
                <Input
                  value={nameDraft}
                  onChange={(e) => setNameDraft(e.target.value)}
                  maxLength={40}
                  showCount
                  allowClear
                  placeholder={detail.orig_name || `历史图片 #${detail.id}`}
                  onPressEnter={saveName}
                  aria-label="记录名称"
                />
                <Button
                  type="primary"
                  icon={<SaveOutlined />}
                  loading={savingName}
                  onClick={saveName}
                >
                  保存
                </Button>
              </Space.Compact>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {detail.custom_name
                  ? `原始文件名：${detail.orig_name || '（旧记录未保存）'}`
                  : '留空则使用原始文件名；命名只影响你自己看到的列表与导出。'}
              </Typography.Text>
            </div>

            {/* 识别结果 */}
            <Descriptions size="small" column={1} colon={false} style={{ marginBottom: 8 }}>
              <Descriptions.Item label="识别结果">
                <Space size={8} wrap>
                  <Tag color={detail.disease_cn && detail.disease_cn.includes('健康') ? 'green' : 'volcano'}>
                    {detail.disease_cn || '—'}
                  </Tag>
                  <Typography.Text strong>{formatPercent(detail.confidence)}</Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    原始 softmax
                  </Typography.Text>
                  {detail.likely_not_leaf === 1 && <Tag color="gold">疑似非叶片</Tag>}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="识别时间">
                <Typography.Text style={{ fontSize: 13 }}>{detail.created_at}</Typography.Text>
              </Descriptions.Item>
            </Descriptions>

            {/* 人工判断（弹窗内也能改，与列表里的标注是同一个接口）。
                语义：用户已经知道真实病害，这里只回答「AI 认对了吗」，
                不需要提交自己的答案 —— 所以要写成「识别正确 / 识别错误」。 */}
            <div style={{ marginBottom: 16 }}>
              <Typography.Text strong style={{ fontSize: 13 }}>
                这次 AI 认对了吗？
              </Typography.Text>
              <div style={{ marginTop: 6 }}>
                <Segmented
                  block
                  value={detail.verdict || 'unset'}
                  options={VERDICT_OPTIONS}
                  disabled={savingVerdict}
                  onChange={changeVerdict}
                />
              </div>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                以你已知的真实病害为准：认错就点「识别错误」，这些图会被筛出来集中处理。
                模型对体系外作物、健康叶也会给高置信度答案，只有你能给真值。
              </Typography.Text>

              {/* 判为「识别错误」时才出现：顺手给出正确答案，这条记录就从
                  「一条错误日志」变成「一条带真值的标注样本」。留空也允许。 */}
              {detail.verdict === 'wrong' && (
                <div style={{ marginTop: 10 }}>
                  <Typography.Text strong style={{ fontSize: 13 }}>
                    实际是哪种病害？
                    <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                      （可选，选了就能直接当标注样本）
                    </Typography.Text>
                  </Typography.Text>
                  <Select
                    style={{ width: '100%', marginTop: 6 }}
                    allowClear
                    showSearch
                    optionFilterProp="label"
                    placeholder="我知道正确答案，选一下"
                    value={detail.ground_truth_class_id ?? undefined}
                    options={diseaseOpts}
                    loading={savingGt}
                    onChange={changeGroundTruth}
                    aria-label="实际病害"
                  />
                  {detail.ground_truth_cn && (
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      已记录：实际为「{detail.ground_truth_cn}」，AI 识别为「
                      {detail.disease_cn}」。
                    </Typography.Text>
                  )}
                  {/* 防矛盾样本：判了「识别错误」却选出与 AI 相同的病害，说明这次其实
                      是识别正确。不阻断（用户可能只是想先记下来），但必须提示 ——
                      否则统计里会混进「判错但答案是同一个」的行，口径自相矛盾。 */}
                  {detail.ground_truth_class_id != null
                    && detail.ground_truth_class_id === detail.pred_class_id && (
                    <Alert
                      type="warning"
                      showIcon
                      style={{ marginTop: 8, padding: '4px 10px' }}
                      message={
                        <span style={{ fontSize: 12 }}>
                          实际病害与 AI 识别结果相同 —— 若确实如此，这次应标为「识别正确」。
                        </span>
                      }
                    />
                  )}
                </div>
              )}
            </div>

            {/* 候选猜测 + 对应病害参考图 */}
            <Typography.Text strong style={{ fontSize: 13 }}>
              当时的候选猜测（Top-{top3.length || 3}）
            </Typography.Text>
            <div style={{ marginTop: 8 }}>
              {top3.length === 0 ? (
                <Alert
                  type="info"
                  showIcon
                  style={{ padding: '6px 10px' }}
                  message={
                    <span style={{ fontSize: 12 }}>
                      这条记录识别于候选存库功能上线之前，只保存了最终结论，没有候选与置信度明细。
                    </span>
                  }
                />
              ) : (
                top3.map((t, i) => (
                  <div
                    key={t.class_id + '-' + i}
                    style={{
                      display: 'flex',
                      gap: 10,
                      alignItems: 'center',
                      padding: '6px 0',
                      borderTop: i === 0 ? 'none' : '1px dashed var(--c-border)',
                    }}
                  >
                    {/* 每类有 3 张参考图，包进 PreviewGroup：点开后可直接左右翻 */}
                    <Image.PreviewGroup>
                      <Image
                        src={diseaseRefs(t.class_id)[0]}
                        alt={`${t.disease_cn}参考图`}
                        width={52}
                        height={52}
                        style={{ objectFit: 'cover', borderRadius: 6, flex: '0 0 auto' }}
                        preview={{ mask: <span style={{ fontSize: 11 }}>3 张</span> }}
                      />
                      {/* 其余两张只注册进预览组，不在页面上占位 */}
                      <div style={{ display: 'none' }}>
                        {diseaseRefs(t.class_id).slice(1).map((src) => (
                          <Image key={src} src={src} alt={`${t.disease_cn}参考图`} />
                        ))}
                      </div>
                    </Image.PreviewGroup>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                        <Space size={6} style={{ minWidth: 0 }}>
                          <Typography.Text
                            style={{ fontSize: 13, fontWeight: i === 0 ? 600 : 400 }}
                            ellipsis
                          >
                            {t.disease_cn}
                          </Typography.Text>
                          {i === 0 && <Tag color="green" style={{ marginInlineEnd: 0 }}>最可能</Tag>}
                          {t.is_healthy === 1 && <Tag style={{ marginInlineEnd: 0 }}>健康</Tag>}
                        </Space>
                        <Typography.Text type="secondary" style={{ fontSize: 12, flex: '0 0 auto' }}>
                          {formatPercent(t.prob)}
                        </Typography.Text>
                      </div>
                      <Progress
                        percent={Number(((t.prob || 0) * 100).toFixed(1))}
                        showInfo={false}
                        size="small"
                        strokeColor={i === 0 ? 'var(--c-primary-500)' : 'var(--c-primary-300)'}
                      />
                    </div>
                  </div>
                ))
              )}
            </div>
          </Col>
        </Row>
      )}
    </Modal>
  )
}
