// 病害知识库页（2026-10-29 改版，方案 A：左栏作物导航 + 右侧卡片墙）
// ---------------------------------------------------------------------------
// 改版动机：
//   1. 旧版是「一长条分组列表」，14 种作物上下排下来，分组标题只是一行 Tag，
//      视觉上根本没有"分区"感；用户想找某个作物要一路滚。
//   2. 旧版数据源是前端硬编码的 mockDiseases，而后端 /diseases 已经有 38 条
//      完整数据（含 cause / treatment）却没人用 —— 两份数据迟早对不上。
//   3. 旧版没有图片，农户看不出"早疫"和"晚疫"到底差在哪。
//
// 现方案：
//   · 左栏常驻作物导航（含每类数量徽标），click 即筛；「全部作物」时按作物分组、
//     每组一个明显的分区标题 —— 这就是"分区更明显"的落点。
//   · 数据全部走后端 GET /api/v1/diseases（唯一来源），mock 模式才退回本地数据。
//   · 每张卡 1 张主图（占位小、信息密度合适），点击看该类别的 3 张参考图 + 完整说明。
//
// 图片走 `src/web/public/disease-ref/<class_id>-<n>.webp`，由
// `scripts/build_disease_refs.py` 从 PlantVillage 测试集抽样生成。
// 直接用 class_id 拼静态路径：不经过后端、不需要鉴权头，也不放大 data/ 的攻击面。
// 关键前提：后端 disease_knowledge.id 与模型类别序号严格一一对应（已校验 38/38）。
// ---------------------------------------------------------------------------
import React, { useEffect, useMemo, useState } from 'react'
import {
  Card, Tag, Input, Typography, Empty, Row, Col, Spin, Alert, Grid, Select, Modal, Divider,
} from 'antd'
import { SearchOutlined, PictureOutlined, AppstoreOutlined } from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockDiseases } from '../api/mock'

const REF_PER_CLASS = 3
const NAV_WIDTH = 216

function diseaseRefs(cid) {
  if (cid === null || cid === undefined || Number.isNaN(Number(cid))) return []
  return Array.from({ length: REF_PER_CLASS }, (_, i) => `/disease-ref/${cid}-${i + 1}.webp`)
}

/**
 * 把后端 /diseases 与本地 mock 两种结构归一化成一个形状。
 *
 * 后端：{ id, crop_cn, disease_cn, is_healthy, symptom, cause, prevention, treatment }
 * mock：{ id, crop, disease, name, isHealthy, symptom, prevention }
 *
 * 归一化的价值：页面只认一种字段，将来换数据源不用改渲染层。
 */
function normalize(d) {
  const isHealthy = d.isHealthy === true || d.is_healthy === 1 || d.is_healthy === true
  const crop = d.crop_cn || d.crop || ''
  const rawName = d.disease_cn || d.name || ''
  // 后端健康类的 disease_cn 就是「健康」二字，直接拼会显示成孤零零的「健康」，
  // 补上作物名才看得出是「番茄（健康）」。
  const name = isHealthy
    ? rawName.includes('健康') && rawName.includes(crop)
      ? rawName
      : `${crop}（健康）`
    : rawName
  return {
    id: d.id,
    crop,
    name,
    isHealthy,
    symptom: d.symptom || '',
    cause: d.cause || '',
    prevention: d.prevention || '',
    treatment: d.treatment || '',
  }
}

/** 参考图：加载失败时退化为占位图标，而不是浏览器默认的破图 */
function RefImage({ src, alt, style }) {
  const [failed, setFailed] = useState(false)
  if (failed || !src) {
    return (
      <div
        style={{
          ...style,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--c-bg-subtle)',
        }}
      >
        <PictureOutlined style={{ fontSize: 22, color: 'var(--c-text-placeholder)' }} />
      </div>
    )
  }
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
      style={{ ...style, display: 'block', objectFit: 'cover' }}
    />
  )
}

export default function KnowledgePage() {
  const screens = Grid.useBreakpoint()
  const isNarrow = !screens.lg
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [list, setList] = useState([])
  const [kw, setKw] = useState('')
  // 选中作物：'' = 全部作物；点击左栏切换
  const [activeCrop, setActiveCrop] = useState('')
  // 详情弹窗：存整条归一化记录，null 为关闭
  const [detail, setDetail] = useState(null)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError('')
    if (USE_MOCK) {
      setList(mockDiseases.map(normalize))
      setLoading(false)
      return () => {
        alive = false
      }
    }
    http
      .get('/diseases')
      .then((res) => {
        if (!alive) return
        const rows = (res.data && res.data) || []
        if (!rows.length) {
          setError('后端未返回病害数据，请检查数据库是否已初始化（disease_knowledge 表）。')
        }
        setList(rows.map(normalize))
      })
      .catch((e) => {
        if (!alive) return
        setError(
          e && e.response
            ? `病害知识加载失败（HTTP ${e.response.status}），请稍后重试。`
            : '无法连接后端（端口 8000），病害知识暂时不可用。',
        )
      })
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [])

  // 作物清单：按类别数降序（与后端 /crops 口径一致），数量在导航里直接可见
  const crops = useMemo(() => {
    const m = new Map()
    list.forEach((d) => {
      if (!d.crop) return
      m.set(d.crop, (m.get(d.crop) || 0) + 1)
    })
    return [...m.entries()]
      .map(([crop, n]) => ({ crop, n }))
      .sort((a, b) => b.n - a.n || a.crop.localeCompare(b.crop, 'zh'))
  }, [list])

  // 搜索 + 作物筛选
  const filtered = useMemo(() => {
    const k = kw.trim()
    return list.filter((d) => {
      if (activeCrop && d.crop !== activeCrop) return false
      if (!k) return true
      return (
        d.name.includes(k) ||
        d.crop.includes(k) ||
        d.symptom.includes(k) ||
        d.cause.includes(k) ||
        d.prevention.includes(k)
      )
    })
  }, [list, kw, activeCrop])

  // 按作物分组：选了单个作物时只有一组；「全部」时每组带明显的分区标题
  const groups = useMemo(() => {
    const m = new Map()
    filtered.forEach((d) => {
      if (!m.has(d.crop)) m.set(d.crop, [])
      m.get(d.crop).push(d)
    })
    return crops
      .filter((c) => m.has(c.crop))
      .map((c) => ({ crop: c.crop, items: m.get(c.crop) }))
  }, [filtered, crops])

  const searching = !!kw.trim()
  const totalCrops = crops.length

  // ---------------- 左栏导航 ----------------
  const navItems = [{ crop: '', label: '全部作物', n: list.length }, ...crops.map((c) => ({
    crop: c.crop,
    label: c.crop,
    n: c.n,
  }))]

  const navNode = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {navItems.map((it) => {
        const active = activeCrop === it.crop
        return (
          <button
            key={it.crop || '__all__'}
            type="button"
            onClick={() => setActiveCrop(it.crop)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 8,
              width: '100%',
              padding: isNarrow ? '6px 12px' : '8px 10px',
              border: 'none',
              borderRadius: 8,
              cursor: 'pointer',
              textAlign: 'left',
              fontSize: 13,
              // 选中项用主色浅底 + 主色文字：这是"当前分区"的唯一视觉锚点，
              // 比加粗更能一眼分辨，也不干扰卡片本身的层级。
              background: active ? 'var(--c-primary-50)' : 'transparent',
              color: active ? 'var(--c-primary-700)' : 'var(--c-text)',
              fontWeight: active ? 600 : 400,
              whiteSpace: 'nowrap',
            }}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
              {it.crop === '' && <AppstoreOutlined style={{ fontSize: 13, opacity: 0.7 }} />}
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{it.label}</span>
            </span>
            <span
              style={{
                flexShrink: 0,
                fontSize: 12,
                color: active ? 'var(--c-primary-700)' : 'var(--c-text-sub)',
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              {it.n}
            </span>
          </button>
        )
      })}
    </div>
  )

  // ---------------- 单张病害卡片 ----------------
  const renderCard = (d) => (
    <Card
      key={`${d.crop}-${d.id}`}
      size="small"
      hoverable
      onClick={() => setDetail(d)}
      styles={{ body: { padding: 0 } }}
      style={{ overflow: 'hidden', height: '100%' }}
    >
      {/* 主图：1:1 裁切，2 列/3 列都稳定；卡片整体可点，图片不单独吞事件 */}
      <div style={{ position: 'relative', width: '100%', aspectRatio: '1 / 1', background: 'var(--c-bg-subtle)' }}>
        <RefImage
          src={diseaseRefs(d.id)[0]}
          alt={`${d.name}示例图`}
          style={{ width: '100%', height: '100%' }}
        />
        <Tag
          color={d.isHealthy ? 'green' : 'volcano'}
          style={{ position: 'absolute', top: 8, left: 8, margin: 0, fontSize: 11 }}
        >
          {d.isHealthy ? '健康' : '病害'}
        </Tag>
      </div>
      <div style={{ padding: '10px 12px 12px' }}>
        <Typography.Text strong style={{ fontSize: 13, display: 'block', lineHeight: 1.45 }}>
          {d.name}
        </Typography.Text>
        <Typography.Paragraph
          type="secondary"
          style={{
            fontSize: 12,
            margin: '6px 0 0',
            lineHeight: 1.55,
            // 症状只给两行摘要：卡片是"索引"，完整内容在弹窗里
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {d.symptom || '暂无描述'}
        </Typography.Paragraph>
      </div>
    </Card>
  )

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Typography.Title level={3} style={{ marginBottom: 4 }}>
          病害知识库
        </Typography.Title>
        <Typography.Text type="secondary" style={{ fontSize: 13 }}>
          收录 <b>{list.length}</b> 个类别（{totalCrops} 种作物），与识别模型支持的 38 类一一对应。
          点击卡片可查看症状参考图与防治建议。
        </Typography.Text>
      </div>

      {error && (
        <Alert type="warning" showIcon style={{ marginBottom: 12 }} message={error} />
      )}

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexDirection: isNarrow ? 'column' : 'row' }}>
        {/* 左栏：作物导航。宽屏常驻 + sticky，滚动时始终能换作物；
            窄屏改为横向可滚动的 chips，避免占掉整屏宽度。 */}
        <Card
          size="small"
          title="作物分类"
          style={
            isNarrow
              ? { width: '100%' }
              : { width: NAV_WIDTH, flexShrink: 0, position: 'sticky', top: 12 }
          }
          styles={{ body: { padding: isNarrow ? 8 : '8px 8px 10px' } }}
        >
          <div
            style={
              isNarrow
                ? { display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 2 }
                : undefined
            }
          >
            {isNarrow ? (
              <Select
                value={activeCrop || ''}
                onChange={setActiveCrop}
                style={{ width: '100%' }}
                options={navItems.map((it) => ({
                  value: it.crop,
                  label: `${it.label}（${it.n}）`,
                }))}
              />
            ) : (
              <Spin spinning={loading} size="small">
                {navNode}
              </Spin>
            )}
          </div>
        </Card>

        {/* 右侧：搜索 + 分区卡片墙 */}
        <div style={{ flex: 1, minWidth: 0, width: '100%' }}>
          <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
            <Input
              prefix={<SearchOutlined />}
              placeholder="搜索作物、病害名或症状关键词，如：晚疫 / 褐色病斑"
              value={kw}
              onChange={(e) => setKw(e.target.value)}
              style={{ flex: 1, minWidth: 240, maxWidth: 420 }}
              allowClear
            />
            {activeCrop && (
              // 当前筛选态必须可见且可一键清除：左栏在窄屏会折叠成下拉，
              // 用户看完卡片后容易忘记自己还带着作物过滤。
              <Tag
                closable
                onClose={() => setActiveCrop('')}
                color="green"
                style={{ alignSelf: 'center', margin: 0, padding: '3px 8px' }}
              >
                仅看：{activeCrop}
              </Tag>
            )}
          </div>

          {loading ? (
            <div style={{ padding: 48, textAlign: 'center' }}>
              {/* antd 5 的 Spin 只有在 nest / fullscreen 模式下才认 tip，
                  单独用 tip 会打一条 warning（控制台噪音），所以文案自己写。 */}
              <Spin size="large" />
              <div style={{ marginTop: 12, fontSize: 13, color: 'var(--c-text-sub)' }}>
                加载病害知识…
              </div>
            </div>
          ) : filtered.length === 0 ? (
            <Empty
              description={
                searching || activeCrop
                  ? `没有匹配「${kw || activeCrop}」的病害，换个关键词试试`
                  : '暂无病害数据'
              }
            />
          ) : (
            groups.map((g) => (
              <section key={g.crop} style={{ marginBottom: 28 }}>
                {/* 分区标题：色块 + 作物名 + 数量。旧版只有一行 Tag，14 个作物滚下来
                    完全分不清边界；这里用左侧竖色条 + 细分隔线，视觉上是一屏一章。 */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    marginBottom: 12,
                    paddingLeft: 10,
                    borderLeft: '3px solid var(--c-primary-500)',
                  }}
                >
                  <Typography.Title level={5} style={{ margin: 0 }}>
                    {g.crop}
                  </Typography.Title>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    共 {g.items.length} 类
                  </Typography.Text>
                </div>
                <Row gutter={[12, 12]}>
                  {g.items.map((d) => (
                    <Col xs={12} sm={8} md={8} lg={8} xl={6} xxl={6} key={`${g.crop}-${d.id}`}>
                      {renderCard(d)}
                    </Col>
                  ))}
                </Row>
              </section>
            ))
          )}
        </div>
      </div>

      {/* 详情弹窗：3 张参考图 + 完整症状/病因/防治/治疗 */}
      <Modal
        open={!!detail}
        onCancel={() => setDetail(null)}
        footer={null}
        width={720}
        title={detail ? detail.name : ''}
      >
        {detail && (
          <>
            <div style={{ display: 'flex', gap: 8 }}>
              {diseaseRefs(detail.id).map((src, i) => (
                <RefImage
                  key={src}
                  src={src}
                  alt={`${detail.name}参考图 ${i + 1}`}
                  style={{
                    width: '33%',
                    aspectRatio: '1 / 1',
                    borderRadius: 8,
                    border: '1px solid var(--c-border)',
                  }}
                />
              ))}
            </div>
            <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 8 }}>
              示例图来自 PlantVillage 公开数据集，同一病害在不同品种、不同发病阶段外观差异较大。
            </Typography.Text>

            <Divider style={{ margin: '14px 0 12px' }} />

            <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
              <Tag color="green">{detail.crop}</Tag>
              <Tag color={detail.isHealthy ? 'green' : 'volcano'}>
                {detail.isHealthy ? '健康' : '病害'}
              </Tag>
              <Tag>类别编号 #{detail.id}</Tag>
            </div>

            {[
              { label: '症状特征', value: detail.symptom },
              { label: '发病原因', value: detail.cause },
              { label: '防治建议', value: detail.prevention },
              { label: '治疗措施', value: detail.treatment },
            ].map((row) =>
              // 后端部分字段（尤其 treatment）可能为空，空行直接跳过，
              // 避免弹窗里出现一排「无」把信息淹没
              row.value ? (
                <div key={row.label} style={{ marginBottom: 12 }}>
                  <Typography.Text strong style={{ fontSize: 13 }}>
                    {row.label}
                  </Typography.Text>
                  <Typography.Paragraph style={{ margin: '4px 0 0', fontSize: 13, lineHeight: 1.7 }}>
                    {row.value}
                  </Typography.Paragraph>
                </div>
              ) : null,
            )}
          </>
        )}
      </Modal>
    </div>
  )
}
