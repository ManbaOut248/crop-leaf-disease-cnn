// 识别历史页：统计概览 + 三张分布图 + 列表 + 详情抽屉 + CSV 导出
//
// 数据来源说明（重要）：
//   列表走 /records（分页），统计走 /stats/overview（后端聚合）。
//   为什么统计不能让前端 reduce 列表得来：/records 分页上限 100 条，
//   记录一多，前端算出的比例分母就是错的，而且是静默算错。
//
// 口径提示：
//   图表里的「置信度」是**原始 softmax**，而列表里的可靠性标签基于温度校准值
//   （calibration_domain.json 的 reliability_high=0.55 / medium=0.25）。
//   两套阈值不可混用，因此图表副标题明确标注「原始 softmax」，不做 high/medium/low 分档。
import React, { useEffect, useMemo, useState } from 'react'
import {
  Table, Button, Tag, Typography, App, Row, Col, Card, Statistic, Alert,
  Progress, Space, Tooltip, Popconfirm, Dropdown, Segmented,
} from 'antd'
import {
  DownloadOutlined, PictureOutlined, EyeInvisibleOutlined, DeleteOutlined, DownOutlined,
} from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockRecords, mockStats } from '../api/mock'
import { formatPercent } from '../utils/format'
import { usePrefs, setPref, resetPrefs, PREF_META } from '../utils/prefs'
import ChartCard from '../components/ChartCard'
import RecordDetailModal from '../components/RecordDetailModal'

// 记录少于这个数就不画分布图：10 条散在 38 类里，每根柱子都是 1，
// 图上看不出任何信息，反而让人以为「系统没数据」。只保留数字卡更诚实。
const MIN_CHART_TOTAL = 10

// 与 main.jsx 的 AntD token 保持一致，不引入第二套色板
const C_PRIMARY = '#24814A'
const C_BRIGHT = '#2E9E5B'
const C_WARN = '#F2A93B'
const C_ERROR = '#D6453E'
const C_AXIS = '#7A8B81'
const C_GRID = '#EEF2EF'

const AXIS_LABEL = { fontSize: 11, color: C_AXIS }

export default function HistoryPage() {
  const { message } = App.useApp()
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState(null)
  const [statsLoading, setStatsLoading] = useState(true)
  // 详情弹窗只存 id：真实数据由弹窗自己拉 /records/{id}（列表行里的缩略图
  // 只有 96px、也没有 Top-3，直接拿来渲染详情必然要二次请求，不如一次到位）
  const [detailId, setDetailId] = useState(null)
  const [detailOpen, setDetailOpen] = useState(false)
  // 服务端分页状态（2026-10-29 修复「历史丢失」）
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [listTotal, setListTotal] = useState(0)
  // 列表筛选（2026-10-29 新增）：只需要「把识别错误的那批挑出来」。
  // 值域与后端 /records?verdict= 对齐：all / wrong / correct / unset。
  const [verdictFilter, setVerdictFilter] = useState('all')
  // 三个档位的条数：来自后端 counts（**全量历史**口径，不是 30 天窗口），
  // 因此筛选器的徽标数字与筛选结果永远一致。
  const [counts, setCounts] = useState({ all: 0, correct: 0, wrong: 0, unset: 0 })
  const prefs = usePrefs()

  // ⚠️ 历史 bug 记录（务必保留这段注释）：
  // 旧实现是 `http.get('/records')` —— 不带任何分页参数，后端默认 size=20，
  // 于是列表**永远只有最新 20 条**；后端返回的 total 被直接丢弃，表格改在前端
  // 分页（pageSize=8），只对那 20 条翻页。实测 user 5 有 180 条记录，界面只能
  // 看到 20 条，页脚却显示「累计识别 180 次」—— 用户看到的就是「历史丢失」。
  // 现在改为受控服务端分页，total 取接口值。
  const load = (p = page, ps = pageSize, vf = verdictFilter) => {
    setLoading(true)
    if (USE_MOCK) {
      // 演示模式没有后端：本地按同样的三档语义过滤，保证界面行为一致
      const all = mockRecords().items
      const kept = all.filter((r) => {
        if (vf === 'all') return true
        if (vf === 'unset') return !r.verdict
        return r.verdict === vf
      })
      setData(kept.slice((p - 1) * ps, p * ps))
      setListTotal(kept.length)
      setCounts({
        all: all.length,
        correct: all.filter((r) => r.verdict === 'correct').length,
        wrong: all.filter((r) => r.verdict === 'wrong').length,
        unset: all.filter((r) => !r.verdict).length,
      })
      setLoading(false)
      return
    }
    http
      .get('/records', { params: { page: p, size: ps, verdict: vf } })
      .then((r) => {
        const d = r.data || {}
        setData(d.items || [])
        setListTotal(d.total || 0)
        if (d.counts) setCounts(d.counts)
      })
      .catch((e) => {
        const msg = e?.response?.data?.msg || '加载失败，请稍后重试'
        message.error(msg)
      })
      .finally(() => setLoading(false))
  }

  // 统计单独加载：即使它失败，列表也应正常显示，不能被一个图表接口拖垮整页
  const loadStats = () => {
    setStatsLoading(true)
    if (USE_MOCK) {
      setStats(mockStats())
      setStatsLoading(false)
      return
    }
    http
      .get('/stats/overview', { params: { days: 30 } })
      .then((r) => setStats(r.data))
      .catch(() => message.warning('统计信息加载失败，列表不受影响'))
      .finally(() => setStatsLoading(false))
  }

  useEffect(() => {
    // 统计与分页无关，只在进页面时取一次
    loadStats()
  }, [])

  // 分页 / 页大小 / 筛选条件变化才重新拉列表
  useEffect(() => {
    load(page, pageSize, verdictFilter)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, pageSize, verdictFilter])

  // 改名 / 判定 / 删除之后：列表与统计都要重取。
  // 只刷列表会让「人工核对准确率」停在上一次的数字，用户以为没生效。
  const refreshAll = () => {
    load(page, pageSize, verdictFilter)
    loadStats()
  }

  // 切筛选：必须回到第 1 页。否则可能停在一个「换了筛选后已不存在」的页码上，
  // 表格空着、页脚却显示共 N 条 —— 又是一次「记录丢了」的错觉。
  const changeFilter = (vf) => {
    if (vf === verdictFilter) return
    setVerdictFilter(vf)
    setPage(1)
  }

  const onDelete = async (row) => {
    try {
      await http.delete(`/records/${row.id}`)
      message.success('已删除该记录')
      // 删的是当前页最后一条时回退一页 —— 否则会停在一页空表格上，
      // 而用户会以为「后面的记录也没了」（这类错觉正是上一轮「历史丢失」的成因）
      if (data.length - 1 === 0 && page > 1) {
        setPage(page - 1)
        loadStats()
      } else {
        refreshAll()
      }
    } catch (e) {
      message.error(e?.response?.data?.msg || '删除失败，请稍后重试')
    }
  }

  // 列表里直接标注对错：先请求成功再刷新，不做乐观更新 ——
  // 标注失败却已变绿，会让统计和界面长期不一致。
  const onVerdict = async (row, verdict) => {
    const next = verdict === 'unset' ? null : verdict
    if ((row.verdict ?? null) === next) return
    try {
      await http.patch(`/records/${row.id}`, { verdict: next })
      message.success(
        next === 'wrong' ? '已标记为「识别错误」' : next === 'correct' ? '已标记为「识别正确」' : '已清除标注',
      )
      // 整页重取而不是只改本地这一行：当前可能停在「仅识别错误」筛选下，
      // 改判成正确后这一行应当从列表消失、徽标数字也要同步减 1。
      refreshAll()
    } catch (e) {
      message.error(e?.response?.data?.msg || '标注失败，请稍后重试')
    }
  }

  // 导出走**后端全量接口**（2026-10-29 修复）。
  // 旧实现是拿列表页手里的 20 条自己拼 CSV —— 用户以为导出了全部历史，
  // 实际只有当前页那 20 条，且没有任何提示。现在口径只有后端一处。
  const onExport = async () => {
    if (USE_MOCK) {
      const header = 'id,image,result,confidence,time\n'
      const body = data
        .map(
          (r) =>
            `${r.id},${r.image_name},${r.disease_cn},${(r.confidence * 100).toFixed(2)}%,${r.created_at}`,
        )
        .join('\n')
      const blob = new Blob([header + body], { type: 'text/csv;charset=utf-8' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = '识别历史.csv'
      a.click()
      message.success('已导出 CSV（演示模式）')
      return
    }
    try {
      // responseType:'blob' 才能拿到二进制；http 拦截器返回的是 res.data 即 Blob
      const blob = await http.get('/records/export', {
        params: { fmt: 'csv' },
        responseType: 'blob',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `识别历史_${listTotal}条.csv`
      document.body.appendChild(a)
      a.click()
      a.remove()
      // 延迟回收：立即 revoke 在部分浏览器会让下载中断
      setTimeout(() => URL.revokeObjectURL(url), 1000)
      message.success(`已导出全部 ${listTotal} 条记录`)
    } catch {
      message.error('导出失败，请稍后重试')
    }
  }

  // ===================== 图表配置 =====================
  // 用 useMemo 固定引用：否则每次渲染都产生新 option 对象，
  // ChartCard 里的 setOption effect 会被无谓触发（图表闪烁）。
  const cropOption = useMemo(() => {
    const items = (stats?.crops || []).slice(0, 8)
    // 横向柱从下往上增长，数据要反转才能让第一条显示在顶部
    const names = items.map((i) => i.crop).reverse()
    const values = items.map((i) => i.count).reverse()
    return {
      grid: { left: 6, right: 36, top: 6, bottom: 6, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (p) => `${p[0].name}：${p[0].value} 次`,
      },
      xAxis: {
        type: 'value',
        minInterval: 1,
        splitLine: { lineStyle: { color: C_GRID } },
        axisLabel: AXIS_LABEL,
      },
      yAxis: {
        type: 'category',
        data: names,
        axisTick: { show: false },
        axisLine: { show: false },
        axisLabel: { fontSize: 12, color: '#4A5850' },
      },
      series: [
        {
          type: 'bar',
          data: values,
          barMaxWidth: 16,
          itemStyle: { color: C_PRIMARY, borderRadius: [0, 4, 4, 0] },
          label: { show: true, position: 'right', fontSize: 11, color: '#4A5850' },
        },
      ],
    }
  }, [stats])

  const confOption = useMemo(() => {
    const buckets = stats?.confidence_buckets || []
    // 低于 50%（复核线）单独用警示色，其余用绿色系由浅到深；
    // 不做红/黄/绿三色「可靠性」语义，避免与校准值口径的分档混淆
    const palette = [C_ERROR, C_WARN, '#C0DD97', '#97C459', '#5DCAA5', C_PRIMARY]
    return {
      grid: { left: 6, right: 8, top: 20, bottom: 4, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (p) => `${p[0].name}：${p[0].value} 条`,
      },
      xAxis: {
        type: 'category',
        data: buckets.map((b) => b.range),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#DDE6E0' } },
        axisLabel: { fontSize: 10, color: C_AXIS, interval: 0 },
      },
      yAxis: {
        type: 'value',
        minInterval: 1,
        splitLine: { lineStyle: { color: C_GRID } },
        axisLabel: AXIS_LABEL,
      },
      series: [
        {
          type: 'bar',
          data: buckets.map((b) => b.count),
          barMaxWidth: 26,
          itemStyle: {
            color: (p) => palette[p.dataIndex] || C_PRIMARY,
            borderRadius: [4, 4, 0, 0],
          },
          label: { show: true, position: 'top', fontSize: 11, color: '#4A5850' },
        },
      ],
    }
  }, [stats])

  const dailyOption = useMemo(() => {
    const daily = stats?.daily || []
    // 标签间隔：30 天全显示会挤成一团，按数据点数算出「每隔几个显示一个」
    const step = Math.max(0, Math.ceil(daily.length / 8) - 1)
    return {
      grid: { left: 6, right: 8, top: 16, bottom: 4, containLabel: true },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (p) => `${p[0].name}：${p[0].value} 条`,
      },
      xAxis: {
        type: 'category',
        data: daily.map((d) => d.date.slice(5)),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#DDE6E0' } },
        axisLabel: { fontSize: 10, color: C_AXIS, interval: step },
      },
      yAxis: {
        type: 'value',
        minInterval: 1,
        splitLine: { lineStyle: { color: C_GRID } },
        axisLabel: AXIS_LABEL,
      },
      series: [
        {
          type: 'bar',
          data: daily.map((d) => d.count),
          barMaxWidth: 16,
          itemStyle: { color: C_BRIGHT, borderRadius: [3, 3, 0, 0] },
        },
      ],
    }
  }, [stats])

  // ===================== 列表列定义 =====================
  // 人工判断的三态文案：与后端 db.VERDICT_VALUES 对齐（correct / wrong / null）。
  // 文案刻意写成「识别正确 / 识别错误」而不是「判断正确 / 判断错误」（2026-10-29）：
  // 被评价的是**模型这次的输出**，用户是拿着已知真实病害来判卷的人。
  // 原写法会被读成「我判断错了」，与功能意图正好相反。
  const VERDICT_UI = {
    correct: { color: 'green', text: '识别正确' },
    wrong: { color: 'red', text: '识别错误' },
    unset: { color: 'default', text: '未核对' },
  }

  const verdictMenu = (row) => ({
    items: [
      { key: 'correct', label: '识别正确（AI 认对了）' },
      { key: 'wrong', label: '识别错误（AI 认错了）' },
      { type: 'divider' },
      { key: 'unset', label: '清除标注' },
    ],
    onClick: ({ key, domEvent }) => {
      // 阻止冒泡：行本身不带点击行为，但避免将来加行点击时误触
      domEvent?.stopPropagation?.()
      onVerdict(row, key)
    },
  })

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
    {
      // 名称三级回落（自定义 > 原始文件名 > 历史图片 #id）由后端统一计算，
      // 前端只负责展示 —— 否则列表/导出/弹窗三处各算一遍迟早漂移。
      title: '名称',
      dataIndex: 'image_name',
      width: 210,
      ellipsis: { showTitle: true },
      render: (t, r) => (
        <Tooltip
          title={
            r.custom_name
              ? `自定义名称：${r.custom_name}（原始文件名：${r.orig_name || '旧记录未保存'}）`
              : t || `历史图片 #${r.id}`
          }
        >
          <span style={{ cursor: 'default' }}>
            {r.custom_name && (
              <Tag color="blue" style={{ marginInlineEnd: 6, padding: '0 4px', fontSize: 11 }}>
                自定义
              </Tag>
            )}
            {t || `历史图片 #${r.id}`}
          </span>
        </Tooltip>
      ),
    },
    {
      title: '识别结果',
      dataIndex: 'disease_cn',
      render: (t, r) => (
        <div>
          <Tag color={t && t.includes('健康') ? 'green' : 'volcano'} style={{ marginInlineEnd: 0 }}>
            {t}
          </Tag>
          {/* 实际病害（2026-10-29）：只在用户判「识别错误」并顺手给出正确答案时出现。
              不加独立列：表格已有 7 列，再加一列会把「名称」挤成省略号；
              紧跟识别结果反而更好对照 —— 一眼看出「认成了 A、其实是 B」。 */}
          {r.ground_truth_cn && (
            <div style={{ marginTop: 2 }}>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                实际：{r.ground_truth_cn}
              </Typography.Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      width: 92,
      render: (v) => formatPercent(v),
    },
    {
      // 人工判断：点标签弹出三选一。
      // 为什么不做「点一下循环切换」：三态循环很容易点过头，而且
      // 「正确→错误」这种切换一旦误触就把统计带偏了，多一次点击换确定性值得。
      title: (
        <Tooltip title="你已经知道真实病害，用这里给 AI 判卷：认对了选「识别正确」，认错了选「识别错误」">
          <span style={{ cursor: 'help', borderBottom: '1px dashed var(--c-border)' }}>
            人工判断
          </span>
        </Tooltip>
      ),
      dataIndex: 'verdict',
      width: 124,
      render: (v, r) => {
        const ui = VERDICT_UI[v] || VERDICT_UI.unset
        return (
          <Dropdown menu={verdictMenu(r)} trigger={['click']} placement="bottomLeft">
            <Tag
              color={ui.color}
              style={{ cursor: 'pointer', marginInlineEnd: 0 }}
              aria-label={`人工判断：${ui.text}，点击修改`}
            >
              {ui.text}
              <DownOutlined style={{ fontSize: 10, marginLeft: 4 }} />
            </Tag>
          </Dropdown>
        )
      },
    },
    { title: '时间', dataIndex: 'created_at', width: 158 },
    {
      title: '操作',
      width: 132,
      render: (_, r) => (
        <Space size={4}>
          <Button
            size="small"
            onClick={() => {
              setDetailId(r.id)
              setDetailOpen(true)
            }}
          >
            详情
          </Button>
          <Popconfirm
            title="删除这条记录？"
            description="记录与已上传的原图会被一并删除，无法恢复。"
            okText="删除"
            okButtonProps={{ danger: true }}
            cancelText="取消"
            onConfirm={() => onDelete(r)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} aria-label="删除记录" />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  // ===================== 派生状态 =====================
  const total = stats?.total ?? 0
  const chartReady = total >= MIN_CHART_TOTAL
  const ood = stats?.ood
  // 分母用 monitored（有判据快照的记录），不是 total —— 历史行 leaf_prob 为 NULL
  const oodRatio = ood && ood.flagged_ratio != null ? ood.flagged_ratio : null
  // 作物图高度按【实际画出来的条数】算（cropOption 里 slice(0,8)），
  // 早期用 stats.crops.length 全量计算，作物超过 8 个时会把卡片撑高、下方留白。
  const cropShown = Math.min(stats?.crops?.length || 0, 8)
  const distHeight = Math.max(180, cropShown * 26 + 24)

  // 人工核对统计（2026-10-29 新增）：官方口径在后端，这里只做展示。
  // accuracy 为 null 表示「一条都没核对过」，不能显示成 0%（0% 会被读成全错）。
  const vd = stats?.verdict
  const verdictAccuracy = vd && vd.accuracy != null ? vd.accuracy : null

  // 第二行三张卡（趋势 / 人工核对 / 非叶片拦截）按可见数量分配宽度：
  // 1 张占满、2 张各半、3 张 12/6/6。硬编码 16/8 在隐藏卡片后会留下半行空白。
  const row2Shown = [prefs.dailyTrend, prefs.verdictCard, prefs.oodCard].filter(Boolean).length
  const spanRow2 =
    row2Shown <= 1
      ? { daily: 24, verdict: 24, ood: 24 }
      : row2Shown === 2
        ? { daily: 12, verdict: 12, ood: 12 }
        : { daily: 12, verdict: 6, ood: 6 }

  // ---------- 图表可见性（第 2 项需求）----------
  // 每张卡片右上角一个「眼睛」按钮，点了就隐藏；恢复入口在个人中心。
  // 为什么不直接删掉不留入口：农户演示时想少点干扰，但事后又想看回来，
  // 隐藏成单向操作会让人不敢点。
  const hideBtn = (key, label) => (
    <Tooltip title={`隐藏「${label}」（个人中心可恢复）`}>
      <Button
        type="text"
        size="small"
        icon={<EyeInvisibleOutlined />}
        aria-label={`隐藏${label}`}
        onClick={() => {
          setPref(key, false)
          message.info(`已隐藏「${label}」，可在「个人中心 → 界面偏好」恢复`)
        }}
      />
    </Tooltip>
  )
  const anyChart = PREF_META.some((m) => prefs[m.key])
  // 两张卡并排时各占一半，只剩一张时占满整行，避免出现半行空白
  const pairSpan = (a, b, w) => (a && b ? w : 24)

  // ---------- 列表筛选器（2026-10-29 新增）----------
  // 这条功能的目的只有一个：把用户判为「识别错误」的那批图挑出来集中复核，
  // 它们是后续补真实域数据最直接的样本来源。
  // 徽标数字取自后端 `counts`（全部历史口径），而统计卡是 30 天窗口 ——
  // 两套数字混用就会出现「徽标 3 条、筛出来 5 条」的老问题，故此处严格区分。
  const FILTER_ITEMS = [
    { key: 'all', text: '全部', color: 'default' },
    { key: 'wrong', text: '仅识别错误', color: 'red' },
    { key: 'correct', text: '仅识别正确', color: 'green' },
    { key: 'unset', text: '未核对', color: 'default' },
  ]
  const filterOptions = FILTER_ITEMS.map((it) => ({
    value: it.key,
    label: (
      <span>
        {it.text}
        <Tag
          color={it.color}
          style={{
            marginInlineStart: 6, marginInlineEnd: 0, padding: '0 5px',
            fontSize: 11, lineHeight: '16px',
          }}
        >
          {counts[it.key] ?? 0}
        </Tag>
      </span>
    ),
  }))
  const filterText = FILTER_ITEMS.find((it) => it.key === verdictFilter)?.text || '全部'

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={3}>识别历史</Typography.Title>
        <Button icon={<DownloadOutlined />} onClick={onExport}>
          导出 CSV
        </Button>
      </div>

      {/* ---------- 概览：4 个数字卡 ----------
          ⚠️ 口径必须写明：这 4 项来自 /stats/overview，是「近 days 天」窗口
          （默认 30 天），而下方列表是**全部历史**。实测 demo 用户近 30 天 171 条、
          全量 180 条：旧标题写「累计识别」，用户看到 171 与列表「共 180 条」对不上，
          第一反应就是「记录丢了」—— 这本身就是「历史丢失」投诉的一部分成因。
          数字没错，是标签骗人。 */}
      <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 8 }}>
        以下 4 项为最近 {stats?.days ?? 30} 天统计；下方列表为全部历史记录。
      </Typography.Text>
      <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
        <Col xs={12} md={6}>
          <Card size="small" loading={statsLoading}>
            <Statistic
              title={`近 ${stats?.days ?? 30} 天识别`}
              value={total}
              suffix="次"
              valueStyle={{ fontSize: 24 }}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small" loading={statsLoading}>
            <Statistic
              title="涉及作物"
              value={stats?.crops?.length || 0}
              suffix="种"
              valueStyle={{ fontSize: 24 }}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small" loading={statsLoading}>
            <Statistic
              title="平均置信度"
              value={stats ? formatPercent(stats.avg_confidence) : '-'}
              valueStyle={{ fontSize: 24 }}
            />
          </Card>
        </Col>
        <Col xs={12} md={6}>
          <Card size="small" loading={statsLoading}>
            <Statistic
              title="需人工复核"
              value={stats?.need_review ?? 0}
              suffix="条"
              valueStyle={{ fontSize: 24, color: (stats?.need_review ?? 0) > 0 ? C_WARN : undefined }}
            />
          </Card>
        </Col>
      </Row>

      {/* ---------- 图表区：不足 10 条时降级为提示 ---------- */}
      {stats && total === 0 && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="还没有识别记录"
          description="去「病害识别」页上传一张叶片照片，这里会出现作物分布、置信度分布和识别趋势。"
        />
      )}

      {stats && total > 0 && total < MIN_CHART_TOTAL && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message={`已记录 ${total} 条，累计到 ${MIN_CHART_TOTAL} 条后显示分布图`}
          description="记录太少时每根柱子都是 1，图表看不出分布，先给数字更实在。"
        />
      )}

      {/* 图表全部被隐藏时给一条可恢复的提示，否则用户会以为页面坏了 */}
      {stats && chartReady && !anyChart && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 12 }}
          message="图表已全部隐藏"
          description="可在「个人中心 → 界面偏好」逐项开启，也可以直接点右侧按钮全部恢复。"
          action={
            <Button
              size="small"
              onClick={() => {
                resetPrefs()
                message.success('已恢复全部图表')
              }}
            >
              全部恢复
            </Button>
          }
        />
      )}

      {/* 图表的显隐各自独立：早先把两行都塞进 `(cropDist || confDist)` 的判断里，
          隐藏前两张图会把趋势/监控卡一起藏掉 —— 用户明明没关它们。 */}
      {stats && chartReady &&
        (prefs.cropDist || prefs.confDist || prefs.dailyTrend || prefs.verdictCard || prefs.oodCard) && (
        <>
          {(prefs.cropDist || prefs.confDist) && (
          <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
            {prefs.cropDist && (
              <Col xs={24} lg={pairSpan(prefs.cropDist, prefs.confDist, 12)}>
                <ChartCard
                  title="作物分布"
                  subtitle="按识别条数排序，最多显示 8 种"
                  option={cropOption}
                  height={distHeight}
                  extra={hideBtn('cropDist', '作物分布')}
                />
              </Col>
            )}
            {prefs.confDist && (
              <Col xs={24} lg={pairSpan(prefs.confDist, prefs.cropDist, 12)}>
                {/* 与左卡同高，避免两列卡片高低不齐 */}
                <ChartCard
                  title="置信度分布"
                  subtitle="原始 softmax 值，未做温度校准"
                  option={confOption}
                  height={distHeight}
                  extra={hideBtn('confDist', '置信度分布')}
                />
              </Col>
            )}
          </Row>
          )}

          {(prefs.dailyTrend || prefs.verdictCard || prefs.oodCard) && (
          <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
            {prefs.dailyTrend && (
              <Col xs={24} lg={spanRow2.daily}>
                <ChartCard
                  title={`近 ${stats.days} 天识别量`}
                  subtitle="无记录日期补零，避免趋势断裂"
                  option={dailyOption}
                  height={170}
                  extra={hideBtn('dailyTrend', '识别趋势')}
                />
              </Col>
            )}
            {prefs.verdictCard && (
              <Col xs={24} lg={spanRow2.verdict}>
                {/* 人工核对统计（2026-10-29）：分母是「已核对条数」而不是总识别数。
                    只有被人看过的那部分记录能给出真值，拿总数当分母会得到一个
                    既不是模型精度也不是人工精度的数字。 */}
                <Card
                  size="small"
                  title={
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 500 }}>人工核对准确率</div>
                      <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                        近 {stats.days} 天 · 仅统计已核对记录
                      </Typography.Text>
                    </div>
                  }
                  extra={hideBtn('verdictCard', '人工核对准确率')}
                  styles={{ body: { padding: '12px 12px 8px' } }}
                  style={{ height: '100%' }}
                >
                  {verdictAccuracy == null ? (
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      还没有核对过任何记录。在你已经知道真实病害的前提下，给 AI 判一次卷
                      （识别正确 / 识别错误），这里会显示核对出来的准确率。
                    </Typography.Text>
                  ) : (
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                        <span style={{ fontSize: 22, fontWeight: 500 }}>
                          {(verdictAccuracy * 100).toFixed(1)}%
                        </span>
                        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                          {vd.correct} / {vd.labeled} 条识别正确
                        </Typography.Text>
                      </div>
                      <Progress
                        percent={Number((verdictAccuracy * 100).toFixed(1))}
                        showInfo={false}
                        strokeColor={verdictAccuracy >= 0.8 ? C_PRIMARY : C_WARN}
                        size="small"
                      />
                      {/* 点数字直接跳到列表的对应筛选（第 2 项需求）：
                          这一卡是「按判卷结果」的唯一入口，看到错误条数后
                          最自然的动作就是想看是哪几条，不该让用户自己去下边找筛选器。 */}
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        <Typography.Link onClick={() => changeFilter('wrong')}>
                          识别错误 {vd.wrong} 条
                        </Typography.Link>
                        <span> · </span>
                        <Typography.Link onClick={() => changeFilter('unset')}>
                          未核对 {vd.unlabeled} 条
                        </Typography.Link>
                        {(vd.with_ground_truth ?? 0) > 0 && (
                          <span> · 已给出正确答案 {vd.with_ground_truth} 条</span>
                        )}
                      </Typography.Text>
                    </Space>
                  )}
                </Card>
              </Col>
            )}
            {prefs.oodCard && (
              <Col xs={24} lg={spanRow2.ood}>
                {/* OOD 监控：运维指标，与记录数无关，始终显示 */}
                <Card
                  size="small"
                  title={
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 500 }}>非叶片拦截</div>
                      <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
                        近 {ood?.days ?? stats.days} 天
                      </Typography.Text>
                    </div>
                  }
                  extra={hideBtn('oodCard', '非叶片拦截')}
                  styles={{ body: { padding: '12px 12px 8px' } }}
                  style={{ height: '100%' }}
                >
                {oodRatio == null ? (
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    暂无判据快照（旧记录不含叶片概率）
                  </Typography.Text>
                ) : (
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                      <span style={{ fontSize: 22, fontWeight: 500 }}>
                        {(oodRatio * 100).toFixed(1)}%
                      </span>
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        {ood.flagged} / {ood.monitored} 条
                      </Typography.Text>
                    </div>
                    <Progress
                      percent={Number((oodRatio * 100).toFixed(1))}
                      showInfo={false}
                      strokeColor={oodRatio > 0.5 ? C_ERROR : C_WARN}
                      size="small"
                    />
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {ood.msp_fallback > 0
                        ? `MSP 回退 ${ood.msp_fallback} 条，检测头未生效，需排查`
                        : 'MSP 回退 0 条，检测头正常'}
                    </Typography.Text>
                    {ood.avg_leaf_prob != null && (
                      <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                        平均叶片概率 {ood.avg_leaf_prob.toFixed(3)}
                      </Typography.Text>
                    )}
                  </Space>
                )}
                </Card>
              </Col>
            )}
          </Row>
          )}
        </>
      )}

      {/* ---------- 列表筛选：把「识别错误」的那批挑出来集中复核 ---------- */}
      <div
        style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexWrap: 'wrap', gap: 8, marginBottom: 8,
        }}
      >
        <Space size={8} wrap>
          <Typography.Text strong style={{ fontSize: 13 }}>
            人工判断筛选
          </Typography.Text>
          <Segmented
            value={verdictFilter}
            options={filterOptions}
            onChange={changeFilter}
            aria-label="按人工判断筛选记录"
          />
        </Space>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          {verdictFilter === 'all'
            ? '徽标为全部历史条数；上方图表为近 30 天窗口，两者口径不同'
            : `当前只看「${filterText}」共 ${listTotal} 条，切换后自动回到第 1 页`}
        </Typography.Text>
      </div>

      <Table
        rowKey="id"
        loading={loading}
        columns={columns}
        dataSource={data}
        // 受控服务端分页：current/pageSize 由我们维护，total 用接口返回值。
        // 早期是 `pagination={{ pageSize: 8 }}` 的前端分页，只对「已经拿到的
        // 20 条」翻页，总条数恒为 20，用户永远看不到更早的历史。
        pagination={{
          current: page,
          pageSize,
          total: listTotal,
          showSizeChanger: true,
          pageSizeOptions: [10, 20, 50, 100],
          showTotal: (t, range) => `第 ${range[0]}-${range[1]} 条 / 共 ${t} 条`,
          onChange: (p, ps) => {
            if (ps !== pageSize) {
              // 换页大小时回到第 1 页，否则可能停在一个已不存在的页码上
              setPageSize(ps)
              setPage(1)
            } else {
              setPage(p)
            }
          },
        }}
      />

      {/* 详情改成居中弹窗（2026-10-29）：Drawer 贴右、宽度受限，叶片图只能挤成
          96px，而看这张图的目的恰恰是对照病斑判断识别准不准。
          弹窗里有 720px 预览图（可放大/缩放/旋转）、Top-3 候选及各自的参考图、
          以及改名和「判断对错」的入口。 */}
      <RecordDetailModal
        open={detailOpen}
        recordId={detailId}
        onClose={() => setDetailOpen(false)}
        onChanged={refreshAll}
        onDeleted={refreshAll}
      />
    </div>
  )
}
