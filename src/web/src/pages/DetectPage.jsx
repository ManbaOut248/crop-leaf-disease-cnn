// 病害识别页：上传图片 → 调 /predict → Top-3 概率 + 病害说明 + 防治建议
// 响应契约对齐 docs/04_API接口文档.md：
//   { pred_class_id, crop_cn, disease_cn, is_healthy, confidence,
//     top3:[{class_id, disease_cn, prob}], need_review, knowledge:{symptom,cause,prevention,treatment},
//     likely_not_leaf, leaf_prob, crop_applied, crop_candidates }
//
// 作物先验（2026-10-19）：后端把 softmax 限制在所选作物的候选类内。
// 作物白名单前置（2026-10-29）：真实识别必须先选择列表内作物，不再提供全 38 类无限制入口。
// 目的不是提升精度，而是在入口声明支持范围，减少未知作物被强行判成体系内病害（docs/25）。
import React, { useState, useEffect, useRef } from 'react'
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
  Modal,
  Segmented,
  Tooltip,
  Spin,
} from 'antd'
import {
  UploadOutlined,
  ExperimentOutlined,
  ScissorOutlined,
  PictureOutlined,
  DeleteOutlined,
  AppstoreOutlined,
  FileImageOutlined,
} from '@ant-design/icons'
import http, { USE_MOCK } from '../api/http'
import { mockPredict } from '../api/mock'
import { formatPercent } from '../utils/format'
import { diseaseRefs } from '../utils/diseaseRefs'
import {
  shrinkToDataUrl,
  dataUrlToFile,
  saveSession,
  loadSession,
  clearSession,
} from '../utils/detectSession'
import ShotGuide from '../components/ShotGuide'
import LeafCropModal from '../components/LeafCropModal'
import BatchItemModal from '../components/BatchItemModal'

const LOW_CONF = 0.5 // 置信度低于此值提示人工复核（对齐 API 文档）

// 批量识别上限，与后端 BATCH_MAX_FILES 保持一致（后端是最终裁决，
// 这里只是提前拦住，避免传 10 张再被 400 退回）。
const BATCH_MAX = 9

// 批量结果卡片的状态标签。优先级刻意与结果区的提示规则一致：
// 「可能不是叶片」压过「置信度低」—— 非叶片图上的高置信度毫无意义，
// 两条都显示只会让用户不知道该信哪个。
function batchStatus(it) {
  if (it.likely_not_leaf) return { color: 'red', text: '可能非叶片' }
  if (it.need_review) return { color: 'gold', text: '置信度低' }
  return it.is_healthy ? { color: 'green', text: '健康' } : { color: 'volcano', text: '病害' }
}

// 把 axios 异常翻译成用户能据此行动的信息。
// 旧实现只有一句「识别失败，请重试」，把「后端没启动」「登录过期」「限流」全糊在一起，
// 用户无法判断该重启服务、重新登录还是稍后再试。
function describeRequestError(e) {
  if (!e || !e.response) {
    return {
      message: '无法连接识别服务，请确认后端已启动（端口 8000）后重试。',
      unauthorized: false,
    }
  }
  const { status, data } = e.response
  const serverMsg = (data && (data.msg || data.detail)) || ''
  if (status === 401) {
    return { message: '登录已过期，请重新登录。', unauthorized: true }
  }
  if (status === 429) {
    return {
      message: serverMsg || '请求过于频繁，请稍等一分钟后再试。',
      unauthorized: false,
    }
  }
  return {
    message: serverMsg
      ? `${serverMsg}（HTTP ${status}）`
      : `识别失败（HTTP ${status}），请重试。`,
    unauthorized: false,
  }
}

export default function DetectPage() {
  const { message, modal } = App.useApp()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  // 原图 dataURL：裁剪框选始终基于原图，避免「裁一次再裁一次」的画质累积损失
  const [sourcePreview, setSourcePreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  // 作物白名单前置：真实识别必须先选择作物，不再允许跳过选择后走全 38 类。
  // 依据 docs/25：未知作物有 88.2% 会得到确定诊断，入口声明支持范围比事后提示更有效。
  const [crop, setCrop] = useState('')
  const [cropOptions, setCropOptions] = useState([])
  const [cropLoading, setCropLoading] = useState(!USE_MOCK)
  const [cropLoadError, setCropLoadError] = useState(false)
  const [singleCandidateCrops, setSingleCandidateCrops] = useState([])
  // 裁剪弹窗开关：把整株远景手动框成单片叶片近景（依据 docs/23 外部域评估的结论）
  const [cropOpen, setCropOpen] = useState(false)
  // 病害参考图弹窗：{ cid, name }，为 null 时关闭（第 3 项需求）
  const [refModal, setRefModal] = useState(null)
  // 会话记忆（第 5 项需求）：fileName 用于历史记录里的「图片名」，
  // restored 标记本次进入页面是从快照恢复的（据此给一条提示 + 重新开始入口）。
  const [fileName, setFileName] = useState('')
  const [restored, setRestored] = useState(false)

  // ===================== 批量识别状态（2026-10-30 新增） =====================
  // mode：单张 / 批量。刻意不做持久化 —— 用户下次进来看到「批量」模式却只有单张的
  // 操作习惯，会点错按钮。每次进页面从单张开始最符合预期。
  const [mode, setMode] = useState('single')
  // 待识别的批量图片：[{ uid, file, name, preview }]
  // preview 是降采样 dataURL（长边 1024），只用于缩略图与裁剪取景；
  // 原图始终留在 file 里上传，绝不用降采样图去推理。
  const [batchFiles, setBatchFiles] = useState([])
  const [batchLoading, setBatchLoading] = useState(false)
  // 后端返回：{ items, errors, summary, crop, top_k }
  const [batchResult, setBatchResult] = useState(null)
  // 当前打开的批量结果项（点卡片进来），item 里带 index 以便回贴
  const [activeItem, setActiveItem] = useState(null)
  // 待裁剪重识别的目标：{ index, src }，src 是该张的降采样 dataURL
  const [recropTarget, setRecropTarget] = useState(null)
  const [recropBusy, setRecropBusy] = useState(false)


  // 作物清单从后端取，不在前端硬编码 —— 类别体系以后端为唯一来源。
  // 清单既是准确率先验，也是产品的支持范围白名单；加载失败时必须明确提示并重试，
  // 不能像旧版一样静默退化为「不选作物也能识别」。
  const loadCrops = () => {
    if (USE_MOCK) return
    setCropLoading(true)
    setCropLoadError(false)
    http
      .get('/crops')
      .then((res) => {
        const d = res.data || {}
        const items = d.items || []
        setCropOptions(items.map((x) => ({ value: x.crop, label: x.crop })))
        setSingleCandidateCrops(d.single_candidate || [])
        if (items.length === 0) setCropLoadError(true)
      })
      .catch(() => {
        setCropOptions([])
        setCropLoadError(true)
      })
      .finally(() => setCropLoading(false))
  }

  useEffect(() => {
    loadCrops()
  }, [])

  // 进页面时恢复上次的会话（第 5 项需求）。
  // 只恢复一次：若恢复后又被 useEffect 重新执行，用户正在输入的新图会被旧快照覆盖。
  useEffect(() => {
    const snap = loadSession()
    if (!snap) return
    setCrop(snap.crop || '')
    setFileName(snap.fileName || '')
    setPreview(snap.preview || '')
    // 恢复后的「原图」只能是降采样图 —— 原图没进快照（体积原因），裁剪取景精度略降，
    // 但导出仍是 448×448，肉眼无差别。
    setSourcePreview(snap.preview || '')
    setResult(snap.result || null)
    if (snap.preview) {
      const f = dataUrlToFile(snap.preview, snap.fileName || 'restored.jpg')
      if (f) setFile(f)
    }
    setRestored(true)
  }, [])

  // 结果 / 图片 / 作物任一变化就落一份快照：
  // 这样「识别完 → 去历史页 → 回识别页」能原样恢复，刷新也不丢。
  // 空状态不写，避免清空后又被这条 effect 写回一份空快照。
  useEffect(() => {
    if (!result && !preview) return
    saveSession({ crop, fileName, preview, result })
  }, [result, preview, crop, fileName])

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
    // 原图 dataURL 只留在内存里给裁剪取景用，不进 sessionStorage：
    // 手机照片的 dataURL 常有 5MB 以上，写进去会直接撑爆配额（约 5MB）。
    const reader = new FileReader()
    reader.onload = (e) => setSourcePreview(e.target.result)
    reader.readAsDataURL(f)
    // 展示与持久化都用降采样图（长边 1024 / JPEG 0.82）：预览位只有 220px 高，
    // 原图存进去纯属浪费，而 1024px 在 2x 屏上依然清晰。
    shrinkToDataUrl(f)
      .then(setPreview)
      .catch(() => {
        // canvas 失败（极少数色彩空间异常）时退回原图，至少保证能看见
        const r2 = new FileReader()
        r2.onload = (e) => setPreview(e.target.result)
        r2.readAsDataURL(f)
      })
    setFile(f)
    // 真实文件名一起记下来 —— 历史页「图片名」显示的就是它（第 4 项需求）
    setFileName(f.name || '')
    setResult(null)
    setRestored(false)
    return false // 阻止自动上传，改为手动“开始识别”
  }

  // 清空本轮识别（图片 + 结果 + 本地快照）。
  // 作物选择刻意保留：连续识别同一作物的场景更多，清掉反而要多点一次。
  const resetAll = () => {
    clearSession()
    setFile(null)
    setPreview('')
    setSourcePreview('')
    setResult(null)
    setFileName('')
    setRestored(false)
  }

  // 识别主流程：显式接收待上传文件，避免裁剪后立刻识别时读到尚未更新的 state。
  // （setFile 是异步的，若 onCropped 里直接调 onDetect，闭包里的 file 仍是裁剪前的旧图。）
  const runDetect = (targetFile) => {
    if (!USE_MOCK && !crop) {
      message.warning('请先选择列表内的作物；列表外作物暂不支持识别')
      return
    }
    if (!targetFile) {
      message.warning('请先选择一张叶片图片')
      return
    }
    // 新请求开始即清空旧结果：否则请求失败时右侧仍展示上一次结论，
    // 用户会把它误读成「本次识别的结果」。
    setResult(null)
    setLoading(true)
    if (USE_MOCK) {
      setTimeout(() => {
        setResult(mockPredict())
        setLoading(false)
      }, 900)
      return
    }
    const fd = new FormData()
    fd.append('file', targetFile)
    http
      .post('/predict', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        // 白名单前置后真实请求必定带作物参数，杜绝绕过选择走全 38 类。
        params: { crop },
      })
      .then((res) => setResult(res.data))
      .catch((e) => {
        const { message: text, unauthorized } = describeRequestError(e)
        message.error(text)
        // token 已被拦截器清除；这里跳登录页，避免用户停在页面上反复点识别。
        if (unauthorized) setTimeout(() => window.location.assign('/login'), 1200)
      })
      .finally(() => setLoading(false))
  }

  // 裁剪完成：替换待上传文件 → 立即用裁剪图重新识别（按钮文案承诺的行为）。
  const onCropped = (croppedFile) => {
    // LeafCropModal 固定导出 `leaf_crop.jpg`。若直接上传，历史列表里每张裁剪图都会
    // 叫同一个名字，用户根本分不清。这里重命名成 `<原名>__裁剪.jpg`，让「图片名」
    // 能自解释（第 4 项需求把文件名真正显示出来之后，这一点才变得重要）。
    const displayName = `${(fileName || '图片').replace(/\.[^.]+$/, '')}__裁剪.jpg`
    const renamed = new File([croppedFile], displayName, {
      type: croppedFile.type || 'image/jpeg',
    })
    shrinkToDataUrl(renamed)
      .then(setPreview)
      .catch(() => {
        const reader = new FileReader()
        reader.onload = (e) => setPreview(e.target.result)
        reader.readAsDataURL(renamed)
      })
    setFile(renamed)
    setFileName(displayName)
    setResult(null)
    setRestored(false)
    setCropOpen(false)
    if (!USE_MOCK && !crop) {
      message.warning('已按单片叶片裁剪，请先选择作物再识别')
      return
    }
    message.success('已按单片叶片裁剪，正在重新识别…')
    // 必须传 renamed：传 croppedFile 的话，multipart 里的 filename 仍是
    // leaf_crop.jpg，后端落库的「图片名」就白改了。
    runDetect(renamed)
  }

  const onDetect = () => runDetect(file)

  // ===================== 批量识别（2026-10-30） =====================
  // 「已选图片」的真值放在 ref 里，state 只负责渲染。
  // 原因：Upload.Dragger 的 beforeUpload 是一次选多张时**逐个同步调用**的，
  // 而 setState 是异步批处理的 —— 若用 state.length 判断上限，一次拖 12 张时
  // 每一次调用看到的都是同一个旧长度，限制形同虚设。
  const batchRef = useRef([])

  const syncBatch = (next) => {
    batchRef.current = next
    setBatchFiles(next)
  }

  const onBatchPick = (f) => {
    const okType = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'].includes(f.type)
    if (!okType) {
      // 同一个 key：一次拖 12 张里有 3 张不合格时只会刷新一条提示，
      // 否则屏幕上会叠三层一模一样的 toast，看着像连点了三次报错
      message.error({ content: `${f.name}：仅支持 JPG / PNG / BMP / WEBP 格式`, key: 'batch-reject' })
      return Upload.LIST_IGNORE
    }
    if (f.size > 10 * 1024 * 1024) {
      message.error({ content: `${f.name}：图片大小不能超过 10MB`, key: 'batch-reject' })
      return Upload.LIST_IGNORE
    }
    if (batchRef.current.length >= BATCH_MAX) {
      message.warning({
        content: `一次最多识别 ${BATCH_MAX} 张，多出的图片已忽略`,
        key: 'batch-overflow',
      })
      return Upload.LIST_IGNORE
    }
    const uid = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    // 先占位（preview 为空）再异步回填，保证卡片顺序与用户选择顺序一致；
    // 若等预览生成完再 push，先选的大图会被后选的小图插队。
    syncBatch([...batchRef.current, { uid, file: f, name: f.name || '未命名.jpg', preview: '' }])
    shrinkToDataUrl(f)
      .then((url) => {
        syncBatch(batchRef.current.map((x) => (x.uid === uid ? { ...x, preview: url } : x)))
      })
      .catch(() => {
        // 预览失败不影响上传与识别（极少数色彩空间异常才会走到这里），
        // 卡片显示占位图标即可，不该因此拦住整批
      })
    return false // 阻止自动上传，改为点「开始批量识别」
  }

  const removeBatchFile = (uid) => syncBatch(batchRef.current.filter((x) => x.uid !== uid))

  const clearBatch = () => {
    syncBatch([])
    setBatchResult(null)
    setActiveItem(null)
  }

  // 后端 summary 与 items 重算：单张重识别之后，卡片里的健康/病害数字必须重算，
  // 否则会出现「统计说 6 张病害、卡片里只有 5 张」这种对不上的情况。
  const summarize = (items, failed = 0) => {
    const healthy = items.filter((x) => x.is_healthy).length
    return {
      total: items.length + failed,
      ok: items.length,
      failed,
      healthy,
      disease: items.length - healthy,
      likely_not_leaf: items.filter((x) => x.likely_not_leaf).length,
      need_review: items.filter((x) => x.need_review).length,
    }
  }

  const doBatch = () => {
    if (USE_MOCK) {
      // 演示模式：直接把同一份 mock 结果铺开，验证网格布局与交互
      setBatchLoading(true)
      setTimeout(() => {
        const items = batchRef.current.map((x, i) => ({
          ...mockPredict(),
          index: i,
          file_name: x.name,
          record_id: null,
        }))
        setBatchResult({ items, errors: [], summary: summarize(items), crop })
        setBatchLoading(false)
      }, 900)
      return
    }
    setBatchLoading(true)
    setBatchResult(null)
    setActiveItem(null)
    const fd = new FormData()
    batchRef.current.forEach((x) => fd.append('files', x.file, x.name))
    http
      .post('/predict/batch', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { crop },
        // 9 张的推理 + 上传远超默认 30s 的余量不多，单独放宽到 3 分钟：
        // 宁可比默认久，也不要在用户已经等了几秒之后抛「网络错误」让他重传一遍。
        timeout: 180000,
      })
      .then((r) => {
        const d = r.data || {}
        setBatchResult(d)
        const s = d.summary || {}
        if (s.failed) {
          message.warning(`已识别 ${s.ok} 张，${s.failed} 张失败（原因见下方列表）`)
        } else {
          message.success(`已识别 ${s.ok} 张，已存入识别历史`)
        }
      })
      .catch((e) => {
        const { message: text, unauthorized } = describeRequestError(e)
        message.error(text)
        if (unauthorized) setTimeout(() => window.location.assign('/login'), 1200)
      })
      .finally(() => setBatchLoading(false))
  }

  // 批量识别前的「同一作物」确认。
  // 为什么不只放一条 Alert：Alert 会被划过去当成装饰。作物先验的代价是「选错比不选更糟」——
  // 混进来的其他作物会被强行判成本作物的病害，而且置信度往往不低，
  // 用户会以为「限定作物后更准」。所以每次开跑都让他当面确认一次。
  const runBatch = () => {
    if (!USE_MOCK && !crop) {
      message.warning('请先选择作物：整批图片会共用这一个作物')
      return
    }
    const n = batchRef.current.length
    if (n === 0) {
      message.warning('请先选择图片')
      return
    }
    modal.confirm({
      title: `确认这 ${n} 张都是「${crop || '同一作物'}」的叶片吗？`,
      width: 460,
      content: (
        <div style={{ fontSize: 13 }}>
          <p style={{ marginBottom: 8 }}>
            整批图片会<b>共用「{crop || '所选作物'}」这一个作物先验</b>，
            模型只在该作物的候选病害范围内比较。
          </p>
          <p style={{ marginBottom: 0, color: 'var(--c-warning-600)' }}>
            如果混入了其他作物，它会被强行判成本作物的病害，而且置信度往往不低 ——
            比不选作物更容易误导。请先去下方缩略图逐张核对。
          </p>
        </div>
      ),
      okText: `确认，识别 ${n} 张`,
      cancelText: '我再检查一下',
      onOk: doBatch,
    })
  }

  // 打开「框选单片叶片」：取该张在内存里的原图（降采样版，与单张恢复会话同样的取舍）
  const openRecrop = (item) => {
    const target = batchRef.current[item.index]
    if (!target) {
      message.warning('这张图已不在本页，请重新上传后再裁剪')
      return
    }
    setRecropTarget({ index: item.index, src: target.preview || '', name: target.name })
  }

  // 裁剪完成后：走**单张**接口重新识别（不重跑整批），并把结果贴回对应卡片。
  // 这是一次独立的识别，会在历史里新增一条 —— 原来那条保留，方便对比两版结果。
  const onRecropCropped = (croppedFile) => {
    const idx = recropTarget?.index
    if (idx === undefined || idx === null) return
    const base = (recropTarget.name || '图片').replace(/\.[^.]+$/, '')
    const renamed = new File([croppedFile], `${base}__裁剪.jpg`, {
      type: croppedFile.type || 'image/jpeg',
    })
    // 缩略图跟着换成裁剪结果：卡片还是原来那张大图会让人以为没生效
    shrinkToDataUrl(renamed)
      .then((url) => {
        syncBatch(batchRef.current.map((x, i) => (i === idx ? { ...x, preview: url } : x)))
      })
      .catch(() => {})

    if (USE_MOCK) {
      setRecropTarget(null)
      message.success('已按单片叶片重新识别（演示模式）')
      return
    }

    setRecropBusy(true)
    const fd = new FormData()
    fd.append('file', renamed, renamed.name)
    http
      .post('/predict', fd, { headers: { 'Content-Type': 'multipart/form-data' }, params: { crop } })
      .then((r) => {
        const d = r.data || {}
        const patched = { ...d, index: idx, file_name: renamed.name }
        setBatchResult((prev) => {
          if (!prev) return prev
          const items = (prev.items || []).map((it) => (it.index === idx ? patched : it))
          return { ...prev, items, summary: summarize(items, (prev.summary || {}).failed || 0) }
        })
        setActiveItem((cur) => (cur && cur.index === idx ? patched : cur))
        message.success('已按单片叶片重新识别，并在识别历史新增一条记录')
      })
      .catch((e) => {
        const { message: text, unauthorized } = describeRequestError(e)
        message.error(text)
        if (unauthorized) setTimeout(() => window.location.assign('/login'), 1200)
      })
      .finally(() => {
        setRecropBusy(false)
        setRecropTarget(null)
      })
  }


  // 把后端契约字段归一化，便于展示
  const top1 = result && {
    name: result.disease_cn,
    confidence: result.confidence || 0,
    is_healthy: !!(result.is_healthy === true || result.is_healthy === 1),
    crop: result.crop_cn,
    // 「一病多名」别名（如 苹果雪松锈病 = 苹果桧锈病），有值时在结果区提示，避免用户误判为识别错误
    alias: (result.knowledge && result.knowledge.alias) || '',
    symptom: result.knowledge && result.knowledge.symptom,
    prevention:
      (result.knowledge && (result.knowledge.prevention || result.knowledge.treatment)) || '',
  }
  const top3 = (result && result.top3) || []
  // 真实域置信度校准：后端用 PlantDoc 拟合的温度缩放后，得到「这个数字在真实田间照片上
  // 到底该信多少」的估计。仅当校准生效（生成了 calibration_domain.json 且开关打开）才有意义。
  const calibrationEnabled = !!(result && result.calibration_enabled)
  const confCal = result && result.confidence_calibrated
  const reliability = (result && result.reliability) || 'unknown'
  // 可靠性分档 → 文案 / 配色（high 绿、medium 金、low 红、校准未生效时灰）
  const RELIABILITY_META = {
    high: { color: 'green', text: '结果较可靠' },
    medium: { color: 'gold', text: '结果仅供参考' },
    low: { color: 'red', text: '结果可能不可靠' },
    unknown: { color: 'default', text: '未校准' },
  }
  const relMeta = RELIABILITY_META[reliability] || RELIABILITY_META.unknown
  // 「大概率不是叶片」时只提示这一件事，不再叠加"置信度低"的橙色提示，避免两条信息打架
  const needReview =
    result &&
    !result.likely_not_leaf &&
    (result.need_review || (top1 && top1.confidence < LOW_CONF))

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 16,
          flexWrap: 'wrap',
        }}
      >
        <div>
          <Typography.Title level={3} style={{ marginBottom: 4 }}>
            病害识别
          </Typography.Title>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            {mode === 'single'
              ? '上传一张农作物叶片照片，系统将自动识别病害类别并给出防治建议。'
              : `一次最多 ${BATCH_MAX} 张，逐张独立识别；每张都会存入识别历史，可事后人工核对。`}
          </Typography.Paragraph>
        </div>
        <Segmented
          value={mode}
          onChange={(v) => {
            setMode(v)
            // 切模式时关掉批量结果详情，否则单张模式下还飘着一个批量弹窗
            setActiveItem(null)
          }}
          options={[
            { label: '单张识别', value: 'single' },
            { label: '批量识别', value: 'batch' },
          ]}
        />
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {/* 左：上传 + 预览（单张）/ 多选上传（批量） */}
        <Card
          title={mode === 'single' ? '上传叶片图片' : `批量上传（最多 ${BATCH_MAX} 张）`}
          style={{ width: 360 }}
        >
          {/* 白名单前置：先声明支持范围，再让用户上传。
              它不能阻止用户故意选错作物，也不提升模型精度；作用是避免列表外作物
              在没有任何提醒的情况下被强行判成 38 类之一（docs/25：88.2% 会硬答）。
              批量模式下压成一行：左栏还要放「同作物警告 + 上传 + 已选缩略图」，
              三条 Alert 叠起来会把缩略图挤到一屏之外 —— 而缩略图正是核对
              「是不是同一作物」的唯一手段，它必须在首屏。 */}
          {mode === 'single' ? (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 12 }}
              message="当前仅支持列表内作物的叶片"
              description="列表外作物暂不支持识别，请勿将其他作物的识别结果用于防治决策。"
            />
          ) : (
            <Typography.Text
              type="secondary"
              style={{ fontSize: 12, display: 'block', marginBottom: 10 }}
            >
              仅支持列表内作物的叶片；列表外作物请勿据此用药。
            </Typography.Text>
          )}
          {/* 批量模式的「同一作物」前置声明（2026-10-30）。
              这条不是客套话：作物先验把 softmax 限制在该作物的候选类里，
              混进来的其他作物会被强行归到本作物，而且置信度通常不低 ——
              结果是「选了作物所以更准」的错觉，比不选作物更容易误导。 */}
          {mode === 'batch' && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 12 }}
              message="整批图片必须是同一种作物"
              description="下面选的作物会套用到全部图片。混入其他作物会被误判成本作物的病害，请在上传前逐张确认。"
            />
          )}
          <div style={{ marginBottom: 14 }}>
            <Typography.Text strong style={{ display: 'block', marginBottom: 6 }}>
              作物类型（必选）
            </Typography.Text>
            <Select
              showSearch
              optionFilterProp="label"
              value={crop || undefined}
              onChange={(v) => {
                setCrop(v)
                setResult(null) // 更换作物后旧结果不再适用，避免误读
              }}
              placeholder="请先选择作物"
              options={cropOptions}
              style={{ width: '100%' }}
              loading={cropLoading}
              disabled={USE_MOCK || cropLoading || cropOptions.length === 0}
            />
            {cropLoadError && (
              <Alert
                type="warning"
                showIcon
                style={{ marginTop: 8 }}
                message="作物清单加载失败，暂不能识别"
                action={
                  <Button size="small" onClick={loadCrops} loading={cropLoading}>
                    重试
                  </Button>
                }
              />
            )}
            {crop && singleCandidateCrops.includes(crop) && (
              <Typography.Text type="warning" style={{ fontSize: 12 }}>
                该作物在模型中只有一个候选类别，结果等同直接给出结论，参考价值有限。
              </Typography.Text>
            )}
            {mode === 'batch' && (
              <Typography.Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 6 }}>
                {crop
                  ? `已选「${crop}」：这批图片会全部按此作物判定，请确认它们都是它。`
                  : '选择作物后才能开始批量识别（作物只选一次，整批共用）。'}
              </Typography.Text>
            )}
          </div>
          {/* 拍摄指南只在单张模式展示：它占 200px 以上，而批量模式下首屏要留给
              「已选缩略图」，否则用户没法核对这 9 张是不是同一种作物。 */}
          {mode === 'single' ? (
            <ShotGuide />
          ) : (
            <Typography.Text
              type="secondary"
              style={{ fontSize: 12, display: 'block', marginBottom: 10 }}
            >
              拍摄建议：单片叶片占满画面。整株 / 远景容易误判，识别后可在结果里对单张框选重识别。
            </Typography.Text>
          )}
          {mode === 'single' ? (
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
          ) : (
            <>
              <div className="drop-zone" style={{ textAlign: 'center' }}>
                <Upload.Dragger
                  multiple
                  beforeUpload={onBatchPick}
                  showUploadList={false}
                  accept="image/*"
                  disabled={batchFiles.length >= BATCH_MAX}
                >
                  {/* 紧凑版上传区：已经有图时，用户的主要动作是「核对」而不是「继续加」，
                      所以收成一行 —— 把纵向空间让给下方的缩略图网格，
                      9 张要在一屏内看全才能逐张核对是不是同一作物。 */}
                  <div style={{ padding: batchFiles.length ? '0' : '6px 0' }}>
                    {!batchFiles.length && (
                      <UploadOutlined
                        style={{ fontSize: 30, color: 'var(--c-text-placeholder)' }}
                      />
                    )}
                    <p
                      className="ant-upload-text"
                      style={{ marginTop: batchFiles.length ? 0 : 6, marginBottom: 0 }}
                    >
                      {batchFiles.length >= BATCH_MAX
                        ? `已满 ${BATCH_MAX} 张`
                        : batchFiles.length
                        ? `继续添加（已选 ${batchFiles.length}/${BATCH_MAX}）`
                        : '点击或拖拽，可一次多选'}
                    </p>
                    {!batchFiles.length && (
                      <p className="ant-upload-hint" style={{ fontSize: 12 }}>
                        JPG / PNG / BMP / WEBP，单张 ≤10MB
                      </p>
                    )}
                  </div>
                </Upload.Dragger>
              </div>
              {/* 已选缩略图：逐张可见是「同一作物」自查的唯一手段 ——
                  只给一个「已选 9 张」的数字，用户没有任何办法确认自己选错了哪张。 */}
              {batchFiles.length > 0 && (
                <div
                  className="batch-thumb-grid"
                  style={{
                    marginTop: 10,
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                    gap: 8,
                  }}
                >
                  {batchFiles.map((x, i) => (
                    // mouseEnterDelay 800ms：9 张时鼠标划过去会连续弹 9 个提示，
                    // 把下面的缩略图盖住，反而妨碍「逐张核对是不是同一作物」
                    <Tooltip key={x.uid} title={x.name} mouseEnterDelay={0.8} placement="top">
                      <div
                        style={{
                          position: 'relative',
                          aspectRatio: '1 / 1',
                          borderRadius: 6,
                          overflow: 'hidden',
                          border: '1px solid var(--c-border)',
                          background: 'var(--c-bg-subtle)',
                        }}
                      >
                        {x.preview ? (
                          <img
                            src={x.preview}
                            alt={x.name}
                            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                          />
                        ) : (
                          <div
                            style={{
                              width: '100%',
                              height: '100%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: 'var(--c-text-placeholder)',
                            }}
                          >
                            <FileImageOutlined />
                          </div>
                        )}
                        <span
                          style={{
                            position: 'absolute',
                            left: 3,
                            top: 3,
                            background: 'rgba(0,0,0,.55)',
                            color: '#fff',
                            fontSize: 11,
                            lineHeight: '15px',
                            padding: '0 5px',
                            borderRadius: 8,
                          }}
                        >
                          {i + 1}
                        </span>
                        {/* 删除键用半透明黑底而不是实心红：9 张时有 9 个红点，
                            视觉噪音盖过图片本身，而这里第一要务是看清叶片。 */}
                        <Button
                          size="small"
                          type="text"
                          icon={<DeleteOutlined />}
                          onClick={() => removeBatchFile(x.uid)}
                          style={{
                            position: 'absolute',
                            right: 2,
                            top: 2,
                            width: 18,
                            height: 18,
                            minWidth: 18,
                            borderRadius: '50%',
                            padding: 0,
                            fontSize: 10,
                            background: 'rgba(0,0,0,.45)',
                            color: '#fff',
                          }}
                        />
                      </div>
                    </Tooltip>
                  ))}
                </div>
              )}
            </>
          )}
          {/* 裁剪入口：只在单张模式、已选图后出现。整株远景 → 框到单片叶片，
              是对 docs/23 里 17.1% 那个结果最直接、且不依赖任何自动检测的补救手段。
              批量模式不提供逐张裁剪（9 张一张张框太慢），改为在结果里对个别不准的图单张重裁。 */}
          {mode === 'single' && preview && (
            <Button
              icon={<ScissorOutlined />}
              block
              onClick={() => setCropOpen(true)}
              style={{ marginTop: 10 }}
            >
              框选单片叶片
            </Button>
          )}
          {mode === 'single' ? (
            <>
              <Button
                type="primary"
                icon={<ExperimentOutlined />}
                block
                loading={loading}
                onClick={onDetect}
                disabled={!USE_MOCK && (!crop || cropOptions.length === 0)}
                style={{ marginTop: 16 }}
              >
                开始识别
              </Button>
              {(preview || result) && (
                <Button type="link" size="small" block onClick={resetAll} style={{ marginTop: 4 }}>
                  清空本张图片与结果
                </Button>
              )}
            </>
          ) : (
            <>
              <Button
                type="primary"
                icon={<AppstoreOutlined />}
                block
                loading={batchLoading}
                onClick={runBatch}
                disabled={
                  (!USE_MOCK && (!crop || cropOptions.length === 0)) || batchFiles.length === 0
                }
                style={{ marginTop: 16 }}
              >
                开始批量识别（{batchFiles.length}/{BATCH_MAX}）
              </Button>
              {batchFiles.length > 0 && (
                <Button type="link" size="small" block onClick={clearBatch} style={{ marginTop: 4 }}>
                  清空已选图片与结果
                </Button>
              )}
            </>
          )}
          {mode === 'single' && result && result.is_mock && (
            <Alert
              style={{ marginTop: 12 }}
              type="info"
              showIcon
              message="演示数据"
              description="模型尚未加载，以下为示例结果。"
            />
          )}
        </Card>

        {/* 右：结果（单张 / 批量） */}
        <Card
          title={mode === 'single' ? '识别结果' : '批量识别结果'}
          style={{ flex: 1, minWidth: 360 }}
        >
          {mode === 'batch' ? (
            <>
              {batchLoading ? (
                <div style={{ padding: '48px 0', textAlign: 'center' }}>
                  <Spin size="large" />
                  <div style={{ marginTop: 12, color: 'var(--c-text-sub)' }}>
                    正在识别 {batchFiles.length} 张，请稍候…
                  </div>
                </div>
              ) : !batchResult ? (
                <Typography.Text type="secondary">
                  尚未识别。请先选择作物、选好这批图片（必须同一作物），再点「开始批量识别」。
                </Typography.Text>
              ) : (
                <>
                  {/* 汇总：先给「成功几张、失败几张」，因为用户最关心的是有没有漏掉 */}
                  <Alert
                    type={batchResult.summary && batchResult.summary.failed ? 'warning' : 'success'}
                    showIcon
                    message={`共 ${batchResult.summary?.total ?? 0} 张：成功 ${
                      batchResult.summary?.ok ?? 0
                    } 张${
                      batchResult.summary?.failed ? `，失败 ${batchResult.summary.failed} 张` : ''
                    }`}
                    description={`作物「${batchResult.crop || crop}」 · 健康 ${
                      batchResult.summary?.healthy ?? 0
                    } · 病害 ${batchResult.summary?.disease ?? 0} · 可能非叶片 ${
                      batchResult.summary?.likely_not_leaf ?? 0
                    } · 置信度偏低 ${batchResult.summary?.need_review ?? 0}`}
                  />
                  {/* 失败清单单独列文件名与原因：只说「失败 1 张」等于让用户自己猜是哪张 */}
                  {(batchResult.errors || []).length > 0 && (
                    <Alert
                      type="warning"
                      showIcon
                      style={{ marginTop: 10 }}
                      message="以下图片未能识别"
                      description={
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                          {(batchResult.errors || []).map((e) => (
                            <li key={e.index}>
                              <b>{e.file_name}</b>：{e.msg}
                            </li>
                          ))}
                        </ul>
                      }
                    />
                  )}
                  <div
                    style={{
                      marginTop: 14,
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                      gap: 12,
                    }}
                  >
                    {(batchResult.items || []).map((it) => {
                      // index 是上传顺序，用它去内存里取对应的缩略图；
                      // 用「成功序号」会在中间有失败时整排错位
                      const img = (batchFiles.find((x, i) => i === it.index) || {}).preview || ''
                      const st = batchStatus(it)
                      return (
                        <div
                          key={it.index}
                          role="button"
                          tabIndex={0}
                          title={`${it.file_name} · 点击查看该张的 Top-3 与参考图`}
                          onClick={() => setActiveItem(it)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              setActiveItem(it)
                            }
                          }}
                          style={{
                            border: '1px solid var(--c-border)',
                            borderRadius: 8,
                            overflow: 'hidden',
                            cursor: 'pointer',
                            background: 'var(--c-bg-elevated, #fff)',
                          }}
                        >
                          <div
                            style={{
                              position: 'relative',
                              aspectRatio: '1 / 1',
                              background: 'var(--c-bg-subtle)',
                            }}
                          >
                            {img ? (
                              <img
                                src={img}
                                alt={it.file_name}
                                style={{
                                  width: '100%',
                                  height: '100%',
                                  objectFit: 'cover',
                                  display: 'block',
                                }}
                              />
                            ) : (
                              <div
                                style={{
                                  width: '100%',
                                  height: '100%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  color: 'var(--c-text-placeholder)',
                                }}
                              >
                                <FileImageOutlined />
                              </div>
                            )}
                            <Tag color={st.color} style={{ position: 'absolute', left: 6, top: 6, margin: 0 }}>
                              {st.text}
                            </Tag>
                          </div>
                          <div style={{ padding: '8px 9px' }}>
                            <div
                              style={{
                                fontWeight: 500,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {it.disease_cn}
                            </div>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                gap: 6,
                                marginTop: 4,
                                fontSize: 12,
                                color: 'var(--c-text-sub)',
                              }}
                            >
                              <span
                                style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                              >
                                {it.file_name}
                              </span>
                              <span style={{ flexShrink: 0 }}>{formatPercent(it.confidence)}</span>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                  <Typography.Text
                    type="secondary"
                    style={{ fontSize: 12, display: 'block', marginTop: 12 }}
                  >
                    点击任意一张可查看该张的 Top-3、参考图与防治建议；识别不准的可在弹窗里
                    <b>框选单片叶片后单独重识别</b>。批量结果不随切页保留，需要回看请到「识别历史」——
                    每一张都已存入历史记录。
                  </Typography.Text>
                </>
              )}
            </>
          ) : (
            <>
          {/* 会话记忆提示（第 5 项需求）：用户从其他板块切回本页时，
              结果是被「恢复」而不是「刚算出来的」，必须说清楚 ——
              否则他可能把上一张图（甚至昨天的）当成本次识别的结论。 */}
          {restored && result && (
            <Alert
              type="success"
              showIcon
              style={{ marginBottom: 12 }}
              message="这是上次识别的结果，已自动保留"
              description={
                <>
                  切到其他页面再回来不会丢失，关闭浏览器标签页后自动清除。如需重新开始，
                  可以直接上传新图片，或
                  {/* 用 Typography.Link 而不是 Button：Button 是 inline-flex 块级元素，
                      会整体换行、把句号甩到下一行单独成行；Link 是行内元素，随文字自然折行。
                      再加 nowrap 防止「清空重来」四个字被拆到两行。 */}
                  <Typography.Link onClick={resetAll} style={{ whiteSpace: 'nowrap' }}>
                    清空重来
                  </Typography.Link>
                  。
                </>
              }
            />
          )}
          {!result && (
            <Typography.Text type="secondary">
              尚未识别，请先选择作物、上传图片，再点击「开始识别」。
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

              {/* 取景提醒（docs/23）：外部域实测显示整株远景会让 Top-1 掉到 17%，
                  而单叶近景是 43.6%。这里不自动判断拍法（三轮实验证明不可靠），
                  只在结果不可信时给一个手动裁剪的快捷入口。 */}
              {(result.likely_not_leaf || reliability === 'low' || needReview) && (
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="如果这张是整株或远距离拍摄，建议先框到单片叶片"
                  description={
                    <div>
                      模型是按「单片叶片占满画面」的照片训练的。把一片叶子框满画面后再识别，
                      结果通常会明显更准。
                      <div style={{ marginTop: 8 }}>
                        <Button size="small" icon={<ScissorOutlined />} onClick={() => setCropOpen(true)}>
                          立即框选单片叶片
                        </Button>
                      </div>
                    </div>
                  }
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
                  {/* 真实域置信度校准：把「实验室风格的 99%」翻译成「田间实测可能只有 18%」，
                      用彩色 Tag 直接告诉用户该不该信这个结果。校准未生效时不显示。 */}
                  {calibrationEnabled && (
                    <Tag color={relMeta.color} style={{ fontSize: 12, marginLeft: 6 }}>
                      校准：{relMeta.text}
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
                  {/* 校准后的置信度：这是后端按真实田间数据重新缩放的数值，更接近真实准确率。
                      与上方原始置信度并排展示，让用户直观看到「模型自报的把握」和
                      「校准后实际可信度」之间的差距（真实田间场景差距极大）。 */}
                  {calibrationEnabled && typeof confCal === 'number' && (
                    <div
                      style={{
                        fontSize: 12,
                        color:
                          reliability === 'low'
                            ? 'var(--c-error-600)'
                            : reliability === 'medium'
                            ? 'var(--c-warning-600)'
                            : 'var(--c-success-600)',
                        marginTop: 4,
                      }}
                    >
                      真实场景校准：{formatPercent(confCal)}
                    </div>
                  )}
                </div>
              </div>

              <Typography.Paragraph strong style={{ marginBottom: 8 }}>
                Top-3 可能性
                <Typography.Text
                  type="secondary"
                  style={{ fontSize: 12, fontWeight: 400, marginLeft: 8 }}
                >
                  点图片看症状示例
                </Typography.Text>
              </Typography.Paragraph>
              {top3.map((t, i) => {
                const refs = diseaseRefs(t.class_id)
                const openRef = () => refs.length && setRefModal({ cid: t.class_id, name: t.disease_cn })
                return (
                  <div key={i} style={{ marginBottom: 10 }}>
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                      {/* 参考图：只给概率数字时，农户分不清「番茄早疫」和「番茄晚疫」差在哪；
                          配一张同类别的真实病叶能显著降低误读。
                          拿不到 class_id（mock 模式）时用占位块，避免破图。 */}
                      <div
                        role={refs.length ? 'button' : undefined}
                        tabIndex={refs.length ? 0 : undefined}
                        title={refs.length ? '查看该病害的参考图' : '暂无参考图'}
                        onClick={openRef}
                        onKeyDown={(e) => {
                          if (refs.length && (e.key === 'Enter' || e.key === ' ')) {
                            e.preventDefault()
                            openRef()
                          }
                        }}
                        style={{
                          flexShrink: 0,
                          width: 48,
                          height: 48,
                          borderRadius: 6,
                          overflow: 'hidden',
                          border: '1px solid var(--c-border)',
                          background: 'var(--c-bg-subtle)',
                          cursor: refs.length ? 'zoom-in' : 'default',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        {refs.length ? (
                          <img
                            src={refs[0]}
                            alt={`${t.disease_cn}参考图`}
                            loading="lazy"
                            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                          />
                        ) : (
                          <PictureOutlined style={{ color: 'var(--c-text-placeholder)' }} />
                        )}
                      </div>

                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                          <span
                            style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          >
                            {i === 0 ? '最可能' : `第 ${i + 1} 可能`}：{t.disease_cn}
                          </span>
                          <span style={{ flexShrink: 0 }}>{formatPercent(t.prob)}</span>
                        </div>
                        <Progress
                          percent={Math.round((t.prob || 0) * 100)}
                          showInfo={false}
                          strokeColor={i === 0 ? 'var(--c-primary-500)' : 'var(--c-primary-300)'}
                        />
                      </div>
                    </div>
                  </div>
                )
              })}

              <Divider />
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="作物">{top1.crop}</Descriptions.Item>
                {/* 一病多名：如「苹果雪松锈病」又称「苹果桧锈病」，不标出来用户会误以为识别错了 */}
                {top1.alias && (
                  <Descriptions.Item label="又称">{top1.alias}</Descriptions.Item>
                )}
                <Descriptions.Item label="症状特征">{top1.symptom}</Descriptions.Item>
                <Descriptions.Item label="防治建议">{top1.prevention}</Descriptions.Item>
              </Descriptions>

              {/* 结果区常驻轻提示（docs/23）。
                  为什么不做成「只在低置信时提醒」：外部域实测里那张整株照被判成
                  「番茄花叶病毒病 76.2%、可靠性：较可靠」—— 置信度不低、各项判据都正常，
                  条件式提示根本不会触发，用户会带着自信的错误结论走掉。
                  既然无法自动识别拍法（三轮实验见 LeafCropModal 注释），
                  就把入口做成常驻但低调，把判断权交给用户。 */}
              <div
                style={{
                  marginTop: 16,
                  paddingTop: 12,
                  borderTop: '1px solid var(--c-border-split)',
                  fontSize: 12,
                  color: 'var(--c-text-sub)',
                }}
              >
                结果和实际情况不符？
                <Button type="link" size="small" style={{ padding: '0 4px', fontSize: 12 }}
                        onClick={() => setCropOpen(true)}>
                  框选单片叶片再识别一次
                </Button>
                —— 整株或远距离拍摄是误判最常见的原因。
              </div>
            </>
          )}
            </>
          )}
        </Card>
      </div>

      <LeafCropModal
        open={cropOpen}
        src={sourcePreview || preview}
        onCancel={() => setCropOpen(false)}
        onOk={onCropped}
      />

      {/* 批量结果项详情：数据全部来自内存（刚算出来的），不请求接口 ——
          匿名场景下批量结果没有 record_id，走 /records/{id} 会直接 404。 */}
      <BatchItemModal
        open={!!activeItem}
        item={activeItem}
        preview={
          activeItem ? ((batchFiles.find((x, i) => i === activeItem.index) || {}).preview || '') : ''
        }
        onClose={() => setActiveItem(null)}
        onRecrop={openRecrop}
      />

      {/* 批量结果里的「单张重裁」：复用同一个 LeafCropModal，只是取图来源换成该张的原图。
          确定后走单张 /predict（新增一条历史），而不是重跑整批 9 张。 */}
      <LeafCropModal
        open={!!recropTarget}
        src={(recropTarget && recropTarget.src) || ''}
        onCancel={() => !recropBusy && setRecropTarget(null)}
        onOk={onRecropCropped}
      />

      {/* 病害参考图放大查看：一次给 3 张，方便对照「病斑形状 / 分布 / 颜色」三个特征 */}
      <Modal
        open={!!refModal}
        onCancel={() => setRefModal(null)}
        footer={null}
        width={640}
        title={refModal ? `${refModal.name} · 症状参考图` : ''}
      >
        {refModal && (
          <>
            <div style={{ display: 'flex', gap: 8 }}>
              {diseaseRefs(refModal.cid).map((src) => (
                <img
                  key={src}
                  src={src}
                  alt={`${refModal.name}参考图`}
                  style={{
                    width: '33%',
                    aspectRatio: '1 / 1',
                    objectFit: 'cover',
                    borderRadius: 8,
                    border: '1px solid var(--c-border)',
                    display: 'block',
                  }}
                />
              ))}
            </div>
            <Typography.Text
              type="secondary"
              style={{ fontSize: 12, display: 'block', marginTop: 12 }}
            >
              示例图来自 PlantVillage 公开数据集。同一病害在不同品种、不同发病阶段外观差异较大，
              请结合田间实际情况判断。
            </Typography.Text>
          </>
        )}
      </Modal>
    </div>
  )
}
