// 界面偏好：用户可隐藏/恢复识别历史页的图表卡片（2026-10-29 新增）
//
// 为什么不做成后端配置：
//   这是纯展示偏好，换台设备看到不同的布局是合理的，也不涉及任何账户数据。
//   存 localStorage 既免去一次网络往返，也避免为 4 个布尔值加一张表。
//
// 为什么用 window 自定义事件而不是 Context：
//   历史页（隐藏按钮）与个人中心（恢复开关）是两个不同路由，正常导航就会
//   重新挂载并重新读 localStorage，单靠 localStorage 已经够用。加事件只是
//   为了让「同标签内已挂载的组件」也能立即同步，成本 3 行，不引入 Provider 层级。
import { useEffect, useState } from 'react'

const STORAGE_KEY = 'leafcare.ui.prefs.v1'
const CHANGE_EVENT = 'leafcare:prefs-changed'

// 默认全部显示：隐藏是用户主动操作，不能因为读不到旧值就少显示内容
export const PREF_META = [
  { key: 'cropDist', label: '作物分布', desc: '按作物统计识别次数（最多 8 种）' },
  { key: 'confDist', label: '置信度分布', desc: '原始 softmax 值的区间分布' },
  { key: 'dailyTrend', label: '识别趋势', desc: `近 30 天每日识别量` },
  { key: 'verdictCard', label: '人工核对准确率', desc: '你给 AI 判卷的结果统计（仅统计已核对记录）' },
  { key: 'oodCard', label: '非叶片拦截', desc: '叶片检测头的运行期监控指标' },
]

export const DEFAULT_PREFS = PREF_META.reduce((acc, m) => ({ ...acc, [m.key]: true }), {})

export function getPrefs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_PREFS }
    const saved = JSON.parse(raw)
    if (!saved || typeof saved !== 'object') return { ...DEFAULT_PREFS }
    // 逐键合并：新版本新增的 key 在旧存档里不存在时回落到 true，
    // 否则升级后会「新增的图表默认不显示」，看起来像功能坏了
    return { ...DEFAULT_PREFS, ...saved }
  } catch {
    // localStorage 被禁用（隐私模式）或存档损坏时不应让整页崩掉
    return { ...DEFAULT_PREFS }
  }
}

export function setPref(key, value) {
  const next = { ...getPrefs(), [key]: !!value }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // 写失败（配额满 / 隐私模式）时静默降级：本次会话内仍会通过事件同步生效
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: next }))
  return next
}

export function resetPrefs() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* 同上 */
  }
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { ...DEFAULT_PREFS } }))
  return { ...DEFAULT_PREFS }
}

/** 订阅偏好变化，返回当前偏好对象。 */
export function usePrefs() {
  const [prefs, setPrefsState] = useState(getPrefs)
  useEffect(() => {
    const onChange = () => setPrefsState(getPrefs())
    window.addEventListener(CHANGE_EVENT, onChange)
    // storage 事件覆盖「另一个标签页改了偏好」的情况
    window.addEventListener('storage', onChange)
    return () => {
      window.removeEventListener(CHANGE_EVENT, onChange)
      window.removeEventListener('storage', onChange)
    }
  }, [])
  return prefs
}
