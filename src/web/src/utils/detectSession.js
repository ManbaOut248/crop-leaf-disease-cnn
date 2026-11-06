// 识别会话记忆（2026-10-29）
// ---------------------------------------------------------------------------
// 问题：识别页的结果只存在组件 state 里。用户点左侧菜单去「历史」或「知识库」，
// DetectPage 被卸载，React 随即丢掉 state；再点回「病害识别」是一个全新的组件实例，
// 图片、作物、Top-3 全部消失 —— 用户视角就是「刚才识别的东西不见了，还得重新上传」。
//
// 为什么不用 localStorage：
//   这里的语义是「这一次会话的临时草稿」，关掉标签页/换设备就不该带过去。
//   localStorage 会让「上次的识别结果」跨天出现，被误读成最新结果。
//
// 为什么不用全局 store（Context/Zustand）：
//   只需跨组件卸载保留，不需要跨页面共享；而且要能扛住刷新。
//   sessionStorage 一次解决「卸载 + 刷新」，且天然按标签页隔离。
//
// 体积控制：原图 dataURL 动辄 5~10MB，sessionStorage 配额只有约 5MB（且按字符计）。
// 所以持久化的是**降采样后的预览图**，不是原图。展示位最大高度 220px，
// 1024px 长边在 2x 屏上依然清晰，单张约 80~200KB。
// ---------------------------------------------------------------------------

const KEY = 'leafcare.detect.session.v1'

// 持久化预览图的长边上限。裁剪功能要求 ≥448px（LeafCropModal 导出 448×448），
// 1024 留了两倍余量，也够覆盖高分屏。
const MAX_PREVIEW_SIDE = 1024
const PREVIEW_QUALITY = 0.82

/**
 * 把图片文件压成长边不超过 MAX_PREVIEW_SIDE 的 JPEG dataURL。
 *
 * 为什么不用 FileReader.readAsDataURL 直接拿原图：
 * 一张手机照片的 dataURL 通常 5MB 以上，写 sessionStorage 必炸配额；
 * 而预览区只有 220px 高，存原图纯属浪费。
 *
 * @param {File|Blob} file 用户选择的图片
 * @returns {Promise<string>} dataURL；失败时 reject（调用方需兜底）
 */
export function shrinkToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      try {
        const scale = Math.min(1, MAX_PREVIEW_SIDE / Math.max(img.width, img.height))
        const w = Math.max(1, Math.round(img.width * scale))
        const h = Math.max(1, Math.round(img.height * scale))
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')
        // 白底再画：PNG/BMP 的透明区域转 JPEG 会变黑，看着像坏图
        ctx.fillStyle = '#fff'
        ctx.fillRect(0, 0, w, h)
        ctx.drawImage(img, 0, 0, w, h)
        resolve(canvas.toDataURL('image/jpeg', PREVIEW_QUALITY))
      } catch (e) {
        reject(e)
      } finally {
        URL.revokeObjectURL(url)
      }
    }
    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('图片解码失败'))
    }
    img.src = url
  })
}

/**
 * 把 dataURL 还原成 File，供「恢复后重新识别」再次上传。
 *
 * 恢复出来的不是原图而是降采样图，所以重新识别时精度会略降 —— 这是可接受的：
 * 用户要的是「还能再跑一次」，而不是「像素级复现原图」。
 *
 * @param {string} dataUrl
 * @param {string} name 原始文件名（用于回填与后端记录）
 */
export function dataUrlToFile(dataUrl, name) {
  const [head, body] = String(dataUrl || '').split(',')
  if (!body) return null
  const mime = (head.match(/data:([^;]+)/) || [])[1] || 'image/jpeg'
  const bin = atob(body)
  const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i += 1) buf[i] = bin.charCodeAt(i)
  return new File([buf], name || 'restored.jpg', { type: mime })
}

/**
 * 写入会话快照。整体覆盖写，避免旧字段残留导致「清空后又冒出来」。
 *
 * @param {{crop?:string, fileName?:string, preview?:string, result?:object|null}} snap
 */
export function saveSession(snap) {
  try {
    const payload = {
      crop: snap.crop || '',
      fileName: snap.fileName || '',
      preview: snap.preview || '',
      result: snap.result || null,
      savedAt: Date.now(),
    }
    // 没有结果也没有图片时不留垃圾快照
    if (!payload.result && !payload.preview) {
      sessionStorage.removeItem(KEY)
      return
    }
    sessionStorage.setItem(KEY, JSON.stringify(payload))
  } catch (e) {
    // QuotaExceededError：降级为「不存预览图」，至少让结果别丢
    try {
      sessionStorage.setItem(
        KEY,
        JSON.stringify({
          crop: snap.crop || '',
          fileName: snap.fileName || '',
          preview: '',
          result: snap.result || null,
          savedAt: Date.now(),
        }),
      )
    } catch (e2) {
      // 连结果都存不下：放弃记忆，不影响识别本身
      sessionStorage.removeItem(KEY)
    }
  }
}

/** 读取会话快照；无快照或内容损坏时返回 null（损坏就清掉，避免每次进页面都报错） */
export function loadSession() {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const snap = JSON.parse(raw)
    if (!snap || (!snap.result && !snap.preview)) return null
    return snap
  } catch (e) {
    try {
      sessionStorage.removeItem(KEY)
    } catch (e2) {
      /* ignore */
    }
    return null
  }
}

/** 清空会话快照（「重新开始」按钮 / 识别失败需要重置时用） */
export function clearSession() {
  try {
    sessionStorage.removeItem(KEY)
  } catch (e) {
    /* ignore */
  }
}
