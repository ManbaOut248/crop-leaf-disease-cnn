// 叶片裁剪弹窗：把「整株远景」手动框到「单片叶片近景」。
//
// 为什么需要它（有实测依据，不是拍脑袋加的）：
//   docs/23 外部域评估显示，印度农户手机实拍的整株番茄照 Top-1 仅 17.1%，
//   而同一模型在单叶近景上是 43.6%。根因是取景方式与训练前提不一致。
//   尝试用轻量启发式自动判断「叶片是否占满画面」三轮全部失败：
//     ① 绿色占比/连通域：近景与远景分布完全重叠（整株的叶子也连成一片）；
//     ② 距离变换/边缘密度：近景 p10 与远景 p90 交叉，无法单阈值切开；
//     ③ 叶片检测头 3×3 分块打分：方向相反（远景 0.89 > 近景 0.67），
//        阈值 0.6 时近景误报 42%、远景命中仅 17%。
//   → 既然自动判断不可靠，就交给用户一次点击确认，零误报、零算法风险。
//
// 交互设计：固定方形取景框，拖动平移 + 滑块缩放，所见即所得。
// 输出：正方形 JPEG Blob（448×448），直接替换上传文件，后端无需改动。
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Modal, Slider, Button, Space, Typography } from 'antd'

const VIEW = 300 // 取景框边长（CSS 像素）
const OUT = 448 // 导出边长，够模型 224 输入且保留细节

export default function LeafCropModal({ open, src, onCancel, onOk }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null) // 缓存的 HTMLImageElement，避免每次重绘都重新解码
  const dragRef = useRef(null) // 拖拽起点（记录指针与当时的偏移）
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [loaded, setLoaded] = useState(false)

  // 打开时加载图片并把取景框居中
  useEffect(() => {
    if (!open || !src) return
    setLoaded(false)
    setZoom(1)
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setOffset({ x: 0, y: 0 })
      setLoaded(true)
    }
    img.src = src
  }, [open, src])

  // 计算当前缩放下的实际绘制尺寸
  const geom = useCallback(() => {
    const img = imgRef.current
    if (!img) return null
    // cover 基准缩放：保证图片始终铺满取景框（不会露出空白）
    const base = Math.max(VIEW / img.width, VIEW / img.height)
    const s = base * zoom
    return { w: img.width * s, h: img.height * s }
  }, [zoom])

  // 把偏移限制在「图片边缘不进入取景框」的范围内
  const clamp = useCallback(
    (o) => {
      const g = geom()
      if (!g) return o
      const maxX = Math.max(0, (g.w - VIEW) / 2)
      const maxY = Math.max(0, (g.h - VIEW) / 2)
      return {
        x: Math.min(maxX, Math.max(-maxX, o.x)),
        y: Math.min(maxY, Math.max(-maxY, o.y)),
      }
    },
    [geom],
  )

  // 重绘取景框
  useEffect(() => {
    const cv = canvasRef.current
    const img = imgRef.current
    if (!cv || !img || !loaded) return
    const g = geom()
    if (!g) return
    const ctx = cv.getContext('2d')
    ctx.clearRect(0, 0, VIEW, VIEW)
    ctx.drawImage(img, VIEW / 2 - g.w / 2 + offset.x, VIEW / 2 - g.h / 2 + offset.y, g.w, g.h)
  }, [offset, zoom, loaded, geom])

  // 缩放变化后偏移可能越界，重新夹紧（否则放大后会露出空白）
  useEffect(() => {
    setOffset((o) => clamp(o))
  }, [zoom, clamp])

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture?.(e.pointerId)
    dragRef.current = { px: e.clientX, py: e.clientY, ox: offset.x, oy: offset.y }
  }

  const onPointerMove = (e) => {
    if (!dragRef.current) return
    const d = dragRef.current
    setOffset(clamp({ x: d.ox + (e.clientX - d.px), y: d.oy + (e.clientY - d.py) }))
  }

  const onPointerUp = () => {
    dragRef.current = null
  }

  // 导出：用同一套变换把可见区域画到 OUT×OUT 画布上
  const handleOk = () => {
    const img = imgRef.current
    const g = geom()
    if (!img || !g) return
    const k = OUT / VIEW // 放大倍数，保证导出分辨率高于取景框
    const cv = document.createElement('canvas')
    cv.width = OUT
    cv.height = OUT
    const ctx = cv.getContext('2d')
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(
      img,
      (VIEW / 2 - g.w / 2 + offset.x) * k,
      (VIEW / 2 - g.h / 2 + offset.y) * k,
      g.w * k,
      g.h * k,
    )
    cv.toBlob(
      (blob) => {
        if (!blob) return
        // 统一命名为 jpg，后端按扩展名校验（ALLOWED_IMAGE_EXT）
        onOk(new File([blob], 'leaf_crop.jpg', { type: 'image/jpeg' }))
      },
      'image/jpeg',
      0.92,
    )
  }

  return (
    <Modal
      open={open}
      title="框选单片叶片"
      onCancel={onCancel}
      destroyOnHidden
      footer={
        <Space style={{ marginTop: 4 }}>
          <Button onClick={onCancel}>取消</Button>
          <Button type="primary" onClick={handleOk} disabled={!loaded}>
            用这张重新识别
          </Button>
        </Space>
      }
    >
      <Typography.Paragraph type="secondary" style={{ fontSize: 13 }}>
        拖动图片把<b>一片叶子</b>移进方框，用滑块放大到叶子基本铺满。背景越少，识别越准。
      </Typography.Paragraph>
      <div
        style={{
          width: VIEW,
          height: VIEW,
          margin: '0 auto',
          borderRadius: 8,
          overflow: 'hidden',
          background: 'var(--c-fill-quaternary, #f5f5f5)',
          cursor: 'grab',
          touchAction: 'none',
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <canvas ref={canvasRef} width={VIEW} height={VIEW} style={{ display: 'block' }} />
      </div>
      <div style={{ marginTop: 12 }}>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          缩放
        </Typography.Text>
        <Slider min={1} max={4} step={0.05} value={zoom} onChange={setZoom} tooltip={{ open: false }} />
      </div>
    </Modal>  )
}
