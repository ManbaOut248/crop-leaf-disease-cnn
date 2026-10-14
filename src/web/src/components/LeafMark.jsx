// 品牌叶子标识（内联 SVG）
//
// 说明：@ant-design/icons 中**不存在**叶子图标（曾误用 LeafOutlined 导致整站白屏），
// 故品牌符号改用自绘 SVG —— 不依赖图标库、跨平台渲染一致、可随尺寸与颜色配置。
export default function LeafMark({ size = 22, color = 'var(--c-primary-500)' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      aria-hidden="true"
      style={{ display: 'block', flexShrink: 0 }}
    >
      {/* 叶片主体：用 style.fill 而非 fill 属性，才能吃到 CSS 变量 */}
      <path d="M17 3C9 3 4 8 3 16c0 1 0 1 1 1 8-1 13-6 13-14z" style={{ fill: color }} />
      {/* 叶脉高光 */}
      <path
        d="M6 14c2-4 6-7 9-9"
        stroke="#fff"
        strokeWidth="1"
        fill="none"
        opacity="0.55"
      />
    </svg>
  )
}
