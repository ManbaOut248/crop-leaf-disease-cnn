// 拍摄要求引导卡：用两张内联 SVG 讲清「怎么拍」。
//
// 依据：docs/23 外部域评估 —— 农户习惯拍整株（Top-1 17.1%），
// 而模型前提是单片叶片占满画面（PlantDoc 单叶近景 43.6%）。
// 自动判断拍法已实测不可行（见 LeafCropModal 注释），所以这里用「示例图 + 文案」
// 做前置引导，配一个「框选单片叶片」的手动裁剪入口。
//
// 用内联 SVG 而非外部图片：不引入静态资源依赖、跟随主题色、任意分辨率不糊。
import React from 'react'
import { Typography } from 'antd'

/** 单张示意卡：ok=true 绿色描边勾，false 红色叉 */
function Sample({ ok, label, hint }) {
  const okColor = 'var(--c-success-600, #389e0d)'
  const badColor = 'var(--c-error-600, #cf1322)'
  const color = ok ? okColor : badColor
  return (
    <div style={{ textAlign: 'center', flex: '0 0 auto' }}>
      <svg width="104" height="78" viewBox="0 0 104 78" role="img" aria-label={label}>
        {/* 画框 */}
        <rect x="1" y="1" width="102" height="76" rx="6" fill="var(--c-fill-quaternary, #fafafa)"
              stroke={color} strokeWidth="1.5" strokeDasharray={ok ? '0' : '4 3'} />
        {ok ? (
          <>
            {/* 一片大叶子铺满画面：叶身 + 中脉 + 两条侧脉 */}
            <path d="M52 12 C78 16 88 34 84 52 C80 66 66 70 52 68 C38 70 24 66 20 52 C16 34 26 16 52 12 Z"
                  fill={okColor} opacity="0.28" stroke={okColor} strokeWidth="1.2" />
            <path d="M52 14 L52 66" stroke={okColor} strokeWidth="1.2" opacity="0.75" />
            <path d="M52 30 L36 22 M52 30 L68 22 M52 44 L34 38 M52 44 L70 38 M52 58 L38 54 M52 58 L66 54"
                  stroke={okColor} strokeWidth="1" opacity="0.55" />
          </>
        ) : (
          <>
            {/* 整株远景：一株主茎 + 多片小叶子 + 底部土壤 */}
            <rect x="8" y="60" width="88" height="16" fill="#000" opacity="0.06" />
            <path d="M52 24 L52 62" stroke={badColor} strokeWidth="1.4" opacity="0.8" />
            {[[40, 30], [64, 34], [36, 42], [68, 46], [44, 50], [60, 56], [34, 26], [70, 30]].map(
              ([x, y], i) => (
                <ellipse key={i} cx={x} cy={y} rx="9" ry="5"
                         fill={badColor} opacity="0.22" stroke={badColor} strokeWidth="0.8" />
              ),
            )}
          </>
        )}
        {/* 角标：勾 / 叉 */}
        <circle cx="90" cy="14" r="9" fill={color} />
        {ok ? (
          <path d="M86 14 L89 17.5 L94.5 10.5" stroke="#fff" strokeWidth="2" fill="none"
                strokeLinecap="round" strokeLinejoin="round" />
        ) : (
          <path d="M86.5 10.5 L93.5 17.5 M93.5 10.5 L86.5 17.5" stroke="#fff" strokeWidth="2"
                strokeLinecap="round" />
        )}
      </svg>
      <div style={{ fontSize: 12, color: color, fontWeight: 600, marginTop: 2 }}>{label}</div>
      <div style={{ fontSize: 11, color: 'var(--c-text-sub, #8c8c8c)', lineHeight: 1.4 }}>{hint}</div>
    </div>
  )
}

export default function ShotGuide() {
  return (
    <div
      style={{
        marginBottom: 14,
        padding: '10px 12px 8px',
        borderRadius: 8,
        background: 'var(--c-fill-quaternary, #fafafa)',
        border: '1px solid var(--c-border-split, #f0f0f0)',
      }}
    >
      <Typography.Text strong style={{ fontSize: 13 }}>
        这样拍识别更准
      </Typography.Text>
      <div style={{ display: 'flex', justifyContent: 'space-around', gap: 8, marginTop: 8 }}>
        <Sample ok label="单片叶片近景" hint="叶子占满画面" />
        <Sample ok={false} label="整株 / 远距离" hint="叶片太小、背景杂乱" />
      </div>
      <ul
        style={{
          margin: '8px 0 0',
          paddingLeft: 18,
          fontSize: 12,
          lineHeight: 1.7,
          color: 'var(--c-text-sub, #595959)',
        }}
      >
        <li>让<b>一片叶子</b>尽量占满画面，别带进整株、竹架、土壤</li>
        <li>病斑处对焦清晰，避免逆光与反光</li>
        <li>已经拍成整株？用下方<b>「框选单片叶片」</b>裁一刀再识别</li>
      </ul>
    </div>
  )
}
