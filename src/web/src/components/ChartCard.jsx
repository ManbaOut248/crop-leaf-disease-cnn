// 通用图表卡片：统一标题 / 副标题 / 空态 / 高度 / 尺寸自适应。
//
// 为什么不直接在页面里 echarts.init：
//   1) 每张图都要写 init → setOption → resize → dispose 一整套，重复且易漏 dispose；
//   2) 漏了 dispose 会在路由切换后残留实例，反复进页面会内存增长；
//   3) 空数据 / 加载中要用同一种样式兜底，散落在各处必然不一致。
//
// 按需引入而非整包 import 'echarts'：整包约 1MB（gzip 330KB），
// 这里只用到柱状图 + 直角坐标系，按需引入后只为实际用到的模块付体积。
import React, { useEffect, useRef } from 'react'
import { Card, Empty, Spin, Typography } from 'antd'
import * as echarts from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, AxisPointerComponent } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'

// SVG 渲染器：数据量小（本例最多几十根柱），矢量图缩放不糊，且导出图片更清晰
echarts.use([BarChart, GridComponent, TooltipComponent, AxisPointerComponent, SVGRenderer])

export default function ChartCard({
  title,
  subtitle,
  option,
  height = 200,
  empty = false,
  emptyText = '暂无数据',
  loading = false,
  extra,
}) {
  const boxRef = useRef(null)
  const instRef = useRef(null)
  // option 放 ref：setOption 用最新值，避免把它放进 init 的依赖导致图表被反复销毁重建
  const optionRef = useRef(option)
  optionRef.current = option

  useEffect(() => {
    if (loading || empty || !boxRef.current) return undefined
    const chart = echarts.init(boxRef.current, null, { renderer: 'svg' })
    instRef.current = chart
    chart.setOption(optionRef.current || {}, true)

    // 侧边栏折叠、窗口缩放、栅格换行都会改变容器宽度，必须重绘
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(boxRef.current)

    return () => {
      ro.disconnect()
      chart.dispose()
      instRef.current = null
    }
  }, [loading, empty])

  useEffect(() => {
    if (instRef.current && option) instRef.current.setOption(option, true)
  }, [option])

  const placeholder = {
    height,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  }

  return (
    <Card
      size="small"
      title={
        <div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>{title}</div>
          {subtitle && (
            <Typography.Text type="secondary" style={{ fontSize: 12, fontWeight: 400 }}>
              {subtitle}
            </Typography.Text>
          )}
        </div>
      }
      extra={extra}
      styles={{ body: { padding: '8px 12px 4px' } }}
    >
      {loading ? (
        <div style={placeholder}>
          <Spin size="small" />
        </div>
      ) : empty ? (
        <div style={placeholder}>
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />
        </div>
      ) : (
        <div ref={boxRef} style={{ height, width: '100%' }} />
      )}
    </Card>
  )
}
