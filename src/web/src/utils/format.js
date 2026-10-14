/**
 * 置信度百分比格式化。
 *
 * 为什么需要它：
 *   直接写 `(v * 100).toFixed(1)` 会把 0.9996 显示成「100.0%」、
 *   把 0.0003 显示成「0.0%」。前者让用户以为模型「绝对正确」，
 *   与产品「辅助识别、需人工复核」的定位相矛盾；后者看起来像没有输出。
 *
 *   这里对超出显示精度的极端值做守卫，用 `>` / `<` 诚实表达，
 *   既避免绝对化表述，也不丢失「模型非常确定」这一信息。
 *
 * @param {number} v       0~1 的概率值
 * @param {number} digits  小数位数（默认 1）
 * @returns {string}       如 "98.8%" / ">99.9%" / "<0.1%" / "0%"
 */
export function formatPercent(v, digits = 1) {
  const n = Number(v)
  if (!Number.isFinite(n) || n <= 0) return '0%'

  const p = n * 100
  // 用「一个最小显示单位」（digits=1 时为 0.1）作为守卫阈值。
  // 注意不能用 0.05 这样的半个单位：99.95 经 toFixed(1) 会进位成 100.0，
  // 反而又造出「>100.0%」这种失真显示。
  const unit = Math.pow(10, -digits)

  if (p < unit) return `<${unit}%`
  if (p > 100 - unit) return `>${(100 - unit).toFixed(digits)}%`
  return `${p.toFixed(digits)}%`
}
