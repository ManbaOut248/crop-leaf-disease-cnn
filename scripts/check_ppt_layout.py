# -*- coding: utf-8 -*-
"""
PPTX 版面自检（无渲染环境下的近似校验）。

本机无 LibreOffice / Office / WPS，无法把 pptx 渲成图片，因此改用
「真实字体度量 + 保守行高估算」反推溢出风险：

- 文本框：用微软雅黑实际字宽逐字折行，估算行数与高度，与形状高度比较
- 表格：逐格估算折行行数 → 行高需求；PowerPoint 会自动增高行，
  所以按「估算总高」与「可用竖向空间」比较
- 图片：检查是否越出画布

只报告问题，不做修改。行高系数取保守值（宁可误报，不要漏报）。
"""
import sys
from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
EMU_IN = 914400
_cache = {}


def font(size_pt, bold=False):
    key = (int(round(size_pt)), bold)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONT_B if bold else FONT, key[0])
    return _cache[key]


def text_width(s, size_pt, bold):
    return font(size_pt, bold).getlength(s)   # 单位≈pt


def wrap_count(text, size_pt, bold, width_pt):
    """逐字贪心折行，返回行数（CJK 与拉丁一视同仁，偏保守）。"""
    if not text:
        return 1
    total = 0
    for para in str(text).split("\n"):
        if not para:
            total += 1
            continue
        cur = ""
        lines = 1
        for ch in para:
            if text_width(cur + ch, size_pt, bold) <= width_pt:
                cur += ch
            else:
                lines += 1
                cur = ch
        total += lines
    return total


def run_size(run, default=12.0):
    return run.font.size.pt if run.font.size else default


def check_text_frame(shape, slide_h_in, page):
    tf = shape.text_frame
    raw = tf.text or ""
    if not raw.strip():
        return []            # 纯色块/横线：无文字，跳过
    w_pt = shape.width / EMU_IN * 72 - 2      # 左右内边距按 0 算，留 2pt 余量
    used = 0.0
    for p in tf.paragraphs:
        text = "".join(r.text for r in p.runs)
        if not text:
            used += 6
            continue
        size = max([run_size(r) for r in p.runs], default=12.0)
        bold = any(r.font.bold for r in p.runs)
        lines = wrap_count(text, size, bold, w_pt)
        sp = p.line_spacing if isinstance(p.line_spacing, float) else 1.2
        used += lines * size * 1.42 * (sp / 1.2 if sp else 1)
        if p.space_after is not None:
            used += p.space_after.pt
    box_pt = shape.height / EMU_IN * 72
    bottom_in = shape.top / EMU_IN + used / 72
    name = raw[:18].replace("\n", " ")
    problems = []
    if used > box_pt + 2.5:      # 2.5pt 容差
        problems.append((page, f"文本框「{name}…」估算需 {used:.0f}pt，框高 {box_pt:.0f}pt", "text-box"))
    if bottom_in > slide_h_in - 0.35 and shape.top / EMU_IN < 6.95:   # 页脚本身贴底，不算越界
        problems.append((page, f"文本框「{name}…」估算底边 {bottom_in:.2f}in 进入页脚区", "text-bottom"))
    return problems


def check_table(shape, slide_h_in, page, limit_in):
    tbl = shape.table
    top_in = shape.top / EMU_IN
    total = 0.0
    problems = []
    col_w = [c.width / EMU_IN for c in tbl.columns]
    if abs(sum(col_w) - shape.width / EMU_IN) > 0.02:
        problems.append((page, f"表格列宽合计 {sum(col_w):.2f}in 与表宽 {shape.width / EMU_IN:.2f}in 不一致", "table-width"))
    for i, row in enumerate(tbl.rows):
        need = 0.0
        for j, cell in enumerate(row.cells):
            text = cell.text_frame.text
            size = 10.0
            bold = False
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size:
                        size = r.font.size.pt
                    bold = bool(r.font.bold) or bold
            width_pt = (col_w[j] - 0.16) * 72
            lines = wrap_count(text, size, bold, width_pt)
            need = max(need, lines * size * 1.32 + 0.07 * 72)
        set_need = row.height / EMU_IN
        total += max(need / 72, set_need)
    bottom = top_in + total
    if bottom > limit_in:
        problems.append((page, f"表格估算底边 {bottom:.2f}in，超出可用边界 {limit_in:.2f}in（{bottom - limit_in:+.2f}in）", "table"))
    return problems


def est_box(shape):
    """返回形状的估算包围盒（英寸）：文本框用度量高度，表格用累加行高。"""
    l = shape.left / EMU_IN
    t = shape.top / EMU_IN
    w = shape.width / EMU_IN
    h = shape.height / EMU_IN
    if shape.has_text_frame and (shape.text_frame.text or "").strip():
        used = 0.0
        for p in shape.text_frame.paragraphs:
            text = "".join(r.text for r in p.runs)
            size = max([run_size(r) for r in p.runs], default=12.0) if p.runs else 12.0
            bold = any(r.font.bold for r in p.runs)
            lines = wrap_count(text, size, bold, w * 72 - 2) if text else 1
            sp = p.line_spacing if isinstance(p.line_spacing, float) else 1.2
            used += lines * size * 1.42 * (sp / 1.2 if sp else 1)
            if p.space_after is not None:
                used += p.space_after.pt
        h = max(h, used / 72)
    elif shape.has_table:
        tbl = shape.table
        col_w = [c.width / EMU_IN for c in tbl.columns]
        total = 0.0
        for row in tbl.rows:
            need = 0.0
            for j, cell in enumerate(row.cells):
                size, bold = 10.0, False
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size:
                            size = r.font.size.pt
                        bold = bool(r.font.bold) or bold
                lines = wrap_count(cell.text_frame.text, size, bold, (col_w[j] - 0.16) * 72)
                need = max(need, (lines * size * 1.32 + 0.07 * 72) / 72)
            total += max(need, row.height / EMU_IN)
        h = total
    return (l, t, l + w, t + h)


def check_overlap(slide, page):
    """同页形状重叠检测：装饰性细线（高 < 0.02in）跳过。"""
    items = []
    for shape in slide.shapes:
        # 纯装饰（顶部色块、分隔细线）不参与重叠判定
        has_text = shape.has_text_frame and (shape.text_frame.text or "").strip()
        if not has_text and not shape.has_table:
            continue
        box = est_box(shape)
        l, t, r, b = box
        if b - t < 0.02:
            continue
        label = (shape.text_frame.text[:14] if shape.has_text_frame else shape.name)
        items.append((box, label))
    problems = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (l1, t1, r1, b1), n1 = items[i]
            (l2, t2, r2, b2), n2 = items[j]
            ow = min(r1, r2) - max(l1, l2)
            oh = min(b1, b2) - max(t1, t2)
            if ow > 0.05 and oh > 0.05 and ow * oh > 0.06:
                problems.append((page, f"形状重叠 {ow * oh:.2f}in²：「{n1}」×「{n2}」", "overlap"))
    return problems


def main():
    path = Path(sys.argv[1])
    prs = Presentation(str(path))
    sw = prs.slide_width / EMU_IN
    sh = prs.slide_height / EMU_IN
    print(f"文件：{path.name}  画布：{sw:.2f}×{sh:.2f} in  共 {len(prs.slides)} 页\n")

    issues = []
    for idx, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            # 越界检查（所有形状）
            l = shape.left / EMU_IN
            t = shape.top / EMU_IN
            r = l + shape.width / EMU_IN
            b = t + shape.height / EMU_IN
            if l < -0.02 or t < -0.02 or r > sw + 0.02 or b > sh + 0.02:
                # 封面整页底色块允许贴边
                if not (idx == 1 and shape.width / EMU_IN >= sw - 0.01):
                    issues.append((idx, f"形状越界：{shape.name} [{l:.2f},{t:.2f}]-[{r:.2f},{b:.2f}]", "bounds"))

            if shape.has_text_frame:
                issues += check_text_frame(shape, sh, idx)
            if shape.has_table:
                issues += check_table(shape, sh, idx, limit_in=sh - 0.35)
        issues += check_overlap(slide, idx)

    if not issues:
        print("✅ 未发现版面溢出（文本框、表格、形状均在画布与可用区域内）")
    else:
        for page, msg, kind in issues:
            print(f"  P{page:02d} [{kind}] {msg}")
        print(f"\n共 {len(issues)} 处需关注")


if __name__ == "__main__":
    main()
