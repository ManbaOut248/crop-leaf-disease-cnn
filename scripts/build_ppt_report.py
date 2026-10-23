# -*- coding: utf-8 -*-
"""
生成《基于 CNN 的农作物叶片病害识别系统》项目汇报 PPT。

设计约束（对齐用户偏好）：
- 16:9 / 1280×720pt 画布；深蓝主色系，沿用 03.ppt 旧版 DESIGN.md 的色板
- **以文字、表格、真实网站截图为主，不画装饰性图形**（避免图形与文字对不齐）
- 所有截图按原始宽高比等比缩放并居中，不拉伸变形
- 表格统一细线边框 + 浅蓝表头，列宽固定，长文本自动换行

运行：用装有 python-pptx 的解释器执行本脚本
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from lxml import etree
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "03.ppt" / "农作物叶片病害识别系统" / "assets_v4"
OUT = ROOT / "03.ppt" / "农作物叶片病害识别系统" / "农作物叶片病害识别系统_项目汇报.pptx"

# ============ 色板（取自旧版 DESIGN.md）============
NAVY = RGBColor(0x0E, 0x3F, 0x8C)      # 藏蓝：顶条 / 主标
BLUE = RGBColor(0x1E, 0x4F, 0xA8)      # 主蓝：页码 / 强调
MID = RGBColor(0x3D, 0x7B, 0xD9)       # 中蓝：次级强调
LIGHT = RGBColor(0xE8, 0xEF, 0xF8)     # 浅蓝：表头底
PALE = RGBColor(0xF0, 0xF5, 0xFC)      # 极浅蓝：交替行
TEXT = RGBColor(0x1A, 0x22, 0x30)      # 正文深
GRAY = RGBColor(0x4A, 0x55, 0x68)      # 副文本
MUTED = RGBColor(0x8B, 0x97, 0xA8)     # 辅助
LINE = RGBColor(0xD6, 0xDC, 0xE5)      # 线框
RED = RGBColor(0xD9, 0x53, 0x4F)       # 警示
YELLOW = RGBColor(0xE0, 0x9B, 0x00)    # 数据强调（深一档，浅底上可读）
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "微软雅黑"

# ============ 版式常量（英寸）============
SW, SH = 13.333, 7.5
ML = 0.45                     # 左边距
CW = SW - ML * 2              # 内容宽 12.43
BAND_H = 0.42
TITLE_Y = 0.54
RULE_Y = 1.33
BODY_Y = 1.43
BODY_BOTTOM = 6.92
FOOTER_Y = 7.02

TOTAL_PAGES = 16


# ---------------- 基础工具 ----------------
def set_font(run, size, bold=False, color=TEXT, name=FONT):
    """设置字号/加粗/颜色，并**同时**指定中文字体（a:ea），否则中文会回退成宋体。"""
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = name
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", name)


def textbox(slide, x, y, w, h, text="", size=13, bold=False, color=TEXT,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split("\n") if text else [""]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        set_font(r, size, bold, color)
    return tb


_FONT_FILE = "C:/Windows/Fonts/msyh.ttc"
_font_cache = {}


def _font(size):
    key = int(round(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_FONT_FILE, key)
    return _font_cache[key]


def measure_lines(text, size, width_in):
    """用真实字体逐字折行，估算行数（用于把文本框高度设成实际需要的高度）。"""
    width_pt = width_in * 72 - 2
    f = _font(size)
    lines = 1
    cur = ""
    for ch in str(text):
        if f.getlength(cur + ch) <= width_pt:
            cur += ch
        else:
            lines += 1
            cur = ch
    return lines


def bullets(slide, x, y, w, items, size=12.5, gap=9, color=TEXT, mark="· "):
    """要点列表：每条一个段落（用文本分隔符而非图形，避免错位）。
    高度按真实字体度量计算，保证形状几何与实际内容一致，便于自检与对齐。"""
    total_pt = 0.0
    for it in items:
        n = measure_lines(mark + it, size, w)
        total_pt += n * size * 1.60 + gap          # 1.60 = 保守行高系数（含 1.35 倍行距）
    h_in = total_pt / 72
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h_in))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.35
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = mark + it
        set_font(r, size, False, color)
    return tb


def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def header(slide, section):
    """顶条：左侧系统名，右侧章节名（纯色块 + 文本，无图标）。"""
    rect(slide, 0, 0, SW, BAND_H, NAVY)
    textbox(slide, ML, 0.105, 8.0, 0.24, "农作物叶片病害识别系统", 10.5, True, WHITE,
            anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, SW - ML - 4.0, 0.105, 4.0, 0.24, section, 10.5, False, WHITE,
            align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def page_title(slide, title, sub=None):
    textbox(slide, ML, TITLE_Y, CW, 0.50, title, 26, True, NAVY)
    if sub:
        textbox(slide, ML, TITLE_Y + 0.54, CW, 0.22, sub, 10.5, False, GRAY)
    # 细横线：用矩形当线条，高度 0.012in，宽度固定，不会错位
    rect(slide, ML, RULE_Y, CW, 0.012, LINE)


def footer(slide, no):
    textbox(slide, ML, FOOTER_Y, 6.0, 0.22, "前沿技术实训 · 2026", 9.5, False, MUTED)
    textbox(slide, SW - ML - 2.0, FOOTER_Y, 2.0, 0.22, f"{no:02d} / {TOTAL_PAGES}",
            9.5, True, BLUE, align=PP_ALIGN.RIGHT)


def picture_fit(slide, path, x, y, max_w, max_h, align="left"):
    """等比缩放放入 (max_w, max_h) 框内，不拉伸；align 控制水平位置。"""
    p = Path(path)
    with Image.open(p) as im:
        iw, ih = im.size
    ratio = iw / ih
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    if align == "center":
        x = x + (max_w - w) / 2
    elif align == "right":
        x = x + (max_w - w)
    y = y + (max_h - h) / 2
    return slide.shapes.add_picture(str(p), Inches(x), Inches(y), Inches(w), Inches(h))


def _cell_border(cell, color=LINE, width=0.75):
    """给单元格加四边细线（python-pptx 无原生 API，直接写 XML）。"""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = tcPr.find(qn(tag))
        if ln is not None:
            tcPr.remove(ln)
        ln = etree.SubElement(tcPr, qn(tag))
        ln.set("w", str(int(width * 12700)))
        ln.set("cap", "flat")
        ln.set("cmpd", "sng")
        ln.set("algn", "ctr")
        fill = etree.SubElement(ln, qn("a:solidFill"))
        clr = etree.SubElement(fill, qn("a:srgbClr"))
        clr.set("val", str(color))
        etree.SubElement(ln, qn("a:prstDash")).set("val", "solid")


def table(slide, x, y, w, col_w, rows, size=10.5, row_h=0.34, header_size=10.5,
          aligns=None, first_col_bold=True, highlight=None):
    """
    统一的表格：浅蓝表头 + 白/极浅蓝交替底 + 细灰边框。
    col_w: 各列宽度（英寸），合计应等于 w
    highlight: {(row_idx, col_idx): RGBColor} 用于强调个别单元格文字颜色
    """
    nrow, ncol = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrow, ncol, Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * nrow))
    tbl = shape.table
    # 关闭默认主题样式，改用自绘边框
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is None:
        style = etree.SubElement(tblPr, qn("a:tableStyleId"))
    style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"

    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nrow):
        tbl.rows[i].height = Inches(row_h)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = LIGHT
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else PALE
            _cell_border(cell)

            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            al = PP_ALIGN.LEFT
            if aligns:
                al = aligns[j]
            p.alignment = al
            p.line_spacing = 1.1
            r = p.add_run()
            r.text = str(val)
            if i == 0:
                set_font(r, header_size, True, NAVY)
            else:
                color = TEXT
                if highlight and (i, j) in highlight:
                    color = highlight[(i, j)]
                set_font(r, size, first_col_bold and j == 0, color)
    return tbl


def note(slide, y, text, color=GRAY, size=9.5):
    """脚注/说明：按实际行数设置框高，避免多行说明被压扁或压到页脚。"""
    n = sum(measure_lines(line, size, CW) for line in text.split("\n"))
    h = n * size * 1.55 / 72
    textbox(slide, ML, y, CW, h, text, size, False, color)


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])   # 6 = 空白版式


# ---------------- 逐页构建 ----------------
def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    no = 0

    # ===== P1 封面 =====
    s = new_slide(prs); no += 1
    rect(s, 0, 0, SW, SH, NAVY)
    rect(s, 0, 4.22, SW, 0.02, MID)
    textbox(s, 1.1, 1.85, 11.2, 0.4, "前沿技术实训 · 项目汇报", 13, False,
            RGBColor(0xC7, 0xD8, 0xEE))
    textbox(s, 1.1, 2.35, 11.2, 0.9, "基于 CNN 的农作物叶片病害识别系统", 40, True, WHITE)
    textbox(s, 1.1, 3.32, 11.2, 0.45, "六大功能模块 · 实现路径 · 实测效果", 18, False,
            RGBColor(0xC7, 0xD8, 0xEE))
    textbox(s, 1.1, 4.55, 11.2, 0.9,
            "14 种作物｜26 类病害 + 12 类健康 = 38 类\n"
            "React 18 + FastAPI + PyTorch 2.5.1 + ResNet18", 13, False,
            RGBColor(0xD8, 0xE3, 0xF3), spacing=1.5)
    textbox(s, 1.1, 6.35, 5.5, 0.3, "2026 年 10 月", 12, False, RGBColor(0xA8, 0xBE, 0xDC))
    textbox(s, SW - 1.1 - 4.0, 6.35, 4.0, 0.3, "实训答辩 · V4.0", 12, False,
            RGBColor(0xA8, 0xBE, 0xDC), align=PP_ALIGN.RIGHT)

    # ===== P2 目录 =====
    s = new_slide(prs); no += 1
    header(s, "目录")
    page_title(s, "汇报内容", "从任务要求 → 系统设计 → 模块实现 → 实测效果 → 交付清单")
    items = [
        ("01", "项目概况与任务要求", "任务书指标 vs 实际交付"),
        ("02", "系统总体设计", "四层架构与一次识别的完整链路"),
        ("03", "六大功能模块与实现", "逐模块演示界面 + 关键实现要点"),
        ("04", "模型效果与关键指标", "实验室域、真实域、拒识、校准四组数据"),
        ("05", "工程亮点与交付清单", "防泄漏、安全加固、文档与产物"),
    ]
    y = BODY_Y + 0.12
    for num, t, sub in items:
        textbox(s, ML, y, 0.72, 0.42, num, 22, True, MID)
        textbox(s, ML + 0.78, y - 0.02, 6.4, 0.3, t, 15, True, NAVY)
        textbox(s, ML + 0.78, y + 0.28, 6.4, 0.26, sub, 10.5, False, GRAY)
        rect(s, ML, y + 0.72, CW, 0.008, LINE)
        y += 0.98
    footer(s, no)

    # ===== P3 项目概况 =====
    s = new_slide(prs); no += 1
    header(s, "01 项目概况")
    page_title(s, "任务要求与实际交付", "任务书 V1.0.0 逐条核对，全部落地")
    rows = [
        ["任务书要求", "实际交付", "状态"],
        ["14 种作物、26 类病害 + 健康状态", "38 类（26 病害 + 12 健康），覆盖 14 种作物", "符合"],
        ["识别准确率 ≥ 85%", "PlantVillage 测试集 99.06%（4914 张）", "超额"],
        ["各类 F1 ≥ 0.80", "全部 ≥ 0.911，宏平均 F1 0.9895", "超额"],
        ["Python + PyTorch + ResNet18", "Python 3.12 + PyTorch 2.5.1 + ResNet18 迁移学习", "符合"],
        ["FastAPI 后端 + React 前端 + SQLite", "FastAPI + Uvicorn｜React 18 + Vite 5 + AntD 5｜SQLite(WAL)", "符合"],
        ["六个功能模块", "6 / 6 实现，均超出最低要求", "完成"],
        ["8 份文档（5.2）", "8 / 8 齐备，另有 4 份附加技术报告", "完成"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [3.45, 7.55, 1.43], rows, size=10.5, row_h=0.42)
    note(s, 5.55, "额外攻坚：真实田间照片（PlantDoc）准确率由 52.9% 提升至 80.0%，并补齐非叶片拒识与置信度校准。", NAVY, 10.5)
    footer(s, no)

    # ===== P4 系统总体设计 =====
    s = new_slide(prs); no += 1
    header(s, "02 系统设计")
    page_title(s, "四层架构", "每层职责单一，层间通过明确的接口约定通信")
    rows = [
        ["层级", "技术组成", "职责"],
        ["前端展示层", "React 18 · Vite 5 · Ant Design 5", "上传预览、结果可视化、历史与知识库浏览；主题 token 化"],
        ["接口服务层", "FastAPI · Pydantic · JWT · 限流", "鉴权、参数校验、统一响应结构、记录落库、CSV/JSON 导出"],
        ["模型推理层", "PyTorch 2.5.1 · ResNet18 · 校准 · 检测头", "预处理→前向→作物先验→非叶片判定→置信度校准→知识卡"],
        ["数据存储层", "SQLite（WAL）· 疾病知识库 · 文件存储", "用户、识别记录、知识库、上传图片"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [1.85, 4.35, 6.23], rows, size=10.5, row_h=0.55)
    textbox(s, ML, 4.35, CW, 0.28, "一次识别的完整链路", 12.5, True, NAVY)
    chain = ("① 浏览器选图 + 选作物  →  ② 前端校验并上传  →  ③ 服务端校验（扩展名/魔数/大小）\n"
             "④ 预处理（EXIF 方向校正 → Resize → CenterCrop → 归一化）  →  ⑤ ResNet18 前向\n"
             "⑥ 作物先验掩码（仅作用于展示概率）  →  ⑦ 非叶片判定  →  ⑧ 温度校准\n"
             "⑨ 组装 Top-3 + 知识卡  →  ⑩ 返回响应并写入识别记录")
    textbox(s, ML, 4.68, CW, 1.9, chain, 11, False, GRAY, spacing=1.5)
    footer(s, no)

    # ===== P5 六大模块总览 =====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "六大功能模块总览", "对应任务书 2.2 的六项要求，逐项落地")
    rows = [
        ["#", "模块", "实现要点"],
        ["1", "图像上传", "拖拽 / 点击上传；扩展名白名单 + 文件头魔数校验 + ≤10MB；上传即预览"],
        ["2", "图像预处理", "默认 224×224（对标任务书）；EXIF 方向校正、超大图预缩放、ImageNet 归一化"],
        ["3", "深度学习识别", "ResNet18 迁移学习，38 类输出，返回 Top-3 概率与类别"],
        ["4", "结果展示", "Top-3 条形图 + 校准置信度 + 可靠性分档 + 症状/防治知识卡 + 免责声明"],
        ["5", "历史记录", "分页列表、缩略图、结果标签、详情对比、CSV / JSON 双格式导出"],
        ["6", "用户管理", "注册登录、改昵称、改密码（校验旧密码）、接口限流、字段级加密"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [0.55, 1.75, 10.13], rows, size=10.5, row_h=0.58)
    note(s, 5.72, "全部模块已在同一套 WebApp 中联调通过：注册 → 登录 → 识别 → 查看历史 → 导出，端到端可演示。", NAVY, 10.5)
    footer(s, no)

    # ===== P6 模块一：用户与鉴权 =====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "模块一：用户管理与数据安全", "注册 / 登录 / 个人信息 / 密码修改")
    picture_fit(s, ASSETS / "01_login.png", ML, BODY_Y, 3.3, 2.08)
    picture_fit(s, ASSETS / "07_profile.png", ML, BODY_Y + 2.35, 3.3, 2.08)
    textbox(s, 4.15, BODY_Y - 0.02, 8.7, 0.26, "怎么实现", 12, True, NAVY)
    bullets(s, 4.15, BODY_Y + 0.32, 8.7, [
        "密码 bcrypt 加盐哈希存储，数据库中不出现明文，也不可逆推",
        "自实现 HMAC-SHA256 JWT；未配置密钥时进程内随机生成，杜绝默认密钥伪造任意账户",
        "登录 / 注册接口限流 60 次/分钟，阻断暴力破解与账号枚举",
        "email 字段 Fernet 加密落库；HTTPS + HSTS（配套自签证书脚本）",
        "改密码需校验旧密码，前后端字段契约已对齐并有回归用例",
    ], size=11.5, gap=10)
    rect(s, 4.15, 5.32, 8.6, 0.012, LINE)
    textbox(s, 4.15, 5.45, 8.7, 0.26, "优点", 12, True, NAVY)
    textbox(s, 4.15, 5.78, 8.7, 0.9,
            "安全不是「加个登录框」：审计发现的 1 项 P0（默认密钥可伪造账户）与 4 项 P1（限流绕过、\n"
            "登录时序侧信道、改密契约错位）全部修复，并附可执行 PoC 与回归用例。",
            11, False, GRAY, spacing=1.4)
    footer(s, no)

    # ===== P7 模块二：识别（核心大图）=====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "模块二：病害识别（核心）", "上传 → 识别 → Top-3 + 知识卡，结果页同时给出可信度提示")
    picture_fit(s, ASSETS / "04_detect_result.png", ML, BODY_Y, 7.55, 4.72)
    x = 8.25
    textbox(s, x, BODY_Y - 0.02, 4.65, 0.26, "结果页读法", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.65, [
        "已限定作物 4 个候选：只在候选类内归一化，避免跨作物误判",
        "Top-1 置信度 + 真实场景校准值：两个数字并排，避免过度自信",
        "可靠性分档（高 / 中 / 低）用颜色标签呈现，低可信时明确提示复核",
        "一病多名别名：如「苹果雪松锈病」又称「苹果桧锈病」",
        "知识卡给出症状特征与防治建议，并常驻免责声明",
    ], size=11, gap=9)
    footer(s, no)

    # ===== P8 识别链路实现 =====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "识别链路的关键实现", "五个环节，每个环节都有明确的校验或约束")
    picture_fit(s, ASSETS / "03_detect_uploaded.png", ML, BODY_Y + 0.05, 4.15, 2.6)
    textbox(s, ML, BODY_Y + 2.78, 4.15, 1.0,
            "上传后即时预览；作物下拉由后端 /crops 下发，\n前端不硬编码，保证与模型类别一致。",
            10, False, GRAY, spacing=1.4)
    rows = [
        ["环节", "实现要点"],
        ["① 上传校验", "扩展名白名单 + 文件头魔数 + 大小上限；拒绝伪造扩展名的非图片文件"],
        ["② 预处理", "EXIF 方向校正 → 短边缩放 → 中心裁剪 224 → ImageNet 均值方差归一化"],
        ["③ 推理", "fp32 单张前向，与评估脚本完全一致的变换，保证线上与离线指标可比"],
        ["④ 后处理", "作物掩码只改展示概率；非叶片判据（leaf_prob）始终用全 38 类 softmax"],
        ["⑤ 落库", "写入识别记录（图片/作物/结果/置信度/时间），供历史页与导出使用"],
    ]
    table(s, 4.85, BODY_Y + 0.05, 7.95, [1.15, 6.8], rows, size=10, row_h=0.62)
    note(s, 6.35, "不变量：作物先验不影响拒识判定 —— 否则指定一个作物即可绕过「非叶片」拦截；已有单测守护。", NAVY, 10)
    footer(s, no)

    # ===== P9 模型训练与优化 =====
    s = new_slide(prs); no += 1
    header(s, "04 模型实现")
    page_title(s, "模型是怎么训出来的", "任务书算法四阶段：数据准备 → 训练 → 优化 → 部署")
    rows = [
        ["阶段", "关键做法", "产出"],
        ["数据准备", "38 类按 8:1:1 切分；用 aHash + 并查集把「同一片叶子的近重复图」归组，整组分到同一侧，杜绝泄漏", "dataset_index.json"],
        ["训练策略", "ResNet18 ImageNet 预训练迁移；类别逆频率加权 + 随机裁剪/翻转/旋转/色彩抖动；交叉熵 + Adam + 余弦退火 + 按宏 F1 早停 + 混合精度", "best_resnet18_38.pth"],
        ["模型优化", "INT8 量化 11.2MB、剪枝 36.1MB、fp32 44.9MB，三后端可切换，附精度-体积对比报告", "三份权重 + 对比报告"],
        ["领域适配", "用真实田间标注（PlantDoc 2341 张）与实验室图混合微调，低学习率、早停守住实验室精度", "finetuned 权重"],
        ["推理部署", "FastAPI 暴露 /predict，返回 Top-K 概率 + 知识卡；后端可切换 fp32 / 剪枝 / 量化", "在线服务"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [1.45, 8.5, 2.48], rows, size=10, row_h=0.62)
    note(s, 5.55, "为什么要做领域适配：实验室图近乎满分，但真实田间照片准确率只有 52.9% —— 差距来自背景、光照、拍摄角度与病斑形态的分布差异。", NAVY, 10.5)
    footer(s, no)

    # ===== P10 指标 =====
    s = new_slide(prs); no += 1
    header(s, "04 模型效果")
    page_title(s, "实测指标：四组数据分开看", "实验室域看能力上限，真实域看可用性，拒识与校准看可信度")
    rows = [
        ["指标", "数据集 / 条件", "实测", "结论"],
        ["Top-1 准确率", "PlantVillage 测试集 4914 张（防泄漏切分）", "99.06%", "远超 ≥85%"],
        ["宏平均 F1", "同上，38 类", "0.9895", "类别均衡"],
        ["最低单类 F1", "同上（玉米灰斑病）", "0.911", "超过 ≥0.80"],
        ["真实域 Top-1", "PlantDoc 225 张去重，指定作物（224 模式）", "76.9%", "微调前 52.9%"],
        ["真实域 Top-1", "同上（320 高精度模式）", "80.0%", "可选增强"],
        ["非叶片拒识", "文档截图 / 图表 / 合成图抽样", "9 / 9 拦截", "防误用"],
        ["校准效果", "真实域，指定作物模式的 ECE", "0.066 → 0.055", "高置信错判 17→11"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [1.85, 5.1, 2.3, 3.18], rows, size=10, row_h=0.44,
          highlight={(1, 2): NAVY, (4, 2): NAVY, (5, 2): NAVY})
    note(s, 5.58, "口径说明：99.06% 为域微调后的测试集结果（微调前 99.27%），用 0.2pp 实验室指标换取真实域 +24~27pp 是刻意取舍；\n拒识 9/9 为 9 张抽样结果，同分布留出集拦截率 99.6%。", GRAY, 9)
    footer(s, no)

    # ===== P11 历史记录 =====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "模块三：识别历史", "可回看、可对比、可导出")
    picture_fit(s, ASSETS / "05_history.png", ML, BODY_Y, 7.55, 4.72)
    x = 8.25
    textbox(s, x, BODY_Y - 0.02, 4.65, 0.26, "怎么实现", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.65, [
        "识别成功即落库，记录 user_id、图片、作物、结果、置信度、时间",
        "列表分页查询，SQL 层按 user_id 过滤，用户间数据严格隔离",
        "缩略图 + 结果标签 + 置信度并列，一眼看出可疑记录",
        "详情抽屉做原图 / 结果对比",
        "支持 CSV 与 JSON 双格式导出，便于后续统计分析",
    ], size=11, gap=9)
    footer(s, no)

    # ===== P12 知识库 =====
    s = new_slide(prs); no += 1
    header(s, "03 功能模块")
    page_title(s, "模块四：病害知识库", "38 类按作物分组，识别结果直接关联知识卡")
    picture_fit(s, ASSETS / "06_knowledge.png", ML, BODY_Y, 7.55, 4.72)
    x = 8.25
    textbox(s, x, BODY_Y - 0.02, 4.65, 0.26, "内容与实现", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.65, [
        "14 种作物、38 个类别，按作物分组浏览，标注病害 / 健康",
        "每类给出症状特征与防治建议，术语与模型类别一一对应",
        "别名映射：12 个常见「一病多名」条目，避免译名差异被误判为识别错误",
        "支持按作物或病害名称检索",
        "知识内容以 JSON 维护，改内容不需要改代码",
    ], size=11, gap=9)
    footer(s, no)

    # ===== P13 工程亮点 =====
    s = new_slide(prs); no += 1
    header(s, "05 工程亮点")
    page_title(s, "主动加固：三个容易被忽略的问题", "都是任务书没要求、但真实使用会踩到的")
    rows = [
        ["问题", "风险", "做法与结果"],
        ["近重复图导致指标虚高", "同一片叶子的多张图被随机分到训练与测试两侧，准确率虚高到 99.9%", "aHash + 并查集按「同叶组」切分，指标回落到 99.27% —— 更低的数字反而更可信"],
        ["非叶片输入仍给结论", "误传截图 / 插画也返回高置信度病害名，误导使用者", "独立「叶片 / 非叶片」检测头（513 参数，不改原模型）；同分布留出集：真实叶误杀 4.6%，非叶片拦截 99.6%"],
        ["自报置信度不可信", "真实域自报 73% 把握、实际只对 17.5%，典型的「自信地错」", "温度缩放校准 + 可靠性分档；真实域 ECE 0.066 → 0.055，高置信错判 17 → 11 张"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [2.35, 4.35, 5.73], rows, size=10, row_h=0.78)
    textbox(s, ML, 5.5, CW, 0.26, "工程与运维", 12, True, NAVY)
    bullets(s, ML, 5.82, CW, [
        "接口限流 60 次/分钟 + 安全响应头；上传校验三层（扩展名 / 魔数 / 大小）",
        "部署三种形态（开发 / 单机 NSSM 服务 / systemd）+ 备份恢复 + 监控与排障手册（docs/13）",
        "15 个 pytest 用例守护契约：校准字段、作物先验与拒识不变量、鉴权与导出",
    ], size=10.5, gap=6)
    footer(s, no)

    # ===== P14 优点总结 =====
    s = new_slide(prs); no += 1
    header(s, "05 工程亮点")
    page_title(s, "这个系统的优点", "每一条都有可核对的数字或产物，不是自我评价")
    rows = [
        ["优点", "证据"],
        ["不只看实验室精度", "单独建真实田间评测集（225 张去重），把 52.9% 做到 80.0%，而不是停在 99% 的实验室数字上"],
        ["敢承认模型会错", "置信度校准 + 可靠性分档：低可信结果明确提示复核，而不是硬给一个高置信度"],
        ["能挡住误用", "非叶片图抽样 9/9 拦截；结果页常驻免责声明；文档写明适用范围与局限"],
        ["工程完整可交付", "8 份任务书文档全部齐备 + 部署运维手册（12 章）+ 安全审计报告 + 15 个回归用例"],
        ["成本可控", "INT8 量化后 11.2MB（fp32 的 25%），CPU 可推理；另提供 320 高精度模式按需切换"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [2.3, 10.13], rows, size=10.5, row_h=0.7)
    note(s, 5.75, "一句话：准确率达标不难，难的是让结果「可被信任、可被交付、可被复现」。", NAVY, 11)
    footer(s, no)

    # ===== P15 交付清单 =====
    s = new_slide(prs); no += 1
    header(s, "05 交付清单")
    page_title(s, "交付物清单", "代码、模型、文档、测试四类产物")
    rows = [
        ["类别", "内容", "位置"],
        ["源代码", "后端 API、模型训练与推理、前端 4 个页面、脚本工具链", "src/api, src/model, src/web, scripts"],
        ["模型产物", "fp32 / 剪枝 / 量化三份权重、校准参数、叶片检测头", "models/"],
        ["8 份文档", "PRD、UI/UX、数据库、API、测试报告、部署手册、AI 协作教程、PPT", "docs/01–05, 07, 13"],
        ["附加报告", "OOD 拒识评估、代码审计、数据集指南、域适应实验、真实域微调、双分辨率", "docs/08–11, 14–17"],
        ["测试与评估", "15 个 pytest 用例、混淆矩阵、分类报告、OOD 跨来源验证", "tests/, outputs/"],
        ["说明与语料", "README（参数与启动方式）、RAG 语料库 225 块 + 提示词", "README.md, rag/"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [1.75, 6.55, 4.13], rows, size=10, row_h=0.6)
    footer(s, no)

    # ===== P16 局限与展望 =====
    s = new_slide(prs); no += 1
    header(s, "06 总结")
    page_title(s, "局限与下一步", "清楚短板，比宣称完美更有价值")
    rows = [
        ["当前局限", "原因", "下一步"],
        ["番茄类内病害区分弱", "真实标注太少（细菌性斑点病仅 8 张训练图），病斑形态相近", "补真实标注或改用更大骨干网络"],
        ["真实域 80% 仍低于实验室 99%", "田间背景、光照、角度、品种与训练分布差异大", "继续扩充田间数据 + 领域自适应训练"],
        ["Firefox / Safari 未实机复测", "本机 Firefox 无头渲染失败，Windows 无 Safari", "按 6 步清单在对应环境补测"],
        ["只做分类，不做定位与分级", "数据集标签只到「作物 + 病害」粒度，无病斑框与严重程度", "后续可做检测 / 分割与轻中重分级"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [3.3, 5.15, 3.98], rows, size=10, row_h=0.62)
    rect(s, ML, 5.35, CW, 0.012, LINE)
    textbox(s, ML, 5.55, 7.2, 0.36, "汇报完毕，请各位老师指正", 20, True, NAVY)
    textbox(s, ML, 6.05, CW, 0.3,
            "演示入口：前端 http://localhost:5173　｜　后端 http://127.0.0.1:8000/docs（交互式接口文档）",
            10.5, False, GRAY)
    footer(s, no)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"已生成：{OUT}")
    print(f"页数：{len(prs.slides.__iter__.__self__._sldIdLst)}")


if __name__ == "__main__":
    build()
