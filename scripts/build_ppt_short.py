# -*- coding: utf-8 -*-
"""
生成精简版项目汇报 PPT（8 页）：以网站界面与测试结果为主体。

复用 build_ppt_report.py 的版式工具（同一套色板、表头、字体与自检口径），
只重写页面内容，保证两份稿子视觉一致、坐标体系一致。

运行：python scripts/build_ppt_short.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_ppt_report as base          # 复用 set_font / header / table / picture_fit ...
from pptx import Presentation
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches

ROOT = base.ROOT
ASSETS = base.ASSETS
OUT = ROOT / "03.ppt" / "农作物叶片病害识别系统" / "农作物叶片病害识别系统_项目汇报.pptx"

# 精简版 8 页：覆盖 base.footer 的页码总数
base.TOTAL_PAGES = 8

header = base.header
page_title = base.page_title
footer = base.footer
rect = base.rect
textbox = base.textbox
bullets = base.bullets
table = base.table
note = base.note
picture_fit = base.picture_fit
new_slide = base.new_slide

ML, CW, BODY_Y = base.ML, base.CW, base.BODY_Y
SW, SH = base.SW, base.SH
NAVY, BLUE, MID = base.NAVY, base.BLUE, base.MID
LIGHT, PALE, TEXT, GRAY, MUTED, LINE = base.LIGHT, base.PALE, base.TEXT, base.GRAY, base.MUTED, base.LINE
RED, YELLOW, WHITE = base.RED, base.YELLOW, base.WHITE
RGBColor = base.RGBColor


def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    # ===== P1 封面 =====
    s = new_slide(prs)
    rect(s, 0, 0, SW, SH, NAVY)
    rect(s, 0, 4.35, SW, 0.02, MID)
    textbox(s, 1.1, 2.05, 11.2, 0.36, "前沿技术实训 · 项目汇报", 13, False, RGBColor(0xC7, 0xD8, 0xEE))
    textbox(s, 1.1, 2.52, 11.2, 0.8, "农作物叶片病害识别系统", 42, True, WHITE)
    textbox(s, 1.1, 3.42, 11.2, 0.42, "网站演示 · 测试结果", 20, False, RGBColor(0xC7, 0xD8, 0xEE))
    textbox(s, 1.1, 4.65, 11.2, 0.6,
            "38 类作物叶片病害识别（14 种作物）｜React 18 + FastAPI + PyTorch 2.5.1 + ResNet18", 12.5, False,
            RGBColor(0xD8, 0xE3, 0xF3), spacing=1.5)
    textbox(s, 1.1, 6.4, 5.0, 0.3, "2026 年 10 月", 12, False, RGBColor(0xA8, 0xBE, 0xDC))

    # ===== P2 网站 · 识别主流程（大图）=====
    s = new_slide(prs)
    header(s, "网站演示")
    page_title(s, "识别页：上传叶片照片 → 输出诊断结果", "左侧选图与作物，右侧结果含 Top-3、校准置信度、可靠性分档与防治建议")
    picture_fit(s, ASSETS / "04_detect_result.png", ML, BODY_Y, 7.65, 4.75)
    x = 8.35
    textbox(s, x, BODY_Y - 0.02, 4.55, 0.26, "结果页信息怎么读", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.55, [
        "限定作物后只在候选类内比较（本页苹果 4 个候选），避免跨作物误判",
        "同时给出原始置信度与真实场景校准值，避免「自报 99% 其实判错」",
        "可靠性分档（高 / 中 / 低）用颜色标签呈现，低可信时提示人工复核",
        "一病多名会标「又称」（苹果雪松锈病 = 苹果桧锈病），避免误判为识别错误",
        "附症状特征与防治建议，并常驻免责声明",
    ], size=10.5, gap=8)
    footer(s, 2)

    # ===== P3 网站 · 上传与选作物 =====
    s = new_slide(prs)
    header(s, "网站演示")
    page_title(s, "从上传到出结果：五个环节都有校验", "前端不做假校验，服务端二次把关，保证送进模型的一定是合法图片")
    picture_fit(s, ASSETS / "03_detect_uploaded.png", ML, BODY_Y + 0.05, 4.25, 2.65)
    textbox(s, ML, BODY_Y + 2.82, 4.25, 0.8,
            "上传即预览；作物下拉由后端 /crops 下发，\n前端不硬编码，保证与模型类别一致。",
            10, False, GRAY, spacing=1.4)
    rows = [
        ["环节", "实现要点"],
        ["上传校验", "扩展名白名单 + 文件头魔数 + ≤10MB，拦截伪造扩展名的非图片"],
        ["预处理", "EXIF 方向校正 → 短边缩放 → 中心裁剪 224 → ImageNet 归一化"],
        ["推理", "ResNet18 单张前向，变换与离线评估完全一致，线上指标可复现"],
        ["后处理", "作物掩码只改展示概率；拒识判据始终用全 38 类 softmax"],
        ["落库", "记录用户、图片、作物、结果、置信度与时间，供历史与导出"],
    ]
    table(s, 4.95, BODY_Y + 0.05, 7.85, [1.15, 6.7], rows, size=10, row_h=0.66)
    note(s, 6.42, "不变量：指定作物不能影响「是否叶片」的判定，否则指定一个作物即可绕过拒识；已有单测守护。", NAVY, 10)
    footer(s, 3)

    # ===== P4 网站 · 历史记录 =====
    s = new_slide(prs)
    header(s, "网站演示")
    page_title(s, "识别历史：可回看、可对比、可导出", "每次识别自动落库，按用户隔离，支持分页与详情对比")
    picture_fit(s, ASSETS / "05_history.png", ML, BODY_Y, 7.65, 4.75)
    x = 8.35
    textbox(s, x, BODY_Y - 0.02, 4.55, 0.26, "页面能力", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.55, [
        "缩略图 + 结果标签 + 置信度 + 时间并列，可疑记录一眼可见",
        "分页查询，SQL 层按 user_id 过滤，用户间数据严格隔离",
        "详情抽屉做原图与结果对比，便于复核",
        "CSV / JSON 双格式导出，方便后续统计",
    ], size=10.5, gap=9)
    footer(s, 4)

    # ===== P5 网站 · 知识库 =====
    s = new_slide(prs)
    header(s, "网站演示")
    page_title(s, "病害知识库：38 类按作物分组", "识别结果直接关联知识卡，术语与模型类别一一对应")
    picture_fit(s, ASSETS / "06_knowledge.png", ML, BODY_Y, 7.65, 4.75)
    x = 8.35
    textbox(s, x, BODY_Y - 0.02, 4.55, 0.26, "内容构成", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 4.55, [
        "14 种作物、38 个类别，按作物分组浏览并标注病害 / 健康",
        "每类给出症状特征与防治建议，可直接用于田间处置",
        "12 个常见「一病多名」别名映射，消除译名歧义",
        "支持按作物或病害名称检索；内容以 JSON 维护，改内容不改代码",
    ], size=10.5, gap=9)
    footer(s, 5)

    # ===== P6 网站 · 账号与安全 =====
    s = new_slide(prs)
    header(s, "网站演示")
    page_title(s, "账号体系：注册登录与个人信息", "密码不可逆存储，接口限流，个人信息可改")
    picture_fit(s, ASSETS / "01_login.png", ML, BODY_Y, 3.35, 2.1)
    picture_fit(s, ASSETS / "07_profile.png", ML, BODY_Y + 2.38, 3.35, 2.1)
    x = 4.2
    textbox(s, x, BODY_Y - 0.02, 8.65, 0.26, "实现要点", 12, True, NAVY)
    bullets(s, x, BODY_Y + 0.32, 8.65, [
        "bcrypt 加盐哈希存储，数据库中不出现明文密码，也不可逆推",
        "自实现 HMAC-SHA256 JWT；未配置密钥时进程内随机生成，杜绝默认密钥伪造账户",
        "登录 / 注册限流 60 次/分钟，阻断暴力破解与账号枚举",
        "email 字段加密落库；生产环境走 HTTPS + HSTS",
        "改密码需校验旧密码；改昵称、改密码均有回归用例",
    ], size=11, gap=9)
    footer(s, 6)

    # ===== P7 测试结果 =====
    s = new_slide(prs)
    header(s, "测试结果")
    page_title(s, "实测数据：实验室域、真实域、拒识、校准", "四组数据分开看，避免用实验室指标掩盖真实场景表现")
    rows = [
        ["指标", "数据集 / 条件", "实测结果", "对照要求"],
        ["Top-1 准确率", "PlantVillage 测试集 4914 张（防泄漏切分）", "99.06%", "≥ 85%"],
        ["宏平均 F1", "同上，38 类", "0.9895", "—"],
        ["最低单类 F1", "同上（最低类为玉米灰斑病）", "0.911", "≥ 0.80"],
        ["真实域 Top-1", "PlantDoc 225 张去重，指定作物（224 模式）", "76.9%", "微调前 52.9%"],
        ["真实域 Top-1", "同上（320 高精度模式）", "80.0%", "可选增强"],
        ["非叶片拒识", "文档截图 / 图表 / 合成图抽样 9 张", "9 / 9 拦截", "防误用"],
        ["置信度校准", "真实域、指定作物模式的 ECE", "0.066 → 0.055", "高置信错判 17 → 11"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [1.85, 5.35, 2.35, 2.88], rows, size=10.5, row_h=0.46,
          highlight={(1, 2): NAVY, (4, 2): NAVY, (6, 2): NAVY})
    note(s, 5.55, "口径：99.06% 为域微调后结果（微调前 99.27%），用 0.2pp 实验室指标换取真实域 +24~27pp；"
                  "拒识 9/9 为 9 张抽样，同分布留出集拦截率 99.6%、真实叶误杀 4.6%。", GRAY, 9)
    footer(s, 7)

    # ===== P8 结论 =====
    s = new_slide(prs)
    header(s, "结论")
    page_title(s, "结论与局限", "能用的系统 = 准确率 + 可信度提示 + 可交付的工程")
    rows = [
        ["结论 / 局限", "说明"],
        ["任务书指标全部达标", "准确率 99.06%（要求 85%）、各类 F1 ≥ 0.911（要求 0.80），六个功能模块 6/6 实现"],
        ["真实场景可用", "真实田间照片从 52.9% 提升到 76.9%（320 模式 80.0%），并给出可信度分档"],
        ["番茄类内病害仍是短板", "细菌性斑点病训练样本仅 8 张，番茄若干病害易互相混淆 —— 需补真实标注"],
        ["只做分类，不做定位与分级", "数据集标签只到「作物 + 病害」粒度，无病斑框与严重程度标注"],
    ]
    table(s, ML, BODY_Y + 0.02, CW, [3.55, 8.88], rows, size=10.5, row_h=0.62)
    rect(s, ML, 4.95, CW, 0.012, LINE)
    textbox(s, ML, 5.18, 8.0, 0.4, "汇报完毕，请各位老师指正", 22, True, NAVY)
    textbox(s, ML, 5.75, CW, 0.5,
            "演示入口：前端 http://localhost:5173　｜　后端接口文档 http://127.0.0.1:8000/docs\n"
            "演示账号：ppt_demo / PptDemo2026（历史记录已有 3 条示例）", 10.5, False, GRAY, spacing=1.5)
    footer(s, 8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"已生成（精简版）：{OUT}")
    print(f"页数：{len(prs.slides)}")


if __name__ == "__main__":
    build()
