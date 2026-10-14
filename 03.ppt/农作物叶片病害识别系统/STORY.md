# STORY.md · 基于 CNN 的农作物叶片病害识别系统（实训答辩）

## ① 用户意图对齐

- **目标受众**：高校实训答辩评审组（指导老师 + 企业导师 + 同组同学），理工科背景，懂深度学习与 Web 开发，但非农学专业。
- **核心目标**：让评审相信三件事——① 这是一个**完整可运行**的前后端 + 模型闭环系统（不是跑分脚本）；② 指标**可信**（主动做了防泄漏，诚实值 99.27%）；③ 工程**规范**（安全和质量基线达标、有自动化测试、有文档）。
- **PPT 长度**：15 页。Hero 页配额 3–4 页（取 4 页：封面 / 数据集与防泄漏 / 识别演示 / 感谢页，占比 26.7%）。
- **视觉调性**：学术严谨 · 蓝白克制 · 证据驱动 · 界面优先（用真实产品截图说话，不用花哨图形）。
- **内容边界**：
  - **必讲**：项目背景与痛点、数据集与防泄漏切分、技术架构、功能演示（真实截图）、模型精度、模型压缩与部署选型、安全与质量基线。
  - **不讲**：训练超参细节、逐行代码实现、前端组件树。
  - **禁碰**：夸大指标、把 99% 说成"田间可用"（必须声明实验室数据集局限）、虚构数据或引用。

---

## ② 页面布局骨架

**页面总数 15 页，分 4 章。**

| 章 | 目录条目（逐字） | 页码区间 | 章节扉页页码 |
|---|---|---|---|
| 01 | 项目背景与目标 | P3–P5 | **P3** |
| 02 | 系统设计方案 | P6–P8 | **P6** |
| 03 | 功能实现与演示 | P9–P11 | **P9** |
| 04 | 成果与总结 | P12–P15 | **P12** |

> 目录（P2）声明 4 章 → 全篇有且仅有 4 个 `type: section` 扉页（P3 / P6 / P9 / P12），编号 01–04 连续，标题与目录逐字一致。✅

**Hero 页定位**：P1 封面（hero）、P5 数据集与防泄漏（hero，中段视觉高潮）、P10 识别演示（hero）、P15 感谢页（hero）。任意两个 Hero 之间至少间隔 3 个 Supporting 页。✅

**rhythm 曲线**：
P1 peak → P2 transition → P3 transition → **P4 valley** → **P5 peak** → P6 transition → P7 valley → P8 valley → P9 transition → **P10 peak** → P11 valley → P12 transition → P13 peak → P14 valley → P15 peak
（无连续 ≥3 页 valley：P7/P8 仅两页连续，随后由 P10 打破。）✅

**非对称版式预算**：13/15 = 87% ≥ 40%。✅ 其中 `左大图+右侧文字`（P7、P10）与 `非对称双栏`（P4）合计 3 页 = 20% ≤ 40%。✅

**对称版式预算**：仅 P14（图表+洞察，实为表格+洞察）与 P15（全屏视觉+大标题）两页，≤ 2 页上限。✅

---

## ③ 页面大纲

### P1 · 封面
| 字段 | 内容 |
|---|---|
| title | 基于 CNN 的农作物叶片病害识别系统 |
| type | cover |
| role | hero |
| rhythm | peak |
| layout | 全幅图+骑线文字（非对称） |
| visual | L1: sample_tomato_late_blight.jpg（右半 46% 铺满） |
| visual_role | atmosphere |
| density | 字数约 62 / 图片 1 张 / 留白约 30% |
| anti_pattern | 禁止党政红金、印章、烟花；禁止把叶片图缩小成 200×70 角标；禁止纯色矩形 + 单行文字的平铺封面 |
| description | 中英文主标 + 副标 + 汇报人/单位/日期；右侧真实病害叶片特写铺底。**判断**：封面即宣告这是一套覆盖 38 类的真实可运行系统，而非课程 demo。 |

### P2 · 目录
| 字段 | 内容 |
|---|---|
| title | 目录 |
| type | catalog |
| role | supporting |
| rhythm | transition |
| layout | 左标题+右内容（非对称） |
| visual | L3: 深蓝编号方块（角标） |
| visual_role | evidence |
| density | 字数约 48 / 图片 0 张 / 留白约 30% |
| anti_pattern | 禁止圆球编号；禁止英文比中文抢眼；禁止堆砌图标 |
| description | 左侧深蓝竖栏放"目录 / CONTENTS"与页码区间，右侧 4 个条目卡（编号 + 中文标题 + 一句话摘要），与 4 个章节扉页一一对应。 |

### P3 · 章节扉页 01
| 字段 | 内容 |
|---|---|
| title | 项目背景与目标 |
| type | section |
| role | transition |
| rhythm | transition |
| layout | 巨型数字+洞察（非对称） |
| visual | L3: 巨型数字 01 |
| visual_role | anchor |
| density | 字数约 78 / 图片 0 张 / 留白约 32% |
| anti_pattern | 禁止四卡片预览；禁止铺满正文段落；禁止 PART 02 字样 |
| description | 巨型数字 01 占左，右侧标题"项目背景与目标"与 3 个要点（行业痛点 / 任务书指标 / 范围收敛 26→38 类）。 |

### P4 · 背景与痛点
| 字段 | 内容 |
|---|---|
| title | 为什么做：基层植保的识别缺口 |
| type | content |
| role | supporting |
| rhythm | valley |
| layout | 非对称双栏（非对称） |
| visual | L2: sample_apple_scab.jpg（占右栏 340×220） |
| visual_role | evidence |
| density | 字数约 260 / 图片 1 张 / 留白约 22% |
| anti_pattern | 禁止 50:50 等分双栏；禁止把叶片图缩小为角标；禁止只摆痛点不给对策 |
| description | 左栏（宽）讲痛点：农技人员经验依赖强、基层力量不足、错过防治窗口；右栏（窄）用真实病斑图佐证"肉眼难分近缘病害"，底部深蓝强调条给出项目定位——**辅助识别工具，替代不了专家**。 |

### P5 · 数据集与防泄漏切分（Hero）
| 字段 | 内容 |
|---|---|
| title | 数据说话：让指标经得起追问 |
| type | content |
| role | hero |
| rhythm | peak |
| layout | 上大图+下方卡片（非对称） |
| visual | L1: sample_corn_gray_leaf_spot.jpg + sample_grape_black_rot.jpg + sample_healthy.jpg（三联 180×180） |
| visual_role | anchor |
| density | 字数约 300 / 图片 3 张 / 留白约 20% |
| anti_pattern | 禁止等宽卡片横排；禁止把核心数字塞进图表角落；禁止 L3 角标顶替 L1 |
| description | 上半部三联真实叶片样本（病斑 / 黑腐 / 健康），下半部两卡对照：**旧随机切分 P=22.5% 精确重复泄漏** → **分组切分后泄漏清零**。判断：指标从 99.82% 降到 99.27%，降的 0.55pp 正是"水分"，剩下的才是真实泛化能力。 |

### P6 · 章节扉页 02
| 字段 | 内容 |
|---|---|
| title | 系统设计方案 |
| type | section |
| role | transition |
| rhythm | transition |
| layout | 巨型数字+洞察（非对称） |
| visual | L3: 巨型数字 02 |
| visual_role | anchor |
| density | 字数约 76 / 图片 0 张 / 留白约 32% |
| anti_pattern | 禁止四卡片预览；禁止铺满正文段落；禁止与封面雷同 |
| description | 巨型数字 02 占左，右侧标题"系统设计方案"与要点（前后端分离 / 模型与推理 / 数据持久化）。 |

### P7 · 技术架构
| 字段 | 内容 |
|---|---|
| title | 前后端分离的三层架构 |
| type | content |
| role | supporting |
| rhythm | valley |
| layout | 左大图+右侧文字（非对称） |
| visual | L1: Diagram(架构分层图)（占左 56%） |
| visual_role | evidence |
| density | 字数约 240 / 图片 0 张 / 留白约 18% |
| anti_pattern | 禁止把架构图缩小成装饰角标；禁止右侧堆成编号列表；禁止 50:50 等分 |
| description | 左侧用 Box 绘制四层架构（前端 Web → 后端 API → 推理引擎 → 数据层），右侧三卡说明各层技术：React18+Vite5+AntD5 / FastAPI+Uvicorn / ResNet18 三后端切换 / SQLite(WAL)。 |

### P8 · 功能模块设计
| 字段 | 内容 |
|---|---|
| title | 五个模块构成完整业务闭环 |
| type | content |
| role | supporting |
| rhythm | valley |
| layout | 左标题+右内容（非对称） |
| visual | L3: FAIcon 模块图标（角标，统一 28×28） |
| visual_role | evidence |
| density | 字数约 250 / 图片 0 张 / 留白约 18% |
| anti_pattern | 禁止等宽四卡；禁止只有功能名没有说明；禁止图标与色阶不统一 |
| description | 左侧深蓝色块写"模块设计"，右侧五行模块条（图标 + 模块名 + 一句话职责）：用户体系 / 图像识别 / 结果与知识 / 历史记录 / 模型优化。底部强调条点出"录入→识别→归档→复核"闭环。 |

### P9 · 章节扉页 03
| 字段 | 内容 |
|---|---|
| title | 功能实现与演示 |
| type | section |
| role | transition |
| rhythm | transition |
| layout | 巨型数字+洞察（非对称） |
| visual | L3: 巨型数字 03 |
| visual_role | anchor |
| density | 字数约 74 / 图片 0 张 / 留白约 32% |
| anti_pattern | 禁止四卡片预览；禁止铺满正文段落；禁止满版色块 |
| description | 巨型数字 03 占左，右侧标题"功能实现与演示"与要点（真实界面截图 / 前后端已联调 / 非 mock 数据）。 |

### P10 · 病害识别演示（Hero）
| 字段 | 内容 |
|---|---|
| title | 上传一张叶片，三步给出诊断 |
| type | content |
| role | hero |
| rhythm | peak |
| layout | 左大图+右侧文字（非对称） |
| visual | L1: detect_result.png（占左 58%，真实界面截图） |
| visual_role | anchor |
| density | 字数约 230 / 图片 1 张 / 留白约 18% |
| anti_pattern | 禁止把产品截图缩小为 200×70；禁止 50:50 等分双栏；禁止只给截图不给结论 |
| description | 左侧铺真实识别结果页截图（Top-3 条形图 + 中文病名 + 防治卡），右侧三步流程（上传 → 推理 → 结果）与结论：Top-1 置信度可视化 + 低置信度触发人工复核提示。 |

### P11 · 历史与知识库
| 字段 | 内容 |
|---|---|
| title | 记录可追溯，病害可查询 |
| type | content |
| role | supporting |
| rhythm | valley |
| layout | 上大图+下方卡片（非对称） |
| visual | L1: history.png（占上 52%）+ L2: knowledge.png（右下 300×170） |
| visual_role | evidence |
| density | 字数约 250 / 图片 2 张 / 留白约 18% |
| anti_pattern | 禁止两图等大并排；禁止把历史截图塞进卡片内部小角；禁止无数据来源说明 |
| description | 上半部铺历史记录页真实截图（登录用户识别记录落库、可导出 CSV / JSON），下半部两卡：历史模块（含导出链路）+ 知识库模块（38 类症状与防治，配小图）。 |

### P12 · 章节扉页 04
| 字段 | 内容 |
|---|---|
| title | 成果与总结 |
| type | section |
| role | transition |
| rhythm | transition |
| layout | 巨型数字+洞察（非对称） |
| visual | L3: 巨型数字 04 |
| visual_role | anchor |
| density | 字数约 72 / 图片 0 张 / 留白约 32% |
| anti_pattern | 禁止四卡片预览；禁止铺满正文段落；禁止与封面雷同 |
| description | 巨型数字 04 占左，右侧标题"成果与总结"与要点（精度达标 / 模型已压缩 / 工程基线合格）。 |

### P13 · 模型精度指标
| 字段 | 内容 |
|---|---|
| title | 精度：达标且留有余量 |
| type | content |
| role | supporting |
| rhythm | peak |
| layout | 左标题+右内容（非对称） |
| visual | 大数字：99.27 / 99.23 / 0.9524（最小 72px） |
| visual_role | evidence |
| density | 字数约 260 / 图片 0 张 / 留白约 18% |
| anti_pattern | 禁止把 KPI 塞进等宽卡片；禁止只给数字不给对标；禁止缺少逐类下限说明 |
| description | 左侧深蓝竖栏放口径说明（4,914 张测试集 / 不使用 AMP），右侧三组 KPI（准确率 99.27% vs ≥85%、宏 F1 99.23% vs ≥0.80、最低类 F1 0.9524）+ 结论条：36 条错误全为同作物近缘病害互混，无跨作物大错。 |

### P14 · 模型压缩与部署选型
| 字段 | 内容 |
|---|---|
| title | 一份模型，三种部署形态 |
| type | content |
| role | supporting |
| rhythm | valley |
| layout | 图表+洞察（对称） |
| visual | Chart(对比表) |
| visual_role | evidence |
| density | 字数约 220 / 图片 0 张 / 留白约 16% |
| anti_pattern | 禁止只给表不给选型判断；禁止表格无表头色；禁止关键格无高亮 |
| description | 表格占主体：fp32 / INT8 / 剪枝 三行的体积、缩减、准确率、延迟、加速比；下方洞察条——**INT8 量化以 −0.04pp 精度换来 4 倍瘦身 + 2 倍加速，是边缘部署首选**。 |

### P15 · 感谢页
| 字段 | 内容 |
|---|---|
| title | 感谢聆听 |
| type | ending |
| role | hero |
| rhythm | peak |
| layout | 全屏视觉+大标题（对称） |
| visual | L3: 页脚线纹（角标） |
| visual_role | atmosphere |
| density | 字数约 40 / 图片 0 张 / 留白约 45% |
| anti_pattern | 禁止烟花、金装饰、印章；禁止感谢页与封面完全不同体系 |
| description | 延续封面深蓝顶条体系，居中"感谢聆听 · 敬请指正"+ 项目信息一行 + 一句免责声明（辅助识别工具，不替代现场农技诊断）。 |

---

## STORY Checklist 自检

- ✅ Hero 页 4 个 / 15 页 = 26.7%，落在 20–30%
- ✅ 无连续 ≥3 页 supporting + valley（最长连续 valley 为 P7–P8 两页）
- ✅ `N卡片横排` 出现 0 次（远低于 ≤2 上限）
- ✅ 非对称版式占比 87% ≥ 40%
- ✅ 任意相邻两页版式均不同（15 页逐对核对通过）
- ✅ `左大图+右侧文字` + `非对称双栏` = 3 页 / 15 = 20% ≤ 40%
- ✅ 每页均含 role / rhythm / visual_role / anti_pattern
- ✅ 含关键数据页（P5 / P13 / P14）均写成"数据 + 判断"两段式
- ✅ `type: section` 扉页 4 个 = 目录声明章节数 4，编号 01–04 连续，标题逐字一致
