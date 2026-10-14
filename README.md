# 基于 CNN 的农作物叶片病害识别系统

> 实训项目 · 农作物叶片病害智能识别 Web 应用（ResNet18 迁移学习 + FastAPI + React）
> 覆盖 **14 种作物、38 类病害及健康状态**，防泄漏诚实指标 **准确率 99.27% / 宏 F1 99.23%**。

---

## 1. 项目简介

传统病害诊断依赖农技人员肉眼观察，经验依赖强、基层力量不足，常错过最佳防治窗口。本项目以 **ResNet18** 为骨干网络，构建一款浏览器端农作物叶片病害识别应用：

- 农户 / 农技人员 **上传叶片照片** → 系统自动识别病害类别；
- 返回 **置信度、Top-3 候选概率分布与防治建议**；
- 登录用户可保存 **识别历史** 并导出（CSV / JSON）。

系统定位为 **辅助识别与科普工具**，结果仅供参考，不能替代现场专业农技诊断。

---

## 2. 核心特性

| 特性 | 说明 |
|------|------|
| 38 类识别 | 14 种作物（苹果/蓝莓/樱桃/玉米/葡萄/柑橘/桃/甜椒/马铃薯/树莓/大豆/南瓜/草莓/番茄）的病害 + 健康子类 |
| Top-3 概率 | 返回 Top-1 与 Top-3 候选，概率条形图可视化，低置信度提示人工复核 |
| 防治知识库 | 38 类病害的「症状特征 / 防治建议」知识卡，识别结果页直接展示 |
| 用户体系 | 注册 / 登录 / 登出（JWT）、个人信息管理，密码 bcrypt 加盐哈希 |
| 历史记录 | 登录用户保存识别记录，支持原图与结果对比、CSV / JSON 导出 |
| 三后端模型 | fp32 / INT8 量化 / 结构化剪枝，通过 `INFER_BACKEND` 一键切换，对调用方透明 |
| 安全基线 | bcrypt 密码、JWT 密钥环境变量、CORS 白名单、真实 HTTP 状态码、静态目录不暴露、SQLite WAL 并发保护 |
| 自动化测试 | `pytest` 11 用例（纯函数 / 字体 / API 冒烟），模型缺失自动跳过 |

---

## 3. 效果指标（防泄漏诚实值）

> 数据集 PlantVillage，38 类 / 约 5.4 万张；采用 **分组切分（group-aware split）** 消除近重复泄漏（详见 `docs/05_测试报告.md`）。

| 指标 | 数值 | 任务书要求 |
|------|------|-----------|
| 整体准确率 | **99.27%** | ≥ 85% ✅ |
| 宏平均 F1 | **99.23%** | 各类 F1 ≥ 0.80 ✅ |
| 校准误差 ECE | 0.0028 | — |
| 38 类最低 F1 | 0.9524（玉米灰斑病） | ≥ 0.80 ✅ |

### 模型压缩对比（边缘 / CPU 部署选型）

| 后端 | 体积 | 体积缩减 | 准确率 | 单图延迟(CPU) | 加速比 |
|------|------|---------|--------|--------------|--------|
| fp32（基线） | 44.9 MB | 0% | 99.27% | 22.75 ms | 1.00× |
| **INT8 量化** | **11.2 MB** | **−75%** | 99.23% | **11.16 ms** | **2.04×** |
| 结构化剪枝 | 36.1 MB | −20% | 99.02% | 18.98 ms | 1.20× |

> **结论**：INT8 量化以几乎无损（仅 −0.04pp）换来 4 倍体积压缩 + 2 倍推理加速，是边缘 / CPU 生产部署首选。详见 `docs/06_模型压缩对比.md`。

---

## 4. 技术架构

前后端分离 B/S 架构：

```
┌─────────────┐     HTTP / JSON     ┌──────────────────┐
│  前端 Web    │ ─────────────────► │   后端 API        │
│ React+Vite  │ ◄───────────────── │   FastAPI         │
│ Ant Design  │                    │   ├─ 认证 / 用户   │
│ (响应式)     │                    │   ├─ 推理 /predict │
│             │                    │   ├─ 历史 /records │
│             │                    │   └─ 知识 /diseases │
└─────────────┘                    └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │  推理引擎         │
                                   │  ResNet18 (38类)  │
                                   │  INFER_BACKEND    │
                                   │  fp32/quant/prune │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │  数据层          │
                                   │  SQLite (WAL)    │
                                   │  用户/历史/知识   │
                                   └──────────────────┘
```

| 层 | 技术 |
|----|------|
| 前端 | React 18 + Vite 5 + Ant Design 5（响应式，PC / 移动端同套界面） |
| 后端 | FastAPI + Uvicorn + Pydantic，RESTful API（`/api/v1`） |
| 模型 | PyTorch + torchvision ResNet18（ImageNet 预训练 + 迁移学习微调） |
| 推理 | `src/model/predict.py`，`INFER_BACKEND` 切换三后端，温度缩放校准置信度 |
| 数据 | SQLite（`WAL` 模式，外键约束，并发保护），用户 / 历史 / 病害知识 / 配置 |

---

## 5. 环境要求

- **Python** 3.12+（训练 / 推理 / 后端；本机测试环境 PyTorch 2.5.1 + CUDA 12.x）
- **Node.js** 18+（前端构建）
- **PyTorch** 2.x（含 torchvision）

依赖见 `requirements.txt`：

```
fastapi
uvicorn[standard]
python-multipart
bcrypt==3.2.0
matplotlib        # 混淆矩阵图
pandas / scikit-learn  # 评估
pytest            # 自动化测试
```

前端依赖见 `src/web/package.json`（antd / axios / react-router-dom 等）。

---

## 6. 快速启动

### 6.1 后端（推理 + API）

```bash
# 1) 安装依赖（建议在虚拟环境）
pip install -r requirements.txt

# 2) 启动 FastAPI（项目根目录执行，确保根目录在 PYTHONPATH）
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3) 浏览器打开接口文档
#    http://127.0.0.1:8000/docs
```

> 推理后端通过 `src/config.py` 的 `INFER_BACKEND` 选择：`"fp32"`（默认）/ `"quantized"` / `"pruned"`。
> 生产环境请设置 `JWT_SECRET` 与 `DATA_ROOT` 环境变量（详见 `src/config.py` 注释）。

### 6.2 前端（Web 客户端）

```bash
cd src/web
npm install
npm run dev          # 开发服务器，默认 http://127.0.0.1:5173
# 或 npm run build && npm run preview  # 产物预览
```

> 前端 `src/web/src/api/http.js` 中 `USE_MOCK` 控制是否走本地 mock 数据（默认 `false`，联调真实后端）。
> 后端未就绪时置 `true` 可独立预览界面。

---

## 7. 目录结构

```
.
├── src/
│   ├── api/            # 后端：FastAPI 路由 / 认证 / 数据库
│   │   ├── main.py     # 应用入口（uvicorn src.api.main:app）
│   │   ├── auth.py     # JWT + bcrypt 用户体系
│   │   ├── db.py       # SQLite 连接（WAL）+ 数据访问
│   │   └── schemas.py  # Pydantic 模型
│   ├── model/          # 模型与推理
│   │   ├── resnet.py   # ResNet18 定义（38 分类）
│   │   ├── predict.py  # 推理引擎（三后端切换 + 预缩放 + 温度校准）
│   │   ├── train.py    # 训练（加权损失 + 早停）
│   │   ├── prune.py    # 结构化通道剪枝
│   │   ├── quantize.py # INT8 静态量化
│   │   ├── evaluate.py # 评估 + 混淆矩阵（字体三级回退）
│   │   └── compare.py  # 三后端对比评测
│   ├── data/           # 预处理 / 数据增强 / 分组切分
│   └── config.py       # 全局配置（路径 / 类别 / 推理后端 / 安全）
├── src/web/            # 前端：React + Vite + Ant Design
│   └── src/pages/      # 识别 / 历史 / 知识库 / 登录 / 个人中心
├── scripts/            # 命令行工具（predict_one / predict_smoke）
├── tests/              # pytest 自动化测试
├── models/             # 训练产出模型权重（best / quantized / pruned）
├── data/               # 运行时数据（上传图 / SQLite / 数据集切分）
├── outputs/            # 评估与对比报告产出
├── docs/               # 项目文档（PRD / UIUX / 数据库 / API / 测试 / 压缩对比）
├── requirements.txt    # Python 依赖
└── pytest.ini          # 测试配置
```

---

## 8. API 概览（`/api/v1`）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/auth/register` | POST | 注册（用户名/邮箱 + 密码） |
| `/auth/login` | POST | 登录，返回 JWT |
| `/auth/me` | GET | 当前用户信息（需鉴权） |
| `/predict` | POST | 上传叶片图，返回 Top-K 病害 + 置信度 + 防治建议（可选 `top_k`） |
| `/diseases` | GET | 38 类病害知识查询 |
| `/records` | GET / POST | 识别历史列表 / 保存 |
| `/records/{rid}` | GET / DELETE | 历史详情 / 删除 |
| `/records/export` | GET | 历史导出（CSV，响应后清理临时文件） |

完整字段与示例见 `docs/04_API接口文档.md`。

---

## 9. 测试与质量基线

```bash
# 自动化测试（模型缺失时相关用例自动 skip）
py -3.12 -m pytest -v

# 三后端推理冒烟（D-3，验证 fp32/quantized/pruned 端到端可用）
py -3.12 scripts/predict_smoke.py
```

质量基线（已修复并验证）：
- 密码 **bcrypt** 加盐哈希（兼容旧 sha256 账户，登录时自动升级）；
- JWT 密钥走 **环境变量**（缺失仅开发告警，不报错退出）；
- **CORS 白名单**，不再通配 `*`；
- 移除 `/static` 挂载，**数据目录（含数据库 / 上传图）不再可公开下载**；
- 错误返回 **真实 HTTP 状态码**（404/400/401/409/413/429/500）；
- SQLite **WAL** 模式 + 超时，缓解并发 `database is locked`。

---

## 10. 文档索引

| 文档 | 内容 |
|------|------|
| `docs/01_PRD.md` | 产品需求文档（背景 / 功能 / NFR / 验收标准） |
| `docs/02_UIUX设计.md` | UI/UX 设计（信息架构 / 页面流 / 视觉规范） |
| `docs/03_数据库设计.md` | 数据库设计（用户 / 历史 / 病害知识表） |
| `docs/04_API接口文档.md` | 后端 API 接口文档 |
| `docs/05_测试报告.md` | 系统测试报告（精度 / 数据泄漏专项 / 错误分析） |
| `docs/06_模型压缩对比.md` | 量化 / 剪枝对比与部署选型 |
| `docs/07_AI协作教程.md` | 本项目 AI 协作开发教程（实训验收用） |

---

## 11. 范围边界

**做**：Web 端 38 类识别、Top-3 可视化、防治知识、用户体系、历史、模型量化 / 剪枝部署。
**不做**：原生移动 App、现场诊断替代、训练集外作物识别、农药电商交易、像素级分割。

---

## 12. 许可证与免责声明

- 数据集：PlantVillage（公开科研数据集）。
- 本系统为 **辅助识别与科普工具**，识别结果仅供参考，**不能替代现场专业农技诊断**；低置信度结果请结合人工经验复核。

---

*本项目为实训交付，完整设计与测试证据见 `docs/` 目录。*
