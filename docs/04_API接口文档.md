# 基于 CNN 的农作物叶片病害识别系统 — API 接口文档

> 版本：V1.0（设计阶段 M1）｜ 框架：FastAPI（RESTful JSON）｜ Base URL：`/api/v1`
> 认证：Bearer JWT（除注册/登录/公开识别外，历史与个人信息需登录）

## 1. 通用约定
- 请求/响应体：`application/json`（文件上传用 `multipart/form-data`）。
- 时间：ISO8601 UTC。
- 成功：`{ "code": 0, "data": {...}, "msg": "ok" }`。
- 失败：`{ "code": <非零>, "msg": "<可读错误>" }`。
- 常用状态码：401 未授权 / 403 禁止 / 422 参数错误 / 413 文件过大 / 429 限流。

## 2. 接口列表

### 2.1 用户认证
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/auth/register` | 注册 |
| POST | `/auth/login` | 登录，返回 access_token |
| POST | `/auth/logout` | 登出（失效 token） |
| GET | `/auth/me` | 获取当前用户信息 |
| PUT | `/auth/me` | 修改昵称/密码 |

**注册请求**
```json
{ "username": "farmer01", "email": "a@b.com", "password": "******" }
```
**登录响应**
```json
{ "code": 0, "data": { "access_token": "<jwt>", "token_type": "bearer", "user": { "id": 1, "username": "farmer01", "nickname": "老农" } } }
```

### 2.2 病害识别（核心）
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 上传叶片图，返回识别结果 |

**请求**：`multipart/form-data`，字段 `file` = 图像。可选**查询参数**（注意是 query，不是 form 字段）：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `top_k` | int | 3 | 返回 Top-K 候选数量（1–38） |
| `crop` | str | 空 | **作物中文名**（如 `番茄`）。指定后只在候选类内归一化。非法值返回 **400**（不静默忽略） |

> **作物先验的收益与前提**：实测 120 张 PlantDoc test 真实照片，多候选作物上
> Top-1 从 **15.8% 提升到 43.6%**（+27.7pp）。前提是**作物由用户提供**——
> 模型自动识别作物只有 32.5% 准确（苹果 8%、樱桃/树莓 0%），
> 拿它做掩码会把正确答案直接排除（收益归零）。
>
> ⚠️ **不变量**：掩码只作用于**展示概率**（`confidence` / `top3[].prob`）。
> OOD 判据（`leaf_prob` / `msp`）始终基于**未掩码的全 38 类 softmax** ——
> 否则候选变少会让置信度虚高，非叶片图片只要指定一个作物就能绕过拒识。
> 该性质已写成单元测试 `test_predict_crop_not_affect_ood`。
>
> 可通过 `CROP_PRIOR_ENABLE=0` 全局关闭。
**响应**
```json
{
  "code": 0,
  "data": {
    "pred_class_id": 31,
    "crop_cn": "番茄",
    "disease_cn": "番茄晚疫病",
    "is_healthy": 0,
    "confidence": 0.93,
    "top3": [
      { "class_id": 31, "disease_cn": "番茄晚疫病", "prob": 0.93 },
      { "class_id": 29, "disease_cn": "番茄早疫病", "prob": 0.04 },
      { "class_id": 30, "disease_cn": "番茄健康", "prob": 0.02 }
    ],
    "need_review": false,
    "likely_not_leaf": false,
    "leaf_prob": 0.9997,
    "leaf_prob_source": "head",
    "mean_chroma": 37.7,
    "grayscale": false,
    "msp": 0.9772,
    "crop_applied": true,
    "crop_candidates": 10,
    "confidence_calibrated": 0.12,
    "reliability": "low",
    "calibration_enabled": true,
    "knowledge": { "symptom": "...", "cause": "...", "prevention": "...", "treatment": "..." }
  }
}
```
> `need_review=true`：当 `confidence < config.threshold_low_conf`（默认 0.5）或预测为训练集外特征时，前端提示「建议人工复核」。

> `likely_not_leaf=true`：**非叶片输入提示**。模型是 38 类闭集 softmax，没有「以上都不是」选项，
> 对插画 / 风景 / 噪声等非叶片图片同样会给出高置信度判断（实测把动漫插画判成「番茄叶霉病 96.87%」）。
>
> 判据按优先级分三级，**判定只由主判据决定**：
>
> | 优先级 | 判据 | 阈值 | 说明 |
> |---|---|---|---|
> | ① 主 | **叶片检测头**（`leaf_prob`） | 0.50 | 冻结模型的 512 维池化特征 → 逻辑回归，直接学「是不是叶片」。真实叶 vs 非叶 AUROC 0.77~0.99。可用环境变量 `OOD_HEAD_THRESHOLD` 覆盖阈值 |
> | ② 回退 | MSP（`msp`） | 0.98 | 仅在检测头缺失或后端为 quantized / pruned 时启用；跨域不可靠（真实田间 AUROC 仅 0.553） |
> | ③ 仅文案 | 平均色度（`mean_chroma`） | 8.0 | 标记灰阶图，用于选择提示语措辞 + 给「灰阶但放行」加温和提醒。**不参与判定** |
>
> 即：`likely_not_leaf = leaf_prob < 阈值`（③ 只影响文案）
>
> > ⚠️ **语义变更（2026-10-19）**：早期版本为 `(leaf_prob < 阈值) or (色度 < 8.0)`。
> > 因色度判据隐含「叶片照片必为彩色」，会把黑白拍照 / 标本扫描的真实叶片 100% 误杀
> > （而灰度增强后的检测头能判对 93%），故已降级为纯展示用途。
>
> 字段说明：
> - `leaf_prob` —— 「是叶片」的把握。判据来源为检测头时是分类器输出，回退时为 MSP
> - `leaf_prob_source` —— `"head"` / `"msp"`，标明实际使用的判据（便于诊断与前端文案分级）
> - `msp` —— 原始 softmax 置信度（诊断 / 日志用）
> - `mean_chroma` —— 平均色度（`mean(max(R,G,B)-min(R,G,B))`）
> - `grayscale` —— 是否为灰阶图。**仅影响提示文案**：判非叶片时用于措辞，判为叶片时触发黄色提醒
> - `crop_applied` —— 作物先验是否生效（传了合法 `crop` 且 `CROP_PRIOR_ENABLE=1` 时为 `true`）。
>   前端据此显示「已限定作物 N 个候选」标签 —— 不标出来用户无从知道概率是在几个候选里算的
> - `crop_candidates` —— 生效时的候选类数；未生效为 0。**为 1 时说明该作物在模型中只有一个类别，
>   约束等于直接给出结论**（如蓝莓 / 树莓 / 大豆 / 南瓜 / 柑橘），参考价值有限
> - `confidence_calibrated` —— **真实域校准后的置信度**（0–1）。用 PlantDoc 田间照片拟合的温度缩放
>   （`T_global≈6.05` / 指定作物时 `T_crop≈9.55`）重新缩放 logits 后的 Top-1 概率。
>   它比 `confidence` 更贴近「真实田间场景下的实际准确率」—— 例如同一张图
>   `confidence=0.93` 但 `confidence_calibrated=0.10`，说明模型在实验室域自报的把握在真实场景里水分很大。
>   仅用于展示，不改变 Top-1 排序，也不参与 OOD 判据
> - `reliability` —— 由 `confidence_calibrated` 映射的**可靠性分档**：`high`（≥0.55，实测判对率 89.7%）/
>   `medium`（0.25–0.55，42.0%）/`low`（<0.25，22.6%）。前端据此显示「校准：结果较可靠 / 仅供参考 / 可能不可靠」标签
> - `calibration_enabled` —— 校准是否生效。未生成 `models/calibration_domain.json` 或
>   `CALIBRATION_DOMAIN_ENABLE=0` 时为 `false`，此时前端退化为原逻辑（不显示校准区）
>
> ⚠️ **两个置信度的用途必须分清**（详见 `docs/14_置信度校准.md`）：
> `confidence` / `top3[].prob` 是 **PlantVillage 域**尺度（T=1.0），用于既有契约与 `msp` OOD 判据；
> `confidence_calibrated` 是 **真实田间域**尺度（T≈6~10），只用于对外展示「该信多少」。
> 同一温度无法同时适配两域，故分两条路径。校准**只改数值、不改 Top-1 排序**。
>
> 判据来源、阈值取舍与实测数据见 `docs/08_OOD拒识评估.md` 与
> `docs/11_域适应与OOD检测实验.md`（现行端到端实测：彩色真实叶片 0/11 误杀，
> 灰阶真实叶片 3/3 放行，漫画 / 截图 / 非叶片照片全部拦截）。

### 2.3 病害知识 / 作物清单
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/crops` | **作物清单**（14 种，含各作物候选类数），供前端「选择作物」下拉使用 |
| GET | `/diseases` | 列表（38 类，支持 `crop` 筛选） |
| GET | `/diseases/{id}` | 单类详情（症状/防治） |

### 2.4 识别历史（需登录）
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/records` | 当前用户历史列表（分页 `?page=&size=`） |
| GET | `/records/{id}` | 历史详情（图 + Top-3 对比） |
| DELETE | `/records/{id}` | 删除单条 |
| GET | `/records/export` | 导出 CSV/JSON（`?fmt=csv|json`） |
| GET | `/records/ood-stats` | **非叶片判据的近 N 天分布**（`?days=7`），用于发现线上输入漂移 |

> 路由声明顺序：`/records/ood-stats` 与 `/records/export` 必须排在 `/records/{id}` **之前**，
> 否则会被当成 `id` 解析（FastAPI 按声明顺序匹配）。

**历史列表响应**（分页）
```json
{
  "code": 0,
  "data": {
    "items": [
      { "id": 12, "image_name": "leaf_012.jpg", "pred_class_id": 31, "disease_cn": "番茄晚疫病", "confidence": 0.93, "created_at": "2026-10-08 09:00:00" }
    ],
    "total": 1, "page": 1, "size": 20
  },
  "msg": "ok"
}
```

**运行期监控响应**（`GET /records/ood-stats?days=7`）
```json
{
  "code": 0,
  "data": {
    "days": 7, "total": 22, "monitored": 2,
    "flagged": 1, "msp_fallback": 0,
    "avg_leaf_prob": 0.514, "avg_chroma": 55.2,
    "flagged_ratio": 0.5, "low_leaf_ratio": 0.5,
    "low_leaf_threshold": 0.2
  },
  "msg": "ok"
}
```

| 字段 | 含义 |
|---|---|
| `monitored` | 有判据快照的记录数。**比例类字段的分母就是它** —— 历史行（加监控列之前写入的）为 NULL，若用 `total` 作分母，比例会被稀释成"一切正常" |
| `flagged_ratio` | 判定为非叶片的占比。突增说明用户在传别的东西，或判据退化 |
| `low_leaf_ratio` | `leaf_prob < low_leaf_threshold` 的占比，比 `flagged` 更敏感，是**早期预警** |
| `msp_fallback` | 走 MSP 回退判据的条数。**>0 说明检测头没生效**（如切了 quantized / pruned 后端），此时真实场景误杀率会显著上升 |

每次 `/predict` 都会把 `leaf_prob` / `leaf_prob_source` / `mean_chroma` / `likely_not_leaf` 一并落库（仅登录用户）。

## 3. 前端调用示例（识别）
```http
POST /api/v1/predict
Authorization: Bearer <jwt>   # 公开识别可不带；保存历史需带
Content-Type: multipart/form-data

file=<二进制图像>
```

## 4. 契约约束（本期预留）
- 本期**不做管理端前端**，但上述接口即管理端可用契约（用户/历史/知识均可经同套 API 管理）。
- 模型版本经 `config.model_version` 暴露，便于灰度与回滚。
- 推理接口保持**无状态**，便于后续横向扩展或替换为量化模型而不改前端。

## 5. 限流与安全
- **接口限流（按真实客户端 IP）**：
  - 识别接口 `/predict`：60 次/分钟；
  - 认证接口 `/auth/login`、`/auth/register`：10 次/分钟（防口令爆破与批量注册）。
  超限统一返回 `429`。
- **限流 IP 取值的部署要求**：服务默认**不信任** `X-Forwarded-For` / `X-Real-IP`，
  因此启动命令须带 `--no-proxy-headers` —— uvicorn 默认会信任来自 `127.0.0.1` 的转发头，
  并在应用层之前覆写客户端 IP，导致限流可被一行请求头绕过。仅当确实部署在可信反向
  代理之后时，才改为 `--proxy-headers --forwarded-allow-ips="代理服务器IP"` 并设 `TRUST_PROXY=1`。
- **认证接口防枚举**：用户名不存在与口令错误返回完全相同的响应（`401` +
  「用户名或密码错误」），且两条分支执行等价的 bcrypt 计算以抹平响应时间差。
- 文件上传校验：类型白名单 + 大小上限（≤10MB）+ PIL 完整性校验（防伪装文件与解压炸弹）。
- 统一注入基础安全响应头：`X-Content-Type-Options=nosniff`、`X-Frame-Options=DENY`、
  `Referrer-Policy=no-referrer`、`Content-Security-Policy=default-src 'none'`。
- 全站 HTTPS（任务书 6.2）。
