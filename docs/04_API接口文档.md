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

**请求**：`multipart/form-data`，字段 `file` = 图像；可选查询参数 `top_k`（1–38，默认 3）控制返回 Top-K 候选数量。
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
    "leaf_prob": 0.9931,
    "knowledge": { "symptom": "...", "cause": "...", "prevention": "...", "treatment": "..." }
  }
}
```
> `need_review=true`：当 `confidence < config.threshold_low_conf`（默认 0.5）或预测为训练集外特征时，前端提示「建议人工复核」。

> `likely_not_leaf=true`：**非叶片输入提示**。模型是 38 类闭集 softmax，没有"以上都不是"选项，对插画 / 风景 / 噪声等非叶片图片同样会给出高置信度判断（实测把动漫插画判成「番茄叶霉病 96.87%」）。后端用最大 softmax 概率（MSP）作兜底判据：`leaf_prob < config.OOD_MSP_THRESHOLD`（默认 0.98）时置为 `true`，前端在结果区顶部显示红色警告，提示用户重新上传。
> 判据来源与阈值取舍见 `docs/08_OOD拒识评估.md`（AUROC 0.9425，误杀率≈5% 时非叶片拦截率≈83%）。

### 2.3 病害知识
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/diseases` | 列表（38 类，支持 `crop` 筛选） |
| GET | `/diseases/{id}` | 单类详情（症状/防治） |

### 2.4 识别历史（需登录）
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/records` | 当前用户历史列表（分页 `?page=&size=`） |
| GET | `/records/{id}` | 历史详情（图 + Top-3 对比） |
| DELETE | `/records/{id}` | 删除单条 |
| GET | `/records/export` | 导出 CSV/JSON（`?fmt=csv|json`） |

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
- 单 IP 识别接口限流（如 60 次/分钟），防止滥用。
- 文件上传校验：类型白名单 + 大小上限（≤10MB）+ 魔数校验（防伪装）。
- 全站 HTTPS（任务书 6.2）。
