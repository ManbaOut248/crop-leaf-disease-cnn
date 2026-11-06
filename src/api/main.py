# -*- coding: utf-8 -*-
"""
FastAPI 推理后端（对齐 docs/04_API接口文档.md）。
功能：用户认证 / 病害识别（核心）/ 病害知识 / 识别历史。
Base URL：/api/v1；统一响应：{code, data, msg}。
"""

import csv
import io
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.background import BackgroundTask

from src.config import (
    PROJECT_ROOT, UPLOAD_DIR, ALLOWED_IMAGE_EXT, MAX_UPLOAD_MB,
    THRESHOLD_LOW_CONF, MODEL_VERSION, CORS_ORIGINS, CORS_ALLOW_CREDENTIALS,
    NUM_CLASSES, CLASS_DIRS, class_name_zh,
    TRUST_PROXY, RATE_LIMIT_LOGIN_PER_MIN,
    REQUIRE_HTTPS, HSTS_MAX_AGE, CROP_PRIOR_ENABLE,
)
from src.api import auth, db, schemas, thumbs

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 解压炸弹防护（审计 P2）：显式下调 PIL 像素上限，超过即抛 DecompressionBombError。
# 默认阈值约 0.89 亿像素且**仅告警不拦截**（2 倍才报错），一张 10MB 的高度压缩
# PNG 可解压出数百 MB 位图 —— 是廉价的内存耗尽途径。50M 像素远大于实际所需。
from PIL import Image as _PILImage  # noqa: E402

_PILImage.MAX_IMAGE_PIXELS = 50_000_000


# ===================== 生命周期 / 中间件 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.set_config("model_version", MODEL_VERSION)
    db.set_config("threshold_low_conf", THRESHOLD_LOW_CONF)
    yield


app = FastAPI(title="农作物叶片病害识别系统 API", version="1.0", lifespan=lifespan)
# CORS：仅放行白名单来源（见 src/config.py 的 CORS_ORIGINS）。
# 本项目用 Bearer Token 鉴权（非 Cookie），故 allow_credentials=False，
# 避免「allow_origins=['*'] + allow_credentials=True」这一非法且危险的跨域组合。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """统一注入基础安全响应头（审计 P2），并在启用时强制 HTTPS（任务书 6.2）。

    本服务只返回 JSON / 文件流、不承载任何页面，故 CSP 收紧到最严
    （default-src 'none'）—— 即便将来某处误回显了用户内容，浏览器也不会执行脚本。

    HTTPS：
      · `REQUIRE_HTTPS=1` 时，HTTP 请求返回 308 跳转到 https（保留方法与请求体）；
      · 判定依据是 `request.url.scheme`，而 Nginx 终止 TLS 的场景下应用看到的是
        http —— 此时必须依赖 `X-Forwarded-Proto`。该头只有在 **TRUST_PROXY=1**
        时才可信（否则客户端可自行伪造），两个开关要成对使用。
      · 已经是 HTTPS 时下发 HSTS，让浏览器后续强制走 TLS。
    """
    if REQUIRE_HTTPS:
        proto = request.url.scheme
        if TRUST_PROXY:
            # 仅信任代理追加的第一个值
            fwd = request.headers.get("X-Forwarded-Proto", "")
            if fwd:
                proto = fwd.split(",")[0].strip().lower()
        if proto != "https":
            url = request.url.replace(scheme="https")
            return RedirectResponse(str(url), status_code=308)

    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    )
    if REQUIRE_HTTPS:
        proto = request.url.scheme
        if TRUST_PROXY and request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https":
            proto = "https"
        if proto == "https":
            resp.headers.setdefault(
                "Strict-Transport-Security", f"max-age={HSTS_MAX_AGE}; includeSubDomains"
            )
    return resp
# 安全说明：刻意不挂载任何静态目录。
# 旧实现曾 app.mount("/static", StaticFiles(directory=data))，会把 data/app.db（全部账户与
# 密码哈希）、data/uploads（他人上传原图）、data/split（5.4 万训练图）全部公开下载，属高危信息泄露。
# 现前端用本地 URL.createObjectURL 预览、历史仅展示图片名文本，无需服务端静态图；
# 若日后需回显原图，应走带鉴权的路由（Depends(get_current_user)）而非公开静态挂载。


# ===================== 限流（简易内存版） =====================
# 注意：进程内字典不跨 worker 共享；多进程部署需改用 Redis/共享存储。
# 单 uvicorn 进程演示场景足够。
_RATE = {}


def _client_ip(request: Request) -> str:
    """取用于限流的客户端 IP。

    安全修复（审计 P1）：仅当 TRUST_PROXY=1（确实部署在可信反向代理之后）时才采信
    转发头，否则一律使用 TCP 直连地址。此前无条件信任 X-Forwarded-For / X-Real-IP，
    攻击者每次请求换一个伪造 IP 即可让限流完全失效（实测连续 65 次请求一次未拦）。
    """
    if TRUST_PROXY:
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
        real = request.headers.get("X-Real-IP")
        if real:
            return real.strip()
    return request.client.host if request.client else "local"


def _rate_ok(ip: str, limit: int = 60, window: int = 60) -> bool:
    now = time.time()
    lst = _RATE.setdefault(ip, [])
    lst[:] = [t for t in lst if now - t < window]
    if len(lst) >= limit:
        return False
    lst.append(now)
    return True


# ===================== 作物先验 =====================
_CROP_NAMES_CACHE = None


def _crop_names():
    """作物中文名清单（模块级缓存）。

    不能在 import 时计算：那时 lifespan 里的 db.init_db() 尚未执行，
    disease_knowledge 表可能还不存在。故首次在请求内调用时再取。
    查询结果为空时**不缓存**，避免「首次调用时表未就绪」把空列表永久固化。
    """
    global _CROP_NAMES_CACHE
    if _CROP_NAMES_CACHE is None:
        names = [r["crop"] for r in db.get_crop_summary()]
        if names:
            _CROP_NAMES_CACHE = names
        return names
    return _CROP_NAMES_CACHE


# ===================== 统一响应 / 依赖 =====================
def _ok(data=None, msg: str = "ok"):
    return {"code": 0, "data": data, "msg": msg}


def _err(code: int, msg: str, status: int = 400):
    """统一错误响应：返回真实 HTTP 状态码（便于网关/监控按 4xx/5xx 区分），
    body 仍保留 {code, msg} 供前端读取具体业务码与文案。"""
    return JSONResponse(status_code=status, content={"code": code, "msg": msg})


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="未提供 token")
    payload = auth.decode_token(authorization.split(" ", 1)[1])
    if not payload:
        raise HTTPException(status_code=401, detail="token 无效或已过期")
    user = db.get_user_by_id(payload["uid"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def get_optional_user(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    payload = auth.decode_token(authorization.split(" ", 1)[1])
    if not payload:
        return None
    return db.get_user_by_id(payload["uid"])


def _is_healthy_zh(name_zh: str) -> int:
    return 1 if name_zh.endswith("（健康）") else 0


def _verdict_cn(verdict) -> str:
    """人工核对结论 → 中文。

    文案是「识别正确 / 识别错误」而不是「正确 / 错误」（2026-10-29 修正）：
    被评价的对象是**模型这一次的识别结果**，不是操作人自己。用户已经知道真实
    病害是什么，来这里只是给 AI 判卷 —— 写成「判断错误」会被读成「我判断错了」，
    与设计意图正好相反（用户原话：「不是我的答案相对识别的选错了」）。
    """
    return {"correct": "识别正确", "wrong": "识别错误"}.get((verdict or "").strip(), "未核对")


# 上传文件名的清洗与展示（2026-10-29）
# 背景：`image_name` 落库的是 uuid 落盘名（防重名 / 防路径穿越），历史页直接展示它
# 就变成 `9f3c...a1.jpg`，用户看不出那是哪张图。故把「磁盘名」与「展示名」分开：
#   · image_name（列）= uuid 落盘名，缩略图靠它定位文件
#   · orig_name （列）= 用户上传时的原始文件名，给人看
# 展示名要做清洗，原因不止美观：
#   · 浏览器上传会带 Windows 假路径 `C:\fakepath\x.jpg`，必须只取 basename；
#   · 换行 / 控制字符会把 CSV 导出顶成多行（CSV 注入的经典形式）；
#   · 超长名（部分手机导出的图长达数百字符）会把表格列撑爆。
_MAX_DISPLAY_NAME = 80


def _display_name(filename) -> str:
    """把上传文件名清洗成可直接展示 / 落库的「原始图片名」。"""
    name = Path(str(filename or "")).name.strip()
    name = "".join(ch for ch in name if ch.isprintable())
    if not name:
        return ""
    if len(name) > _MAX_DISPLAY_NAME:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 6:
            # 减 2 是给「…」和「.」留位，否则截断后仍会超出上限一位
            keep = _MAX_DISPLAY_NAME - len(ext) - 2
            name = f"{stem[:keep]}…{dot}{ext}"
        else:
            name = name[:_MAX_DISPLAY_NAME]
    return name


def _record_image_name(orig_name, rid, custom_name=None) -> str:
    """历史记录对外展示的名称。

    三级回落（自定义命名 > 上传原名 > 占位名）：
      1. `custom_name` —— 用户自己起的名字，优先级最高，否则改名没意义；
      2. `orig_name` —— 上传时的原始文件名；
      3. `历史图片 #id` —— 2026-10-29 之前的老记录两者都没有。直接回显 uuid
         会让用户误以为「系统把我的图片名改掉了」。

    为什么三级都放在后端算：导出 CSV / 详情弹窗 / 列表三处都要用同一个名字，
    放在前端算三遍必然有一处漏改（此前就出现了「列表是新名字、导出还是 uuid」）。
    """
    for candidate in (custom_name, orig_name):
        name = (candidate or "").strip()
        if name:
            return name
    return f"历史图片 #{rid}"


def _enrich_record(r, with_preview=False):
    """把一条原始记录补齐成前端可直接渲染的结构（就地修改并返回）。

    列表与详情共用：病名、展示名、缩略图三处口径必须一致，各写一份迟早漂移。
    `with_preview=True` 额外内联 720px 预览图 —— 只有详情弹窗需要，
    列表若也带上，100 条一页会多传十几 MB。
    """
    cid = r.get("pred_class_id")
    try:
        r["disease_cn"] = class_name_zh(CLASS_DIRS[cid])
    except Exception:
        r["disease_cn"] = ""
    # 展示名 / 磁盘名分离：file_name 是 uuid 落盘名（取图用，绝不展示），
    # image_name 覆盖成可读名（docs/04 契约里「图片名」就是它）。
    disk_name = r.get("image_name")
    r["file_name"] = disk_name
    r["image_name"] = _record_image_name(r.get("orig_name"), r.get("id"), r.get("custom_name"))
    # orig_name 单独回传：弹窗里要能说明「原始文件名是 xxx」，
    # 否则用户改完名就再也认不出这是哪张图。
    r["orig_name"] = (r.get("orig_name") or "").strip() or None
    r["thumb"] = thumbs.thumb_for(disk_name)
    # top3_json 是内部存储格式，与详情里的 top3 重复，对外一律不返回
    # （老记录里它还是 NULL，前端拿去做任何判断都只会得到 undefined）
    r.pop("top3_json", None)
    # 实际病害（2026-10-29）：只有用户判了「识别错误」并顺手选了正确答案才有值。
    # 与 top3 同口径，按 class_id 现算中文名，类别改名后历史记录自动跟随。
    gid = r.get("ground_truth_class_id")
    if gid is None or gid == "":
        r["ground_truth_cn"] = None
        r["ground_truth_class_id"] = None
    else:
        try:
            r["ground_truth_cn"] = class_name_zh(CLASS_DIRS[int(gid)])
            r["ground_truth_class_id"] = int(gid)
        except Exception:
            # 库里有越界脏值时降级为「未填写」，而不是让整页 500
            r["ground_truth_cn"] = None
            r["ground_truth_class_id"] = None
    if with_preview:
        r["preview"] = thumbs.preview_for(disk_name)
    # 判定结论的中文文案统一在这里补（2026-10-29）：
    # PATCH 响应此前不带 verdict_cn，只有详情接口单独补了一行，导致同一个字段
    # 在不同接口时有时无 —— 前端一旦依赖它做展示，就会出现「改完标注标签变空」。
    r["verdict_cn"] = _verdict_cn(r.get("verdict"))
    return r


def _record_top3(r):
    """把 top3_json 解析成带病名与健康标记的候选列表。

    落库时只存了 `{class_id, prob}`（2026-10 之前），病名是按 class_id 现算的 ——
    这样类别调整后老记录的病名会自动跟着变，不会出现历史里存着旧名的情况。

    老记录（233 条里有 155 条）的 top3_json 是 NULL：那是候选存库功能上线前
    写入的，无法反推，返回空列表由前端提示「旧记录未保存候选」，而不是假装有。
    """
    raw = r.get("top3_json")
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return []
    out = []
    for it in items or []:
        try:
            cid = int(it.get("class_id"))
            name = class_name_zh(CLASS_DIRS[cid])
        except Exception:
            continue
        out.append({
            "class_id": cid,
            "disease_cn": name,
            "is_healthy": _is_healthy_zh(name),
            "prob": it.get("prob"),
        })
    return out


# ===================== 路由 =====================
@app.get("/api/v1/health")
def health():
    return _ok({"status": "up", "model_version": MODEL_VERSION})


# ---------- 2.1 用户认证 ----------
@app.post("/api/v1/auth/register")
def register(body: schemas.RegisterRequest, request: Request = None):
    # 注册限流（审计 P1）：此前该接口完全无限流，可被批量注册或用于用户名枚举探测。
    ip = _client_ip(request) if request else "local"
    if not _rate_ok(f"register:{ip}", limit=RATE_LIMIT_LOGIN_PER_MIN):
        return _err(429, "注册请求过于频繁，请稍后再试", 429)
    if db.get_user_by_username(body.username):
        return _err(1001, "用户名已存在", 409)
    salt, pwd_hash = auth.hash_password(body.password)
    uid = db.create_user(body.username, pwd_hash, salt, body.email, body.nickname)
    return _ok({"user_id": uid}, "注册成功")


@app.post("/api/v1/auth/login")
def login(body: schemas.LoginRequest, request: Request = None):
    # 登录限流（审计 P1）：此前登录接口无限流，配合「最短 6 位口令」可被在线爆破。
    ip = _client_ip(request) if request else "local"
    if not _rate_ok(f"login:{ip}", limit=RATE_LIMIT_LOGIN_PER_MIN):
        return _err(429, "登录尝试过于频繁，请稍后再试", 429)
    user = db.get_user_by_username(body.username)
    if not user:
        # 用户不存在时也做一次等价 bcrypt 计算，抹平与「用户存在」分支的耗时差，
        # 防止通过响应时间枚举已注册用户名（审计 P2，实测差异可达 300ms+）。
        auth.dummy_verify(body.password)
        return _err(1002, "用户名或密码错误", 401)
    if not auth.verify_password(body.password, user["salt"], user["password_hash"]):
        return _err(1002, "用户名或密码错误", 401)
    # 旧 sha256 账户登录成功后，自动升级为 bcrypt（一次性迁移，提升存储安全性）
    if auth.is_legacy_hash(user["password_hash"]):
        new_salt, new_hash = auth.hash_password(body.password)
        db.update_user(user["id"], password_hash=new_hash, salt=new_salt)
    token = auth.create_token({"uid": user["id"], "username": user["username"]})
    return _ok({
        "access_token": token, "token_type": "bearer",
        "user": {"id": user["id"], "username": user["username"], "nickname": user["nickname"]},
    })


@app.get("/api/v1/auth/me")
def me(user=Depends(get_current_user)):
    return _ok({
        "id": user["id"], "username": user["username"],
        "nickname": user["nickname"], "email": user["email"],
        "created_at": user["created_at"],
    })


@app.put("/api/v1/auth/me")
def update_me(body: schemas.UpdateMeRequest, user=Depends(get_current_user)):
    # 昵称为 None 时保留原值（审计 P1 防御）：db.update_user 会无条件执行
    # `SET nickname=?`，若调用方未提供昵称就会把用户昵称写成 NULL。
    new_nickname = body.nickname if body.nickname is not None else user["nickname"]
    if body.new_password:
        if not body.old_password or not auth.verify_password(body.old_password, user["salt"], user["password_hash"]):
            return _err(1003, "旧密码错误", 400)
        salt, pwd_hash = auth.hash_password(body.new_password)
        db.update_user(user["id"], nickname=new_nickname, password_hash=pwd_hash, salt=salt)
    else:
        db.update_user(user["id"], nickname=new_nickname)
    return _ok({}, "资料已更新")


@app.post("/api/v1/auth/logout")
def logout():
    # JWT 无状态，服务端无需失效；客户端丢弃 token 即可
    return _ok({}, "已登出（客户端丢弃 token 即可）")


# ---------- 2.2 病害识别（核心，公开；登录后自动存历史） ----------
# 批量识别（2026-10-30）：一次最多 9 张，正好铺满前端 3×3 网格。
# 为什么限 9：单张 10MB 上限，9 张的 multipart 请求体可达 90MB；
# 且 CPU 推理约 0.2~0.3s/张，9 张约 2~3 秒 —— 再往上用户就会以为是卡死了。
BATCH_MAX_FILES = 9
# 批量限流按「请求次数」计，所以必须比单张更严：单张是 60 次/分钟（≈60 张），
# 批量若沿用 60，就成了 540 张/分钟，等于给单张限流开了个九倍的后门。
# 6 次 × 9 张 = 54 张/分钟，略低于单张口径。
RATE_LIMIT_BATCH_PER_MIN = 6


class _AnalyzeError(Exception):
    """识别流程的内部错误，携带对外业务码与 HTTP 状态码。

    为什么需要它：单张接口与批量接口要共用同一段「校验 → 落盘 → 推理」代码，
    但两者对错误的处理方式相反 —— 单张要立刻把错误返回给调用方，批量则要把错误
    记成「第 N 张失败」，不能让一张坏图毁掉另外 8 张。用异常传递，
    公共函数就不必知道调用方是谁，也就不必返回 (data, err) 这种双返回值。
    """

    def __init__(self, code, msg, status=400):
        super().__init__(msg)
        self.code = code
        self.msg = msg
        self.status = status


def _build_predict_data(results, ood):
    """把推理结果组装成对外响应体（单张与批量共用）。

    抽出来的原因：两个接口各拼一份的话，字段一旦增减就会出现
    「单张有 reliability、批量没有」这类静默不一致 —— 前端是同一个组件渲染两种数据，
    少一个字段就表现为某处渲染成空白或 undefined，而且不会有任何报错。

    `confidence` 保持 PV 域尺度的原始值（契约不变），真实场景校准值另给三个字段，
    详见 docs/04 与 docs/14。
    """
    top1 = results[0]
    return {
        "pred_class_id": top1["class_id"],
        "crop_cn": top1["crop"],
        "disease_cn": top1["name_zh"],
        "is_healthy": _is_healthy_zh(top1["name_zh"]),
        "confidence": top1["confidence"],
        "top3": [{"class_id": r["class_id"], "disease_cn": r["name_zh"], "prob": r["confidence"]}
                 for r in results],
        "need_review": top1["confidence"] < THRESHOLD_LOW_CONF,
        # 非叶片输入提示：38 类闭集 softmax 没有「以上都不是」选项，对非叶片图片
        # （插画 / 风景 / 噪声）也会给出高置信度判断。判据优先级：
        #   叶片检测头（主，512 维特征二分类）→ MSP（回退）→ 平均色度（始终叠加，抓灰阶图）。
        # 详见 predict.py 的 predict_image_with_ood 说明。
        "likely_not_leaf": ood["likely_not_leaf"],
        "leaf_prob": ood["leaf_prob"],
        "leaf_prob_source": ood["leaf_prob_source"],
        "mean_chroma": ood["mean_chroma"],
        "grayscale": ood["grayscale"],
        "msp": ood["msp"],
        # 作物先验：crop_applied=True 表示概率已在指定作物的候选类内归一化
        # （此时 crop_candidates 为候选类数）。前端据此说明「已按 X 作物约束」。
        "crop_applied": ood["crop_applied"],
        "crop_candidates": ood["crop_candidates"],
        # 真实域置信度校准：confidence 仍是 PV 域尺度的原始值（契约不变），
        # 下面三个字段是新增维度，告诉前端「这个数字在真实田间照片上到底该信多少」。
        #   confidence_calibrated —— 用 PlantDoc 拟合温度缩放后的置信度（更贴近真实准确率）
        #   reliability          —— 分档 high/medium/low（阈值见 docs/14）
        #   calibration_enabled  —— 校准是否生效（未生成校准文件时为 false，前端退化为原逻辑）
        "confidence_calibrated": ood["confidence_calibrated"],
        "reliability": ood["reliability"],
        "calibration_enabled": ood["calibration_enabled"],
        "knowledge": {
            # 「一病多名」别名（如 苹果雪松锈病 = 苹果桧锈病），供前端提示，避免用户误判为识别错误
            "alias": top1.get("alias", ""),
            "symptom": top1.get("symptom", ""),
            "cause": top1.get("cause", ""),
            "prevention": top1.get("prevention", ""),
            "treatment": top1.get("treatment", ""),
        },
    }


def _analyze_one(raw, filename, crop, top_k):
    """校验 + 落盘 + 推理一张图，返回 (data, ood, save_name)。

    调用方负责：限流、crop 合法性校验、按登录态落库 —— 这三件事单张与批量的
    处理方式不同（批量把所有图共用一次 crop 校验、逐张落库），所以不放在这里。

    校验顺序刻意「先校验再落盘」：失败时不留孤儿文件。旧实现在图像校验前就写了盘，
    非法文件会在 data/uploads 里堆出永不清理的垃圾。
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise _AnalyzeError(400, f"不支持的文件类型（仅允许 {sorted(ALLOWED_IMAGE_EXT)}）")

    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise _AnalyzeError(413, f"文件超过 {MAX_UPLOAD_MB}MB 上限", 413)

    # 先校验图像合法性（内存解码），确认无误再落盘，避免非法文件产生孤儿文件（#8）
    try:
        from PIL import Image
        probe = Image.open(io.BytesIO(raw))
        probe.verify()  # 仅校验文件完整性，不解码像素
        img = Image.open(io.BytesIO(raw))  # verify 后需重新打开方可读取像素
    except Exception:
        raise _AnalyzeError(400, "文件不是合法的图片或已损坏")

    save_name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / save_name).write_bytes(raw)
    try:
        results, ood = predict_image_with_ood(img, top_k=top_k, crop=crop)
    except FileNotFoundError:
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        raise _AnalyzeError(503, "模型尚未就绪，请稍候（训练进行中或权重缺失）", 503)
    except Exception as e:  # noqa: BLE001
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        # 不回传内部异常细节（可能含服务器路径 / 库版本等实现信息），仅记入服务端日志。
        logging.getLogger("uvicorn.error").exception("识别失败: %s", e)
        raise _AnalyzeError(500, "识别失败，请稍后重试", 500)

    return _build_predict_data(results, ood), ood, save_name


def _top3_json(data):
    """从响应体反推落库用的候选数组（只存 class_id 与概率，病名按 id 现算）。"""
    return [{"class_id": t["class_id"], "prob": t["prob"]} for t in data["top3"]]


@app.post("/api/v1/predict")
def predict(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=NUM_CLASSES, description="返回 Top-K 候选数量，默认 3"),
    crop: str = Query("", description="作物中文名（可选）。指定后只在候选类内归一化，可显著提升准确率"),
    user=Depends(get_optional_user),
    request: Request = None,
):
    ip = _client_ip(request) if request else "local"
    if not _rate_ok(ip):
        return _err(429, "请求过于频繁，请稍后再试", 429)

    # 作物名合法性：非法值不静默忽略，而是明确报错 —— 静默忽略会让用户以为
    # 约束生效了、实际仍走全 38 类，属于难以察觉的错误。
    crop = (crop or "").strip()
    if crop:
        crop_names = _crop_names()
        if crop not in crop_names:
            return _err(400, f"未知作物「{crop}」，可选：{', '.join(crop_names)}")

    raw = file.file.read()
    try:
        data, ood, save_name = _analyze_one(raw, file.filename, crop, top_k)
    except _AnalyzeError as e:
        return _err(e.code, e.msg, e.status)

    if user:
        # 连同 OOD 判据一起落库：否则「线上输入是否漂移」无从观测，
        # 改完模型也只能靠离线实验集自说自话（2026-10-19 加入运行期监控）。
        # orig_name 单独落一列（2026-10-29）：save_name 是 uuid，历史页显示它
        # 对用户毫无意义，真实文件名必须留一份可读的。
        rid = db.add_record(user["id"], save_name, data["pred_class_id"], data["confidence"],
                            _top3_json(data), ood=ood, orig_name=_display_name(file.filename))
        # 回传记录 id（2026-10-30）：批量接口每项都带 record_id，单张不带的话
        # 前端「识别完点去历史看这条」就得靠猜（按时间倒序找最新一条）。
        data["record_id"] = rid

    return _ok(data)


@app.post("/api/v1/predict/batch")
def predict_batch(
    files: List[UploadFile] = File(..., description=f"待识别图片，1~{BATCH_MAX_FILES} 张"),
    top_k: int = Query(3, ge=1, le=NUM_CLASSES, description="每张返回 Top-K 候选数量"),
    crop: str = Query("", description="作物中文名。批量识别要求必填 —— 整批共用同一个作物先验"),
    user=Depends(get_optional_user),
    request: Request = None,
):
    """批量识别（2026-10-30 新增）：一次上传多张**同一种作物**的叶片，逐张独立识别。

    为什么强制同一个作物：
      作物先验的收益来自「把 softmax 限制在该作物的候选类内」。若这一批里混进别的
      作物（比如选了番茄却传了一张马铃薯），它会被强行归到番茄的候选类里，并且
      置信度通常还不低 —— 这比不选作物更糟，因为用户会以为「选了作物所以更准」。
      前端因此必须显式提示「整批需为同一作物」，后端这里再要求 crop 必填兜底。

    为什么单张失败不整体失败：
      9 张里有一张是截图 / 损坏文件很常见。如果整体 400，用户得自己找出是哪张删掉
      才能继续；改为「能识别的先给结果，失败的逐条报文件名」，一次就能交付 8 张。
    """
    ip = _client_ip(request) if request else "local"

    # 先校验参数、后计配额（2026-10-30）。
    # 顺序很关键：限流放前面的话，用户连续传错 6 次「10 张超限」或「忘了选作物」
    # 就会被锁 1 分钟 —— 而这类错误只花了服务端几百微秒做字符串比较，没有任何
    # 推理成本。真正贵的（图像解码 + ResNet 推理）才该计入配额。
    if not files:
        return _err(400, "请至少选择一张图片")

    if len(files) > BATCH_MAX_FILES:
        # 明确报错而不是静默截断：静默丢掉后面几张，用户会以为「上传失败」，
        # 反复重试也找不到原因（历史 bug：/records 的 size 越界曾被静默重置为 20）。
        return _err(400, f"一次最多识别 {BATCH_MAX_FILES} 张，当前选了 {len(files)} 张")

    crop = (crop or "").strip()
    if not crop:
        return _err(400, "批量识别必须先选择作物：整批图片共用同一个作物")
    crop_names = _crop_names()
    if crop not in crop_names:
        return _err(400, f"未知作物「{crop}」，可选：{', '.join(crop_names)}")

    if not _rate_ok(f"batch:{ip}", limit=RATE_LIMIT_BATCH_PER_MIN):
        return _err(429, f"批量识别过于频繁，请稍等一分钟后再试（每分钟最多 {RATE_LIMIT_BATCH_PER_MIN} 批）", 429)

    items = []
    errors = []
    for idx, f in enumerate(files):
        shown = _display_name(f.filename) or f"第 {idx + 1} 张"
        try:
            raw = f.file.read()
            data, ood, save_name = _analyze_one(raw, f.filename, crop, top_k)
        except _AnalyzeError as e:
            # 逐条记账，不中断：第 3 张坏了，第 4~9 张照样要识别
            errors.append({"index": idx, "file_name": shown, "msg": e.msg})
            continue

        record_id = None
        if user:
            record_id = db.add_record(
                user["id"], save_name, data["pred_class_id"], data["confidence"],
                _top3_json(data), ood=ood, orig_name=shown,
            )
        items.append({
            # index 是「前端上传顺序」，不是成功序号 —— 前端靠它把结果贴回对应缩略图，
            # 用成功序号的话，中间失败一张后面的图会全部错位。
            "index": idx,
            "file_name": shown,
            "record_id": record_id,
            **data,
        })

    healthy = sum(1 for it in items if it["is_healthy"])
    return _ok({
        "items": items,
        "errors": errors,
        "summary": {
            "total": len(files),
            "ok": len(items),
            "failed": len(errors),
            "healthy": healthy,
            "disease": len(items) - healthy,
            # 下面两个与单张接口的判定口径完全一致，前端不用二次推导
            "likely_not_leaf": sum(1 for it in items if it["likely_not_leaf"]),
            "need_review": sum(1 for it in items if it["need_review"]),
        },
        "crop": crop,
        "top_k": top_k,
    })


# ---------- 2.3 病害知识 ----------
@app.get("/api/v1/crops")
def crops():
    """作物清单（含各作物的候选类别数），供前端「选择作物」下拉使用。

    为什么需要它：指定作物后 /predict 只在候选类内归一化，实测多类别作物
    Top-1 从 15.8% 提升到 43.6%。候选数与类别体系必须以后端为唯一来源，
    否则前端硬编码会在类别调整后静默失真。
    """
    rows = db.get_crop_summary()
    return _ok({
        "items": rows,
        # 说明：n_classes == 1 的作物（蓝莓 / 树莓 / 大豆 / 南瓜）只有一个候选类，
        # 指定作物等于直接给出结论，前端应据此提示，避免用户误以为「约束后很准」。
        "single_candidate": [r["crop"] for r in rows if r["n_classes"] == 1],
        "enabled": CROP_PRIOR_ENABLE,
    })


@app.get("/api/v1/diseases")
def diseases(crop: str = Query(None, description="按作物中文名筛选，如 番茄")):
    """病害知识库（38 类）。

    额外回传 `class_id`（2026-10-29）：知识库主键由 `_import_knowledge` 按
    `CLASS_DIRS` 的顺序写入，天然就是模型类别 id。前端「实际病害」下拉需要
    的是**类别 id**（落库字段就是它），直接透传比让前端猜 `id == class_id`
    要诚实 —— 一旦将来插入新类导致顺序变化，至少调用方拿到的是显式字段。
    """
    rows = db.get_disease_list(crop)
    for r in rows:
        r["class_id"] = r.get("id")
    return _ok(rows)


@app.get("/api/v1/diseases/{did}")
def disease_detail(did: int):
    row = db.get_disease_by_id(did)
    if not row:
        return _err(404, "未找到该病害", 404)
    return _ok(row)


# ---------- 2.4 识别历史（需登录） ----------
@app.get("/api/v1/records")
def records(
    user=Depends(get_current_user),
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    size: int = Query(20, ge=1, le=100, description="每页条数，1~100"),
    verdict: str = Query(None, pattern="^(all|correct|wrong|unset)$",
                         description="按人工核对结论筛选：all/correct/wrong/unset"),
):
    """分页取识别记录，可只取「识别错误 / 识别正确 / 未核对」的子集。

    `verdict` 过滤与 `counts`（2026-10-29 新增）配合使用：
    用户判错的记录是唯一能拿去补数据的那批，必须能一键筛出来；
    `counts` 是**全量历史**口径（不带时间窗口），直接做筛选器的数字徽标 ——
    若改用 /stats/overview 的 30 天数字，会出现「徽标 3 条、筛出 5 条」。

    历史 bug（2026-10-29 修复）：早期实现是
        `if size < 1 or size > 100: size = 20`
    即把「超过 100」**重置为默认 20** 而不是截断到 100。前端若传 200 想一次
    拿全，只会静默拿到 20 条，页面上看起来就是「历史只有 20 条、其余丢了」，
    且没有任何错误可循。现在用 Query(ge=1, le=100)：越界直接 422 报错，
    让契约错误暴露在调用方，而不是静默降级。
    """
    rows, total = db.get_records_by_user(user["id"], page, size, verdict=verdict)
    for r in rows:
        _enrich_record(r)
    return _ok({
        "items": rows,
        "total": total,
        "page": page,
        "size": size,
        "verdict": verdict or "all",
        # 三个筛选档位的条数（全量历史）。放同一个响应里而不是另开接口：
        # 两处查询在时间上不一致时，徽标会偶尔比列表多一条，用户会当成 bug。
        "counts": db.count_records_by_verdict(user["id"]),
    })


@app.get("/api/v1/records/export")
def record_export(user=Depends(get_current_user), fmt: str = Query("json")):
    """导出全部识别记录（不分页）。

    2026-10-29 修复两件事：
      1. 前端此前是拿列表页已有的 20 条自己拼 CSV —— 用户以为导出了全部历史，
         实际只有当前页。现在前端改为直接下载本接口，口径只有一处。
      2. CSV 原先是 `id,pred_class_id,confidence,created_at`：pred_class_id 是
         0~37 的数字，农户看不懂，也没图片名与中文病名。改为中文表头 + 百分比。
    """
    rows, _ = db.get_records_by_user(user["id"], 1, 1000000)
    # 补中文病名与展示名：与 /records 列表口径一致，避免导出后用户还要自己对照编号
    for r in rows:
        try:
            r["disease_cn"] = class_name_zh(CLASS_DIRS[r["pred_class_id"]])
        except Exception:
            r["disease_cn"] = ""
        # 导出同样只给可读名：写 uuid 落盘名等于给用户一串无意义字符
        r["image_name"] = _record_image_name(
            r.get("orig_name"), r.get("id"), r.get("custom_name"))
        # 实际病害列：与 /records 同口径现算，避免导出里只有「识别错误」
        # 却看不出正确答案是什么
        gid = r.get("ground_truth_class_id")
        try:
            r["ground_truth_cn"] = class_name_zh(CLASS_DIRS[int(gid)]) if gid not in (None, "") else ""
        except Exception:
            r["ground_truth_cn"] = ""
    if fmt == "csv":
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        try:
            # utf-8-sig：带 BOM，Excel 双击打开中文才不乱码
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                # 「人工判断」列（2026-10-29 新增）：导出的目的是拿去汇报/归档，
                # 只有预测结果没有人工核对结论，等于把模型自说自话当结论。
                # 「实际病害」列（同日）：判错时用户已给出正确答案，导出缺了就白标。
                w.writerow(["编号", "图片名", "识别结果", "置信度", "人工判断", "实际病害", "识别时间"])
                for r in rows:
                    conf = r.get("confidence")
                    w.writerow([
                        r["id"],
                        r.get("image_name") or "",
                        r.get("disease_cn") or "",
                        f"{conf * 100:.2f}%" if conf is not None else "",
                        _verdict_cn(r.get("verdict")),
                        r.get("ground_truth_cn") or "",
                        r.get("created_at") or "",
                    ])
        except Exception:
            os.close(fd)
            raise
        return FileResponse(
            path, filename="识别历史.csv", media_type="text/csv",
            # 响应发送完成后删除临时文件，避免每次导出在服务器留一个泄漏文件（#C）
            background=BackgroundTask(os.unlink, path),
        )
    return _ok(rows)


@app.get("/api/v1/records/ood-stats")
def record_ood_stats(days: int = Query(7, ge=1, le=365), user=Depends(get_current_user)):
    """近 N 天「非叶片判据」的分布，用于发现线上输入漂移。

    注意路由声明顺序：必须排在 `/records/{rid}` **之前**，
    否则 `ood-stats` 会被当成 rid 去解析（FastAPI 按声明顺序匹配）。
    """
    return _ok(db.get_ood_stats(user["id"], days=days))


@app.get("/api/v1/stats/overview")
def stats_overview(days: int = Query(30, ge=1, le=365), user=Depends(get_current_user)):
    """识别历史聚合统计，一次返回前端全部图表所需数据。

    为什么不让前端自己算：`/records` 分页上限 100 条，前端拿不到全量，
    reduce 出来的比例分母是错的；而且 4 张图要发 4 个请求，口径还可能不一致。

    ⚠️ `confidence` 是**原始 softmax**，不是温度校准值 —— 因此这里不返回
    high/medium/low 可靠性分档（那是校准值口径，见 calibration_domain.json）。
    """
    return _ok(
        db.get_stats_overview(user["id"], days=days, low_conf=THRESHOLD_LOW_CONF)
    )


@app.get("/api/v1/records/{rid}")
def record_detail(rid: int, user=Depends(get_current_user)):
    """单条记录的完整详情（详情弹窗用）。

    与列表的差别只有三处，其余口径复用 `_enrich_record`：
      · `preview` —— 720px 预览图（列表只有 96px 缩略图，放大就糊）；
      · `top3`    —— 解析后的候选列表（含病名/健康标记），列表不需要；
      · `verdict` —— 人工核对结论，弹窗里可改。
    """
    row = db.get_record_by_id(rid, user["id"])
    if not row:
        return _err(404, "未找到该记录", 404)
    # 先解析 top3 再 enrich：enrich 会把 top3_json 这个内部字段剔除
    row["top3"] = _record_top3(row)
    _enrich_record(row, with_preview=True)
    # verdict_cn 已由 _enrich_record 统一补齐，此处不再单独赋值
    return _ok(row)


@app.patch("/api/v1/records/{rid}")
def record_update(rid: int, req: schemas.UpdateRecordRequest,
                  user=Depends(get_current_user)):
    """修改记录的自定义命名 / 人工核对结论。

    为什么用 PATCH 而不是 PUT：这里只改两个字段，其余字段（预测结果、置信度）
    是识别时的客观事实，不允许事后篡改 —— PUT 的语义是整条替换，会诱使调用方
    以为可以连预测结果一起改。

    字段三态：JSON 里不出现 = 不改；显式传 null = 清空（取消命名 / 取消核对）。
    用 `model_fields_set` 判断是否出现，而不是 `is not None` —— 后者分不清
    「没传」和「传了 null」。
    """
    sent = req.model_fields_set
    try:
        row = db.update_record(
            rid, user["id"],
            custom_name=req.custom_name if "custom_name" in sent else db._UNSET,
            verdict=req.verdict if "verdict" in sent else db._UNSET,
            # 实际病害：不传 = 交给 db 层按 verdict 自动推断（判错保留、判对/清空连带清空）
            ground_truth_class_id=(
                req.ground_truth_class_id if "ground_truth_class_id" in sent else db._UNSET),
        )
    except ValueError as e:
        return _err(400, str(e))
    if not row:
        return _err(404, "未找到该记录", 404)
    _enrich_record(row)
    return _ok(row, "已保存")


@app.delete("/api/v1/records/{rid}")
def record_delete(rid: int, user=Depends(get_current_user)):
    """删除一条记录，并清理磁盘上的原图。

    原图不删会一直占盘（上传目录只增不减是审计里挂了很久的问题）。
    删除文件失败**不回滚**数据库删除：记录已经不可见了，为一张删不掉的文件
    把记录留在列表里反而更让人困惑 —— 记日志由运维侧清理即可。
    """
    ok, image_name = db.delete_record(rid, user["id"])
    if not ok:
        return _err(404, "未找到该记录", 404)
    if image_name:
        path = UPLOAD_DIR / Path(image_name).name
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logging.getLogger("uvicorn.error").warning("删除原图失败 %s: %s", path, e)
    return _ok({}, "已删除")


# 延迟导入，避免应用启动时即触发模型加载
from src.model.predict import predict_image_with_ood  # noqa: E402
