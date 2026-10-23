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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return _err(400, f"不支持的文件类型（仅允许 {sorted(ALLOWED_IMAGE_EXT)}）")

    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return _err(413, f"文件超过 {MAX_UPLOAD_MB}MB 上限", 413)

    # 先校验图像合法性（内存解码），确认无误再落盘，避免非法文件产生孤儿文件（#8）
    try:
        from PIL import Image
        probe = Image.open(io.BytesIO(raw))
        probe.verify()  # 仅校验文件完整性，不解码像素
        img = Image.open(io.BytesIO(raw))  # verify 后需重新打开方可读取像素
    except Exception:
        return _err(400, "文件不是合法的图片或已损坏")

    save_name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / save_name).write_bytes(raw)
    try:
        results, ood = predict_image_with_ood(img, top_k=top_k, crop=crop)
    except FileNotFoundError:
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        return _err(503, "模型尚未就绪，请稍候（训练进行中或权重缺失）", 503)
    except Exception as e:  # noqa: BLE001
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        # 不回传内部异常细节（可能含服务器路径 / 库版本等实现信息），仅记入服务端日志。
        logging.getLogger("uvicorn.error").exception("识别失败: %s", e)
        return _err(500, "识别失败，请稍后重试", 500)

    top1 = results[0]
    is_healthy = _is_healthy_zh(top1["name_zh"])
    data = {
        "pred_class_id": top1["class_id"],
        "crop_cn": top1["crop"],
        "disease_cn": top1["name_zh"],
        "is_healthy": is_healthy,
        "confidence": top1["confidence"],
        "top3": [{"class_id": r["class_id"], "disease_cn": r["name_zh"], "prob": r["confidence"]} for r in results],
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

    if user:
        top3_json = [{"class_id": r["class_id"], "prob": r["confidence"]} for r in results]
        # 连同 OOD 判据一起落库：否则「线上输入是否漂移」无从观测，
        # 改完模型也只能靠离线实验集自说自话（2026-10-19 加入运行期监控）。
        db.add_record(user["id"], save_name, top1["class_id"], top1["confidence"],
                      top3_json, ood=ood)

    return _ok(data)


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
    rows = db.get_disease_list(crop)
    return _ok(rows)


@app.get("/api/v1/diseases/{did}")
def disease_detail(did: int):
    row = db.get_disease_by_id(did)
    if not row:
        return _err(404, "未找到该病害", 404)
    return _ok(row)


# ---------- 2.4 识别历史（需登录） ----------
@app.get("/api/v1/records")
def records(user=Depends(get_current_user), page: int = 1, size: int = 20):
    if page < 1:
        page = 1
    if size < 1 or size > 100:
        size = 20
    rows, total = db.get_records_by_user(user["id"], page, size)
    # 每条记录补中文病名（对齐 docs/04「历史列表项含 disease_cn」契约）：
    # recognition_records 只存 pred_class_id，前端需可读病名，故在此按类别映射补齐。
    for r in rows:
        cid = r.get("pred_class_id")
        try:
            r["disease_cn"] = class_name_zh(CLASS_DIRS[cid])
        except Exception:
            r["disease_cn"] = ""
        # 缩略图以 base64 内联返回：既不再公开 data/ 静态目录，
        # 也免去前端为 <img> 无法带 Authorization 头而做的 blob 处理。
        # 详见 src/api/thumbs.py 顶部说明。
        r["thumb"] = thumbs.thumb_for(r.get("image_name"))
    return _ok({"items": rows, "total": total, "page": page, "size": size})


@app.get("/api/v1/records/export")
def record_export(user=Depends(get_current_user), fmt: str = Query("json")):
    rows, _ = db.get_records_by_user(user["id"], 1, 100000)
    if fmt == "csv":
        fd, path = tempfile.mkstemp(suffix=".csv", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "pred_class_id", "confidence", "created_at"])
                for r in rows:
                    w.writerow([r["id"], r["pred_class_id"], r["confidence"], r["created_at"]])
        except Exception:
            os.close(fd)
            raise
        return FileResponse(
            path, filename="records.csv", media_type="text/csv",
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


@app.get("/api/v1/records/{rid}")
def record_detail(rid: int, user=Depends(get_current_user)):
    row = db.get_record_by_id(rid, user["id"])
    if not row:
        return _err(404, "未找到该记录", 404)
    return _ok(row)


@app.delete("/api/v1/records/{rid}")
def record_delete(rid: int, user=Depends(get_current_user)):
    if not db.delete_record(rid, user["id"]):
        return _err(404, "未找到该记录", 404)
    return _ok({}, "已删除")


# 延迟导入，避免应用启动时即触发模型加载
from src.model.predict import predict_image_with_ood  # noqa: E402
