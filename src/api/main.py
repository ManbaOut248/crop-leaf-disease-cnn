# -*- coding: utf-8 -*-
"""
FastAPI 推理后端（对齐 docs/04_API接口文档.md）。
功能：用户认证 / 病害识别（核心）/ 病害知识 / 识别历史。
Base URL：/api/v1；统一响应：{code, data, msg}。
"""

import csv
import io
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from src.config import (
    PROJECT_ROOT, UPLOAD_DIR, ALLOWED_IMAGE_EXT, MAX_UPLOAD_MB,
    THRESHOLD_LOW_CONF, MODEL_VERSION, CORS_ORIGINS, CORS_ALLOW_CREDENTIALS,
    NUM_CLASSES,
)
from src.api import auth, db, schemas

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    """取真实客户端 IP：优先反向代理透传的 X-Forwarded-For / X-Real-IP，
    否则退回直连 IP。避免反向代理下 request.client.host 恒为 127.0.0.1
    导致限流 key 全部撞同一 IP 而失效或误伤。"""
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
def register(body: schemas.RegisterRequest):
    if db.get_user_by_username(body.username):
        return _err(1001, "用户名已存在", 409)
    salt, pwd_hash = auth.hash_password(body.password)
    uid = db.create_user(body.username, pwd_hash, salt, body.email, body.nickname)
    return _ok({"user_id": uid}, "注册成功")


@app.post("/api/v1/auth/login")
def login(body: schemas.LoginRequest):
    user = db.get_user_by_username(body.username)
    if not user or not auth.verify_password(body.password, user["salt"], user["password_hash"]):
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
    if body.new_password:
        if not body.old_password or not auth.verify_password(body.old_password, user["salt"], user["password_hash"]):
            return _err(1003, "旧密码错误", 400)
        salt, pwd_hash = auth.hash_password(body.new_password)
        db.update_user(user["id"], nickname=body.nickname, password_hash=pwd_hash, salt=salt)
    else:
        db.update_user(user["id"], nickname=body.nickname)
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
    user=Depends(get_optional_user),
    request: Request = None,
):
    ip = _client_ip(request) if request else "local"
    if not _rate_ok(ip):
        return _err(429, "请求过于频繁，请稍后再试", 429)

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
        results = predict_image(img, top_k=top_k)
    except FileNotFoundError:
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        return _err(503, "模型尚未就绪，请稍候（训练进行中或权重缺失）", 503)
    except Exception as e:  # noqa: BLE001
        (UPLOAD_DIR / save_name).unlink(missing_ok=True)
        return _err(500, f"识别失败：{e}", 500)

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
        "knowledge": {
            "symptom": top1.get("symptom", ""),
            "cause": top1.get("cause", ""),
            "prevention": top1.get("prevention", ""),
            "treatment": top1.get("treatment", ""),
        },
    }

    if user:
        top3_json = [{"class_id": r["class_id"], "prob": r["confidence"]} for r in results]
        db.add_record(user["id"], save_name, top1["class_id"], top1["confidence"], top3_json)

    return _ok(data)


# ---------- 2.3 病害知识 ----------
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
from src.model.predict import predict_image  # noqa: E402
