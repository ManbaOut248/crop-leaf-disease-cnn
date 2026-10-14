# -*- coding: utf-8 -*-
"""识别记录缩略图：带缓存的内联 base64 缩略图生成。

为什么这么做（设计取舍）：
    安全加固 A 方案已移除 `app.mount("/static", ...)`（避免整个 `data/` 目录
    ——含数据库、他人上传原图、5.4 万张训练图——被公开下载）。因此历史列表
    的缩略图不能再走静态 URL。

    同时 `<img src>` 无法携带 `Authorization` 头，若新增受鉴权的缩略图路由，
    前端就得改用 fetch + Blob 手动管理 objectURL 生命周期，复杂度明显上升。

    故采用「后端按需生成 base64 内联缩略图」：
      · 不新增任何公开静态路由，不放大攻击面；
      · 前端一行 `<img src={thumb}>` 即可，无需处理鉴权与 objectURL 回收；
      · 缩略图固定 64×64 / JPEG q60，单条约 1.5–2.5 KB，
        20 条列表额外体积约 40 KB，本地与局域网部署完全可接受。
"""
import base64
import warnings
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.config import UPLOAD_DIR

# 缩略图规格：96px 在 48px 展示位上仍有 2x 余量（高分屏不糊），
# 单条约 3–4 KB；64px 实测在叶片居中构图上会糊成一团，故不用。
THUMB_SIZE = 96
THUMB_QUALITY = 70


@lru_cache(maxsize=512)
def _cached_thumb(abs_path: str, mtime: float) -> str:
    """生成缩略图 data URI。

    缓存键包含文件 mtime —— 同名文件被覆盖时缓存自动失效，
    不会出现「图变了但缩略图还是旧的」。

    Args:
        abs_path: 图片绝对路径
        mtime: 文件修改时间（仅用于缓存失效，函数内不直接使用）

    Returns:
        `data:image/jpeg;base64,...` 字符串；失败时返回空字符串
    """
    try:
        with Image.open(abs_path) as im:
            # draft() 让 JPEG 解码器直接按比例降采样，大图可提速数倍，
            # 避免把 6000×4000 的原图完整解码进内存只为做 64×64 缩略图。
            im.draft("RGB", (THUMB_SIZE * 3, THUMB_SIZE * 3))
            im = im.convert("RGB")
            im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # 缩略图失败不应影响历史列表本身
        warnings.warn(f"缩略图生成失败 {abs_path}: {exc}")
        return ""


def thumb_for(image_name):
    """按上传文件名取缩略图 data URI。

    Args:
        image_name: 记录中的 `image_name` 字段（落盘文件名，如 `abc123.jpg`）

    Returns:
        data URI 字符串；文件不存在或生成失败时返回空字符串
    """
    if not image_name:
        return ""
    # 只取文件名部分，防止 `../../` 之类路径穿越读到 uploads 之外的文件
    safe_name = Path(image_name).name
    path = Path(UPLOAD_DIR) / safe_name
    if not path.is_file():
        return ""
    return _cached_thumb(str(path), path.stat().st_mtime)
