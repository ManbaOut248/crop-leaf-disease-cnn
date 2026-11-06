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

# 详情弹窗用的预览图规格（2026-10-29 新增）：
#   96px 缩略图放大到弹窗尺寸必然糊成马赛克，但也没必要回传原图 ——
#   手机原图 3~5MB，base64 后膨胀 1.33 倍，一次开弹窗就吃十几 MB 内存。
#   720px 长边在 900px 宽的弹窗里 1:1 显示清晰，放大到 2 倍仍可辨认病斑。
PREVIEW_SIZE = 720
PREVIEW_QUALITY = 82

# 预览图缓存独立一份：与缩略图 key 不同尺寸，共用缓存会互相顶掉。
# 数量上限比缩略图小（详情是逐条打开的，不会一次要几十张），
# 但要能容下用户来回翻看十几条记录。
_PREVIEW_CACHE_MAX = 128


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


@lru_cache(maxsize=_PREVIEW_CACHE_MAX)
def _cached_preview(abs_path: str, mtime: float) -> str:
    """生成详情弹窗用的长边 720px 预览图 data URI。

    与缩略图用同一套缓存策略（键含 mtime，文件被覆盖即失效），
    但尺寸与质量不同 —— 详情要能看清病斑，缩略图只要能认出来是同一张图。
    """
    try:
        with Image.open(abs_path) as im:
            im.draft("RGB", (PREVIEW_SIZE * 2, PREVIEW_SIZE * 2))
            im = im.convert("RGB")
            im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=PREVIEW_QUALITY, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        warnings.warn(f"预览图生成失败 {abs_path}: {exc}")
        return ""


def preview_for(image_name):
    """按落盘文件名取详情用预览图 data URI。

    Args:
        image_name: 记录中的 `image_name` 字段（落盘文件名，如 `abc123.jpg`）

    Returns:
        data URI 字符串；文件不存在或生成失败时返回空字符串
        （文件被手工清理掉时前端显示占位图，不影响其余字段）
    """
    if not image_name:
        return ""
    # 与 thumb_for 同样只取文件名部分，挡住 `../../` 路径穿越
    path = Path(UPLOAD_DIR) / Path(image_name).name
    if not path.is_file():
        return ""
    return _cached_preview(str(path), path.stat().st_mtime)


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
