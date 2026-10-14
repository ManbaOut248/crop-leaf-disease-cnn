# -*- coding: utf-8 -*-
"""_prescale 纯函数单测：不加载模型，秒级完成。"""
from PIL import Image

from src.model.predict import _prescale


def _img(w, h):
    return Image.new("RGB", (w, h), (123, 200, 80))


def test_no_upscale_when_small():
    img = _img(100, 80)
    out = _prescale(img, 512)
    assert out.size == (100, 80), "小图不应被放大"


def test_boundary_unchanged():
    img = _img(512, 400)
    out = _prescale(img, 512)
    assert out.size == (512, 400), "长边恰等于上限时应原样返回"


def test_large_landscape_scaled_to_max_side():
    img = _img(1500, 1000)
    out = _prescale(img, 512)
    assert max(out.size) == 512, "长边应被缩放到上限 512"
    # 宽高比保持（允许 1px 取整误差）
    assert abs(out.size[0] / out.size[1] - 1.5) < 0.01


def test_large_square_scaled_to_max_side():
    img = _img(2000, 2000)
    out = _prescale(img, 512)
    assert out.size == (512, 512)


def test_min_size_guard():
    # 极端窄图也不应产生 0 尺寸
    img = _img(10, 3)
    out = _prescale(img, 512)
    assert min(out.size) >= 1
