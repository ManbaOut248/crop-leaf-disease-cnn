# -*- coding: utf-8 -*-
"""evaluate.py 中文字体注册健壮性单测：不依赖具体平台，确认不抛异常。"""
import warnings

from src.model.evaluate import setup_cjk_font


def test_setup_cjk_font_never_raises():
    # 无论当前平台是否有中文字体，都应返回 str 或 None，且绝不抛异常
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 无中文字体时会发告警，这里只验证不崩溃
        result = setup_cjk_font()
    assert result is None or isinstance(result, str)
