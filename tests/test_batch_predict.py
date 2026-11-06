# -*- coding: utf-8 -*-
"""批量识别相关的纯函数单测（2026-10-30）。

只测「不依赖模型权重、不依赖网络」的部分：字段契约与前后端常量一致性。
真实推理路径由 backups/_regress_1030_batch.py 的 39 项 HTTP 回归覆盖。
"""
import re
from pathlib import Path

import pytest

from src.api import main

ROOT = Path(__file__).resolve().parents[1]


def _fake_result(cid=31, name="番茄晚疫病", crop="番茄", conf=0.93):
    return {
        "class_id": cid, "crop": crop, "name_zh": name, "confidence": conf,
        "alias": "", "symptom": "症状", "cause": "病因", "prevention": "防治", "treatment": "治疗",
    }


def _fake_ood(**over):
    d = {
        "likely_not_leaf": False, "leaf_prob": 0.99, "leaf_prob_source": "head",
        "mean_chroma": 40.0, "grayscale": False, "msp": 0.98,
        "crop_applied": True, "crop_candidates": 10,
        "confidence_calibrated": 0.52, "reliability": "medium", "calibration_enabled": True,
    }
    d.update(over)
    return d


def test_build_predict_data_keeps_full_contract():
    """单张与批量共用同一份组装函数：这里的字段集合就是对外契约。

    少一个字段不会报错，只会让前端某处渲染成空白 —— 所以用断言把契约钉住。
    新增字段时同步改这里，等于强制过一遍「前端要不要用」的思考。
    """
    d = main._build_predict_data([_fake_result()], _fake_ood())
    must = {
        "pred_class_id", "crop_cn", "disease_cn", "is_healthy", "confidence", "top3",
        "need_review", "likely_not_leaf", "leaf_prob", "leaf_prob_source",
        "mean_chroma", "grayscale", "msp", "crop_applied", "crop_candidates",
        "confidence_calibrated", "reliability", "calibration_enabled", "knowledge",
    }
    assert must.issubset(d.keys())
    assert d["disease_cn"] == "番茄晚疫病"
    assert d["is_healthy"] == 0
    assert set(d["knowledge"]) == {"alias", "symptom", "cause", "prevention", "treatment"}


def test_top3_json_drops_display_fields():
    """落库只存 class_id 与概率：病名若也存进去，类别改名后历史里就会是旧名。"""
    d = main._build_predict_data([_fake_result(), _fake_result(cid=30, name="番茄早疫病", conf=0.05)],
                                 _fake_ood())
    rows = main._top3_json(d)
    assert len(rows) == 2
    assert all(set(r) == {"class_id", "prob"} for r in rows)
    assert rows[1]["prob"] == pytest.approx(0.05)


def test_analyze_error_carries_code_and_status():
    e = main._AnalyzeError(400, "坏图")
    assert (e.code, e.msg, e.status) == (400, "坏图", 400)
    assert main._AnalyzeError(413, "太大", 413).status == 413


def test_analyze_one_rejects_unknown_ext_before_touching_disk():
    """扩展名不合法要在落盘之前就拒掉，否则 data/uploads 会堆孤儿文件。"""
    with pytest.raises(main._AnalyzeError) as ei:
        main._analyze_one(b"whatever", "note.txt", "番茄", 3)
    assert ei.value.code == 400 and "不支持的文件类型" in ei.value.msg


def test_analyze_one_rejects_corrupt_image():
    with pytest.raises(main._AnalyzeError) as ei:
        main._analyze_one(b"not an image at all", "x.jpg", "番茄", 3)
    assert ei.value.code == 400 and "不是合法的图片" in ei.value.msg


def test_batch_max_matches_frontend():
    """前后端的上限常量必须一致。

    不一致的表现很隐蔽：前端说「最多 9 张」却允许选到 12 张，
    或者前端只让选 6 张而后端能收 9 张 —— 前者会白等一次上传再被 400 退回来。
    """
    src = (ROOT / "src" / "web" / "src" / "pages" / "DetectPage.jsx").read_text(encoding="utf-8")
    m = re.search(r"const BATCH_MAX = (\d+)", src)
    assert m, "前端 DetectPage.jsx 里找不到 BATCH_MAX"
    assert int(m.group(1)) == main.BATCH_MAX_FILES
