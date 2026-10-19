# -*- coding: utf-8 -*-
"""API 冒烟测试（slow）：用 TestClient 跑通核心接口；模型缺失时整体跳过。"""
import io

import numpy as np
import pytest
from PIL import Image

from src.config import BEST_MODEL_PATH, NUM_CLASSES
import src.api.main as api

pytestmark = pytest.mark.slow

if not BEST_MODEL_PATH.exists():
    pytest.skip("best 模型权重缺失（models/best_resnet18_38.pth），跳过 API 冒烟", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(api.app) as c:
        yield c


def _png():
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"), "RGB").save(buf, "PNG")
    buf.seek(0)
    return buf


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["data"]["status"] == "up"


def test_predict_default_top3(client):
    r = client.post("/api/v1/predict", files={"file": ("leaf.png", _png(), "image/png")})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data["top3"]) == 3
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_topk_param(client):
    for k in (1, 5, NUM_CLASSES):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", _png(), "image/png")},
            params={"top_k": k},
        )
        assert r.status_code == 200, f"top_k={k} 失败"
        assert len(r.json()["data"]["top3"]) == k, f"top_k={k} 返回条数不符"


def test_predict_invalid_file_type(client):
    r = client.post(
        "/api/v1/predict",
        files={"file": ("x.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert r.status_code == 400


def test_predict_topk_out_of_range(client):
    r = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", _png(), "image/png")},
        params={"top_k": 999},
    )
    assert r.status_code == 422


# ---------- 作物先验（2026-10-19） ----------
def test_crops_endpoint(client):
    """作物清单接口：14 种作物，且标出只有单一候选的作物（约束等于强制作答）。"""
    r = client.get("/api/v1/crops")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["enabled"] is True
    assert len(data["items"]) == 14
    crops = {x["crop"]: x["n_classes"] for x in data["items"]}
    assert crops["番茄"] == 10
    # 只有一个候选类的作物必须被标出来，否则前端不会提示「结果等同直接给结论」
    assert set(data["single_candidate"]) == {c for c, n in crops.items() if n == 1}


def test_predict_crop_prior(client):
    """指定作物后：Top-3 必须全部落在该作物内，且 crop_applied 为 True。"""
    r = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", _png(), "image/png")},
        params={"crop": "番茄", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["crop_applied"] is True
    assert data["crop_candidates"] == 10
    assert all(t["disease_cn"].startswith("番茄") or "番茄" in t["disease_cn"]
               for t in data["top3"]), data["top3"]


def test_predict_crop_single_candidate(client):
    """单候选作物：候选只有 1 个，返回条数必须夹紧到 1（不能越界返回 3 条）。"""
    r = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", _png(), "image/png")},
        params={"crop": "蓝莓", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["crop_candidates"] == 1
    assert len(data["top3"]) == 1
    assert data["confidence"] == 1.0  # 唯一候选，归一化后必为 1


def test_predict_crop_invalid(client):
    """非法作物名要明确报错，不能静默退化成「不限制作物」。"""
    r = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", _png(), "image/png")},
        params={"crop": "西瓜"},
    )
    assert r.status_code == 400
    assert "未知作物" in r.json()["msg"]


def test_predict_crop_not_affect_ood(client):
    """核心安全属性：作物掩码只改展示概率，**不得**改变 OOD 判据。

    若掩码影响了 leaf_prob/msp，候选变少会让 softmax 更尖锐、置信度虚高，
    非叶片图片只要指定一个作物就能绕过拒识 —— 这是必须守住的不变量。
    """
    # 必须用同一份字节：_png() 每次生成随机数组，各调一次等于拿两张不同的图比，
    # 数值天然不同（第一次写这个测试就踩到了，误报成产品缺陷）。
    raw = _png().getvalue()
    base = client.post(
        "/api/v1/predict", files={"file": ("leaf.png", io.BytesIO(raw), "image/png")}
    ).json()["data"]
    masked = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.png", io.BytesIO(raw), "image/png")},
        params={"crop": "番茄"},
    ).json()["data"]
    # OOD 判据必须完全一致
    assert base["leaf_prob"] == masked["leaf_prob"]
    assert base["msp"] == masked["msp"]
    assert base["likely_not_leaf"] == masked["likely_not_leaf"]
    # 而展示置信度**应该**变化 —— 否则说明掩码压根没生效，这条测试就等于没测
    assert base["confidence"] != masked["confidence"]
