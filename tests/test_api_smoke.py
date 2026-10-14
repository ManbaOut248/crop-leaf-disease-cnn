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
