# -*- coding: utf-8 -*-
"""OOD 运行期监控的单元测试：表结构迁移 + 聚合比例口径。

为什么要测这两点：
  1. 迁移是「一次性但必须幂等」的操作 —— 老库升级失败会导致整个服务起不来，
     且重复执行不能报错（每次启动都会跑）。
  2. 聚合的**分母**曾是个坑：`total` 含历史 NULL 行，如果拿它当分母，
     `flagged_ratio` 会被历史数据稀释，看起来"一切正常"。必须用 monitored。
"""
import sqlite3

import pytest

from src.api import db


def test_migrate_adds_columns_and_is_idempotent():
    """旧结构的表跑迁移后应补齐 4 列，且可重复执行。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE recognition_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_name TEXT,
        pred_class_id INTEGER NOT NULL,
        confidence REAL NOT NULL,
        top3_json TEXT,
        created_at TEXT
    );
    """)
    cur = conn.cursor()

    db._migrate_records(cur)
    cols = {r["name"] for r in cur.execute("PRAGMA table_info(recognition_records)")}
    assert {"leaf_prob", "leaf_prob_source", "mean_chroma", "likely_not_leaf"} <= cols

    # 重复执行不应抛错（服务每次启动都会调用）
    db._migrate_records(cur)
    conn.close()


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """把 db.DB_PATH 指向临时库，避免污染开发库。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    uid = db.create_user("u1", "hash", "salt")
    yield uid
    # 连接都是用完即关，无需清理；临时库随 tmp_path 回收


def test_ood_stats_uses_monitored_as_denominator(temp_db):
    """有快照的 2 条里 1 条被判定为非叶片 → 比例应为 50%，不受历史 NULL 行影响。"""
    uid = temp_db
    db.add_record(uid, "a.jpg", 1, 0.99, [], ood=None)          # 历史行：无快照
    db.add_record(uid, "b.jpg", 2, 0.90, [],
                  ood={"leaf_prob": 0.98, "leaf_prob_source": "head",
                       "mean_chroma": 40.1, "likely_not_leaf": False})
    db.add_record(uid, "c.jpg", 3, 0.10, [],
                  ood={"leaf_prob": 0.05, "leaf_prob_source": "head",
                       "mean_chroma": 12.0, "likely_not_leaf": True})

    s = db.get_ood_stats(uid, days=7)
    assert s["total"] == 3
    assert s["monitored"] == 2                      # 只统计有判据快照的
    assert s["flagged"] == 1
    assert s["flagged_ratio"] == pytest.approx(0.5)  # 1/2，而不是 1/3
    assert s["low_leaf_ratio"] == pytest.approx(0.5)  # leaf_prob < 0.2 的占比
    assert s["msp_fallback"] == 0
    assert s["avg_leaf_prob"] == pytest.approx(0.515, abs=1e-3)


def test_ood_stats_empty_returns_none_ratios(temp_db):
    """没有任何记录时分母为 0，比例应返回 None 而不是除零或 0。"""
    s = db.get_ood_stats(temp_db, days=7)
    assert s["total"] == 0
    assert s["flagged_ratio"] is None
    assert s["avg_leaf_prob"] is None


def test_ood_stats_scoped_to_user(temp_db):
    """统计必须按用户隔离，不能把别人的记录算进来。"""
    uid = temp_db
    other = db.create_user("u2", "hash", "salt")
    db.add_record(uid, "a.jpg", 1, 0.9, [],
                  ood={"leaf_prob": 0.9, "leaf_prob_source": "head",
                       "mean_chroma": 30.0, "likely_not_leaf": False})
    db.add_record(other, "b.jpg", 1, 0.9, [],
                  ood={"leaf_prob": 0.01, "leaf_prob_source": "head",
                       "mean_chroma": 0.0, "likely_not_leaf": True})

    s = db.get_ood_stats(uid, days=7)
    assert s["total"] == 1
    assert s["flagged"] == 0
