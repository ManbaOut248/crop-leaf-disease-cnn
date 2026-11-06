# -*- coding: utf-8 -*-
"""识别记录的人工判卷结论（verdict）与实际病害（ground_truth_class_id）联动。

为什么要单独测这一组：这两个字段有**从属关系**（实际病害只在「识别错误」时成立），
联动逻辑写在 db 层，一旦哪次重构把清理条件写错，表现是「统计里混进判对但写了
实际病害的矛盾行」——不报错、不崩溃，只在很久以后的数据分析里发现脏。

隔离方式：把 db.DB_PATH 指到 tmp_path 再 init_db()，不碰 data/app.db 里的演示数据。
"""
import pytest

from src.api import db
from src.config import CLASS_DIRS


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """临时库 + 一个用户 + 一条记录。"""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    uid = db.create_user("t_user", "hash", "salt", None, "测试用户")
    rid = db.add_record(uid, "a.jpg", 30, 0.99, [{"class_id": 30, "prob": 0.99}])
    return uid, rid


def test_wrong_then_correct_clears_ground_truth(env):
    uid, rid = env
    # 判错 + 补实际病害
    db.update_record(rid, uid, verdict="wrong", ground_truth_class_id=30)
    row = db.get_record_by_id(rid, uid)
    assert row["verdict"] == "wrong"
    assert row["ground_truth_class_id"] == 30

    # 改判为「识别正确」且**没有**显式给实际病害 → 必须连带清空，
    # 否则留下「判对却写着实际病害是别的」的矛盾行
    db.update_record(rid, uid, verdict="correct")
    row = db.get_record_by_id(rid, uid)
    assert row["verdict"] == "correct"
    assert row["ground_truth_class_id"] is None


def test_unset_verdict_also_clears_ground_truth(env):
    uid, rid = env
    db.update_record(rid, uid, verdict="wrong", ground_truth_class_id=12)
    db.update_record(rid, uid, verdict=None)
    row = db.get_record_by_id(rid, uid)
    assert row["verdict"] is None
    assert row["verdict_at"] is None          # 时间戳跟着结论一起清
    assert row["ground_truth_class_id"] is None


def test_explicit_ground_truth_wins(env):
    """调用方显式给了实际病害时，以调用方为准（三态：不传=自动 / None=清 / 值=设）。"""
    uid, rid = env
    row = db.update_record(rid, uid, verdict="correct", ground_truth_class_id=7)
    assert row["verdict"] == "correct"
    assert row["ground_truth_class_id"] == 7


def test_ground_truth_out_of_range_rejected(env):
    uid, rid = env
    for bad in (-1, len(CLASS_DIRS), 999):
        with pytest.raises(ValueError):
            db.update_record(rid, uid, ground_truth_class_id=bad)
    # 脏值不能落库
    assert db.get_record_by_id(rid, uid)["ground_truth_class_id"] is None


def test_filter_total_matches_rows(env):
    """筛选后的 total 必须与过滤条件一致 —— 只过滤 SELECT 会让页脚显示全量。"""
    uid, rid = env
    r2 = db.add_record(uid, "b.jpg", 4, 0.6, None)
    db.update_record(rid, uid, verdict="wrong")

    rows, total = db.get_records_by_user(uid, page=1, size=10, verdict="wrong")
    assert total == 1 and [r["id"] for r in rows] == [rid]

    rows, total = db.get_records_by_user(uid, page=1, size=10, verdict="correct")
    assert total == 0 and rows == []

    rows, total = db.get_records_by_user(uid, page=1, size=10, verdict="unset")
    assert total == 1 and [r["id"] for r in rows] == [r2]

    counts = db.count_records_by_verdict(uid)
    assert counts == {"all": 2, "correct": 0, "wrong": 1, "unset": 1, "with_ground_truth": 0}


def test_custom_name_does_not_touch_verdict(env):
    """只改名字时，判定与实际病害必须原样保留（三态哨兵的回归点）。"""
    uid, rid = env
    db.update_record(rid, uid, verdict="wrong", ground_truth_class_id=30)
    row = db.update_record(rid, uid, custom_name="东区3号棚")
    assert row["custom_name"] == "东区3号棚"
    assert row["verdict"] == "wrong"
    assert row["ground_truth_class_id"] == 30
