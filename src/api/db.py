# -*- coding: utf-8 -*-
"""
SQLite 数据库层（对齐 docs/03_数据库设计.md）。
四张表：users / recognition_records / disease_knowledge / system_config
提供连接、初始化建表、以及用户/记录/知识的相关 CRUD。
"""

import json
import sqlite3
from pathlib import Path

from src.config import PROJECT_ROOT, DISEASE_KNOWLEDGE_PATH, CLASS_DIRS, class_name_zh

DB_PATH = PROJECT_ROOT / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    """获取一个开启了 Row 工厂的 SQLite 连接。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表并首次导入病害知识库（38 类，与模型输出 0~37 对齐）。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        email TEXT,
        nickname TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')),
        updated_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);

    CREATE TABLE IF NOT EXISTS recognition_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_name TEXT,
        pred_class_id INTEGER NOT NULL,
        confidence REAL NOT NULL,
        top3_json TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_records_user ON recognition_records(user_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS disease_knowledge (
        id INTEGER PRIMARY KEY,
        crop_en TEXT NOT NULL,
        crop_cn TEXT NOT NULL,
        disease_en TEXT,
        disease_cn TEXT NOT NULL,
        is_healthy INTEGER NOT NULL,
        symptom TEXT,
        cause TEXT,
        prevention TEXT,
        treatment TEXT
    );

    CREATE TABLE IF NOT EXISTS system_config (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT DEFAULT (datetime('now','localtime'))
    );
    """)
    conn.commit()

    # 首次导入知识库（仅当为空时）
    cur.execute("SELECT COUNT(*) AS c FROM disease_knowledge")
    if cur.fetchone()["c"] == 0:
        _import_knowledge(cur)
    conn.commit()
    conn.close()


def _split_crop_disease(en, zh, info):
    """从类目录名/中文名/知识库正确拆分「作物」与「病害」。

    - 健康类：crop_cn=作物名，disease_cn='健康'
    - 病害类：crop_cn=作物名（取自知识库 crop 字段），disease_cn=病害全称
    兜底：知识库缺 crop 时，按中文名粗拆（健康类去「（健康）」后缀；病害类取前 2 字）。
    """
    is_healthy = 1 if en.endswith("healthy") else 0
    crop_cn = (info.get("crop") or "").strip()
    if not crop_cn:
        if is_healthy:
            crop_cn = zh.replace("（健康）", "").strip()
        else:
            crop_cn = zh[:2]  # 多数作物名为 2 字（苹果/番茄/玉米/草莓…），兜底用
    if is_healthy:
        disease_cn = "健康"
    else:
        disease_cn = zh  # 病害全称，如「苹果黑星病」
    return crop_cn, disease_cn, is_healthy


def _import_knowledge(cur):
    """把 disease_knowledge.json 与 CLASS_DIRS 合并导入到 disease_knowledge 表。"""
    raw = {}
    if DISEASE_KNOWLEDGE_PATH.exists():
        with open(DISEASE_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    for idx, en in enumerate(CLASS_DIRS):
        info = raw.get(en, {})
        zh = class_name_zh(en)
        crop_cn, disease_cn, is_healthy = _split_crop_disease(en, zh, info)
        cur.execute(
            """INSERT OR REPLACE INTO disease_knowledge
               (id, crop_en, crop_cn, disease_en, disease_cn, is_healthy,
                symptom, cause, prevention, treatment)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (idx, en, crop_cn, en, disease_cn, is_healthy,
             info.get("symptom", ""), info.get("cause", ""),
             info.get("prevention", ""), info.get("treatment", "")),
        )


# ===================== 用户 CRUD =====================
def create_user(username, password_hash, salt, email=None, nickname=None):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, email, nickname) VALUES (?,?,?,?,?)",
            (username, password_hash, salt, email, nickname or username),
        )
        uid = cur.lastrowid
        conn.commit()
        return uid
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_user(uid, nickname=None, password_hash=None, salt=None):
    conn = get_conn()
    cur = conn.cursor()
    if password_hash is not None:
        cur.execute(
            "UPDATE users SET nickname=?, password_hash=?, salt=?, updated_at=datetime('now','localtime') WHERE id=?",
            (nickname, password_hash, salt, uid),
        )
    else:
        cur.execute(
            "UPDATE users SET nickname=?, updated_at=datetime('now','localtime') WHERE id=?",
            (nickname, uid),
        )
    conn.commit()
    conn.close()


# ===================== 识别记录 CRUD =====================
def add_record(user_id, image_name, pred_class_id, confidence, top3):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO recognition_records
           (user_id, image_name, pred_class_id, confidence, top3_json)
           VALUES (?,?,?,?,?)""",
        (user_id, image_name, pred_class_id, confidence, json.dumps(top3, ensure_ascii=False)),
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_records_by_user(user_id, page=1, size=20):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) AS c FROM recognition_records WHERE user_id=?", (user_id,))
    total = cur.fetchone()["c"]
    cur.execute(
        """SELECT * FROM recognition_records WHERE user_id=?
           ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (user_id, size, (page - 1) * size),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows, total


def get_record_by_id(rid, user_id=None):
    conn = get_conn()
    cur = conn.cursor()
    if user_id is not None:
        cur.execute("SELECT * FROM recognition_records WHERE id=? AND user_id=?", (rid, user_id))
    else:
        cur.execute("SELECT * FROM recognition_records WHERE id=?", (rid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_record(rid, user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM recognition_records WHERE id=? AND user_id=?", (rid, user_id))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0


# ===================== 病害知识查询 =====================
def get_disease_list(crop=None):
    conn = get_conn()
    cur = conn.cursor()
    if crop:
        cur.execute("SELECT * FROM disease_knowledge WHERE crop_cn LIKE ? ORDER BY id", (f"%{crop}%",))
    else:
        cur.execute("SELECT * FROM disease_knowledge ORDER BY id")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_disease_by_id(did):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM disease_knowledge WHERE id=?", (did,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ===================== 配置读写 =====================
def get_config(key, default=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM system_config WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else default


def set_config(key, value):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO system_config (key, value, updated_at) VALUES (?,?,datetime('now','localtime'))",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("数据库初始化完成：", DB_PATH)
    print("病害知识条数：", len(get_disease_list()))
