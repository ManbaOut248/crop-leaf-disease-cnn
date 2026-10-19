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
from src.api.crypto import cipher

DB_PATH = PROJECT_ROOT / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    """获取一个开启了 Row 工厂的 SQLite 连接。

    并发保护（#D）：
    - timeout=30 + check_same_thread=False：多请求并发时等待而非立即报
      'database is locked'（本项目每请求独立建连/关闭，不跨线程共享连接，安全）。
    - journal_mode=WAL：写入不再锁全库，读可并发，显著降低锁冲突。
    - busy_timeout=30000：写冲突时阻塞等待而非立即失败。
    - foreign_keys=ON：使 recognition_records 的 ON DELETE CASCADE 真正生效
      （SQLite 默认关闭外键约束，此前删用户不会级联删记录）。
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
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
        -- 非叶片判据的运行期快照（2026-10-19 加入）：
        -- 用于观察线上输入分布是否漂移。历史行留 NULL，聚合查询会自动跳过。
        leaf_prob REAL,
        leaf_prob_source TEXT,
        mean_chroma REAL,
        likely_not_leaf INTEGER,
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
    _migrate_records(cur)
    _migrate_encrypt_pii(cur)
    conn.commit()

    # 首次导入知识库（仅当为空时）
    cur.execute("SELECT COUNT(*) AS c FROM disease_knowledge")
    if cur.fetchone()["c"] == 0:
        _import_knowledge(cur)
    conn.commit()
    conn.close()


def _migrate_records(cur):
    """为已存在的旧库补齐 OOD 监控列。

    SQLite 不支持 `ADD COLUMN IF NOT EXISTS`，只能先读 `PRAGMA table_info` 再补。
    索引必须在这之后建 —— 否则在旧库上 `CREATE INDEX ... (leaf_prob)` 会因
    列不存在直接抛错，导致整个 init_db 失败。
    """
    cur.execute("PRAGMA table_info(recognition_records)")
    have = {r["name"] for r in cur.fetchall()}
    added = []
    for name, typ in (("leaf_prob", "REAL"),
                      ("leaf_prob_source", "TEXT"),
                      ("mean_chroma", "REAL"),
                      ("likely_not_leaf", "INTEGER")):
        if name not in have:
            cur.execute(f"ALTER TABLE recognition_records ADD COLUMN {name} {typ}")
            added.append(name)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_records_leafprob "
                "ON recognition_records(created_at DESC, leaf_prob)")
    if added:
        print(f"[db] recognition_records 已补列: {', '.join(added)}")


def _migrate_encrypt_pii(cur):
    """把历史明文的 `users.email` 就地加密（幂等）。

    为什么需要迁移而不是只改写入路径：加密封装上线前已有用户的 email 是明文，
    如果只改 create_user，历史数据仍然裸存，等于「新用户加密、老用户没加密」。
    这里按前缀 `enc:v1:` 判断，已加密的跳过，可重复执行。
    """
    if not cipher.enabled:
        return 0
    cur.execute("SELECT id, email FROM users WHERE email IS NOT NULL AND email <> ''")
    rows = [dict(r) for r in cur.fetchall()]
    n = 0
    for r in rows:
        if cipher.is_ciphertext(r["email"]):
            continue
        cur.execute("UPDATE users SET email=? WHERE id=?", (cipher.encrypt(r["email"]), r["id"]))
        n += 1
    if n:
        print(f"[db] users.email 已加密迁移 {n} 行（DB_FIELD_KEY 已启用）")
    return n


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
            # email 属 PII，按任务书 6.2 做静态加密后落库（未配置密钥时原样写入）
            (username, password_hash, salt, cipher.encrypt(email), nickname or username),
        )
        uid = cur.lastrowid
        conn.commit()
        return uid
    finally:
        conn.close()


def _dec_user(row):
    """把用户行里的 PII 列解密后再交给上层（缺失密钥时为 None，不返回密文）。"""
    d = dict(row)
    if "email" in d:
        d["email"] = cipher.decrypt(d["email"])
    return d


def get_user_by_username(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return _dec_user(row) if row else None


def get_user_by_id(uid):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return _dec_user(row) if row else None


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
def add_record(user_id, image_name, pred_class_id, confidence, top3, ood=None):
    """写入一条识别记录。

    ood: `predict_image_with_ood()` 返回的判据字典（可选）。
        落库目的是**运行期监控** —— 只有把 leaf_prob 等值存下来，
        才能回答「线上输入是否偏离训练分布」，也才能为下一轮重训积累素材。
        传 None 时四列为 NULL（兼容旧调用方与单元测试）。
    """
    ood = ood or {}
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO recognition_records
           (user_id, image_name, pred_class_id, confidence, top3_json,
            leaf_prob, leaf_prob_source, mean_chroma, likely_not_leaf)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (user_id, image_name, pred_class_id, confidence,
         json.dumps(top3, ensure_ascii=False),
         ood.get("leaf_prob"),
         ood.get("leaf_prob_source"),
         ood.get("mean_chroma"),
         None if "likely_not_leaf" not in ood else int(bool(ood["likely_not_leaf"]))),
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_ood_stats(user_id, days=7, low_leaf=0.2):
    """近 N 天的非叶片判据分布，用于发现线上数据漂移。

    关心三个信号：
      · `flagged_ratio` —— 被判定为非叶片的占比。突增说明用户在传别的东西，
        或模型/检测头退化了（例如切了量化后端，`msp_fallback` 会同步上涨）。
      · `low_leaf_ratio` —— `leaf_prob < low_leaf` 的占比，比 flagged 更敏感，
        能在正式拦截之前给出早期预警。
      · `msp_fallback` —— 有多少条走了 MSP 回退判据。>0 说明检测头没生效，
        此时真实场景误杀率会显著上升，需要立刻排查。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN leaf_prob IS NOT NULL THEN 1 ELSE 0 END) AS monitored,
                  SUM(CASE WHEN likely_not_leaf = 1 THEN 1 ELSE 0 END) AS flagged,
                  SUM(CASE WHEN leaf_prob < ? THEN 1 ELSE 0 END) AS low,
                  SUM(CASE WHEN leaf_prob_source = 'msp' THEN 1 ELSE 0 END) AS msp_fallback,
                  AVG(leaf_prob) AS avg_leaf_prob,
                  AVG(mean_chroma) AS avg_chroma
           FROM recognition_records
           WHERE user_id = ?
             AND created_at >= datetime('now','localtime','-' || ? || ' days')""",
        (low_leaf, user_id, int(days)),
    )
    r = dict(cur.fetchone())
    conn.close()

    total = r["total"] or 0
    monitored = r["monitored"] or 0
    return {
        "days": int(days),
        "total": total,
        "monitored": monitored,
        "flagged": r["flagged"] or 0,
        "msp_fallback": r["msp_fallback"] or 0,
        "avg_leaf_prob": round(r["avg_leaf_prob"], 4) if r["avg_leaf_prob"] is not None else None,
        "avg_chroma": round(r["avg_chroma"], 1) if r["avg_chroma"] is not None else None,
        # 分母用 monitored（有判据快照的记录），避免把历史 NULL 行算成"正常"
        "flagged_ratio": round((r["flagged"] or 0) / monitored, 4) if monitored else None,
        "low_leaf_ratio": round((r["low"] or 0) / monitored, 4) if monitored else None,
        "low_leaf_threshold": low_leaf,
    }


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


def get_crop_summary():
    """作物清单及各作物的候选类别数，供前端「选择作物」下拉使用。

    返回按类别数降序排列，形如：
        [{"crop": "番茄", "n_classes": 10, "n_diseases": 9}, ...]

    为什么要前端从后端取而不是硬编码：类别体系（38 类）的唯一来源是数据库 /
    知识库；硬编码在前端意味着改类别时要改两处，且前端不知道哪个作物只有
    「健康」一个候选（此时作物约束等于强制作答，应提示用户）。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT crop_cn AS crop,
               COUNT(*) AS n_classes,
               SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) AS n_diseases
        FROM disease_knowledge
        GROUP BY crop_cn
        ORDER BY n_classes DESC, crop_cn
    """)
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
