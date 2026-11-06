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
        -- 用户上传时的原始文件名（2026-10-29 加入）：
        -- image_name 存的是落盘用的 uuid 名（防重名/防路径穿越），对用户毫无意义，
        -- 历史页因此一直显示 `a3f1...c9.jpg`。真正的「图片名」必须单独存一列。
        orig_name TEXT,
        -- 用户自定义命名（2026-10-29 加入）：
        -- 上传名（orig_name）往往是一串相机编号或 IMG_0001，用户想按自己的
        -- 意思备注（「东区3号棚-晚疫」）就必须可改。空串与 NULL 都表示未命名，
        -- 展示时回落到 orig_name，而不是把空串显示成空白行。
        custom_name TEXT,
        -- 人工核对结论（2026-10-29 加入）：
        -- NULL=未核对 / 'correct'=用户认为识别正确 / 'wrong'=识别错误。
        -- 之所以不由模型自己算「对不对」：闭集模型对体系外作物、健康叶会给出高
        -- 置信度错误答案（docs/25 实测未知作物 88.2% 硬答），只有人能给真值。
        -- 这也是唯一能积累真实域标注的入口。
        verdict TEXT,
        verdict_at TEXT,
        -- 人工确认的「实际病害」类别（2026-10-29 加入）：
        -- 只在 verdict='wrong'（用户判定 AI 识别错误）时才有意义 —— 用户既然
        -- 已经知道正确答案，顺手给出是哪一类，这条记录就从「一条错误日志」
        -- 升级成「一条带真值的标注样本」，是真实域补数据最省成本的入口。
        -- 存 class_id(0~37) 而非病名文本：类别体系调整后名称自动跟随，
        -- 与 top3_json 存 class_id 的理由一致。NULL = 没给 / 不适用。
        ground_truth_class_id INTEGER,
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
                      ("likely_not_leaf", "INTEGER"),
                      # 原始文件名列（2026-10-29）：旧库补列后历史行仍为 NULL，
                      # 查询层会退化显示，不会因缺列直接 500。
                      ("orig_name", "TEXT"),
                      # 自定义命名 / 人工核对（2026-10-29）：同样只补列不填值，
                      # 旧记录自然落在「未命名 / 未核对」状态。
                      ("custom_name", "TEXT"),
                      ("verdict", "TEXT"),
                      ("verdict_at", "TEXT"),
                      # 实际病害：旧库补列后历史行为 NULL，即「没提供」，
                      # 前端按「未填写」展示，不参与任何统计分母。
                      ("ground_truth_class_id", "INTEGER")):
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
def add_record(user_id, image_name, pred_class_id, confidence, top3, ood=None,
               orig_name=None):
    """写入一条识别记录。

    image_name: 落盘文件名（uuid.ext），用于取缩略图，**不是给用户看的**。
    orig_name:  用户上传时的原始文件名（如 `番茄晚疫病.jpg`），历史列表展示用。
                放关键字参数且默认 None —— 旧调用方与单测不受影响，
                缺失时历史页退化为占位文案而不是报错。
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
           (user_id, image_name, orig_name, pred_class_id, confidence, top3_json,
            leaf_prob, leaf_prob_source, mean_chroma, likely_not_leaf)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (user_id, image_name, orig_name, pred_class_id, confidence,
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


def get_records_by_user(user_id, page=1, size=20, verdict=None):
    """按页取某个用户的识别记录，返回 (rows, total)。

    `verdict` 过滤（2026-10-29 新增）：
      · None / 'all' → 不过滤（默认，保持旧行为）
      · 'correct' / 'wrong' → 只看已核对为正确 / 错误的
      · 'unset' → 只看还没核对过的
    判定为「识别错误」的记录是用户最想集中处理的那批（挑出来复核 / 补数据），
    只能在界面上一页页翻 = 这个功能等于不存在。

    ⚠️ 排序必须带 `id DESC` 二级键（2026-10-29 修复）。
    此前只有 `ORDER BY created_at DESC`，而 `created_at` 是秒级文本时间戳：
    同一秒内写入的多条记录（实测 user 5 在 2026-10-26 11:24:49 一次写入 6 条）
    在 SQLite 里顺序未定义，翻页时同一行可能既出现在第 1 页又出现在第 2 页，
    而另一行两条页都取不到 —— 表现为「历史记录偶尔少一条 / 翻页对不上」，
    且无任何报错。加 id 二级排序后顺序全序且稳定。

    ⚠️ 过滤条件必须同时进 COUNT 与 SELECT：只过滤 SELECT 会让页脚总数
    仍是全量，用户看到「共 182 条」但翻两页就空了。
    """
    where = "user_id=?"
    args = [user_id]
    v = (verdict or "").strip().lower()
    if v in ("correct", "wrong"):
        where += " AND verdict=?"
        args.append(v)
    elif v == "unset":
        # verdict 为 NULL 才是「未核对」；空串是脏值，也一并算未核对
        where += " AND (verdict IS NULL OR verdict='')"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) AS c FROM recognition_records WHERE {where}", args)
    total = cur.fetchone()["c"]
    cur.execute(
        f"""SELECT * FROM recognition_records WHERE {where}
            ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""",
        args + [size, (page - 1) * size],
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows, total


def count_records_by_verdict(user_id):
    """全量历史按人工核对结论分组计数（不受分页 / 时间窗口影响）。

    为什么不用 `/stats/overview.verdict`：那是**近 N 天**窗口的统计，而列表是
    全部历史。实测 demo 用户近 30 天 171 条 / 全量 182 条 —— 拿窗口数字当
    筛选徽标，用户会看到「识别错误 3 条」却筛出 5 条，又是一次「数字对不上」。
    这里刻意不带时间条件，与列表口径严格一致。

    返回：{"all": n, "correct": n, "wrong": n, "unset": n, "with_ground_truth": n}
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(*) AS total,
                  COALESCE(SUM(CASE WHEN verdict='correct' THEN 1 ELSE 0 END), 0) AS correct,
                  COALESCE(SUM(CASE WHEN verdict='wrong' THEN 1 ELSE 0 END), 0) AS wrong,
                  COALESCE(SUM(CASE WHEN ground_truth_class_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS with_gt
           FROM recognition_records WHERE user_id=?""",
        (user_id,),
    )
    r = dict(cur.fetchone())
    conn.close()
    total = r["total"] or 0
    correct = r["correct"] or 0
    wrong = r["wrong"] or 0
    return {
        "all": total,
        "correct": correct,
        "wrong": wrong,
        # 相减而不是再查一次：三档互斥且穷尽（correct / wrong / 其余），
        # 单独 SUM(IS NULL) 一旦碰到脏值就会与前两档之和不符
        "unset": max(0, total - correct - wrong),
        "with_ground_truth": r["with_gt"] or 0,
    }


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
    """删除一条记录，返回 (是否删除成功, 落盘文件名或 None)。

    为什么要把文件名带出来：历史删掉了、`data/uploads/` 里的原图还在，
    磁盘只会单调增长（审计里「匿名上传文件无清理」是同一个问题）。
    但清理文件属于 I/O、不属于事务 —— 放在路由层做，数据库删除保持纯粹，
    单测不需要造真实文件也能跑。

    `user_id` 条件不能省：只按 id 删等于任何登录用户都能删别人的记录
    （越权删除比越权读取更不可接受，且不可恢复）。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT image_name FROM recognition_records WHERE id=? AND user_id=?",
        (rid, user_id),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, None
    image_name = row["image_name"]
    cur.execute("DELETE FROM recognition_records WHERE id=? AND user_id=?", (rid, user_id))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0, image_name


# 用于 update_record 区分「没传这个字段」与「显式传 None 清空」。
# 用 None 当哨兵会把「清空自定义命名」误判成「不要改」，这是典型的三态陷阱。
_UNSET = object()

# 人工核对只允许这两种取值，其余一律拒绝 —— 存进脏值会让统计口径静默失真
VERDICT_VALUES = ("correct", "wrong")

# 自定义命名的长度上限：列表列宽 200px 大约能放 20 个汉字，
# 留 40 字符给弹窗与导出，再长就只能靠省略号，反而更难认。
_MAX_CUSTOM_NAME = 40


def update_record(rid, user_id, custom_name=_UNSET, verdict=_UNSET,
                  ground_truth_class_id=_UNSET):
    """就地修改一条记录的自定义命名 / 人工核对结论 / 实际病害，返回更新后的行。

    只允许改这几列：预测结果、置信度是识别时的客观事实，事后改动等于伪造日志，
    真要修正请重新识别一条。

    `ground_truth_class_id`（2026-10-29 新增）与 `verdict` 有从属关系：
      · verdict='wrong' 且给了实际病害 → 记录一条「AI 认错、正确答案是 X」的样本；
      · verdict 改成 'correct' / null 而调用方**没显式**给实际病害 → 连带清空，
        否则会留下「已判定识别正确，却写着实际病害是别的」这种自相矛盾行，
        统计时还得额外加条件过滤，迟早漏。
      显式传了值就以调用方为准（三态：不传=自动 / None=清空 / 值=设置）。

    Returns:
        更新后的行 dict；记录不存在或不属于该用户时返回 None（路由层转 404，
        不区分「不存在」与「不是你的」，避免用 404/403 差异探测他人记录 id）。
    """
    fields = []
    params = []
    if custom_name is not _UNSET:
        # 空串统一存 NULL：空串会让「是否有自定义名」的真值判断变成两套写法
        value = (custom_name or "").strip() or None
        if value is not None and len(value) > _MAX_CUSTOM_NAME:
            value = value[:_MAX_CUSTOM_NAME]
        fields.append("custom_name = ?")
        params.append(value)
    verdict_value = _UNSET
    if verdict is not _UNSET:
        v = (verdict or "").strip().lower() or None
        if v is not None and v not in VERDICT_VALUES:
            raise ValueError(f"verdict 只能是 {'/'.join(VERDICT_VALUES)} 或 null")
        verdict_value = v
        fields.append("verdict = ?")
        params.append(v)
        # 时间戳跟着结论走：取消核对（置空）时一并清掉，否则会留下
        # 「未核对但有核对时间」的自相矛盾行
        fields.append("verdict_at = datetime('now','localtime')" if v else "verdict_at = NULL")
    if ground_truth_class_id is _UNSET:
        # 结论不再是「识别错误」→ 实际病害失去意义，顺带清掉
        if verdict_value is not _UNSET and verdict_value != "wrong":
            fields.append("ground_truth_class_id = NULL")
    else:
        gid = None
        if ground_truth_class_id is not None and str(ground_truth_class_id).strip() != "":
            try:
                gid = int(ground_truth_class_id)
            except (TypeError, ValueError):
                raise ValueError("ground_truth_class_id 必须是整数类别 id")
            # 越界直接拒绝：存进 0~37 之外的值，前端 class_name_zh 取不到名，
            # 展示成空白，而统计又把它算成「已标注」，是最难查的一类脏数据
            if gid < 0 or gid >= len(CLASS_DIRS):
                raise ValueError(f"ground_truth_class_id 必须在 0~{len(CLASS_DIRS) - 1} 之间")
        fields.append("ground_truth_class_id = ?")
        params.append(gid)
    if not fields:
        return get_record_by_id(rid, user_id)

    conn = get_conn()
    cur = conn.cursor()
    params.extend([rid, user_id])
    cur.execute(
        f"UPDATE recognition_records SET {', '.join(fields)} WHERE id=? AND user_id=?",
        params,
    )
    affected = cur.rowcount
    conn.commit()
    conn.close()
    if not affected:
        return None
    return get_record_by_id(rid, user_id)


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


# ===================== 统计聚合（前端图表用） =====================
# 置信度直方图的桶边界：<0.5 / 0.5-0.6 / ... / 0.9-1.0，共 6 档。
# 为什么从 0.5 起单独一档：0.5 是 THRESHOLD_LOW_CONF（需人工复核线），
# 把低于复核线的记录单独成桶，图上能直接看出「有多少条是不该轻信的」。
_CONF_BUCKETS = [
    ("<50%", 0.0, 0.5),
    ("50-60%", 0.5, 0.6),
    ("60-70%", 0.6, 0.7),
    ("70-80%", 0.7, 0.8),
    ("80-90%", 0.8, 0.9),
    ("90-100%", 0.9, 1.0001),   # 上界留一点余量，避免 confidence 恰好 1.0 落不进任何桶
]


def get_stats_overview(user_id, days=30, low_conf=0.5):
    """识别历史的聚合统计，供前端图表使用。

    为什么必须后端聚合：`/records` 分页上限 100 条，前端拿不到全量数据，
    自己 reduce 会在记录变多后静默算错（比例分母不对），且要发 N 个请求。

    统计窗口：最近 days 天（含今天），所有子项用同一窗口，避免「总数 50 但
    分部之和 47」这类对不上的问题。

    ⚠️ 口径提醒：`confidence` 存的是**原始 softmax**，不是温度校准后的值。
    因此本函数**不**给出 high/medium/low 可靠性分档 —— 那是基于校准值的
    （calibration_domain.json 的 reliability_high=0.55 / medium=0.25），
    两套阈值混用会让「历史页标签」和「图表分档」自相矛盾。
    图表只呈现原始分布，口径差异在接口文档与前端副标题中注明。
    """
    conn = get_conn()
    cur = conn.cursor()
    window = max(1, min(int(days), 365))
    # 作用域条件只拼整数，无注入风险；集中一处便于四个查询保持口径一致
    scope = (
        "FROM recognition_records r "
        "LEFT JOIN disease_knowledge k ON k.id = r.pred_class_id "
        "WHERE r.user_id = ? "
        f"AND r.created_at >= datetime('now','localtime','-{window} days')"
    )

    # ① 总量 / 平均置信度 / 需复核数 / 健康叶数
    cur.execute(
        "SELECT COUNT(*) AS total,"
        "       COALESCE(AVG(r.confidence), 0) AS avg_conf,"
        "       COALESCE(SUM(CASE WHEN r.confidence < ? OR r.likely_not_leaf = 1"
        "                        THEN 1 ELSE 0 END), 0) AS need_review,"
        "       COALESCE(SUM(CASE WHEN k.is_healthy = 1 THEN 1 ELSE 0 END), 0) AS healthy "
        + scope,
        (low_conf, user_id),
    )
    head = dict(cur.fetchone())
    total = head["total"] or 0

    # ② 作物分布（按识别条数降序）
    cur.execute(
        "SELECT COALESCE(k.crop_cn, '未知') AS crop, COUNT(*) AS n " + scope +
        " GROUP BY crop ORDER BY n DESC",
        (user_id,),
    )
    crops = [
        {"crop": r["crop"], "count": r["n"], "ratio": round(r["n"] / total, 4) if total else 0}
        for r in cur.fetchall()
    ]

    # ③ 病害分布 Top 10（同一种病在不同作物上算不同项，故按 作物+病害 分组）
    cur.execute(
        "SELECT COALESCE(k.crop_cn, '未知') AS crop,"
        "       COALESCE(k.disease_cn, '未知') AS disease,"
        "       COALESCE(k.is_healthy, 0) AS is_healthy,"
        "       COUNT(*) AS n " + scope +
        " GROUP BY crop, disease ORDER BY n DESC LIMIT 10",
        (user_id,),
    )
    diseases = [
        {
            "crop": r["crop"],
            "disease": r["disease"],
            # 知识库里两类命名规则不同：
            #   · 病害类 disease_cn 已含作物前缀（"苹果黑星病"）→ 直接用
            #   · 健康类 disease_cn 只是"健康" → 补成「苹果（健康）」，与列表页一致
            # 曾经无条件拼 crop_cn，得到「番茄番茄晚疫病」，已修正。
            "name": f"{r['crop']}（健康）" if r["is_healthy"] else r["disease"],
            "is_healthy": r["is_healthy"],
            "count": r["n"],
        }
        for r in cur.fetchall()
    ]

    # ④ 置信度直方图（CASE 归桶，Python 侧补零，保证 6 档恒定存在）
    cur.execute(
        "SELECT CASE"
        "         WHEN r.confidence < 0.5 THEN 0"
        "         WHEN r.confidence < 0.6 THEN 1"
        "         WHEN r.confidence < 0.7 THEN 2"
        "         WHEN r.confidence < 0.8 THEN 3"
        "         WHEN r.confidence < 0.9 THEN 4"
        "         ELSE 5 END AS idx,"
        "       COUNT(*) AS n " + scope + " GROUP BY idx",
        (user_id,),
    )
    bucket_counts = {r["idx"]: r["n"] for r in cur.fetchall()}
    confidence_buckets = [
        {"range": label, "count": bucket_counts.get(i, 0)}
        for i, (label, _, _) in enumerate(_CONF_BUCKETS)
    ]

    # ⑤ 日趋势：SQL 只返回有记录的日子，Python 补零到完整窗口，
    #    否则前端画柱状/折线时会出现「跳日期」的错觉。
    cur.execute(
        "SELECT date(r.created_at) AS d, COUNT(*) AS n " + scope + " GROUP BY d",
        (user_id,),
    )
    daily_map = {r["d"]: r["n"] for r in cur.fetchall()}
    daily = []
    for back in range(window - 1, -1, -1):
        cur.execute("SELECT date('now','localtime',?) AS d", (f"-{back} days",))
        day = cur.fetchone()["d"]
        daily.append({"date": day, "count": daily_map.get(day, 0)})

    # ⑥ 人工核对结论（2026-10-29 新增）
    #    准确率的分母必须是「已核对条数」，不是 total —— 只有 20% 的记录被人看过时，
    #    拿 total 当分母会得到一个既不是模型精度、也不是人工精度的数字。
    #    未核对条数单独返回，让前端能显式说明「样本量还小」。
    cur.execute(
        "SELECT COALESCE(SUM(CASE WHEN r.verdict = 'correct' THEN 1 ELSE 0 END), 0) AS correct,"
        "       COALESCE(SUM(CASE WHEN r.verdict = 'wrong' THEN 1 ELSE 0 END), 0) AS wrong,"
        # 「判错且顺手给了正确答案」的条数：这才是可回流训练的那部分样本，
        # 光有 wrong 只是「知道错了」，不知道对的是什么，标注还得重来一遍
        "       COALESCE(SUM(CASE WHEN r.ground_truth_class_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS with_gt "
        + scope,
        (user_id,),
    )
    v = dict(cur.fetchone())
    labeled = (v["correct"] or 0) + (v["wrong"] or 0)
    verdict = {
        "labeled": labeled,
        "correct": v["correct"] or 0,
        "wrong": v["wrong"] or 0,
        "with_ground_truth": v["with_gt"] or 0,
        "unlabeled": total - labeled,
        # 没有核对记录时返回 None 而不是 0：0% 会被读成「全错」，
        # None 前端才好显示「暂无核对」
        "accuracy": round((v["correct"] or 0) / labeled, 4) if labeled else None,
    }

    conn.close()

    return {
        "days": window,
        "total": total,
        "avg_confidence": round(head["avg_conf"], 4),
        "need_review": head["need_review"],
        "healthy": head["healthy"],
        "diseased": total - head["healthy"],
        "crops": crops,
        "diseases": diseases,
        "confidence_buckets": confidence_buckets,
        "daily": daily,
        "verdict": verdict,
        # 复用已有的 OOD 监控聚合，避免两处各写一份判据统计而口径漂移
        "ood": get_ood_stats(user_id, days=window),
    }


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
