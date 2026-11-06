# -*- coding: utf-8 -*-
"""
演示数据生成：为指定用户批量补齐历史识别记录，让历史页的图表有内容可看。

为什么需要它：
    新库/新账号只有零星几条记录时，作物分布图每根柱子都是 1、置信度直方图只有
    一个桶、日趋势几乎是空的 —— 图表反而暴露"数据不足"，答辩演示会很难看。
    本脚本只在【本地演示库】使用，不参与线上逻辑。

设计要点：
  1. **幂等**：默认行为是「补齐到目标条数」，记录数已够就跳过，可重复运行。
     要重置必须显式 --reset --yes，且只删该用户自己的记录。
  2. **图片复用**：从 data/uploads 现有文件里随机取 image_name，不复制新文件，
     既不涨磁盘，缩略图又是真实存在可渲染的。
  3. **分布贴近真实**：番茄权重最高（真实场景番茄拍摄最多）、置信度偏高端但
     保留约 5% 低于复核线、时间上近 7 天更密集。

用法（Windows CMD，整条复制）：
    X:\\Anconda3\\python.exe scripts\\seed_demo_records.py
    X:\\Anconda3\\python.exe scripts\\seed_demo_records.py --user demo --count 180 --days 30
    X:\\Anconda3\\python.exe scripts\\seed_demo_records.py --user demo --reset --yes

前置：后端至少启动过一次（data/app.db 已建表、知识库已导入）。
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_DIRS, UPLOAD_DIR  # noqa: E402
from src.api import db  # noqa: E402

# 作物权重：番茄在真实使用里拍得最多（也是病害类最多的作物），
# 苹果/葡萄/玉米次之，蓝莓/树莓/大豆这类只有健康类的作物权重最低。
CROP_WEIGHT = {
    "Tomato": 6.0,
    "Apple": 3.0,
    "Grape": 2.0,
    "Corn": 2.0,
    "Potato": 1.6,
    "Pepper": 1.2,
    "Peach": 1.0,
    "Cherry": 0.6,
    "Strawberry": 0.6,
    "Squash": 0.5,
    "Blueberry": 0.4,
    "Raspberry": 0.3,
    "Soybean": 0.3,
    "Orange": 0.3,
}


def crop_prefix(class_dir):
    """从文件夹名取作物前缀，如 'Corn_(maize)___X' -> 'Corn'。"""
    return class_dir.split("___")[0].split("_(")[0]


def build_class_pool():
    """构造加权类别池：(class_id, weight)。"""
    pool = []
    for idx, name in enumerate(CLASS_DIRS):
        pool.append((idx, CROP_WEIGHT.get(crop_prefix(name), 0.3)))
    return pool


def pick_class(pool):
    """按权重随机选一个类别 id。"""
    total = sum(w for _, w in pool)
    r = random.uniform(0, total)
    acc = 0.0
    for idx, w in pool:
        acc += w
        if r <= acc:
            return idx
    return pool[-1][0]


def pick_confidence():
    """置信度三段分布：主体高分、少量中等、约 5% 低于 0.5 复核线。"""
    r = random.random()
    if r < 0.05:
        return round(random.uniform(0.32, 0.5), 4)      # 需人工复核
    if r < 0.16:
        return round(random.uniform(0.5, 0.85), 4)      # 中等
    return round(random.uniform(0.85, 0.995), 4)        # 高置信


def pick_created_at(days):
    """最近 days 天的随机时刻；近 7 天权重更高（模拟答辩前集中使用）。"""
    now = datetime.now()
    weights = []
    for back in range(days):
        weights.append(3.0 if back < 7 else 1.0)
    # 按权重抽一个「几天前」
    total = sum(weights)
    r = random.uniform(0, total)
    acc = 0.0
    back = 0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            back = i
            break
    day = now - timedelta(days=back)
    moment = day.replace(
        hour=random.randint(7, 20),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )
    # 不允许出现「未来时间」：back=0 时是今天，随机小时可能晚于当前时刻，
    # 会让日趋势图画出超过今天的数据、列表按时间倒序也会怪。钳到当前时刻。
    if moment > now:
        moment = now
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def pick_ood_snapshot():
    """叶片判据快照：95% 正常放行，5% 判为非叶片（用于 OOD 监控卡有数据）。"""
    if random.random() < 0.05:
        return {
            "leaf_prob": round(random.uniform(0.03, 0.3), 4),
            "leaf_prob_source": "head",
            "mean_chroma": round(random.uniform(3, 28), 1),
            "likely_not_leaf": 1,
        }
    return {
        "leaf_prob": round(random.uniform(0.7, 0.995), 4),
        "leaf_prob_source": "head",
        "mean_chroma": round(random.uniform(35, 95), 1),
        "likely_not_leaf": 0,
    }


def list_upload_images():
    """取 data/uploads 下现有图片名，用于复用为记录的 image_name。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not UPLOAD_DIR.exists():
        return []
    names = [p.name for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in exts]
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description="为演示账号补齐历史识别记录")
    ap.add_argument("--user", default="demo", help="用户名（默认 demo）")
    ap.add_argument("--count", type=int, default=180, help="目标记录总数（默认 180）")
    ap.add_argument("--days", type=int, default=30, help="时间跨度天数（默认 30）")
    ap.add_argument("--reset", action="store_true", help="先删除该用户全部记录")
    ap.add_argument("--yes", action="store_true", help="配合 --reset 使用，跳过确认")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不写库")
    args = ap.parse_args()

    conn = db.get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE username=?", (args.user,))
    user = cur.fetchone()
    if not user:
        conn.close()
        print(f"[错误] 未找到用户 {args.user!r}。现有用户：")
        for r in db.get_conn().cursor().execute("SELECT username FROM users"):
            print("   ", r["username"])
        return 1
    uid = user["id"]

    cur.execute("SELECT COUNT(*) AS c FROM recognition_records WHERE user_id=?", (uid,))
    existing = cur.fetchone()["c"]

    if args.reset:
        if not args.yes:
            conn.close()
            print(f"[中止] --reset 将删除用户 {args.user}(id={uid}) 的 {existing} 条记录。")
            print("       确认无误请追加 --yes 重跑。")
            return 1
        if not args.dry_run:
            cur.execute("DELETE FROM recognition_records WHERE user_id=?", (uid,))
            conn.commit()
        print(f"[重置] 已删除 {existing} 条记录")
        existing = 0

    need = args.count - existing
    if need <= 0:
        conn.close()
        print(f"[跳过] 用户 {args.user} 已有 {existing} 条记录（目标 {args.count}），无需补齐。")
        return 0

    images = list_upload_images()
    if not images:
        print("[警告] data/uploads 为空，记录将不含缩略图（列表显示占位图标）。")
    pool = build_class_pool()

    rows = []
    for _ in range(need):
        cid = pick_class(pool)
        conf = pick_confidence()
        ood = pick_ood_snapshot()
        rows.append(
            (
                uid,
                random.choice(images) if images else f"demo_{random.randint(1, 9999)}.jpg",
                cid,
                conf,
                None,  # top3_json：列表与图表都不读，留空避免伪造不存在的信息
                ood["leaf_prob"],
                ood["leaf_prob_source"],
                ood["mean_chroma"],
                ood["likely_not_leaf"],
                pick_created_at(args.days),
            )
        )

    if args.dry_run:
        conn.close()
        print(f"[dry-run] 将插入 {need} 条（用户 {args.user} id={uid}，跨度 {args.days} 天）")
        return 0

    cur.executemany(
        """INSERT INTO recognition_records
           (user_id, image_name, pred_class_id, confidence, top3_json,
            leaf_prob, leaf_prob_source, mean_chroma, likely_not_leaf, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()

    # 插入后回读统计，作为结果自证（不是凭猜测报告成功）
    cur.execute(
        "SELECT COUNT(*) AS c, COUNT(DISTINCT pred_class_id) AS k,"
        "       MIN(created_at) AS lo, MAX(created_at) AS hi,"
        "       AVG(confidence) AS avg_c"
        "  FROM recognition_records WHERE user_id=?",
        (uid,),
    )
    s = cur.fetchone()
    conn.close()

    print(f"[完成] 用户 {args.user}（id={uid}）新增 {need} 条，现共 {s['c']} 条")
    print(f"       覆盖 {s['k']} 个类别，时间 {s['lo']} ~ {s['hi']}")
    print(f"       平均置信度 {s['avg_c']:.4f}")
    print("       打开网站「识别历史」页即可看到图表。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
