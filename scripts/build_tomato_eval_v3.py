# -*- coding: utf-8 -*-
"""P0：把番茄真实域评估集从未泄漏 186 张扩到 400+ 张（docs/22 第八节遗留项）。

数据源（P0 选定）
---------------------------------------------------------------
Mendeley `zfv4jj7855`（CC BY-SA 4.0，10 个番茄类与本项目 38 类一一对应，
且含 Target_Spot —— 真实域训练 0 样本、评估 0 样本的关键短板类）。
来源页: https://data.mendeley.com/datasets/zfv4jj7855

防泄漏（三层，缺一不可）
---------------------------------------------------------------
1. md5 精确重复 —— 对 mixed_v2（train/val/test）与 realtest_v2 全量比对；
2. 感知哈希（aHash 64bit）近重复 —— 外部集若是 PV 换皮/重压缩，
   md5 会变但像素结构不变，aHash + 镜像 aHash 能兜住；
3. 集内近重复 —— 新增图之间互查，防止同图多份进入评估集。

关键红线
---------------------------------------------------------------
只要一张新增图与「部署权重见过的任何图」近重复，就必须剔除；
aHash 汉明距离阈值取 4（64bit 下经验值，宁可错杀）。
若存活率过低（说明外部集是 PV 换皮），脚本会明确报出，不硬凑数量。

用法
    python -m scripts.build_tomato_eval_v3          # 构建并落盘 v3 索引
    python -m scripts.build_tomato_eval_v3 --dry-run  # 只统计不落盘
"""

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASS_DIRS  # noqa: E402

ZIP_PATH = ROOT / "data" / "_downloads" / "mendeley_zfv4jj7855" / "TomatoDataset.zip"
EXTRACT_DIR = ROOT / "data" / "_downloads" / "mendeley_zfv4jj7855" / "extracted"
MATERIAL_DIR = ROOT / "data" / "realtest_extra" / "mendeley_zfv4jj7855"

INDEX_DIR = ROOT / "outputs" / "index"
MIXED_V2 = INDEX_DIR / "dataset_index_mixed_v2.json"
REALTEST_V2 = INDEX_DIR / "dataset_index_realtest_v2.json"
V3_INDEX = INDEX_DIR / "dataset_index_realtest_v3.json"
V3_MANIFEST = INDEX_DIR / "tomato_eval_v3_manifest.json"
HASH_CACHE = INDEX_DIR / "ahash_cache_v3.json"

# 外部文件夹名 → 38 类类名
FOLDER_TO_CLASS = {
    "Tomato_healthy": "Tomato___healthy",
    "Bacterial_spot": "Tomato___Bacterial_spot",
    "Early_blight": "Tomato___Early_blight",
    "Late_blight": "Tomato___Late_blight",
    "Leaf_Mold": "Tomato___Leaf_Mold",
    "Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Spider_mites_Two-spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Target_Spot": "Tomato___Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

PER_CLASS_CAP = 30      # 每类最多新增张数（10 类 × 30 = 上限 300，合计 186+300=486）
HAMMING_MAX = 4         # aHash 汉明距离阈值
MIN_SIDE = 64           # 最小边长


def md5_of(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def ahash(img, size=8):
    """64bit 平均哈希：8x8 灰度，大于均值记 1。"""
    g = img.convert("L").resize((size, size), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v > avg else 0)
    return bits


def mirror_hash(img):
    return ahash(img.transpose(Image.FLIP_LEFT_RIGHT))


def hamming(a, b):
    return bin(a ^ b).count("1")


def norm(p):
    return str(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ---------- 1) 解压 ----------
    if not ZIP_PATH.exists():
        raise SystemExit(f"❌ 找不到 {ZIP_PATH}，请先下载数据集 zip")
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True)
        print(f"解压 {ZIP_PATH.name} → {EXTRACT_DIR} ...")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)

    # ---------- 2) 收集候选图 ----------
    candidates = []  # (folder, path, class_name)
    for sub in sorted(EXTRACT_DIR.rglob("*")):
        if sub.is_file() and sub.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            rel = sub.relative_to(EXTRACT_DIR).parts
            folder = next((r for r in rel if r in FOLDER_TO_CLASS), None)
            if folder:
                candidates.append((folder, sub, FOLDER_TO_CLASS[folder]))
    print(f"候选图总数: {len(candidates)}")
    by_folder = defaultdict(int)
    for f, _, _ in candidates:
        by_folder[f] += 1
    for f, n in sorted(by_folder.items()):
        print(f"  {f:<42}{n:>6}")

    # ---------- 3) 参考哈希：见过的所有图 ----------
    ref_hashes = []          # [(ahash, ahash_mirror, tag)]
    md5_seen = set()
    for idx_path, tag in [(MIXED_V2, "mixed_v2"), (REALTEST_V2, "realtest_v2")]:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            for p, _ in idx.get(split, []):
                md5_seen.add(norm(p))
    print(f"参考路径（md5 比对用）: {len(md5_seen)}")

    cache = json.loads(HASH_CACHE.read_text(encoding="utf-8")) if HASH_CACHE.exists() else {}
    tids = {i for i, n in enumerate(CLASS_DIRS) if n.startswith("Tomato")}
    mixed = json.loads(MIXED_V2.read_text(encoding="utf-8"))
    rt = json.loads(REALTEST_V2.read_text(encoding="utf-8"))
    seen_items = [(p, l) for split in ("train", "val") for p, l in mixed.get(split, [])]
    seen_items += [(p, l) for split in ("train", "val", "test") for p, l in rt.get(split, [])]
    ref_imgs = [p for p, _ in seen_items
                if int(_) in tids or "real_mix" in norm(p) or "realtest" in norm(p)]
    # 只对「真实域 + 番茄」算感知哈希（PV 实验室图换皮可能性低且量大；
    # md5 层已覆盖全部参考图）
    ref_imgs = sorted({norm(p) for p in ref_imgs})
    print(f"感知哈希参考图（真实域+番茄）: {len(ref_imgs)}")

    new_hashes = [h for h, k in cache.items() if k == "ref"] if False else None  # 占位
    ref_list = []
    for p in ref_imgs:
        key = "ref::" + p
        if key in cache:
            h, hm = cache[key]
        else:
            try:
                im = Image.open(p).convert("RGB")
                h, hm = ahash(im), mirror_hash(im)
            except Exception:
                continue
            cache[key] = [h, hm]
        ref_list.append((h, hm))
    print(f"参考哈希就绪: {len(ref_list)}")

    # ---------- 4) 逐张审候选 ----------
    accepted = []
    stats = defaultdict(lambda: defaultdict(int))
    seen_new_hashes = []

    def too_close(h):
        for rh, rhm in ref_list:
            if hamming(h, rh) <= HAMMING_MAX or hamming(h, rhm) <= HAMMING_MAX:
                return True
        for oh, ohm in seen_new_hashes:
            if hamming(h, oh) <= HAMMING_MAX or hamming(h, ohm) <= HAMMING_MAX:
                return True
        return False

    for folder, path, cname in candidates:
        st = stats[cname]
        st["候选"] += 1
        cls_id = CLASS_DIRS.index(cname)
        if sum(1 for _, l in accepted if l == cls_id) >= PER_CLASS_CAP:
            st["超类内上限"] += 1
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.width < MIN_SIDE or im.height < MIN_SIDE:
                    st["尺寸过小"] += 1
                    continue
                h, hm = ahash(im), mirror_hash(im)
        except Exception:
            st["损坏"] += 1
            continue
        if md5_of(path) in _md5_of_refs(md5_seen):
            st["md5 命中"] += 1
            continue
        if too_close(h):
            st["感知近重复"] += 1
            continue
        seen_new_hashes.append((h, hm))
        accepted.append((str(path), cls_id, folder))
        st["接受"] += 1

    # ---------- 5) 汇总 / 落盘 ----------
    print("\n=== 逐类审核结果 ===")
    total_new = 0
    for cname in sorted(stats):
        st = stats[cname]
        total_new += st["接受"]
        print(f"  {cname:<46} 候选{st['候选']:>5} 接受{st['接受']:>4} "
              f"近重复{st['感知近重复']:>5} md5命中{st['md5命中']:>4} "
              f"小图{st['尺寸过小']:>4} 损坏{st['损坏']:>3} 超限{st['超类内上限']:>4}")
    print(f"\n新增合计: {total_new}（原番茄 186 → {186 + total_new}）")

    if total_new < 80:
        print("⚠️ 存活率过低 —— 该外部集疑似 PV 换皮，不建议入评估集。"
              "请改用真实田间来源（如 Mendeley bpfd9cns5g / 自采）。")

    if args.dry_run:
        return

    # 物化：复制进 data/realtest_extra/（索引路径要求稳定存在）
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)
    final_items = []
    for p, cls_id, folder in accepted:
        dst = MATERIAL_DIR / folder
        dst.mkdir(parents=True, exist_ok=True)
        d = dst / Path(p).name
        if not d.exists():
            d.write_bytes(Path(p).read_bytes())
        final_items.append([str(d), cls_id])

    rt_test = rt["test"] + final_items
    json.dump({"train": [], "val": [], "test": rt_test},
              V3_INDEX.open("w", encoding="utf-8"), ensure_ascii=False)
    json.dump({
        "source": "Mendeley zfv4jj7855 (CC BY-SA 4.0)",
        "source_url": "https://data.mendeley.com/datasets/zfv4jj7855",
        "leak_check": "md5 全量 + aHash64(+镜像) 汉明<=4 vs 真实域参考图 + 集内互查",
        "per_class_cap": PER_CLASS_CAP,
        "hamming_max": HAMMING_MAX,
        "n_new": total_new,
        "n_total_tomato": 186 + total_new,
        "stats": {k: dict(v) for k, v in stats.items()},
    }, V3_MANIFEST.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    HASH_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"\n产物:\n  {V3_INDEX}\n  {V3_MANIFEST}")


_md5_cache = None


def _md5_of_refs(path_set):
    """参考图 md5 集合（惰性计算并缓存）。注意：md5 只对「外部 zip 内文件」比对时
    才有意义 —— 参考图自己也要算 md5。这里在首次调用时批量计算真实域参考图。"""
    global _md5_cache
    if _md5_cache is None:
        _md5_cache = set()
        n = 0
        for p in path_set:
            fp = ROOT / p
            if fp.exists():
                try:
                    _md5_cache.add(md5_of(fp))
                    n += 1
                except Exception:
                    pass
        print(f"参考图 md5 计算完成: {n}")
    return _md5_cache


if __name__ == "__main__":
    main()
