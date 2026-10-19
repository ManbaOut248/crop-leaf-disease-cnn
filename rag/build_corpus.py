# -*- coding: utf-8 -*-
"""构建 RAG 语料库（任务书 5.2：语料库 + 用于构建 RAG 助手的提示词）。

产物
---------------------------------------------------------------
    rag/corpus/corpus.jsonl    每行一个知识块（JSON）
    rag/corpus/stats.json      来源统计（便于内容审查）

三类来源
---------------------------------------------------------------
1. **病害知识库**（`src/data/disease_knowledge.json`，38 类）
   每个病害拆成两个块：「症状与病因」「防治措施」。
   为什么拆：用户提问通常只涉及其中一面（"这病怎么防" vs "这是什么病"），
   整条塞进一个块会稀释相似度；拆开后检索命中更准，引用也更聚焦。

2. **项目文档**（`docs/*.md`）
   按 Markdown 标题层级切块，单块上限 800 字。
   作用是让助手能回答"系统能力边界""为什么会有非叶片提示""数据从哪来"
   这类**关于系统本身**的问题，而不是只会背病害知识。

3. **常见问题 FAQ**（本文件内手写）
   用户实际最常问、但文档里分散的内容，直接写成问答块。

块结构
---------------------------------------------------------------
    {
      "id":     "disease:Apple___Apple_scab#symptom",   # 全局唯一，也用于引用
      "source": "disease_knowledge.json",               # 来源标识
      "kind":   "disease" | "doc" | "faq",
      "title":  "苹果黑星病（苹果）· 症状与病因",
      "text":   "……",
      "meta":   {"class_id": 0, "crop": "苹果"}
    }

用法
    py -3.12 rag/build_corpus.py
"""

import json
import os
import re
from pathlib import Path

# 构建语料需要 import src.config 拿 CLASS_DIRS（保证 class_id 与模型输出对齐）。
# 但 config 在缺少 JWT_SECRET 时会发"随机密钥"告警 —— 这里本就不签发 token，
# 预置一个占位值即可让构建日志保持干净。
os.environ.setdefault("JWT_SECRET", "corpus-build-only-not-a-real-secret")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "rag" / "corpus"
KNOWLEDGE = ROOT / "src" / "data" / "disease_knowledge.json"
DOCS_DIR = ROOT / "docs"

MAX_DOC_CHUNK = 800        # 单块字符上限（约 500~600 token）

# ===================== 1) 病害知识库 =====================
def build_disease_chunks():
    data = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    # 与模型 class_id 对齐（CLASS_DIRS 顺序即模型输出索引）
    import sys
    sys.path.insert(0, str(ROOT))
    from src.config import CLASS_DIRS

    chunks = []
    for cid, en in enumerate(CLASS_DIRS):
        info = data.get(en)
        if not info:
            continue
        zh = info.get("name_zh") or en
        crop = info.get("crop") or ""
        # name_zh 多数已含作物名（如「番茄叶霉病」），再拼一次会出现「番茄番茄…」
        label = zh if (crop and crop in zh) else f"{crop}{zh}"
        is_healthy = en.endswith("healthy")

        if is_healthy:
            text = (f"{label}指未表现明显病害症状的健康植株。"
                    f"判断要点：叶面平整无病斑、无霉层、无萎蔫或黄化；叶色均匀，"
                    f"叶脉清晰。若出现斑点、霉层、卷曲或大面积黄化，需考虑病害或虫害。")
            chunks.append({
                "id": f"disease:{en}#healthy",
                "source": "disease_knowledge.json",
                "kind": "disease",
                "title": f"{label} · 判断要点",
                "text": text,
                "meta": {"class_id": cid, "crop": crop, "healthy": True},
            })
            continue

        sym = " ".join(x for x in (info.get("symptom"), info.get("cause")) if x)
        chunks.append({
            "id": f"disease:{en}#symptom",
            "source": "disease_knowledge.json",
            "kind": "disease",
            "title": f"{label} · 症状与病因",
            "text": sym,
            "meta": {"class_id": cid, "crop": crop, "healthy": False},
        })

        prev = " ".join(x for x in (info.get("prevention"), info.get("treatment")) if x)
        chunks.append({
            "id": f"disease:{en}#control",
            "source": "disease_knowledge.json",
            "kind": "disease",
            "title": f"{label} · 防治与用药",
            "text": prev,
            "meta": {"class_id": cid, "crop": crop, "healthy": False},
        })
    return chunks


# ===================== 2) 项目文档 =====================
HEAD_RE = re.compile(r"^(#{1,4})\s+(.*)$")
CODE_FENCE = re.compile(r"```.*?```", re.S)


def split_markdown(md, doc_name):
    """按标题切块。返回 [(标题路径, 正文)]。

    代码块会被剥离：RAG 语料面向自然语言问答，命令行片段会引入大量噪声 token，
    且容易让检索把"部署命令"匹配到"病害症状"这类问题上去。
    """
    lines = md.splitlines()
    chunks, cur_head, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            # 过长时按段落再切
            for part in _slice(body, MAX_DOC_CHUNK):
                chunks.append((" / ".join(cur_head) or doc_name, part))
        buf.clear()

    for line in lines:
        m = HEAD_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            cur_head = cur_head[:level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()

    # 剥离代码块后再做一次清洗
    return [(t, CODE_FENCE.sub("[代码示例略]", b).strip()) for t, b in chunks]


def _slice(text, limit):
    """按段落把长文本切成 <= limit 的块，避免切断句子。"""
    if len(text) <= limit:
        return [text]
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= limit:
            cur = f"{cur}\n\n{p}".strip()
        else:
            if cur:
                out.append(cur)
            # 单段仍超长：硬切
            while len(p) > limit:
                out.append(p[:limit])
                p = p[limit:]
            cur = p
    if cur:
        out.append(cur)
    return out


def _is_table_heavy(text, threshold=0.6):
    """判断块是否以 Markdown 表格为主。

    为什么剔除：表格切块后是一串 `| 32 | 番茄 | 番茄叶霉病 | 0 |` 形式的碎片，
    既读不通，又因为密集出现病名/作物名而**把真正的知识块挤出结果**
    （实测"番茄叶子有褐色斑点"的第二名被数据库表结构抢走）。
    这类内容适合结构化查询，不适合作为自然语言问答语料。
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.lstrip().startswith("|"))
    return table_lines / len(lines) >= threshold


def build_doc_chunks():
    chunks = []
    for p in sorted(DOCS_DIR.glob("*.md")):
        for i, (title, body) in enumerate(split_markdown(p.read_text(encoding="utf-8"), p.stem)):
            if len(body) < 30:          # 太短的块（如只有一句标题下文）没有检索价值
                continue
            if _is_table_heavy(body):   # 纯表格碎片会把知识块挤出结果
                continue
            chunks.append({
                "id": f"doc:{p.stem}#{i}",
                "source": f"docs/{p.name}",
                "kind": "doc",
                "title": title,
                "text": body,
                "meta": {"file": p.name, "index": i},
            })
    return chunks


# ===================== 3) FAQ =====================
FAQ = [
    ("怎么判断该上传什么样的照片？",
     "上传单一叶片的清晰特写，背景尽量干净、光线均匀。整株、多叶堆叠、远距离拍摄都会降低准确率。"
     "系统对非叶片图片（插画、截图、图表、文档）会给出提示，因为模型是 38 类闭集分类，"
     "没有「以上都不是」这个选项，任何图片都会被归到某个病害。"),
    ("为什么结果显示「>99.9%」而不是 100%？",
     "四舍五入会把 0.9996 显示成 100.0%，暗示模型绝对正确，这与「辅助识别」的定位矛盾。"
     "因此对超出显示精度的值如实标注：>99.9% 表示极高但不等于确定；<0.1% 表示极低但不等于零。"),
    ("提示「这张图片可能不是叶片照片」是什么意思？",
     "系统用一个独立的「叶片/非叶片」检测头判断输入是否是叶片特写。"
     "命中时会提示重新上传，但仍显示识别结果供参考——因为误判真实叶片的代价（重拍一张）"
     "小于漏放非叶片图片的代价（可能据此用药）。"),
    ("提示「识别结果可能不可靠」和「不是叶片照片」有什么区别？",
     "前者来自低置信度判据，多出现在背景杂乱的田间照片上，此时图片确实是叶片，只是与训练样本差异大；"
     "后者来自灰阶判据（黑白图/线稿），系统可以确定它不是正常彩色叶片照片。"),
    ("模型准确率 99% 为什么田里拍的照片经常认不准？",
     "99.27% 是在 PlantVillage 实验室数据集上测得的：黑底、居中、单一叶片。"
     "真实田间背景复杂、光照多变、常有遮挡，属于分布外数据。"
     "实测 651 张真实田间照片的病害分类准确率只有 15~35%，这是已知局限，不能作为用药依据。"),
    ("系统能识别哪些作物？",
     "14 种作物共 38 个类别（26 类病害 + 12 类健康）：苹果、蓝莓、樱桃、玉米、葡萄、柑橘、桃、"
     "甜椒、马铃薯、树莓、大豆、南瓜、草莓、番茄。不在训练集内的作物或病害无法识别。"),
    ("历史记录能导出吗？",
     "登录后可在识别历史页导出 CSV 与 JSON 两种格式；导出内容包含记录 ID、预测类别、置信度与时间。"),
    ("为什么有些黑白照片识别不了？",
     "病害识别依赖颜色信息（病斑常呈黄、褐、橙等），黑白照片丢失了这部分线索。"
     "系统允许灰度图识别但会给出提醒，建议上传彩色照片。"),
    ("识别结果能直接用来决定打什么药吗？",
     "不能。本系统仅作辅助识别与科普工具，不能替代植保人员的现场诊断。"
     "具体药剂、剂量与安全间隔期应遵循当地植保部门指导与农药标签，"
     "并注意不同作物、不同生育期的用药差异。"),
    ("系统会保存我上传的图片吗？",
     "登录用户的识别记录会保存图片文件名与识别结果，用于历史回看；匿名上传的文件不写入记录。"
     "生产部署应启用 HTTPS 传输与数据库字段加密，并定期清理无记录引用的孤儿文件。"),
    ("支持哪些浏览器？",
     "前端为响应式 Web 应用，支持 Chrome / Edge / Firefox / Safari 最新两个大版本。"
     "已实测 Chromium 内核在 1440 / 1024 / 390 三种视口下无布局溢出。"),
    ("本地怎么启动这个系统？",
     "两个终端分别启动：后端 `py -3.12 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 "
     "--no-proxy-headers`，前端 `cd src/web && npm run dev`，然后访问 http://127.0.0.1:5173。"
     "`--no-proxy-headers` 不能省略，否则接口限流可被伪造的 X-Forwarded-For 绕过。"),
]


def build_faq_chunks():
    return [{
        "id": f"faq:{i:02d}",
        "source": "faq",
        "kind": "faq",
        "title": q,
        "text": a,
        "meta": {"index": i},
    } for i, (q, a) in enumerate(FAQ)]


# ===================== 主流程 =====================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = {
        "disease": build_disease_chunks(),
        "doc": build_doc_chunks(),
        "faq": build_faq_chunks(),
    }
    all_chunks = [c for v in parts.values() for c in v]

    # 校验：id 唯一、无空文本
    ids = [c["id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "chunk id 重复"
    for c in all_chunks:
        assert c["text"].strip(), f"空块: {c['id']}"

    out = OUT_DIR / "corpus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_chunks),
        "by_kind": {k: len(v) for k, v in parts.items()},
        "avg_chars": round(sum(len(c["text"]) for c in all_chunks) / len(all_chunks), 1),
        "max_chars": max(len(c["text"]) for c in all_chunks),
        "sources": sorted({c["source"] for c in all_chunks}),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print("RAG 语料库构建完成")
    print("=" * 66)
    print(f"  总块数   {stats['total']}")
    for k, v in stats["by_kind"].items():
        print(f"    {k:<8} {v}")
    print(f"  平均长度 {stats['avg_chars']} 字   最长 {stats['max_chars']} 字")
    print(f"  产物     {out.relative_to(ROOT)}  ({out.stat().st_size/1024:.1f} KB)")
