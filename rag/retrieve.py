# -*- coding: utf-8 -*-
"""RAG 检索 demo（离线可跑，不依赖任何外部 API）。

用途
---------------------------------------------------------------
这是"语料库 + 提示词"的最小可用验证：给出问题，返回最相关的知识块，
用来直观判断**语料切分是否合理、召回是否命中**。
真正接大模型时，检索这一层可以整体替换（向量库 / BM25 / 混合检索），
但 corpus.jsonl 的 schema 与 system_prompt 的用法保持不变。

为什么用字符级 TF-IDF 而不是向量模型
---------------------------------------------------------------
1. 离线可用：本机拉不到 embedding 模型（GitHub/外网受限），
   而 sklearn 已在环境内，零下载。
2. 中文不需要分词：`analyzer="char_wb", ngram_range=(2,3)` 直接以 2~3 字滑窗建索引，
   避免引入 jieba 等额外依赖与词典维护。
3. 作为**基线**足够：它的召回失败恰恰说明语料缺内容或切分不合理，
   先修语料，再上向量检索，顺序不能反。

用法
    py -3.12 rag/retrieve.py "番茄叶子有褐色斑点是什么病"
    py -3.12 rag/retrieve.py "为什么田里拍的照片认不准" -k 5
    py -3.12 rag/retrieve.py "系统怎么启动" --json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "rag" / "corpus" / "corpus.jsonl"


def load_corpus(path=CORPUS):
    if not path.exists():
        raise SystemExit(f"语料不存在：{path}，请先运行 py -3.12 rag/build_corpus.py")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Retriever:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else load_corpus()
        # 标题重复一次：标题往往就是用户提问的浓缩，加权后明显提升命中率
        docs = [f"{r['title']} {r['title']} {r['text']}" for r in self.rows]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        self.mat = self.vec.fit_transform(docs)

    def search(self, query, k=3):
        q = self.vec.transform([query])
        # 稀疏矩阵乘法得到余弦相似度（TfidfVectorizer 已做 L2 归一化）
        scores = (self.mat @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [{
            "score": round(float(scores[i]), 4),
            "id": self.rows[i]["id"],
            "kind": self.rows[i]["kind"],
            "source": self.rows[i]["source"],
            "title": self.rows[i]["title"],
            "text": self.rows[i]["text"],
        } for i in top if scores[i] > 0]


def main():
    ap = argparse.ArgumentParser(description="RAG 检索 demo（TF-IDF 基线）")
    ap.add_argument("query", help="用户问题")
    ap.add_argument("-k", type=int, default=3, help="返回条数，默认 3")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出（便于接大模型）")
    args = ap.parse_args()

    r = Retriever()
    hits = r.search(args.query, args.k)

    if args.json:
        print(json.dumps({"query": args.query, "hits": hits}, ensure_ascii=False, indent=2))
        return

    print("=" * 76)
    print(f"问题：{args.query}")
    print(f"语料：{len(r.rows)} 块")
    print("=" * 76)
    if not hits:
        print("未检索到相关内容 —— 说明语料缺这个主题，应补充而不是硬答。")
        sys.exit(1)
    for i, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        print(f"\n[{i}] 相似度 {h['score']:.4f}  ({h['kind']} · {h['source']})")
        print(f"    {h['title']}")
        print(f"    {snippet}")
    print("\n提示：把上面命中的正文拼进 system_prompt 的【检索片段】，"
          "即可交给大模型作答并标注引用编号。")


if __name__ == "__main__":
    main()
