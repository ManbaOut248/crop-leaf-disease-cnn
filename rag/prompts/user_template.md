# 用户消息模板（User Template）

> 与 `system_prompt.md` 配合使用。变量用 `{{ }}` 标记，接入时做字符串替换。

---

## 模板 A：标准问答（推荐）

```
【检索片段】
{{context}}

【用户问题】
{{question}}
```

`{{context}}` 由检索结果拼接，**每条片段必须带序号**，否则模型无法正确标注引用：

```
[1] （来源：disease_knowledge.json · 番茄叶霉病 · 症状与病因）
叶片正面生淡黄色斑，背面生灰紫色霉层；严重时叶片卷曲枯死。……

[2] （来源：faq · 为什么结果显示 >99.9% 而不是 100%？）
四舍五入会把 0.9996 显示成 100.0%……

[3] （来源：docs/11_域适应与OOD检测实验.md · 局限）
真实田间 651 张照片的病害分类准确率仅 15~35%……
```

拼接规则（建议直接复用 `rag/retrieve.py --json` 的输出）：

| 规则 | 说明 |
|---|---|
| 条数 | 3~5 条；太少容易缺依据，太多会稀释注意力并抬高成本 |
| 相似度阈值 | 低于 0.05 的片段宁可不给，让模型走"资料未涉及"分支 |
| 去重 | 同一 `title` 只保留一条，避免重复占位 |
| 顺序 | 按相似度降序，编号即引用编号 |

---

## 模板 B：带识别结果的多模态追问

用户刚上传一张叶片照片、拿到系统结果后继续提问时使用。

```
【系统识别结果】
- Top-1：{{top1_name}}（置信度 {{top1_conf}}）
- Top-3：{{top3_summary}}
- 是否触发非叶片提示：{{likely_not_leaf}}
- 提示来源：{{leaf_prob_source}}

【检索片段】
{{context}}

【用户问题】
{{question}}

要求：若用户问题与识别结果相关，请结合两者作答；
若识别结果可能不可靠（置信度低或触发非叶片提示），须先提醒用户这一点再回答。
```

> 为什么要把识别结果一并给出：用户问"这个病怎么治"时，指代的是屏幕上那个结果，
> 而不是一句话描述。缺了这块上下文，助手容易答非所问。

---

## 模板 C：无检索命中（降级）

检索为空或全部低于阈值时，**不要调用大模型硬答**，直接返回固定话术：

```
资料中没有找到与「{{question}}」相关的内容。

本系统覆盖 14 种作物、38 个类别（26 类病害 + 12 类健康）。
你可以：
1. 上传叶片照片，由识别模型给出候选类别；
2. 换个说法描述症状（部位、颜色、形状、是否有霉层）；
3. 联系当地农技推广机构现场诊断。
```

> 这一条比"让模型自由发挥"重要得多：知识库没有的内容，
> 模型一旦开始补，就会编造药剂和剂量，这在农业场景是**实质性风险**。

---

## 组装示例（Python 伪代码）

```python
from rag.retrieve import Retriever
import pathlib

SYSTEM = pathlib.Path("rag/prompts/system_prompt.md").read_text(encoding="utf-8")

# 模板 A 的正文（与上文保持一致）
TPL_A = """【检索片段】
{{context}}

【用户问题】
{{question}}"""

def build_messages(question, k=4, min_score=0.05):
    hits = [h for h in Retriever().search(question, k) if h["score"] >= min_score]
    if not hits:
        return None                      # 走模板 C 固定话术，不调用模型

    ctx = "\n\n".join(
        f"[{i}] （来源：{h['source']} · {h['title']}）\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    user = TPL_A.replace("{{context}}", ctx).replace("{{question}}", question)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
```

接入任意 OpenAI 兼容接口（DeepSeek / 通义 / 本地 vLLM 等）即可，
`messages` 结构通用。**模型选择不影响本目录的语料与提示词设计。**
