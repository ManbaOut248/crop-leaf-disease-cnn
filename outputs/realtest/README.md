# 真实场景测试图片（PlantDoc）

本目录用于**真实田间场景**的测试与阈值标定，与 `data/split/`（PlantVillage 实验室图）互补。

## 来源与许可

| 项 | 内容 |
|---|---|
| 数据集 | **PlantDoc** — *PlantDoc: A Dataset for Visual Plant Disease Detection*（CoDS-COMAD 2020） |
| 仓库 | https://github.com/pratikkayal/PlantDoc-Dataset |
| 类别 | `train/Tomato leaf late blight`（番茄晚疫病） |
| 许可 | **CC BY 4.0** — 使用时必须署名原作者 |
| 署名 | Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, 249–253. |

> 与 PlantVillage 的关键差异：PlantDoc 是**从互联网爬取的真实照片**（复杂背景、真实光照、多角度），
> 而 PlantVillage 是实验室黑底居中拍摄。因此本目录图片用于检验模型**在真实场景下的表现落差**。

## 下载方式（本机实测）

GitHub 直连不通，经 `ghproxy.net` 代理 raw 文件下载：

```bash
BASE="https://ghproxy.net/https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/train/Tomato%20leaf%20late%20blight"
curl -sL -o "outputs/realtest/600.jpg" "$BASE/600.jpg"
```

## 测试结果（2026-10-16，首轮：番茄晚疫病 13 张有效）

| 指标 | 真实田间（本目录） | 实验室（PlantVillage 测试集） |
|---|---:|---:|
| 识别正确率 | **84.6%**（11/13 张有效） | 99.27% |
| 被判「非叶片」 | **53.8%**（7/13） | 6.2% |

**关键发现**：OOD 误杀率在真实场景下涨了近 9 倍（6.2% → 53.8%），
有 7 张**识别正确却被拦**（其中一张置信度高达 0.9772）。
印证了 `docs/08_OOD拒识评估.md` 的预警：**阈值是基于实验室图标定的，真实场景必须重标**。

> 注：色度判据未参与任何一次误杀（这些图色度 31–67，远离阈值 8）——
> 误杀全部由 MSP 判据触发。说明**色度判据是安全的，MSP 判据需要针对真实场景重新标定**。

## 后续实验（同日，用本目录扩充后的 127 张 / 3 类作物）

为 B 与 C1-a 两轮实验扩了数据：下游按 `tomato_late_blight`(37) / `corn_rust`(57) /
`grape_black_rot`(34) 三个子目录组织，共 **127 张真实田间照片**。

| 实验 | 做法 | 结果 |
|---|---|---|
| **B：重标阈值** | 用 127 张真实叶片作「分布内」重扫 MSP 操作点 | ❌ 真实叶片 MSP 中位 0.805 vs 非叶片 0.672 —— **几乎完全重叠**；最佳操作点（0.80）也是"拦下 10 张非叶片就误伤 8 张真叶片"。**任何阈值都不可用** |
| **C1-a：AdaBN** | 只重估 20 个 BN 层的统计量（纯前向、无梯度、无标签，55 秒） | ❌ AUROC **0.5864 → 0.5045**，反而更差；两类 MSP 中位从 0.80/0.67 挤成 0.71/0.72 |

**根因**：`AUROC 对单调变换不变`。BN 重估与温度缩放都近似单调映射，
**不改变样本间排序**；而 OOD 分离完全依赖排序。基线 AUROC 仅 0.586
（实验室域 0.94）说明输出端已无可用排序信息。
→ 要改善必须动**卷积核或分类头**（C1-b / C2），且需先把真实数据扩充到覆盖更多作物，
否则微调会偏科。

交互演示：`outputs/bn_adapt_demo.html`（可拖动阈值查看两类重叠程度）。

## 复现

```bash
# ① 下载数据（本目录图片不入库，可一键重建）
py -3.12 scripts/fetch_plantdoc.py

# ② 单类识别测试
py -3.12 backups/_realtest.py

# ③ 阈值重标定（实验 B）
py -3.12 scripts/tune_ood_threshold.py --in-dir outputs/realtest

# ④ BN 重估实验（C1-a）+ 生成交互演示 HTML
py -3.12 backups/_gen_bn_demo.py
```
