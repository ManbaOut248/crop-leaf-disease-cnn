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

## 测试结果（14 张，2026-10-16）

| 指标 | 真实田间（本目录） | 实验室（PlantVillage 测试集） |
|---|---:|---:|
| 识别正确率 | **84.6%**（11/13 张有效） | 99.27% |
| 被判「非叶片」 | **53.8%**（7/13） | 6.2% |

**关键发现**：OOD 误杀率在真实场景下涨了近 9 倍（6.2% → 53.8%），
有 7 张**识别正确却被拦**（其中一张置信度高达 0.9772）。
印证了 `docs/08_OOD拒识评估.md` 的预警：**阈值是基于实验室图标定的，真实场景必须重标**。

> 注：色度判据未参与任何一次误杀（这些图色度 31–67，远离阈值 8）——
> 误杀全部由 MSP 判据触发。说明**色度判据是安全的，MSP 判据需要针对真实场景重新标定**。

## 复现

```bash
py -3.12 backups/_realtest.py
```
