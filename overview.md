# 真实域精准度提升 — 交付概览

> 日期：2026-10-20 | 方案：Plan A 混合微调 + 温度缩放校准

## 核心效果（同一评估集 data/real_mix_test，带作物先验）

| 指标 | 微调前 (best) | 微调后 (finetuned) | 变化 |
|---|---|---|---|
| **真实田间 Top-1（去重 225 张）** | 52.9% | **75.6%** | **+22.7pp** |
| 同上（目录 497 含重复） | 54.1% | 76.7% | +22.6pp |
| 实验室域 val_f1 | 99.27% | **99.33%** | 不退化 |
| 真实域裸 acc | — | 46.7% | — |

置信度校准（温度缩放）：ECE 0.066 → 0.055（crop 模式），高置信错判 17→11 张。
**意外收益**：微调后模型不再过自信，校准温度从 `T=6.05/9.55` 降到 `T=1.10/1.20`。

## 做了什么

1. **克隆公开 PlantDoc 数据集**（pratikkayal/PlantDoc-Dataset，28 类 CC BY 4.0），导出绕过 Windows 文件名三连坑
2. **混合微调**：PlantDoc 真实标注（补齐旧数据漏的 6 个病害类）+ PV 合成图保底 13 类，低 LR + 早停守实验室域
3. **重拟合校准温度**：权重 logits 变了，必须重做（否则置信度校准失效）
4. **端到端回归**：pytest 契约 / 真实接口 / 前端 build 全过

## 改动文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `scripts/export_plantdoc.py` | 新增 | 绕过 Windows 文件名坑导出数据集 |
| `scripts/prepare_plantdoc_mix.py` | 新增 | 构建混合索引 + 真实域评估集（防泄漏） |
| `scripts/finetune_domain.py` | 新增 | 混合微调训练（早停守 PV 精度） |
| `scripts/eval_realtest.py` | 新增 | 真实域带先验准确率评估 |
| `scripts/calibrate_domain.py` | 改 | 数据源改指向新真实域 |
| `models/calibration_domain.json` | 改 | 新校准温度 T=1.10/1.20（运行时依赖） |
| `docs/14_置信度校准.md` | 改 | 更新温度值 |
| `docs/15_真实域混合微调.md` | 新增 | 微调报告 |
| `models/best_resnet18_38.pth` | 替换 | ← finetuned（被 gitignore，本地部署） |
| `.gitignore` | 改 | 忽略生成数据 data/ + 临时文件 |

## 复跑

```bash
python -m scripts.export_plantdoc && python -m scripts.prepare_plantdoc_mix \
 && python -m scripts.finetune_domain && python -m scripts.calibrate_domain \
 && python -m scripts.eval_realtest --weights models/finetuned_resnet18_38.pth
```

## 注意事项

- 权重文件（*.pth）被 gitignore，不入库；本地部署用本地替换，原权重备份在 `backups/2026-10-20-pre-finetune/`
- 生成数据 `data/plantdoc_raw`、`data/real_mix`、`data/real_mix_test` 已 gitignore（脚本可重建）
- 后端须用绝对路径 `X:\Anconda3\python.exe -m uvicorn ... --no-proxy-headers` 启动
