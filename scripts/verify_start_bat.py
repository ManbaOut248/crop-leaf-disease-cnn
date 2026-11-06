# -*- coding: utf-8 -*-
"""
对 启动网站.bat 做静态校验 + 生成 dry-run 副本。

校验项：
1. 文件是否为 GBK 编码（CMD 代码页 936 下中文不乱码）
2. 是否全部 CRLF 换行
3. if 块括号是否配平（bat 的 ( ) 不配平会导致解析失败）
4. start 行里的引号/路径是否能正确解析（目标目录是否存在）
5. 生成把 start 替换为 echo 的 dry-run 副本，可安全执行不真的起服务
"""

import sys
from pathlib import Path

BAT = Path(r"F:\实训产品\基于CNN的农作物叶片病害识别系统\启动网站.bat")


def main() -> int:
    if not BAT.exists():
        print("找不到", BAT)
        return 2

    raw = BAT.read_bytes()
    print(f"文件大小: {len(raw)} 字节")

    # 1) 编码
    try:
        text = raw.decode("gbk")
        print("[1] GBK 解码: 通过")
    except UnicodeDecodeError as e:
        print("[1] GBK 解码: 失败 ->", e)
        return 1

    # 2) 换行
    crlf = raw.count(b"\r\n")
    lf_only = raw.count(b"\n") - crlf
    print(f"[2] 换行: CRLF {crlf} 处, 裸 LF {lf_only} 处 ->", "通过" if lf_only == 0 else "存在裸 LF")

    # 3) 括号配平（忽略 REM 行）
    left = right = 0
    for ln in text.splitlines():
        s = ln.strip()
        if s.lower().startswith("rem"):
            continue
        left += s.count("(")
        right += s.count(")")
    print(f"[3] 括号: ( {left} / ) {right} ->", "配平" if left == right else "不配平")

    # 4) 关键路径是否真实存在
    root = BAT.parent
    checks = {
        "后端入口 main.py": root / "10_完整工程源码" / "src" / "api" / "main.py",
        "前端 package.json": root / "10_完整工程源码" / "src" / "web" / "package.json",
        "前端 vite 配置": root / "10_完整工程源码" / "src" / "web" / "vite.config.js",
        "模型权重": root / "10_完整工程源码" / "models" / "best_resnet18_38.pth",
        "数据集 PlantVillage": root / "PlantVillage",
        "Python 解释器": Path(r"X:\Anconda3\python.exe"),
    }
    print("[4] 路径检查:")
    bad = 0
    for name, p in checks.items():
        ok = p.exists()
        print(f"    {'OK  ' if ok else 'MISS'} {name}: {p}")
        if not ok:
            bad += 1

    # 5) dry-run 副本：把 start 行替换成 echo，去掉 pause
    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith("start "):
            out.append("echo [DRYRUN] " + ln)
        elif s == "pause":
            out.append("REM pause skipped")
        else:
            out.append(ln)
    dry = BAT.parent / "_dryrun_start.bat"
    dry.write_bytes(("\r\n".join(out) + "\r\n").encode("gbk"))
    print(f"[5] dry-run 副本: {dry}")

    print()
    print("结论:", "全部通过" if (lf_only == 0 and left == right and bad == 0) else "有问题需修")
    return 0 if (lf_only == 0 and left == right and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
