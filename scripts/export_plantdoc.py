#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 PlantDoc 数据集到本地安全文件名结构（稳健版）。

原因：pratikkayal/PlantDoc-Dataset 仓库大量文件名带 URL 残留字符
（`?1507122477.jpg`、`%20`、空格等），Windows 禁止这些字符作文件名，
`git checkout` 会报 invalid path 中断；而 `git archive` 在沙箱里返回空流。

本脚本改用：
  1) `git ls-tree -r HEAD` 拿 (sha, path) 清单（path 只是字符串，不受文件系统限制）
  2) `git cat-file --batch` 按 blob sha 一次性批量导出字节流
  3) 写盘时用安全文件名（非法字符 -> '_'），保留 train/ test 两级目录

完全在 Python 内控制文件名，绕开 Windows 对非法字符的限制。
"""
import subprocess, re, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "plantdoc_raw" / "PlantDoc-Dataset"
OUT = ROOT / "data" / "plantdoc_raw" / "exported"

IMG_SUF = {".jpg", ".jpeg", ".png"}
_SAFE = re.compile(r"[^A-Za-z0-9._\-/]")


def safe(p: str) -> str:
    return "/".join(_SAFE.sub("_", part) for part in p.split("/"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) 列出所有 blob
    ls = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-tree", "-r", "HEAD"], text=True
    )
    entries = []  # (sha, path)
    for line in ls.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        mode, typ, sha, path = parts
        if typ != "blob":
            continue
        if not (path.startswith("train/") or path.startswith("test/")):
            continue
        if Path(path).suffix.lower() not in IMG_SUF:
            continue
        entries.append((sha, path))
    print(f"待导出图片（train/test）: {len(entries)} 张")

    # 2) 批量 cat-file：把 sha 清单写入文件，git 从文件读 stdin，
    #    Python 只从 stdout 读 —— 避免「双向全缓冲管道」死锁
    #    （一次性 write 全部 stdin 会阻塞在 PIPE 缓冲，git 也因 stdout
    #     缓冲满而阻塞，双方互等导致死锁、0 产出）。
    sha_file = OUT / "_shas_tmp.txt"
    sha_file.write_text("".join(f"{sha}\n" for sha, _ in entries), encoding="utf-8")
    f_in = open(sha_file, "rb")
    proc = subprocess.Popen(
        ["git", "-C", str(REPO), "cat-file", "--batch"],
        stdin=f_in, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out = proc.stdout

    count = 0
    for sha, path in entries:
        header = out.readline()
        if not header:
            break
        hp = header.split()
        if len(hp) < 3 or hp[1] == b"missing":
            continue
        size = int(hp[2])
        data = out.read(size)
        out.read(1)  # 消费尾随换行
        tgt = OUT / safe(path)
        # 截断超长文件名：Windows MAX_PATH=260，原始 PlantDoc 文件名常 >150 字符，
        # 触发 OS 误导性的 FileNotFoundError。截断 stem 并加哈希去重（文件名对训练无意义）。
        if len(tgt.name) > 60:
            h = hashlib.md5(tgt.stem.encode("utf-8")).hexdigest()[:8]
            tgt = tgt.with_name(tgt.stem[:40] + "_" + h + tgt.suffix)
        if tgt.exists():
            h = hashlib.md5(data[:2048]).hexdigest()[:6]
            tgt = tgt.with_name(f"{tgt.stem}__{h}{tgt.suffix}")
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_bytes(data)
        count += 1
    proc.wait()
    f_in.close()
    sha_file.unlink(missing_ok=True)
    print(f"导出完成: {count} 张 -> {OUT}")
    print(f"  train 类: {len([p for _, p in entries if p.startswith('train/')])} / "
          f"test 类: {len([p for _, p in entries if p.startswith('test/')])}")


if __name__ == "__main__":
    main()
