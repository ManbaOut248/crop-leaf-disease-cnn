# -*- coding: utf-8 -*-
"""
Mendeley 数据集多线程分块下载器：
- 已有 TomatoDataset.zip 前缀（约292MB）保留，剩余区间分 6 段并行；
- 每段独立断点续传（part 文件记录进度）、异常退避重试、每轮重新解析新鲜 S3 直链；
- 全部完成后按段拼接 → 终验（总大小 + zipfile 完整性）。
"""
import os
import time
import threading
import zipfile
import urllib.request

ORIG = "https://data.mendeley.com/public-files/datasets/zfv4jj7855/files/5bec472d-1728-4dd0-b81a-233d04ad026d/file_downloaded"
OUT = r"F:\WorkBuddy工作空间\基于CNN的农作物叶片病害识别系统\data\_downloads\mendeley_zfv4jj7855\TomatoDataset.zip"
TARGET = 668089588
N = int(os.environ.get("DL_WORKERS", "8"))   # 并发段数，可用环境变量覆盖
UA = {"User-Agent": "Wget/1.21.4"}  # 默认 Python UA 被 WAF 403


def resolve_s3():
    """跟随 Mendeley 短链 302，拿新鲜 S3 预签名直链（有时效，每轮都要重取）。"""
    req = urllib.request.Request(ORIG, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.url


def worker(seg_id: int, start: int, end: int):
    """下载 [start, end) 到 part 文件；断点续传 + 无限重试。"""
    part = OUT + f".part{seg_id}"
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    backoff = 10
    while start + pos < end:
        try:
            s3 = resolve_s3()
            req = urllib.request.Request(
                s3, headers={**UA, "Range": f"bytes={start + pos}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
            backoff = 10  # 有进展重置退避
        except Exception as e:
            pos = os.path.getsize(part) if os.path.exists(part) else 0
            print(f"[seg{seg_id}] 异常: {e}（{start + pos}/{end}，退避{backoff}s）",
                  flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    print(f"[seg{seg_id}] 完成 {end - start} 字节", flush=True)


def main():
    prefix = os.path.getsize(OUT)
    remain = TARGET - prefix
    seg = remain // N
    bounds = [(prefix + i * seg, (prefix + (i + 1) * seg) if i < N - 1 else TARGET)
              for i in range(N)]
    print(f"前缀 {prefix}，剩余 {remain}，分段 {[e - s for s, e in bounds]}", flush=True)

    threads = [threading.Thread(target=worker, args=(i, s, e), daemon=True)
               for i, (s, e) in enumerate(bounds)]
    t0 = time.time()
    for t in threads:
        t.start()
    # 主线程打进度
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        done = sum(os.path.getsize(OUT + f".part{i}") if os.path.exists(OUT + f".part{i}") else 0
                   for i in range(N))
        rate = done / max(time.time() - t0, 1) / 1024
        print(f"[进度] {done + prefix}/{TARGET} ({(done + prefix) * 100 // TARGET}%) "
              f"均速 {rate:.0f}KB/s", flush=True)
    for t in threads:
        t.join()

    # 拼接：前缀文件 + 6 个 part
    with open(OUT, "ab") as f:
        for i in range(N):
            with open(OUT + f".part{i}", "rb") as p:
                while True:
                    buf = p.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
    total = os.path.getsize(OUT)
    assert total == TARGET, f"总大小不符 {total} != {TARGET}"
    # 终验 zip 完整性
    z = zipfile.ZipFile(OUT)
    bad = z.testzip()
    print("ZIP 校验:", ("损坏@" + bad) if bad else "OK",
          "条目:", len(z.namelist()), flush=True)
    if not bad:
        for i in range(N):
            os.remove(OUT + f".part{i}")


if __name__ == "__main__":
    main()
