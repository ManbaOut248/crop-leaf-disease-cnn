# -*- coding: utf-8 -*-
"""下载 PlantDoc **test 划分**用于独立的准确性评估（任务书 2.2-3 的实测验证）。

为什么用 test 划分
---------------------------------------------------------------
训练/阈值标定只用了 `train/`。`test/` 从未参与任何训练、微调或阈值选择，
是唯一能给出「无偏准确性」的来源。用训练集图片自测属于自证，没有意义。

与 train 下载脚本的差别（编码处理）
---------------------------------------------------------------
test 集文件名普遍带 URL 残留（`%20`、`%28`、`+`）。Git 中的文件名是**已解码**的
真实名字（如 `apple scab leaf.jpg`），所以正确做法是：
    unquote(名字) → quote(..., safe="")
直接 quote 原名会二次编码（`%20` → `%2520`）导致 404 —— 这是本次踩过的坑。

用法
    py -3.12 scripts/fetch_plantdoc_test.py
    py -3.12 scripts/fetch_plantdoc_test.py --workers 8
产物
    outputs/realtest_test/<本地类别>/<图片>
    outputs/realtest_test/manifest.jsonl   （path / class_id / 期望类别）
"""

import argparse
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "realtest_test"
PROXY = "https://ghproxy.net/"
RAW = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/test/"
TIMEOUT = 90

# (GitHub test 目录, 本地子目录, 项目 class_id, [文件名…])
# 文件名取自 GitHub API 列目录结果（每个目录前 5~6 个），不做人工挑选
JOBS = [
    ("Apple Scab Leaf", "apple_scab", 0, [
        "052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg",
        "1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg",
        "28-500x375.jpg", "816.jpg", "apple%20scab%20leaf.jpg"]),
    ("Apple leaf", "apple_healthy", 3, [
        "20180511_090912-14gtw8a-e1526047952754.jpg",
        "20180511_091133-24l1vhg-e1526047988236.jpg",
        "20180511_091252-1gy5xf5-e1526048000596.jpg",
        "4120978-single-green-leaf-of-apple-tree.jpg",
        "Apple-Leaf-Wallpaper-17.jpg"]),
    ("Apple rust leaf", "apple_cedar_rust", 2, [
        "02.-Rust-2017-207u24s.jpg", "0605_Rust-induced_leafspot.jpg",
        "185161-004-EAF28842.jpg", "2011-011.jpg", "20130519cedarapplerust.jpg"]),
    ("Bell_pepper leaf spot", "pepper_bacterial_spot", 18, [
        "00.jpg", "000.jpg", "01.jpg", "Bacterial-spot-pepper2.jpg",
        "CMVpepperLeafShock-copy-50QUALITY-1ge8umw.jpg"]),
    ("Bell_pepper leaf", "pepper_healthy", 19, [
        "10148582-green-leaf-of-pepper.jpg", "4825902504_5fa80b65b8_z.jpg",
        "GREEN%20BELL%20PLANT%20YELLOW%20LEAF.JPG.jpg", "RnGnC.jpg"]),
    ("Blueberry leaf", "blueberry_healthy", 4, [
        "P1200665+Vaccinium+alaskaense+vs+ovalifolium+leaf+and++twig+cr.jpg",
        "P1200701+Vaccinium+ovalifolium+Oval-leaf+Blueberry+leaf+margin+Heather+Meadows+cr.jpg",
        "blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg",
        "blueberrysilverleaf16-1372b.jpg", "leaf-blueberry-15281271.jpg"]),
    ("Cherry leaf", "cherry_healthy", 6, [
        "d7b79dbc0bca0eb3e63fbe5dfb256ccd53d90a4eb1413_1260x1260.jpg",
        "e52bbb9a08ede2948ba0c006f02fe7b053ad7831d1df6_1260x1260.jpg",
        "ed12ff77ce021883de241830dee55e2e53d2c93231699_1260x1260.jpg",
        "fc729af279e9087d00bd32d89a1552a756a3ad9de7f2b_1260x1260.jpg",
        "five-cherry-tree-green-leaves-white-isolated-background-34493587.jpg"]),
    ("Corn Gray leaf spot", "corn_gray_leaf_spot", 7, [
        "IMG_42231.jpg", "rsz0803Figure6.jpg", "strom5.jpg"]),
    ("Corn leaf blight", "corn_northern_leaf_blight", 9, [
        "02c.jpg", "0796.20graylssymt.jpg", "07c.jpg", "0c.jpg", "1321189.jpg"]),
    ("Corn rust leaf", "corn_rust", 8, [
        "0796.39maizerust.jpg", "0796.40comrust.jpg", "0796.47southrust.jpg",
        "0796.52srusttelia.jpg", "12-19striprustJIM.jpg"]),
    ("Peach leaf", "peach_healthy", 17, [
        "00pe.jpg",
        "7190f27ead55c0f6e3ff8b982972810a5446a9713a49d_1260x1260.jpg",
        "depositphotos_3786198-Peach-leaf.jpg", "peach-leaf-10223898.jpg",
        "peach-leaf-16690421.jpg"]),
    ("Potato leaf early blight", "potato_early_blight", 20, [
        "3023.jpg", "backus-056-potato-blight.jpg",
        "early-blight-alternaria-alternata-leaf-spotting-on-potato-leaf-X91XJE.jpg",
        "early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-AXK6AY.jpg",
        "earlyblightpotato.jpg"]),
    ("Potato leaf late blight", "potato_late_blight", 21, [
        "20090710-lateblight.jpg", "2540_600.jpg",
        "5816740026_d42ef24413_Phytophthora-Infestans.jpg",
        "B2750109-Late_blight_on_a_potato_plant-SPL.jpg"]),
    ("Raspberry leaf", "raspberry_healthy", 23, [
        "Raspberry-leaf.jpg", "depositphotos_1323264-Raspberry-leaf-on-white.jpg",
        "depositphotos_1323551-stock-photo-raspberry-leaves.jpg",
        "depositphotos_2795312-stock-photo-green-raspberry-leaf.jpg",
        "iron-deficiency-raspberry-leaf-chlorosis-isolated-32457798.jpg"]),
    ("Soyabean leaf", "soybean_healthy", 24, [
        "07feb_ma_sbr3.JPG.jpg", "Soybean%20edamame%20leaf1.JPG.jpg",
        "fresh-soy-leaves-isolated-on-260nw-667193923.jpg",
        "leaf-raspberry-isolated-on-a-white-stock-photography-image-10106222-1625198.jpg",
        "resistance1c.jpg"]),
    ("Squash Powdery mildew leaf", "squash_powdery_mildew", 25, [
        "Powdery-Mildew-on-squash.jpg",
        "powdery-mildew-erysiphe-plantani-on-young-sycamore-leaves-b774tm.jpg",
        "powdery-mildew-on-pumpkin-leaves-close-up-2.jpg",
        "powdery-mildew-sphaerotheca-fuliginea-infection-on-pumpkin-leaves-ajdkkx.jpg",
        "squash-leave-w-powd-mild.jpg"]),
    ("Strawberry leaf", "strawberry_healthy", 27, [
        "strawberry-leaf--stock-photo-1431216.jpg", "strawberry-leaf-22360816.jpg",
        "strawberry-leaves-collection-white-background-36684676.jpg",
        "strawberry-leaves-isolated-white-17473157.jpg",
        "strawberry-leaves-isolated-white-34722259.jpg"]),
    ("Tomato Early blight leaf", "tomato_early_blight", 29, [
        "039b47d574bc4bb8a14259a1cd96a741.jpg", "100983448.jpg",
        "1234080-Early-Blight.jpg", "Shoemaker_7068.JPG.jpg", "bugs-and-blight-061.jpg"]),
    ("Tomato Septoria leaf spot", "tomato_septoria_leaf_spot", 32, [
        "0.jpg", "2STEM.jpg", "2septoria-tomato.jpg",
        "55830b60a7cf2.image.jpg", "6134794031202304.jpeg.jpg"]),
    ("Tomato leaf bacterial spot", "tomato_bacterial_spot", 28, [
        "%2320+Bacterial+Spot+and+Speck.jpg",
        "07.17.18-Common_Tomato_Diseases_Canker-258x300.jpg",
        "Bacterial_spots563.jpg", "Tom_Bact3.jpg", "Tom_Bact7.jpg"]),
    ("Tomato leaf late blight", "tomato_late_blight", 30, [
        "110822-206-Tomato-blight.jpg", "18c.jpg",
        "636368370164094909-late-blight-15-.jpg", "70.jpg",
        "E%29Tomato-late-blight-foliar.jpg"]),
    ("Tomato leaf mosaic virus", "tomato_mosaic_virus", 36, [
        "9511.img.jpg", "D%29AlfMov.jpg", "Tomato-Mosaic-1wdr2jx-300x225.jpg",
        "everystockphoto-21099455-l.jpg",
        "ill-tomato-of-tomato-mosaic-virus-DBE17N.jpg"]),
    ("Tomato leaf yellow virus", "tomato_yellow_leaf_curl_virus", 35, [
        "11-40580_5.jpg", "17fc47.jpg", "8226402877_9abf151b5b_b.jpg",
        "fwa2013313115151.jpg", "glyphosate.jpg"]),
    ("Tomato leaf", "tomato_healthy", 37, [
        "1684.jpg", "2013-08-20-06.jpg", "90e2c0.jpg",
        "DSCN1015.JPG.jpg", "IMG_1246.jpg"]),
    ("Tomato mold leaf", "tomato_leaf_mold", 31, [
        "49%20Leafmold%20Bottom.jpg", "50%20Leafmold%20Top.jpg", "BIGSD.jpg",
        "black_leaf_mold_in_zina_tina.jpg", "dscn6689.jpg"]),
    ("grape leaf black rot", "grape_black_rot", 11, [
        "03gb.jpg", "35589125035_662dd5b258_b.jpg", "5-29black-rot-chardRR.jpg",
        "Black%20rot%20on%20foliage.jpg", "Black%20rot%20on%20foliage2.jpg"]),
    ("grape leaf", "grape_healthy", 14, [
        "0000.jpg", "730-grape-leaf-2560x1600-nature-wallpaper.jpg",
        "DSCN7317.JPG.jpg", "P1130286.jpg",
        "depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg"]),
]


def fetch_one(url, dest, attempt=0):
    """下载并做完整性校验。返回 'ok' 或错误标记。"""
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return f"http{r.status_code}"
        data = r.content
        if len(data) < 1024:
            return "tiny"
        dest.write_bytes(data)
        # 必须真正 load()：verify() 抓不住截断的 JPEG
        with Image.open(dest) as im:
            im.load()
        return "ok"
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(1)
            return fetch_one(url, dest, 1)
        return f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for gh_dir, local, cid, names in JOBS:
        d = OUT / local
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            if "/" in n or len(n) > 160:
                continue
            # Git 里的文件名是已解码的真实名字 → 先 unquote 再 quote，避免二次编码
            real = urllib.parse.unquote(n)
            url = PROXY + RAW + urllib.parse.quote(gh_dir, safe="") + "/" + \
                urllib.parse.quote(real, safe="")
            # 本地名用安全的短名，避免 Windows 路径里的特殊字符问题
            dest = d / f"{urllib.parse.quote(real, safe='')[:80]}.jpg"
            tasks.append((url, dest, local, cid, gh_dir, real))

    print(f"待下载 {len(tasks)} 个文件（{len(JOBS)} 个类别），并发 {args.workers}")
    t0 = time.time()
    ok, fail = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, u, d): (d, cid, gh, real) for u, d, loc, cid, gh, real in tasks}
        done = 0
        for f in as_completed(futs):
            dest, cid, gh, real = futs[f]
            res = f.result()
            done += 1
            if res == "ok":
                ok.append((dest, cid, gh, real))
            else:
                fail.setdefault(res, 0)
                fail[res] += 1
            if done % 20 == 0:
                print(f"  进度 {done}/{len(tasks)}  ok={len(ok)}  {time.time()-t0:.0f}s")

    manifest = OUT / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for dest, cid, gh, real in sorted(ok):
            f.write(json.dumps({
                "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "class_id": cid,
                "gh_dir": gh,
                "origin": real,
            }, ensure_ascii=False) + "\n")

    print(f"\n完成：{len(ok)} 张有效，{sum(fail.values())} 张失败，用时 {time.time()-t0:.0f}s")
    for k, v in sorted(fail.items()):
        print(f"  失败类型 {k}: {v}")
    print(f"清单：{manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
