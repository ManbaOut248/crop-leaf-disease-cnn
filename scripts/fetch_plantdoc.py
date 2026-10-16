# -*- coding: utf-8 -*-
"""批量下载 PlantDoc 真实田间叶片图（经 ghproxy 代理 raw 文件）。

用途：为「方案 B（重标阈值）」与「方案 C（领域适应）」准备真实场景数据。
按作物/病害分子目录组织，便于后续按类统计。

注意：只挑选纯 ASCII、无空格的文件名，避免 URL 编码问题。
"""
import concurrent.futures as cf
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("outputs/realtest")
GH = ("https://ghproxy.net/https://raw.githubusercontent.com/"
      "pratikkayal/PlantDoc-Dataset/master/train")

GROUPS = {
    "tomato_late_blight": ("Tomato%20leaf%20late%20blight", [
        "118A_Tom_Late.jpg", "20150619d.jpg", "600.jpg", "924977LateBlightLeaf.jpg",
        "IMG_1526.jpg", "IMG_2254.JPG.jpg", "IMG_4182.JPG.jpg", "IMG_5360.jpg",
        "IMG_5368.jpg", "IMG_5813.jpg", "LB-2-225x300.jpg", "LBleaf2INIA_0.jpg",
        "Late-Blight-A.jpg", "Late-Blight-on-Tomato-4JDill.jpg",
        "Late-Blight-on-a-tomato-leaf.jpg", "LateBlightLeaf.jpg", "LateBlt09_05.jpg",
        "Late_Blight_2.jpg", "Pest2369.jpg", "Tomato-Late-BlightBM.jpg",
        "TomatoBlight.JPG.jpg", "TomatoLateBlightTop.jpg", "dsc_4621c.jpg",
        "dsc_4624s.jpg", "dsc_8140cs.jpg", "late-blight-TomC2-10.jpg",
        "late-blight-tom-leaf.jpg", "late-blight-tomato-leaf.jpg", "late_blight1.jpg",
        "lateblight2.jpg", "lateblightTom_lesion.jpg", "lateblightTom_lesion3.jpg",
        "lb2.jpg", "lb_leaf_lesions2x1200.jpg", "lbleaf.jpg", "p1020076cs.jpg",
        "photo-2.jpg",
    ]),
    "corn_rust": ("Corn%20rust%20leaf", [
        "36687400253_a227c2ca28_b.jpg", "37310253326_7f32b316d1_b.jpg",
        "37310253806_04fee91918_b.jpg", "37356969601_4208f8bb7f_b.jpg",
        "5833328024_724f1df3f2_b.jpg", "COURTESY_southernrust-greyleaf.jpg",
        "Common-rust-east-central-Indiana.jpg", "Common-rust-of-corn-Tedford.jpg",
        "Corn-Common-Rust-300x225.jpg", "Corn-Common-Rust.jpg",
        "Corn-Rusts-wKBishop-story-622x467.jpg", "Corn-Southern-Rust-7-17-F1.jpg",
        "Corn-southern-rust-C-Schleicher-7-31-F1.jpg",
        "Corn-southern-rust-advanced-F1b-8-7-15.jpg",
        "Corn-southern-rust-early-F1a-8-7-15.jpg", "CornRust1.jpg", "CornRustFS1.jpg",
        "CornRustFS3.jpg", "Figure-12.jpg", "Figure-4-stripe-rust.jpg",
        "IMG_1360-1.jpg", "IMG_1376-300x300.jpg", "IMG_1762.jpg", "IMG_1841.jpg",
        "IMG_20130620_190316.jpg", "IMG_5011-15vruqb.jpg",
        "Resized_DSC_0191_Rust-Closeup.jpg", "SCR-2015.jpg", "SCR.jpg",
        "So-Corn-Rust.jpg", "Southern-Corn-Rust1.jpg", "Southern-Rust4-300x200.jpg",
        "Southern-rust-2_Wise.jpg", "Southern-rust-underside-of-leaf-1.jpg",
        "Southern-rust-upper-side-of-leaf-with-pustules-1.jpg",
        "common-rust-lower-leaf-side.jpg", "common-rust-upper-leaf-side.jpg",
        "common-rust3.jpg", "commonrust-2016-a-shirley.jpg", "commonrust2-1024x768.jpg",
        "corn-common-rust-f5-R.jpg", "corn-leaf-rust.jpg", "corn-rust-2.jpg",
        "corn-rust-3.jpg", "corn-rust-4.jpg", "corn-rust-5.jpg", "corn-rust.jpg",
        "corn-southern-rust-2.jpg", "corn-southern-rust-spores.jpg", "corn_rust.jpg",
        "corn_rust1_l.jpg", "imag4090.jpg", "markjones-corn-001.jpg", "rust-3.jpg",
        "rust.jpg", "southern-rust-2.jpg", "southern-rust-als.jpg",
        "southern-rust-close.jpg", "southern-rust-lower-side.jpg",
        "southernrust-seminole-june1.jpg", "southernrust.jpg",
        "web1_cusoutherncornrust.jpg",
    ]),
    "grape_black_rot": ("grape%20leaf%20black%20rot", [
        "000gb.jpg", "00gb.jpg", "01gb.jpg", "02gb.jpg", "0gb.jpg",
        "150px-Plasmopara_viticola_000.jpg", "220px-Guignardia_bidwellii_08.jpg",
        "35327886795_6f608e0eaa_b.jpg", "80111791.jpg",
        "Black_Rot_of_Grapes1153.jpg", "Black_Rot_of_Grapes1154.jpg",
        "Black_Rot_of_Grapes1155.jpg", "Black_Rot_of_Grapes1157.jpg",
        "Black_Rot_of_Grapes1166.jpg", "Black_Rot_of_Grapes1167.jpg",
        "Black_Rot_of_Grapes1168.jpg", "Black_Rot_of_Grapes1317.jpg",
        "Black_Rot_of_Grapes151.jpg", "Black_Rot_of_Grapes152.jpg",
        "Black_rot_on_grape_leaves5.jpg", "Concord-Grapes.jpg",
        "DCDl_p8UQAAf2hP.jpg", "DSCN1228.jpg", "Guignardia-vigne44.jpg",
        "black-rot-leaves.jpg", "black-rot-of-grape_figure-1.jpg", "black_rot_1.jpg",
        "br-ocw-dm.jpg", "br-ocw-phom.jpg", "brshoot2_zoom.jpg", "fungus_rust.jpg",
        "guignardia-bidwellii-05.jpg", "image66.jpg", "june-9-0181.jpg",
        "muscals2.jpg", "p_319.jpg",
    ]),
}

# 先把散布在 realtest 根目录的旧图归入对应子目录（首次运行时）
_OLD_TOMATO = ["118A_Tom_Late.jpg", "20150619d.jpg", "600.jpg",
               "924977LateBlightLeaf.jpg", "IMG_1526.jpg", "IMG_5360.jpg",
               "IMG_5368.jpg", "IMG_5813.jpg", "LateBlightLeaf.jpg", "dsc_4621c.jpg",
               "lateblight2.jpg", "lb2.jpg", "lbleaf.jpg", "photo-2.jpg"]
dest0 = ROOT / "tomato_late_blight"
dest0.mkdir(parents=True, exist_ok=True)
for name in _OLD_TOMATO:
    src = ROOT / name
    if src.is_file():
        shutil.move(str(src), str(dest0 / name))


def fetch(task):
    url, dest = task
    if dest.exists() and dest.stat().st_size > 5000:
        return ("skip", dest.name, dest.stat().st_size)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if len(data) < 5000 or not data[:2] == b"\xff\xd8":
            return ("bad", dest.name, len(data))
        dest.write_bytes(data)
        return ("ok", dest.name, len(data))
    except Exception as e:  # noqa: BLE001
        return ("fail", dest.name, str(e)[:40])


tasks = []
for sub, (enc, names) in GROUPS.items():
    d = ROOT / sub
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        tasks.append((f"{GH}/{enc}/{n}", d / n))

print(f"待处理 {len(tasks)} 张（4 并发，经 ghproxy）")
t0 = time.time()
res = {"ok": 0, "skip": 0, "bad": 0, "fail": 0}
fails = []
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    for i, (st, name, extra) in enumerate(ex.map(fetch, tasks), 1):
        res[st] = res.get(st, 0) + 1
        if st == "fail":
            fails.append((name, extra))
        if i % 20 == 0:
            print(f"  {i}/{len(tasks)}  ok={res['ok']} fail={res['fail']}", flush=True)

print()
print(f"完成，用时 {time.time()-t0:.0f}s：{res}")
if fails:
    print("失败清单（前 10）：")
    for n, e in fails[:10]:
        print(f"  {n}  {e}")
print()
print("各类实际图片数：")
total = 0
for sub in GROUPS:
    c = len([p for p in (ROOT / sub).iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stat().st_size > 5000])
    total += c
    print(f"  {sub:<22} {c} 张")
print(f"  合计 {total} 张")
