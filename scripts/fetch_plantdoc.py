# -*- coding: utf-8 -*-
"""
PlantDoc 真实田间叶片图批量下载（经 ghproxy 代理 raw 文件）。

为什么需要
---------------------------------------------------------------
为「真实域适应」实验（C1-b）提供**多类别**真实叶片数据。
只在 3 个类别上微调分类头会让其余类别偏科，必须覆盖尽量多的作物类别。

网络前提（本机实测）
---------------------------------------------------------------
    GitHub 直连          HTTP 000（完全不通）
    ghproxy.net 代理 raw 可用，约 8~138 KB/s —— 取样本够用，下大 ZIP 不行
    Zenodo / challenger.ai 不可达；ModelScope / hf-mirror / 天池 可达

为什么只下载「安全文件名」
---------------------------------------------------------------
PlantDoc 的文件名大量带 URL 残留（`?w=600&h=408`、`%20`、`%253E` 等），
这些字符经代理转发时存在编码歧义（`?` 会被当查询串、`%` 需要二次编码），
极易 404。因此只取仅含 [A-Za-z0-9._-] 的文件名 —— 每类都有 20+ 个，足够。

图片来源：PlantDoc-Dataset (pratikkayal)，CC BY 4.0，引用需署名。
"""

import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "realtest"
PROXY = "https://ghproxy.net/"
RAW = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/train/"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
WORKERS = 8
TIMEOUT = 90
MAX_PER_CLASS = 25

# (GitHub 目录名, 本地子目录, 项目 class_id, [文件名…])
JOBS = [
    ("Apple Scab Leaf", "apple_scab", 0, [
        "11NovMW_W4-A_4b.jpg", "12-20_apple_scab1.jpg", "Apple-Scab-Mulrooney.jpg",
        "Apple-Scab-image-01.jpg", "Apple-Scab-image-02.jpg", "Apple-Scab.jpg",
        "Apple-Scab1.jpg", "Apple-Scab_1_1.jpg", "Apple-scab-on-crab-apple.jpg",
        "AppleScab02.jpg", "AppleScab03.jpg", "AppleScab14.jpg", "Apple_Scab163.jpg",
        "Apple_scab_symptoms_on_leaf.jpg", "DSC00005.JPG.jpg", "DSCF1248-1gz1wpc.jpg",
        "IMG_3455-xjgwni-1024x1024.jpg", "Malus-Scab-disease-225x300.jpg",
        "acfsca026527.jpg", "apple-scab-disease.jpg", "apple-scab-leaf-big.jpg",
        "apple_scab.jpg", "apple_scab1.jpg", "apple_scab2.jpg", "applescab.jpg",
    ]),
    ("Apple leaf", "apple_healthy", 3, [
        "20130519yellowingappleleaves.jpg", "AppleLeavesInRain.jpg", "Apple_tree_leaf_J1.jpg",
        "CRABAPPLE2_leaves.jpg", "CRABAPPLE3_leaves.jpg", "CRABAPPLE_leaves.jpg",
        "Dwarf-Apple-Pink-Lady-2545.jpeg.jpg", "apple-tree-leaves.jpg",
        "autumn-red-apple-leaves.jpg", "crab_apple_03_leaves.jpg",
        "fruit_apple_apple-tree_wallpaper_EA60026.jpg", "green-apple-leaf-26110796.jpg",
        "green-leaf-apple-tree-14300049.jpg", "malus-floribunda-le-amoro2-a.jpg",
        "red-apple-leaves-309078.jpg", "red-apple-one-leaf-13226302.jpg",
        "yellow-apple-leaves-isolated-10868989.jpg", "yellow-ripe-apple-leaf-27092797.jpg",
        "appleleaf+trial+007.JPG.jpg", "apple-tree-leaf-curl-apple-tree-leaf-curl-aphids-apple-tree-leaves-curling-too-much-water.jpg",
    ]),
    ("Apple rust leaf", "apple_cedar_rust", 2, [
        "20130610_110525.jpg", "20130802_111648.jpg", "Apple_RustSpots2.jpg",
        "Apple_Scab164.jpg", "CAR1.jpg", "CAR4.jpg", "C11.11.jpg", "CedarApple_Rust168.jpg",
        "DAxpYpoWAAQxi1f.jpg", "LIFH3.jpg", "P1020430.jpg", "apple-rust.jpg", "apple_F14b.jpg",
        "applerust-500x383.jpg", "car-apple1.jpg", "car-apple9.jpg", "car.jpg", "car1.jpg",
        "car2.jpg", "cedapp04_lg.jpg", "cedar-apple-rust.jpg", "cedar-apple-rust_fig-1.jpg",
        "cedar-apple-rust_fig-2.jpg", "cedar-apple-rust_fig-3.jpg", "image1_14.jpeg.jpg",
    ]),
    ("Potato leaf early blight", "potato_early_blight", 20, [
        "00009.jpg", "0042241.jpg", "0061.jpg", "15hrt15a1f1.jpg", "18028_1.jpg",
        "20100816earlyblight.jpg", "2D1A9552.jpg", "45035_1.jpg", "80109626.jpg",
        "87A_Pot_Early.jpg", "Blght.jpg", "IMG_0014.jpg", "IMG_0020.jpg", "IMG_0022.jpg",
        "IMG_9055.JPG.jpg", "Potato-earlyblight2.jpg", "a_solani.jpg", "blight-2x.jpg",
        "blight-leaf-top.jpg", "blighted-potatoe-leave.jpg", "cq5dam.web.16-9.8.jpg",
        "early-blight-2.jpg", "early-blight-potatoes.jpg", "early_blight1.jpg", "eb2.jpg",
    ]),
    ("Tomato Early blight leaf", "tomato_early_blight", 29, [
        "100_46341.jpg", "2373320.jpg", "30053_original.jpg", "3606630.jpg", "5086-early.jpg",
        "Diseased-Tomato-Plant.jpg", "EBTomato.jpg", "Early-bligh_t.jpg",
        "Early-blight-660x440.jpg", "Early-blight.jpg", "Early_Blight_of_Tomato1687.jpg",
        "FS547-2-big.jpg", "IMG_0591.jpg", "IMG_2436w.jpg",
        "Tomato-Alternaria-Early-Blight.jpg",
        "Tomato-Early-Blight-Leaf-Examples-Cornell-University.jpg",
        "Tomato-Early-Blight-Leaves-Cornell-University.jpg", "Tomato-Early-Blight.jpg",
        "Tomato-blight-large.jpg", "early-blight-2x.jpg", "early-blight1.jpg",
        "early-tomato-blight.jpg", "early_blight_spots_l.jpg", "early_blight_tomato.jpg",
        "earlyblighttomato.jpg",
    ]),
    ("Corn Gray leaf spot", "corn_gray_leaf_spot", 7, [
        "0796.16graylspot.jpg", "1192.jpg", "1193.jpg", "1355.51grayleafspot.jpg",
        "CGrayLeafSpot.jpg", "Close-up-of-GLS.jpg", "Corn-Gray-leafspot-lesions-300x225.jpg",
        "Corn-gray-leaf-spot-8-15-F1-660.jpg", "FJ_010_F17220_B00010.jpg", "GRAY_LE2.jpg",
        "Gray-Leaf-Spot-on-Corn-um9znc.jpg", "Gray-leaf-spot-Adam-Sisson-20.jpg",
        "Gray-leaf-spot.jpg", "GrayLeafSpot_Chilvers.jpg", "Greyleafspot.jpg",
        "IMG_1359.JPG.jpg", "IMG_5251.jpg", "corn-gray-leaf-spot-backlit-2.jpg",
        "corn-gray-leaf-spot-f4.jpg", "corn-gray-leaf-spot.jpg", "cornGLS2.jpg",
        "g1902-1.jpg", "glfspot.jpg", "grayspot.jpg", "grey_leaf_spot.jpg",
    ]),
    ("Bell_pepper leaf spot", "pepper_bacterial_spot", 18, [
        "0.jpg", "0000.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg", "06.jpg",
        "100_1657.jpg", "534a.jpg", "BacterialSpot08.jpg", "BacterialSpot09.jpg",
        "BacterialSpot10.jpg", "BacterialSpot17.jpg", "Habanero1.jpg", "IMG_0945.jpg",
        "P5270397.JPG.jpg", "Pepper-BLS-Leaf_93303-e1370119902259.jpg", "Pepper-problem.jpg",
        "PepperPhytophLeaf3-2f94629.jpg", "e4d0e5.jpg", "ed8e20.jpg", "figure1.jpg",
        "may27pepleaes001.jpg", "may27pepleaes002.jpg", "thrips001.jpg",
    ]),
    ("Bell_pepper leaf", "pepper_healthy", 19, [
        "004-10.jpg", "007.JPG.jpg", "2477703364_0c94646b48.jpg", "28a593.jpg", "387502.jpg",
        "951d30.jpg", "Bell-pepper-leaves.jpg", "Bell-pepper-plant.JPG.jpg",
        "DSC04477.JPG.jpg", "DSCN0652.jpg", "DSCN3768.JPG.jpg", "IMG_1495.JPG.jpg",
        "IMG_2855-1.jpg", "M7JV1.jpg", "P4190299aS.jpg",
        "Seminis-Turnpike-Bell-Pepper-1024x756.jpg", "bb2581.jpg", "dscf1776.jpg",
        "fotolia_4311376_XS.jpg", "green-pepper-leaf-parks-3302644.jpg",
        "green-pepper-leaves-11364990.jpg", "img_2509_b.jpg", "leaves.jpg", "leaves11.jpg",
        "october-bell-peppers.jpg",
    ]),
    ("Blueberry leaf", "blueberry_healthy", 4, [
        "15rnn8k.jpg", "179288748.jpg", "2198823523_4b8fdb3f03_b_d.jpg",
        "4851522550_02e3b9f3b4.jpg", "5920105029_2c657564ba_b.jpg",
        "6348380125_ae2d097496_z.jpg", "7-blueberry-leaves.jpg",
        "9167d851305baf06649c57aea130c8fb.jpg", "BLUEBERRY-BLUE-CROP-LEAF-500x500.jpg",
        "BLUEBERRY-SPARTAN-LEAF-500x500.jpg", "Blueberry-Leaf.jpg", "Blueberry-leaves.jpg",
        "Blueberry07.jpg", "Blueberry165.JPG.jpg", "Blueberry169.JPG.jpg",
        "Chlorotic-Blueberry-Leaf.jpg", "Fruit-Blueberry-Northblue.jpg", "IMG_2003.JPG.jpg",
        "IMG_5310.JPG.jpg", "IMG_7119.jpg", "IMG_8502.JPG.jpg", "Red-Blueberry-Leaves.jpg",
        "blueberry-leaf.jpg", "blueberryleaves.jpg", "ovalifolium1.jpg",
    ]),
    ("Cherry leaf", "cherry_healthy", 6, [
        "100_0123_2.jpg", "100_4608.jpg", "BlackCherry04.jpg", "CHERRY_CHOKE2_leaves.jpg",
        "CHERRY_CHOKE_leaves.jpg", "ECE47.jpg", "IMG_4280.JPG.jpg", "canada-red-cherry-leaves.jpg",
        "cherry-fertilizer.jpg", "cherry-tree-leaves-and-fruits.jpg", "cherry-tree-leaves.jpg",
        "cherrytree-leaf.jpg", "img_3088.jpg", "prunsero_twig3.jpg", "prunus-avium-le-rvid-c.jpg",
        "prunus-pensylvanica-007.jpg", "sunburst_cherry.jpg", "yoshino-weeping-cherry-tree2.jpg",
        "yoshino-weeping-cherry-tree9.jpg", "black-brazilian-cherry-4988.jpeg.jpg",
    ]),
    ("Peach leaf", "peach_healthy", 17, [
        "0052.jpg", "020.JPG.jpg", "0pe.jpg", "0peach.jpg", "1343.jpg",
        "17558-peach-and-leaves.jpg", "1921.jpg", "24570793-peach-leaf.jpg", "300117.jpg",
        "300117_1.jpg", "366.jpg", "78517936.jpg", "DSC02329.jpg", "DSCN1913.JPG.jpg",
        "P1010703.JPG.jpg", "P1015750.JPG.jpg", "P1015775.JPG.jpg", "Peach-Leaf.jpg",
        "Peach-leaves.jpg", "Peach_leaves_raw.jpg", "dsc06413.jpg", "peach-1024x754.jpg",
        "peach-leaf.jpg", "peach.jpg", "product_Peach-leaf.jpg",
    ]),
    ("Squash Powdery mildew leaf", "squash_powdery_mildew", 25, [
        "100367_max.jpg", "100_08631.jpg", "207.jpg", "628x471.jpg", "DSC05432.JPG.jpg",
        "DSC05433.JPG.jpg", "DSC_0183.JPG.jpg", "IMG_0702.jpg", "IMG_1725.JPG.jpg",
        "IMG_1726.JPG.jpg", "IMG_1802-1024x768.jpg", "IMG_3829-1024x684.jpg", "IMG_4003.jpg",
        "IMG_5457.jpg", "IMG_5488.JPG.jpg", "IMG_5606.jpg", "IMG_5937.jpg", "IW000047.jpg",
        "Mildew-on-cucumber-leaf-1024x768.jpg", "Moldy-Leaf-Pumpkin-plant.jpg",
        "P1080620.JPG.jpg", "P8230073.JPG.jpg", "Powdery1.jpg",
        "PowderyMildewSquashPlant12010.jpg", "Squash_PowderMildew.jpg",
    ]),
    ("Corn leaf blight", "corn_northern_leaf_blight", 9, [
        "00c.jpg", "011.jpg", "01c.jpg", "022.jpg", "03c.jpg", "04c.jpg", "05c.jpg",
        "06c.jpg", "09c.jpg", "1-495x400.jpg", "0805Fig3400.jpg", "20140826_131630.jpg",
        "203725159.jpg", "28380-1024x768.jpg", "28430_rel.jpg", "28430_web.jpg",
        "80102321.jpg", "CORN-DISEASE.jpg", "Cochliobolus_heterostrophus.jpg",
        "Corn-Disease-NCLB-A-Nelson-6-18-2015-F1.jpg", "Corn-Gray-Leaf-Spot-7-31-F2.jpg",
        "Corn-GLS-Picture2-e1403817927977-1024x416.jpg",
        "Corn-disease-small-NCLB-lesions-A-Nelson-6-16-2015-F2.jpg",
        "Corn-northern%20corn%20leaf%20blight-8-21.jpg", "Corn020Fig2.jpg",
    ]),
    ("Raspberry leaf", "raspberry_healthy", 23, [
        "1-flowering-raspberry.jpg", "1029980206_21e73f8c57.jpg", "12972raspberry_leaf.jpg",
        "18183295-H.jpg", "2904.jpg", "2905.jpg", "53840458-raspberry-leaf.jpg",
        "5944718_f1024.jpg", "800px_COLOURBOX4311119.jpg", "Bee29A.jpg", "Beetles-Berry_3998.jpg",
        "DC77r.jpg", "FsSYF.jpg", "IMG_1982.jpg", "Lexicon_Raspberry.jpg",
        "Raspberry-Suckers1.jpg", "Red-Rasp-Leaf.jpg", "a61a6-raspberry_leaf_tea_pregnancy.jpg",
        "basketofraspberryleaves.jpg", "benefits-red-raspberry-leaf.jpg", "black-raspberry-01.jpg",
        "c-204-raspberry.jpg", "dsc_0127.jpg", "edible_weeds_raspberry_leaf.jpg",
        "feuille-de-framboisier.jpg",
    ]),
    ("Soyabean leaf", "soybean_healthy", 24, [
        "487977347.jpg", "6898762496_5b9b6c22d3_z.jpg", "DM-on-soy.jpg", "IMG_1138.jpg",
        "IMG_20150714_122008907.jpg", "IMG_4780.jpg", "IMG_4810.jpg", "IMG_fig-4.jpg",
        "KDeficientSoyBean.jpg", "Kdefsoybean.JPG.jpg", "Leaves-of-Soybean.jpg",
        "Soybean_leaves.jpg", "agron10.jpg", "asoybeanleafcloseup.JPG.jpg",
        "depositphotos_15928367-stock-photo-soy-leaf.jpg", "glma4_leaf_001.jpg",
        "kdeficiency2.jpg", "kdeficiency4.jpg", "soy-964327_960_720.jpg",
        "soy-leaf-27914090.jpg", "soybean-21357451.jpg", "soybean-leaf-single.jpg",
        "soybean-magnesium-2.jpg", "soybean-plant-seedling.jpg", "soybean_leaves_01.jpg",
    ]),
    ("Strawberry leaf", "strawberry_healthy", 27, [
        "102_0829.JPG.jpg", "13400strawberry_leaves.jpg", "18423_original.jpg", "41.jpg",
        "7.jpg", "DSCN5936.JPG.jpg", "F-vesca-leaf1.jpg", "Fragaria-virginiana-6.jpg",
        "IMG_1309.JPG.jpg", "IMG_2780.jpg", "IMG_4056.JPG.jpg", "IMG_4695.jpg",
        "IMG_4696.jpg", "IMG_5392.JPG.jpg", "IMG_7107.jpg", "Leaves-of-Strawberries.jpg",
        "Strawberry-Leaves.jpg", "giant-leaves.jpg", "img_0034.jpg", "img_0164.jpg",
        "img_20110302_170719.jpg", "indian-strawberry-leaf.jpg",
        "leaf-wild-strawberry-plant-25253212.jpg", "leaves_comparison.jpg",
        "single-green-strawberry-leaf-isolated-white-12750533.jpg",
    ]),
    ("Tomato Septoria leaf spot", "tomato_septoria_leaf_spot", 32, [
        "1b.jpg", "20100824septoria.jpg", "20809116-standard.jpg",
        "28018723549_3a16c1467b_b.jpg", "3893.jpg", "4853086777_bc41d29801.jpg",
        "5008922249_880e15c21f.jpg", "Bean-rust-on-leaves.jpg", "IMG_0312_original.jpg",
        "IMG_3175.jpg", "Sept9-300x225.jpg", "Septoria-461x346.jpg", "Septoria-Tomato.jpg",
        "Septoria-leaf-spot-hc.jpg", "SeptoriaLeaf_T3.jpg", "SeptoriaTomato-B.jpg",
        "Septoria_Leaf_Spot.jpg", "Septoria_Leaf_Spot_of_Tomato186.jpg", "Tom_Sept1.jpg",
        "Tom_Sept4.jpg", "bigseptoria01.jpg", "bnX8Q.jpg", "classic-septoria.jpg",
        "closeup-of-tomato-leaf-y4af51-1024x768.jpg", "early-blight-septoria-ls-fig-3.jpg",
    ]),
    ("grape leaf", "grape_healthy", 14, [
        "000g.jpg", "0g.jpg", "10178grape_leaf.jpg", "4635766-green-grapevine-leaves.jpg",
        "6cy5kMMEi.jpg", "GrapeLeaf.jpg", "Grape_leaf_01.jpg", "Grolleau_noir_leaves.jpg",
        "IMG_20150511_095517.jpg", "IMG_20150511_111942.jpg", "IMG_2666.jpg", "P1130239.jpg",
        "WildGrapeLeaf_cutout.jpg", "Wild_Grape_Leaf.jpg", "dc9rk7rKi.jpg",
        "depositphotos_6812535-stock-photo-grape-leaves.jpg", "dsc00849.jpg",
        "grape-leaf-15796930.jpg", "grape-leaf-19616502.jpg", "grape-leaf-detail-1457175.jpg",
        "grape-leaf-dew-465884.jpg", "grape-leaves-1.jpg", "grape-leaves-2.jpg",
        "grape-leaves1.jpg", "grapeleaf4.JPG.jpg",
    ]),
    ("Tomato leaf mosaic virus", "tomato_mosaic_virus", 36, [
        "1422_0.jpeg.jpg", "51196-004-61703A1B.jpg", "DSC02246.JPG.jpg", "TMV-3.jpg",
        "TNRV2.jpg", "ab40cce958c60d8b6e0ec2e4fdb8f526.jpg",
        "d41c86a650e49b8066d6b347d8e9d3ad.jpg", "dahliaviralsymptoms2lowresjpg.jpg",
        "dummen-tmv.jpg", "fig1-2k11-163-165.jpg", "fig2-2k11-1-12.jpg", "image018.jpg",
        "image022.jpg", "image024.jpg", "mosaic-virus-foliage.jpg", "mosaic_virus_tomato.jpg",
        "mosiac-virus3.jpg", "thumb-500.jpg", "tmv-tomato-16.jpg", "tomato-mosaic-virus-1.jpg",
        "tomato-mosaic-virus-F0BA26.jpg", "tomato-mosaic-virus-leaf.jpg",
        "tomato-mosaic-virus.jpg", "tomato_mosaic_virus.jpg", "tomato_virus_01_zoom.jpg",
    ]),
    ("Tomato leaf yellow virus", "tomato_yellow_leaf_curl_virus", 35, [
        "072.jpg", "14456.img.jpg", "18071.img.jpg", "28619919393_7ef44acccc_b.jpg",
        "29994445671_b6914b44d0_b.jpg", "5146.jpg", "534ddf3a932e62dd929a3ac1b29bfa29.jpg",
        "7585001780_4b89e826b7.jpg", "8226402609_62d16884a7_b.jpg",
        "8227475116_0d40e8f21c_b.jpg", "8e60bae9f613f360403943f108a03083.jpg",
        "DSC01999.JPG.jpg", "DSC02006.JPG.jpg", "DSC02007.JPG.jpg",
        "DSC_3019_320x209.jpg", "Figure-4.jpg", "IMG_2684.JPG.jpg", "IMG_2686.JPG.jpg",
        "IMG_2687.JPG.jpg", "IU-TYLCV_img_6.jpg", "IU-TYLCV_img_8.jpg", "TYLCV1.jpg",
        "TYLCV2.jpg", "TYLCV4.jpg", "TYLCVDFerrinlr.jpg",
    ]),
    ("Tomato leaf", "tomato_healthy", 37, [
        "003.JPG.jpg", "100_8283.jpg", "1686.jpg", "1688.jpg", "17371378-tomato-leaf.jpg",
        "20160428_080421.jpg", "CTG514794.jpg", "DSCN1016.JPG.jpg", "Fig-2.jpg",
        "IMG_2061.JPG.jpg", "IMG_3011.jpg", "IMG_3192.JPG.jpg", "L.peruv-assort.jpg",
        "Leaves-of-Tomato.jpg", "PxWBE.jpg", "curling-issues.jpg", "flies.jpg",
        "fpls-08-01602-g001.jpg", "happier-inside.jpg", "img_1028.jpg", "img_6635.jpg",
        "p1011399.jpg", "tomato-leaf-20728368.jpg", "tomato-leaf-6291945.jpg",
        "tomato-leaf.jpg",
    ]),
    ("Tomato mold leaf", "tomato_leaf_mold", 31, [
        "100_4542.jpg", "119B_Tom_Mold.jpg", "2195890.jpg", "603-116-0.jpg",
        "6032331.JPG.jpg", "607-155-0.jpg", "607-155-1.jpg", "BlackLeafMold2.jpg",
        "DSCN2697.jpg", "GIVXB.jpg", "HighTunnel_TLM5.jpg", "HighTunnel_TLM6.jpg",
        "IMG_1261.jpg", "OLU13.jpg", "Ormvn.jpg", "P1020365.jpg", "Problem-2a.jpg",
        "dscf2712.jpg", "dscn6690.jpg", "image233.jpg", "leaf_mold1.jpg", "leaf_mold2.jpg",
        "leafmold.jpg", "leafmoldtomato1.jpg", "leafmoldtomato2.jpg",
    ]),
]


def fetch_one(url, dest):
    """下载单张图片。已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code != 200:
                return f"http{r.status_code}"
            data = r.content
            if len(data) < 1024:
                return "tiny"
            dest.write_bytes(data)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                return f"err:{type(e).__name__}"
            time.sleep(1)
    return "err"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT}")
    print(f"代理: {PROXY}  并发: {WORKERS}\n")

    tasks = []
    for gh_dir, local, cid, names in JOBS:
        dest_dir = OUT / local
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_names = [n for n in names[:MAX_PER_CLASS] if SAFE.match(n)]
        for n in safe_names:
            url = PROXY + RAW + urllib.parse.quote(gh_dir) + "/" + urllib.parse.quote(n, safe="")
            tasks.append((url, dest_dir / n, gh_dir, local, cid))

    print(f"待下载 {len(tasks)} 个文件\n")

    t0 = time.time()
    stats = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, u, d): (gh, loc) for u, d, gh, loc, _ in tasks}
        done = 0
        for f in as_completed(futs):
            gh, loc = futs[f]
            res = f.result()
            stats.setdefault(loc, []).append(res)
            done += 1
            if done % 25 == 0:
                print(f"  进度 {done}/{len(tasks)}  ({time.time()-t0:.0f}s)")

    print(f"\n下载完成，用时 {time.time()-t0:.0f}s\n")
    print(f"{'本地目录':<26} {'ok':>5} {'skip':>5} {'失败':>5}")
    total_ok = 0
    for gh_dir, local, cid, _ in JOBS:
        rs = stats.get(local, [])
        ok = sum(1 for x in rs if x == "ok")
        sk = sum(1 for x in rs if x == "skip")
        bad = len(rs) - ok - sk
        total_ok += ok + sk
        print(f"{local:<26} {ok:>5} {sk:>5} {bad:>5}   → class {cid}")
    print(f"\n合计可用 {total_ok} 张")

    # 汇总各本地目录实际文件数
    print("\n各目录实际文件数:")
    for gh_dir, local, cid, _ in JOBS:
        n = len([p for p in (OUT / local).glob("*") if p.is_file() and p.stat().st_size > 0])
        print(f"  {local:<26} {n:>3} 张")


if __name__ == "__main__":
    sys.exit(main())
