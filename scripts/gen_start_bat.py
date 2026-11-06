# -*- coding: utf-8 -*-
"""
生成一键启动脚本 启动网站.bat（GBK + CRLF）。

为什么用 Python 生成而不直接写 .bat：
1. CMD 默认代码页 936，用 UTF-8 保存的 bat 中文会乱码，必须转 GBK；
2. 换行必须是 CRLF，否则某些 CMD 环境下 goto/if 块解析异常。
"""

BAT = r"""@echo off
chcp 936 >nul
setlocal
title 农作物叶片病害识别系统 - 一键启动

set "ROOT=%~dp0"
set "PRJ=%ROOT%10_完整工程源码"
set "WEB=%PRJ%\src\web"
set "PY=X:\Anconda3\python.exe"

echo ============================================================
echo   农作物叶片病害识别系统 - 一键启动
echo   后端 8000  /  前端 5173
echo ============================================================
echo.

REM ---------- 环境检查 ----------
if not exist "%PRJ%\src\api\main.py" (
  echo [错误] 未找到后端代码：%PRJ%\src\api\main.py
  echo 请把本文件放在交付包根目录，与 10_完整工程源码 同级。
  pause
  exit /b 1
)

if not exist "%PY%" (
  echo [错误] 未找到 Python：%PY%
  echo 请右键编辑本文件，把 set "PY=..." 改成你的 python.exe 完整路径。
  pause
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo [错误] 未检测到 Node.js，前端无法启动。请先安装 Node.js 18 或更高版本。
  pause
  exit /b 1
)

if not exist "%ROOT%PlantVillage" (
  echo [提示] 未找到数据集目录 PlantVillage，识别功能不受影响，但训练与建索引会失败。
  echo.
)

REM ---------- 首次运行安装前端依赖 ----------
if not exist "%WEB%\node_modules" (
  echo [1/3] 首次运行，正在安装前端依赖，约 1-3 分钟，请稍候...
  pushd "%WEB%"
  call npm install
  popd
  echo.
)

REM ---------- 启动后端 ----------
echo [2/3] 启动后端 http://127.0.0.1:8000
echo       注意：--no-proxy-headers 不能省略，否则限流可被 X-Forwarded-For 绕过。
start "后端 8000" /D "%PRJ%" cmd /k %PY% -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload --no-proxy-headers

timeout /t 3 /nobreak >nul

REM ---------- 启动前端 ----------
echo [3/3] 启动前端 http://127.0.0.1:5173
start "前端 5173" /D "%WEB%" cmd /k npm run dev

timeout /t 7 /nobreak >nul
start "" http://localhost:5173

echo.
echo 启动完成：已弹出「后端 8000」和「前端 5173」两个窗口。
echo 关闭这两个窗口即停止服务；浏览器应已自动打开 http://localhost:5173
echo.
echo 首次使用请先注册账号（数据存本地 SQLite，不联网）。
pause
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 默认输出到交付包根目录，可用第一个参数覆盖
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"F:\实训产品\基于CNN的农作物叶片病害识别系统\启动网站.bat"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="gbk", newline="\r\n") as f:
        f.write(BAT)
    print(f"已生成 {out}  ({len(BAT.encode('gbk'))} 字节, GBK+CRLF)")
