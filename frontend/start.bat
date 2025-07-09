@echo off
echo ========================================
echo EasySteer - Steer Vector Control Panel
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python
    pause
    exit /b 1
)

REM 安装依赖
echo [1/3] 检查并安装依赖...
pip install -r requirements.txt

echo.
echo [2/3] 启动后端服务器...
start cmd /k "python app.py"

REM 等待服务器启动
echo [*] 等待服务器启动...
timeout /t 3 /nobreak >nul

echo.
echo [3/3] 启动前端服务器...
start cmd /k "python -m http.server 8000"

REM 等待前端服务器启动
timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo 启动成功！
echo.
echo 后端API: http://localhost:5000
echo 前端界面: http://localhost:8000/
echo.
echo 正在打开浏览器...
echo ========================================

REM 打开浏览器
start http://localhost:8000/index.html

echo.
echo 按任意键关闭此窗口（服务器将继续运行）
pause >nul 