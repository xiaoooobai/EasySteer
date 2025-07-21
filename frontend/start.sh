#!/bin/bash

echo "========================================"
echo "EasySteer - Steer Vector Control Panel"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python3"
    exit 1
fi

# 安装依赖
echo "[1/3] 检查并安装依赖..."
pip3 install -r requirements.txt

echo ""
echo "[2/3] 启动后端服务器..."
python3 app.py &
BACKEND_PID=$!

# 等待服务器启动
echo "[*] 等待服务器启动..."
sleep 3

echo ""
echo "[3/3] 启动前端服务器..."
python3 -m http.server 8111 &
FRONTEND_PID=$!

# 等待前端服务器启动
sleep 2

echo ""
echo "========================================"
echo "启动成功！"
echo ""
echo "后端API: http://localhost:5000"
echo "前端界面: http://localhost:8000/"
echo ""
echo "正在打开浏览器..."
echo "========================================"

# 根据操作系统打开浏览器
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:8000/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://localhost:8000/index.html 2>/dev/null || \
    sensible-browser http://localhost:8000/index.html 2>/dev/null || \
    echo "请手动打开浏览器访问: http://localhost:8000/index.html"
fi

echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

# 等待用户中断
trap "echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait 