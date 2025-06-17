#!/bin/bash

# 天气预测项目启动脚本 (Linux/Mac)

# 定义颜色常量
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python环境，请先安装Python 3.8+${RESET}"
    exit 1
fi

# 检查依赖是否安装
check_dependency() {
    if ! pip3 show $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# 安装所需依赖
if ! check_dependency fastapi; then
    echo -e "${YELLOW}正在安装FastAPI依赖...${RESET}"
    pip3 install fastapi uvicorn numpy pydantic
fi

# 启动后端服务
echo -e "${GREEN}正在启动FastAPI后端服务...${RESET}"
cd api && uvicorn main:app --reload --port 8001 &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端服务
echo -e "${GREEN}正在启动前端服务...${RESET}"
cd ../frontend && python3 -m http.server 8000 &
FRONTEND_PID=$!

# 显示访问信息
echo -e "\n${GREEN}服务启动成功!${RESET}\n"
echo -e "${YELLOW}后端API地址:${RESET} http://localhost:8001"
echo -e "${YELLOW}API文档地址:${RESET} http://localhost:8001/docs"
echo -e "${YELLOW}前端页面地址:${RESET} http://localhost:8000"
echo -e "\n${YELLOW}提示: 使用 Ctrl+C 停止服务${RESET}\n"

# 等待用户中断
wait $BACKEND_PID $FRONTEND_PID