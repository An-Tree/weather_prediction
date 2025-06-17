<# 天气预测项目启动脚本 #>

# 定义颜色常量
$GREEN = "\e[32m"
$YELLOW = "\e[33m"
$RED = "\e[31m"
$RESET = "\e[0m"

# 检查Python是否安装
try {
    python --version 2>&1 | Out-Null
} catch {
    Write-Host "${RED}错误: 未找到Python环境，请先安装Python 3.8+${RESET}"
    exit 1
}

# 检查依赖是否安装
function Check-Dependency {
    param([string]$package)
    try {
        pip show $package 2>&1 | Out-Null
        return $true
    } catch {
        return $false
    }
}

# 安装所需依赖
if (-not (Check-Dependency -package "fastapi")) {
    Write-Host "${YELLOW}正在安装FastAPI依赖...${RESET}"
    pip install fastapi uvicorn numpy pydantic
}

# 启动后端服务
Write-Host "${GREEN}正在启动FastAPI后端服务...${RESET}"
Start-Process powershell -ArgumentList "cd api; uvicorn main:app --reload --port 8001; pause"

# 等待后端启动
Start-Sleep -Seconds 2

# 启动前端服务
Write-Host "${GREEN}正在启动前端服务...${RESET}"
Start-Process powershell -ArgumentList "cd frontend; python -m http.server 8000; pause"

# 显示访问信息
Write-Host "
${GREEN}服务启动成功!${RESET}
"
Write-Host "${YELLOW}后端API地址:${RESET} http://localhost:8001"
Write-Host "${YELLOW}API文档地址:${RESET} http://localhost:8001/docs"
Write-Host "${YELLOW}前端页面地址:${RESET} http://localhost:8000"
Write-Host "
${YELLOW}提示: 关闭所有PowerShell窗口即可停止服务${RESET}"