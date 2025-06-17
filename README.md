# 天气预测应用

一个基于FastAPI和模拟MLP模型的天气预测系统，提供7天预报、小时级预报、实时天气和生活指数等功能。

## 项目概述
这是一个基于FastAPI和模拟MLP模型的天气预测应用，提供7天天气预报、小时级预报、实时天气和生活指数等功能。前端使用HTML、JavaScript、Tailwind CSS和Chart.js实现数据可视化，后端采用Python FastAPI框架提供API服务。

## ✨ 功能特点
- 🌤️ 7天天气预报展示
- ⏰ 24小时小时级温度和降水概率预测
- 🌡️ 实时天气数据查询
- 📊 多种生活指数推荐（紫外线、舒适度、穿衣、洗车、运动）
- 🔍 城市搜索和位置管理
- ⚙️ 模型状态监控和参数调整
- 7天天气预报展示
- 24小时小时级温度和降水概率预测
- 实时天气数据查询
- 多种生活指数推荐（紫外线、舒适度、穿衣、洗车、运动）
- 城市搜索和位置管理
- 模型状态监控和参数调整

## 项目结构
```
weather_prediction/
├── README.md           # 项目文档
├── start.ps1           # Windows启动脚本
├── model/              # 模型相关代码
├── frontend/           # 前端代码
│   ├── index.html      # 前端页面
│   └── app.js          # 前端逻辑
├── api/                # API服务代码
│   └── main.py         # FastAPI后端服务
├── Backend/            # 后端相关代码
├── data/               # 数据存储
└── __pycache__/
```

## 快速开始
### 环境要求
- Python 3.8+ 
- 现代浏览器

### 一键启动 (Windows)
1. 双击运行 `start.ps1` 脚本，或在PowerShell中执行:
```powershell
.\start.ps1
```

2. 脚本将自动完成:
   - 检查Python环境
   - 安装所需依赖
   - 启动后端API服务 (端口8001)
   - 启动前端静态服务器 (端口8000)

3. 服务启动后，可通过以下地址访问:
   - 前端页面: http://localhost:8000
   - API文档: http://localhost:8001/docs
   - 后端服务: http://localhost:8001

### 手动启动
#### 后端服务
```bash
cd api
uvicorn main:app --reload --port 8001
```

#### 前端页面
```bash
cd frontend
python -m http.server 8000
```

## API接口文档

### 1. 7天天气预报
- **路径**: `/api/v1/forecast/weekly`
- **方法**: GET
- **参数**:
  | 参数名 | 类型   | 必须 | 描述           | 默认值 |
  |--------|--------|------|----------------|--------|
  | city   | string | 是   | 城市名称       | 无     |
  | days   | int    | 否   | 预测天数(1-15) | 7      |

- **输出格式**:
```json
{
  "code": 200,
  "data": [
    {
      "date": "2023-11-15",
      "temp_high": 22,
      "temp_low": 15,
      "precipitation": 30.5,
      "wind_speed": 3.2,
      "uv_index": 5,
      "condition": "多云"
    },
    // ...更多日期数据
  ],
  "message": "success"
}
```

### 2. 小时级天气预报
- **路径**: `/api/v1/forecast/hourly`
- **方法**: GET
- **参数**:
  | 参数名   | 类型   | 必须 | 描述               | 默认值 |
  |----------|--------|------|--------------------|--------|
  | city     | string | 是   | 城市名称           | 无     |
  | hours    | int    | 否   | 预测小时数(1-48)   | 24     |
  | interval | int    | 否   | 时间间隔(1-6小时)  | 1      |

- **输出格式**:
```json
{
  "code": 200,
  "data": [
    {
      "hour": "9:00",
      "temperature": 18.5,
      "precipitation": 10.2
    },
    // ...更多小时数据
  ],
  "message": "success"
}
```

### 3. 当前天气
- **路径**: `/api/v1/weather/current`
- **方法**: GET
- **参数**:
  | 参数名 | 类型   | 必须 | 描述     | 默认值 |
  |--------|--------|------|----------|--------|
  | city   | string | 是   | 城市名称 | 无     |

- **输出格式**:
```json
{
  "code": 200,
  "data": {
    "date": "2023-11-15",
    "temp_high": 22,
    "temp_low": 15,
    "precipitation": 30.5,
    "wind_speed": 3.2,
    "uv_index": 5,
    "condition": "多云",
    "current_temp": 19.3,
    "humidity": 65.2,
    "wind_direction": "东南",
    "visibility": 10.5,
    "pressure": 1005.3,
    "dew_point": 12.1
  },
  "message": "success"
}
```

### 4. 生活指数
- **路径**: `/api/v1/weather/life-indices`
- **方法**: GET
- **参数**:
  | 参数名       | 类型   | 必须 | 描述                                     | 默认值 |
  |--------------|--------|------|------------------------------------------|--------|
  | city         | string | 是   | 城市名称                                 | 无     |
  | indices_type | array  | 否   | 指数类型(uv, comfort, dressing, car_washing, exercise) | 全部   |

- **输出格式**:
```json
{
  "code": 200,
  "data": {
    "uv": {
      "level": 5,
      "description": "中等，注意防护",
      "suggestion": "外出时建议涂抹防晒霜"
    },
    "comfort": {
      "temperature": 18.5,
      "humidity": 65.2,
      "description": "舒适",
      "suggestion": "感觉舒适，适宜打开窗户通风"
    },
    // ...更多指数数据
  },
  "message": "success"
}
```

### 5. 城市搜索
- **路径**: `/api/v1/location/search`
- **方法**: GET
- **参数**:
  | 参数名 | 类型   | 必须 | 描述               | 默认值 |
  |--------|--------|------|--------------------|--------|
  | query  | string | 是   | 城市名称搜索关键词 | 无     |
  | limit  | int    | 否   | 最大返回结果数     | 10     |

- **输出格式**:
```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "city": "北京",
      "latitude": 39.9042,
      "longitude": 116.4074
    },
    // ...更多城市数据
  ]
}
```

### 6. 保存位置
- **路径**: `/api/v1/location/save`
- **方法**: POST
- **参数**:
  | 参数名    | 类型    | 必须 | 描述               | 默认值 |
  |-----------|---------|------|--------------------|--------|
  | user_id   | string  | 是   | 用户ID             | 无     |
  | city      | string  | 是   | 城市名称           | 无     |
  | lat       | float   | 是   | 纬度               | 无     |
  | lon       | float   | 是   | 经度               | 无     |
  | is_default| boolean | 否   | 是否设为默认位置   | false  |

- **输出格式**:
```json
{
  "code": 200,
  "message": "位置保存成功",
  "data": [
    {
      "city": "北京",
      "latitude": 39.9042,
      "longitude": 116.4074,
      "is_default": true,
      "saved_at": "2023-11-15T09:30:00.123456"
    },
    // ...更多位置数据
  ]
}
```

### 7. 模型状态
- **路径**: `/api/v1/model/status`
- **方法**: GET
- **参数**: 无

- **输出格式**:
```json
{
  "code": 200,
  "data": {
    "model_name": "SimpleMLP",
    "version": "1.0",
    "status": "running",
    "last_trained": "2023-11-01T12:00:00Z",
    "prediction_count": 3562,
    "accuracy": 0.87
  },
  "message": "success"
}
```

### 8. 模型参数调整
- **路径**: `/api/v1/model/parameters`
- **方法**: POST
- **参数**:
  | 参数名        | 类型   | 必须 | 描述               | 默认值 |
  |---------------|--------|------|--------------------|--------|
  | learning_rate | float  | 否   | 学习率(0 < x < 1)  | 无     |
  | batch_size    | int    | 否   | 批次大小(1-1024)   | 无     |
  | epochs        | int    | 否   | 训练轮次(1-100)    | 无     |

- **输出格式**:
```json
{
  "code": 200,
  "data": {
    "message": "模型参数更新成功",
    "updated_parameters": {
      "learning_rate": 0.001,
      "batch_size": 32
    },
    "timestamp": "2023-11-15T09:35:00.123456"
  },
  "message": "success"
}
```

### 9. 健康检查
- **路径**: `/api/v1/health`
- **方法**: GET
- **参数**: 无

- **输出格式**:
```json
{
  "status": "healthy",
  "timestamp": "2023-11-15T09:40:00.123456",
  "version": "1.0"
}
```

## 技术栈
- **前端**: HTML, JavaScript, Tailwind CSS, Chart.js
- **后端**: Python, FastAPI, NumPy
- **开发工具**: Uvicorn

## 注意事项
- 本项目使用模拟MLP模型生成天气数据，实际应用中可替换为真实训练的预测模型
- 城市数据存储在内存中，重启服务后用户保存的位置信息将丢失
- 生产环境中建议添加用户认证和授权机制

## 许可证
[MIT](LICENSE)