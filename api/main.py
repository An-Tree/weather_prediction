from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="Weather Prediction API")

# 模拟数据库 - 城市数据
CITY_DATA = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2796, 120.1590),
    "南京": (32.0603, 118.7969),
    "武汉": (30.5928, 114.3055),
    "成都": (30.5723, 104.0665),
    "重庆": (29.4316, 106.9123),
    "西安": (33.4299, 108.9401),
    "沈阳": (41.7968, 123.4294),
    "哈尔滨": (45.8038, 126.5349),
    "济南": (36.6512, 117.1200),
    "郑州": (34.7472, 113.6250),
    "长沙": (28.1127, 112.9822),
    "合肥": (31.8611, 117.2831),
    "福州": (26.0999, 119.2958),
    "厦门": (24.4798, 118.0894),
    "昆明": (25.0406, 102.7123)
}

# 模拟用户位置数据库
USER_LOCATIONS = {}

# 模拟MLP模型
class SimpleMLP:
    @staticmethod
    def predict(date):
        # 基于日期生成伪随机但一致的天气数据
        np.random.seed(int(date.strftime('%Y%m%d')))
        
        # 生成基本天气参数
        temp_high = np.random.randint(15, 35)
        temp_low = np.random.randint(5, temp_high - 5)
        precipitation = round(np.random.uniform(0, 80), 1)
        wind_speed = round(np.random.uniform(1, 15), 1)
        uv_index = np.random.randint(1, 11)
        
        # 基于降水概率确定天气状况
        weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "大雨"]
        if precipitation < 10:
            condition = weather_conditions[0] if np.random.random() > 0.3 else weather_conditions[1]
        elif precipitation < 30:
            condition = weather_conditions[1] if np.random.random() > 0.5 else weather_conditions[2]
        elif precipitation < 50:
            condition = weather_conditions[3] if np.random.random() > 0.4 else weather_conditions[4]
        else:
            condition = weather_conditions[5] if np.random.random() > 0.5 else weather_conditions[4]
        
        return {
            "date": date.strftime('%Y-%m-%d'),
            "temp_high": temp_high,
            "temp_low": temp_low,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "uv_index": uv_index,
            "condition": condition
        }

# 定义响应模型
class WeatherResponse(BaseModel):
    date: str
    temp_high: int
    temp_low: int
    precipitation: float
    wind_speed: float
    uv_index: int
    condition: str

@app.get("/predict/7days", response_model=list[WeatherResponse])
def predict_7days():
    """预测未来7天的天气数据"""
    predictions = []
    for i in range(7):
        date = datetime.now() + timedelta(days=i)
        prediction = SimpleMLP.predict(date)
        predictions.append(prediction)
    return predictions
@app.get("/")
def read_root():
    return {"message": "Hello, World"}
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="Weather Prediction API", version="1.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # 允许前端地址访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟MLP模型
class SimpleMLP:
    @staticmethod
    def predict(date, city=None):
        # 基于位置和日期生成伪随机但一致的天气数据
        seed = int(date.strftime('%Y%m%d'))
        if city:
            seed += hash(city) % 1000

        np.random.seed(seed)
        
        # 生成基本天气参数
        temp_high = np.random.randint(15, 35)
        temp_low = np.random.randint(5, temp_high - 5)
        precipitation = round(np.random.uniform(0, 80), 1)
        wind_speed = round(np.random.uniform(1, 15), 1)
        uv_index = np.random.randint(1, 11)
        
        # 基于降水概率确定天气状况
        weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "大雨"]
        if precipitation < 10:
            condition = weather_conditions[0] if np.random.random() > 0.3 else weather_conditions[1]
        elif precipitation < 30:
            condition = weather_conditions[1] if np.random.random() > 0.5 else weather_conditions[2]
        elif precipitation < 50:
            condition = weather_conditions[3] if np.random.random() > 0.4 else weather_conditions[4]
        else:
            condition = weather_conditions[5] if np.random.random() > 0.5 else weather_conditions[4]
        
        return {
            "date": date.strftime('%Y-%m-%d'),
            "temp_high": temp_high,
            "temp_low": temp_low,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "uv_index": uv_index,
            "condition": condition
        }

# 响应模型
class DailyForecast(BaseModel):
    date: str
    temp_high: int
    temp_low: int
    precipitation: float
    wind_speed: float
    uv_index: int
    condition: str

class HourlyForecast(BaseModel):
    hour: str
    temperature: float
    precipitation: float

class WeeklyForecastResponse(BaseModel):
    code: int = 200
    data: List[DailyForecast]
    message: str = "success"

class HourlyForecastResponse(BaseModel):
    code: int = 200
    data: List[HourlyForecast]
    message: str = "success"

class CurrentWeatherResponse(BaseModel):
    code: int = 200
    data: dict
    message: str = "success"

class LifeIndicesResponse(BaseModel):
    code: int = 200
    data: dict
    message: str = "success"

class ModelStatusResponse(BaseModel):
    code: int = 200
    data: dict
    message: str = "success"

class ModelParametersResponse(BaseModel):
    code: int = 200
    data: dict
    message: str = "success"


@app.get("/api/v1/forecast/weekly", response_model=WeeklyForecastResponse)
def get_weekly_forecast(
    city: str = Query(..., description="城市名称"),
    days: int = Query(7, ge=1, le=15, description="预测天数")
):
    """获取未来N天的天气预报"""
    # 参数验证
    if not city:
        raise HTTPException(
            status_code=400,
            detail={"code": 400, "message": "必须提供城市名称参数", "data": None}
        )
        
    predictions = []
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        prediction = SimpleMLP.predict(date, city, lat, lon)
        predictions.append(prediction)
        
    return {"data": predictions}

@app.get("/api/v1/forecast/hourly", response_model=HourlyForecastResponse)
def get_hourly_forecast(
    city: str = Query(..., description="城市名称"),
    hours: int = Query(24, ge=1, le=48, description="预测小时数"),
    interval: int = Query(1, ge=1, le=6, description="时间间隔(小时)")
):
    """获取未来N小时的小时级天气预报"""
    # 生成种子确保结果一致性
    seed = hash(city) % 10000 + int(datetime.now().strftime('%Y%m%d'))
    np.random.seed(seed)
    
    base_temp = np.random.randint(15, 25)
    hourly_data = []
    
    for i in range(0, hours, interval):
        # 模拟昼夜温差
        hour = datetime.now().hour + i
        if hour >= 24:
            hour -= 24
        
        # 正弦曲线模拟温度变化 (下午2点左右温度最高)
        temp_variation = 5 * np.sin((hour - 14) * np.pi / 12)
        temperature = round(base_temp + temp_variation, 1)
        
        # 模拟降水概率
        precipitation = round(np.random.uniform(0, 60) if (hour > 18 or hour < 6) else np.random.uniform(0, 30), 1)
        
        hourly_data.append({
            "hour": f"{hour}:00",
            "temperature": temperature,
            "precipitation": precipitation
        })
    
    return {"data": hourly_data}

@app.get("/api/v1/health")
def health_check():
    """服务健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }

@app.get("/api/v1/weather/current", response_model=CurrentWeatherResponse)
def get_current_weather(
    city: str = Query(..., description="城市名称")
):
    """获取当前天气信息"""
    if not city and (lat is None or lon is None):
        raise HTTPException(
            status_code=400,
            detail={"code": 400, "message": "必须提供城市名称或经纬度参数", "data": None}
        )
        
    # 获取当前天气数据
    current_date = datetime.now()
    weather_data = SimpleMLP.predict(current_date, city, lat, lon)
    
    # 添加实时天气特有字段
    current_hour = current_date.hour
    weather_data.update({
        "current_temp": round((weather_data["temp_high"] + weather_data["temp_low"]) / 2 + 5 * np.sin((current_hour - 14) * np.pi / 12), 1),
        "humidity": round(np.random.uniform(30, 80), 1),
        "wind_direction": np.random.choice(["东", "南", "西", "北", "东南", "东北", "西南", "西北"]),
        "visibility": round(np.random.uniform(5, 15), 1),
        "pressure": round(np.random.uniform(980, 1020), 1),
        "dew_point": round(np.random.uniform(5, 20), 1)
    })
    
    return {"data": weather_data}

@app.get("/api/v1/weather/life-indices", response_model=LifeIndicesResponse)
def get_life_indices(
    city: str = Query(..., description="城市名称"),
    indices_type: Optional[List[str]] = Query(None, description="生活指数类型，可选值：uv, comfort, dressing, car_washing, exercise")
):
    """获取生活指数"""
    # 生成种子确保结果一致性
    seed = hash(city) % 10000 + int(datetime.now().strftime('%Y%m%d'))
    np.random.seed(seed)
    
    # 获取基础天气数据
    weather_data = SimpleMLP.predict(datetime.now(), city)
    
    # 生成生活指数
    indices = {}
    
    # 紫外线指数
    if not indices_type or "uv" in indices_type:
        uv_level = weather_data["uv_index"]
        if uv_level <= 2:
            uv_desc = "低，无需特殊防护"
            uv_suggestion = "适宜户外活动"
        elif uv_level <= 5:
            uv_desc = "中等，注意防护"
            uv_suggestion = "外出时建议涂抹防晒霜"
        elif uv_level <= 7:
            uv_desc = "较高，加强防护"
            uv_suggestion = "避免长时间暴露在阳光下"
        elif uv_level <= 10:
            uv_desc = "高，尽可能避免外出"
            uv_suggestion = "尽量待在室内，必须外出时做好全面防护"
        else:
            uv_desc = "极高，危险"
            uv_suggestion = "应尽可能避免所有室外活动"
        
        indices["uv"] = {
            "level": uv_level,
            "description": uv_desc,
            "suggestion": uv_suggestion
        }
    
    # 舒适度指数
    if not indices_type or "comfort" in indices_type:
        temp = (weather_data["temp_high"] + weather_data["temp_low"]) / 2
        humidity = np.random.uniform(40, 70)
        
        if temp < 18:
            comfort_desc = "凉爽"
        elif temp < 24:
            comfort_desc = "舒适"
        elif temp < 28:
            comfort_desc = "温暖"
        else:
            comfort_desc = "炎热"
        
        indices["comfort"] = {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "description": comfort_desc,
            "suggestion": "感觉{0}，{1}打开窗户通风".format(comfort_desc, "适宜" if 18 <= temp <= 26 else "建议")
        }
    
    # 穿衣指数
    if not indices_type or "dressing" in indices_type:
        temp_range = weather_data["temp_high"] - weather_data["temp_low"]
        
        if weather_data["temp_low"] < 10:
            dressing_desc = "寒冷"
            dressing_suggestion = "建议穿着羽绒服、厚毛衣等保暖衣物"
        elif weather_data["temp_low"] < 16:
            dressing_desc = "凉爽"
            dressing_suggestion = "建议穿着外套、毛衣等保暖衣物"
        elif weather_data["temp_low"] < 22:
            dressing_desc = "温和"
            dressing_suggestion = "建议穿着薄外套、长袖衬衫等"
        else:
            dressing_desc = "温暖"
            dressing_suggestion = "建议穿着短袖、短裤等清凉衣物"
        
        if temp_range > 10:
            dressing_suggestion += "，昼夜温差大，注意增减衣物"
        
        indices["dressing"] = {
            "description": dressing_desc,
            "suggestion": dressing_suggestion
        }
    
    # 洗车指数
    if not indices_type or "car_washing" in indices_type:
        if weather_data["precipitation"] < 20:
            car_desc = "适宜"
            car_suggestion = "天气晴朗，适宜洗车"
        elif weather_data["precipitation"] < 50:
            car_desc = "较适宜"
            car_suggestion = "降水概率较低，可以洗车"
        else:
            car_desc = "不适宜"
            car_suggestion = "降水概率较高，建议不要洗车"
        
        indices["car_washing"] = {
            "description": car_desc,
            "suggestion": car_suggestion
        }
    
    # 运动指数
    if not indices_type or "exercise" in indices_type:
        if weather_data["precipitation"] > 50:
            exercise_desc = "不适宜"
            exercise_suggestion = "降水概率高，建议室内活动"
        elif weather_data["temp_high"] > 32 or weather_data["temp_low"] < 5:
            exercise_desc = "较不适宜"
            exercise_suggestion = "温度过高或过低，建议减少户外活动时间"
        else:
            exercise_desc = "适宜"
            exercise_suggestion = "天气良好，适宜户外运动"
        
        indices["exercise"] = {
            "description": exercise_desc,
            "suggestion": exercise_suggestion
        }
    
    return {"data": indices}

@app.get("/api/v1/location/search")
def search_city(
    query: str = Query(..., description="城市名称搜索关键词"),
    limit: int = Query(10, ge=1, le=50, description="最大返回结果数")
):
    """搜索城市信息"""
    results = []
    query = query.lower()
    
    for city, (lat, lon) in CITY_DATA.items():
        if query in city.lower():
            results.append({
                "city": city,
                "latitude": lat,
                "longitude": lon
            })
            if len(results) >= limit:
                break
    
    return {
        "code": 200,
        "message": "success",
        "data": results
    }

@app.post("/api/v1/location/save")
def save_location(
    user_id: str = Query(..., description="用户ID"),
    city: str = Query(..., description="城市名称"),
    lat: float = Query(..., ge=-90, le=90, description="纬度"),
    lon: float = Query(..., ge=-180, le=180, description="经度"),
    is_default: bool = Query(False, description="是否设为默认位置")
):
    """保存用户位置信息"""
    if user_id not in USER_LOCATIONS:
        USER_LOCATIONS[user_id] = []
        
    # 检查是否已存在该城市
    location_exists = False
    for loc in USER_LOCATIONS[user_id]:
        if loc["city"] == city:
            loc["latitude"] = lat
            loc["longitude"] = lon
            loc["is_default"] = is_default
            location_exists = True
            break
    
    if not location_exists:
        # 如果设为默认，先取消其他默认
        if is_default:
            for loc in USER_LOCATIONS[user_id]:
                loc["is_default"] = False
        
        USER_LOCATIONS[user_id].append({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "is_default": is_default,
            "saved_at": datetime.now().isoformat()
        })
    
    return {
        "code": 200,
        "message": "位置保存成功",
        "data": USER_LOCATIONS[user_id]
    }

@app.get("/api/v1/model/status", response_model=ModelStatusResponse)
def get_model_status():
    """获取模型状态信息"""
    return {
        "data": {
            "model_name": "SimpleMLP",
            "version": "1.0",
            "status": "running",
            "last_trained": "2023-11-01T12:00:00Z",
            "prediction_count": np.random.randint(1000, 5000),
            "accuracy": round(np.random.uniform(0.7, 0.95), 3)
        }
    }

@app.post("/api/v1/model/parameters", response_model=ModelParametersResponse)
def update_model_parameters(
    learning_rate: Optional[float] = Query(None, gt=0, lt=1, description="学习率"),
    batch_size: Optional[int] = Query(None, ge=1, le=1024, description="批次大小"),
    epochs: Optional[int] = Query(None, ge=1, le=100, description="训练轮次")
):
    """更新模型参数"""
    # 在实际应用中，这里会更新真实模型的参数
    # 这里仅做模拟
    updated_params = {}
    if learning_rate is not None:
        updated_params["learning_rate"] = learning_rate
    if batch_size is not None:
        updated_params["batch_size"] = batch_size
    if epochs is not None:
        updated_params["epochs"] = epochs
    
    if not updated_params:
        return {
            "data": {
                "message": "未提供任何参数更新"
            }
        }
    
    return {
        "data": {
            "message": "模型参数更新成功",
            "updated_parameters": updated_params,
            "timestamp": datetime.now().isoformat()
        }
    }


def predict_hourly():
    """预测未来24小时的小时级天气数据"""
    np.random.seed(int(datetime.now().strftime('%Y%m%d')))
    base_temp = np.random.randint(15, 25)
    hourly_data = []
    
    for i in range(24):
        # 模拟昼夜温差
        hour = datetime.now().hour + i
        if hour >= 24:
            hour -= 24
        
        # 凌晨温度较低，下午温度较高
        temp_variation = 5 * np.sin((hour - 14) * np.pi / 12)  # 正弦曲线模拟温度变化
        temperature = round(base_temp + temp_variation, 1)
        
        # 模拟降水概率
        precipitation = round(np.random.uniform(0, 60) if (hour > 18 or hour < 6) else np.random.uniform(0, 30), 1)
        
        hourly_data.append({
            "hour": f"{hour}:00",
            "temperature": temperature,
            "precipitation": precipitation
        })
    
    return hourly_data