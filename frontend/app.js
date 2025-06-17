// 从API获取天气数据
let weatherData = {
    current: {},
    hourly: [],
    weekly: [],
    alert: {
        show: false,
        type: "",
        message: ""
    }
};

// 获取未来7天天气预报数据
async function fetchWeeklyData() {
    try {
        const response = await fetch('http://localhost:8001/predict/7days');
        if (!response.ok) throw new Error('网络响应不正常');
        return await response.json();
    } catch (error) {
        console.error('获取周预报数据失败:', error);
        // 失败时返回模拟数据
        return [
            { date: '2023-06-15', temp_high: 28, temp_low: 17, precipitation: 0, wind_speed: 3, uv_index: 5, condition: '晴' },
            { date: '2023-06-16', temp_high: 30, temp_low: 19, precipitation: 10, wind_speed: 4, uv_index: 7, condition: '多云' },
            { date: '2023-06-17', temp_high: 26, temp_low: 20, precipitation: 50, wind_speed: 5, uv_index: 3, condition: '小雨' },
            { date: '2023-06-18', temp_high: 24, temp_low: 19, precipitation: 30, wind_speed: 3, uv_index: 2, condition: '阴' },
            { date: '2023-06-19', temp_high: 27, temp_low: 18, precipitation: 0, wind_speed: 2, uv_index: 5, condition: '晴' },
            { date: '2023-06-20', temp_high: 29, temp_low: 19, precipitation: 0, wind_speed: 3, uv_index: 8, condition: '晴' },
            { date: '2023-06-21', temp_high: 31, temp_low: 21, precipitation: 10, wind_speed: 4, uv_index: 8, condition: '多云' }
        ];
    }
}

// 获取小时预报数据
async function fetchHourlyData() {
    try {
        const response = await fetch('http://localhost:8001/predict/hourly');
        if (!response.ok) throw new Error('网络响应不正常');
        return await response.json();
    } catch (error) {
        console.error('获取小时预报数据失败:', error);
        // 失败时返回模拟数据
        return Array.from({length: 24}, (_, i) => ({
            hour: `${(new Date().getHours() + i) % 24}:00`,
            temperature: Math.floor(18 + Math.random() * 12),
            precipitation: Math.floor(Math.random() * 60)
        }));
    }
}

// 获取天气图标信息
function getWeatherIcon(condition) {
    switch(condition) {
        case '晴': return { icon: 'fa-sun-o', color: 'text-warning' };
        case '多云': return { icon: 'fa-cloud', color: 'text-gray-400' };
        case '阴': return { icon: 'fa-cloud', color: 'text-gray-500' };
        case '小雨': return { icon: 'fa-tint', color: 'text-primary' };
        case '中雨': return { icon: 'fa-tint', color: 'text-primary' };
        case '大雨': return { icon: 'fa-tint', color: 'text-primary' };
        default: return { icon: 'fa-question', color: 'text-gray-300' };
    }
}

// 获取紫外线指数描述
function getUvDescription(index) {
    if (index <= 2) return '低';
    if (index <= 5) return '中等';
    if (index <= 7) return '高';
    return '极高';
}

// DOM元素加载完成后执行
document.addEventListener('DOMContentLoaded', async function() {
    // 设置当前日期
    setCurrentDate();
    
    try {
        // 并行获取天气数据
        const [weeklyData, hourlyData] = await Promise.all([
            fetchWeeklyData(),
            fetchHourlyData()
        ]);
        
        // 处理周预报数据
        weatherData.weekly = weeklyData.map((day, index) => {
            const date = new Date(day.date);
            const weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
            const iconInfo = getWeatherIcon(day.condition);
            
            return {
                day: index === 0 ? '今天' : weekdays[date.getDay()],
                date: `${date.getMonth() + 1}/${date.getDate()}`,
                condition: day.condition,
                icon: iconInfo.icon,
                iconColor: iconInfo.color,
                tempHigh: day.temp_high,
                tempLow: day.temp_low,
                precipitation: day.precipitation,
                windSpeed: day.wind_speed,
                uvIndex: getUvDescription(day.uv_index)
            };
        });
        
        // 处理小时预报数据
        weatherData.hourly = hourlyData;
        
        // 更新当前天气（使用第一天的数据）
        const today = weatherData.weekly[0];
        weatherData.current = {
            city: "北京市",
            temperature: today.tempHigh,
            condition: today.condition,
            conditionIcon: today.icon,
            conditionColor: today.iconColor,
            precipitation: today.precipitation,
            windSpeed: today.windSpeed,
            uvIndex: today.uvIndex,
            aqi: Math.floor(50 + Math.random() * 70),
            aqiLevel: "良好",
            humidity: Math.floor(30 + Math.random() * 40),
            pressure: 1008 + Math.floor(Math.random() * 10),
            sunrise: "05:13",
            sunset: "19:45",
            updateTime: new Date().toLocaleString()
        };
        
        // 渲染UI
        renderCurrentWeather();
        renderWeatherAlert();
        renderWeeklyForecast();
        initHourlyChart();
    } catch (error) {
        console.error('初始化天气数据失败:', error);
        alert('获取天气数据失败，请刷新页面重试');
    }
    
    // 初始化聊天对话框
    initChatDialog();
});

// 渲染当前天气
function renderCurrentWeather() {
    const currentWeather = document.getElementById('current-weather');
    if (!currentWeather || !weatherData.current) return;
    
    currentWeather.innerHTML = `
        <div class="flex items-center justify-between mb-6">
            <div>
                <h2 class="text-2xl font-bold">${weatherData.current.city}</h2>
                <p class="text-gray-500">${weatherData.current.updateTime}</p>
            </div>
            <div class="flex items-center">
                <div class="${weatherData.current.conditionColor} text-5xl mr-3"><i class="fa ${weatherData.current.conditionIcon}"></i></div>
                <div class="text-right">
                    <div class="text-4xl font-bold">${weatherData.current.temperature}°</div>
                    <div>${weatherData.current.condition}</div>
                </div>
            </div>
        </div>
        
        <div class="grid grid-cols-3 gap-4 mb-6">
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">降水</div>
                <div class="font-bold">${weatherData.current.precipitation}%</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">风速</div>
                <div class="font-bold">${weatherData.current.windSpeed}m/s</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">紫外线</div>
                <div class="font-bold">${weatherData.current.uvIndex}</div>
            </div>
        </div>
        
        <div class="grid grid-cols-3 gap-4">
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">空气质量</div>
                <div class="font-bold">${weatherData.current.aqi} (${weatherData.current.aqiLevel})</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">湿度</div>
                <div class="font-bold">${weatherData.current.humidity}%</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="text-gray-500 text-sm">气压</div>
                <div class="font-bold">${weatherData.current.pressure}hPa</div>
            </div>
        </div>
    `;
}

// 设置当前日期
function setCurrentDate() {
    const now = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    document.getElementById('current-date').textContent = now.toLocaleDateString('zh-CN', options);
}

// 渲染天气预警
function renderWeatherAlert() {
    const alertElement = document.getElementById('weather-alert');
    if (weatherData.alert.show) {
        alertElement.classList.remove('hidden');
        alertElement.querySelector('h3').textContent = weatherData.alert.type;
        alertElement.querySelector('p').textContent = weatherData.alert.message;
    }
}

// 渲染7天天气预报
function renderWeeklyForecast() {
    const container = document.getElementById('weekly-forecast');
    container.innerHTML = '';
    
    weatherData.weekly.forEach((day, index) => {
        const dayCard = document.createElement('div');
        dayCard.className = 'bg-white rounded-xl p-4 weather-card-shadow text-center hover:scale-105 transition-transform cursor-pointer';
        dayCard.setAttribute('data-day-index', index);
        
        // 点击每日卡片打开聊天对话框
        dayCard.addEventListener('click', function() {
            openChatDialog(index);
        });
        
        dayCard.innerHTML = `
            <div class="font-medium mb-1">${day.day}</div>
            <div class="text-gray-500 text-sm mb-2">${day.date}</div>
            <div class="${day.iconColor} text-2xl mb-2"><i class="fa ${day.icon}"></i></div>
            <div class="font-bold mb-1">${day.tempHigh}°/${day.tempLow}°</div>
            <div class="text-xs text-gray-500 mb-1">降水: ${day.precipitation}%</div>
            <div class="text-xs text-gray-500 mb-1">风速: ${day.windSpeed}m/s</div>
            <div class="text-xs text-gray-500">紫外线: ${day.uvIndex}</div>
        `;
        
        container.appendChild(dayCard);
    });
}

// 初始化小时预报图表
function initHourlyChart() {
    const ctx = document.getElementById('hourly-chart').getContext('2d');
    
    const hours = weatherData.hourly.map(item => item.hour);
    const temps = weatherData.hourly.map(item => item.temp);
    const precipitations = weatherData.hourly.map(item => item.precipitation);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{
                label: '温度 (°C)',
                data: temps,
                borderColor: '#165DFF',
                backgroundColor: 'rgba(22, 93, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                yAxisID: 'y'
            }, {
                label: '降水概率 (%)',
                data: precipitations,
                borderColor: '#36CFC9',
                backgroundColor: 'rgba(54, 207, 201, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: '温度 (°C)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '降水概率 (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

// 初始化聊天对话框
function initChatDialog() {
    const chatElement = document.getElementById('weather-chat');
    const chatHeader = document.getElementById('chat-header');
    const closeChat = document.getElementById('close-chat');
    const minimizeChat = document.getElementById('minimize-chat');
    const chatInput = chatElement.querySelector('input');
    const sendButton = chatElement.querySelector('button');
    const chatMessages = document.getElementById('chat-messages');
    
    // 点击头部切换聊天框显示/隐藏
    chatHeader.addEventListener('click', function(e) {
        if (!e.target.closest('button')) {
            toggleChatDialog();
        }
    });
    
    // 关闭聊天框
    closeChat.addEventListener('click', function() {
        chatElement.classList.add('translate-y-full');
    });
    
    // 最小化聊天框
    minimizeChat.addEventListener('click', function() {
        chatElement.classList.add('translate-y-full');
    });
    
    // 发送消息
    sendButton.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });
    
    // 发送聊天消息
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (message) {
            // 添加用户消息
            addChatMessage(message, 'user');
            chatInput.value = '';
            
            // 模拟AI回复
            setTimeout(() => {
                const replies = [
                    "今天天气晴朗，非常适合户外活动，但紫外线强度中等，记得做好防晒措施哦！",
                    "根据天气预报，今天不会下雨，您可以放心出门，无需携带雨具。",
                    "当前温度28°C，体感舒适，风力较小，非常适合开窗通风。"
                ];
                const randomReply = replies[Math.floor(Math.random() * replies.length)];
                addChatMessage(randomReply, 'ai');
            }, 1000);
        }
    }
    
    // 添加聊天消息
    function addChatMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = sender === 'user' ? 'flex items-start justify-end mb-4' : 'flex items-start mb-4';
        
        if (sender === 'user') {
            messageElement.innerHTML = `
                <div class="bg-primary text-white rounded-lg rounded-tr-none p-3 max-w-[80%]">
                    <p class="text-sm">${text}</p>
                </div>
                <div class="bg-primary rounded-full p-2 ml-3"><i class="fa fa-user text-white"></i></div>
            `;
        } else {
            messageElement.innerHTML = `
                <div class="bg-gray-light rounded-full p-2 mr-3"><i class="fa fa-robot text-gray-500"></i></div>
                <div class="bg-gray-light rounded-lg rounded-tl-none p-3 max-w-[80%]">
                    <p class="text-sm">${text}</p>
                </div>
            `;
        }
        
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // 切换聊天框显示/隐藏
    function toggleChatDialog() {
        chatElement.classList.toggle('translate-y-full');
    }
    
    // 打开聊天对话框并设置当前日期
    window.openChatDialog = function(dayIndex) {
        const day = weatherData.weekly[dayIndex];
        chatElement.classList.remove('translate-y-full');
        
        // 在实际应用中，这里可以根据选择的日期更新聊天内容
        if (dayIndex !== 0) {
            addChatMessage(`您正在查看${day.day}(${day.date})的天气信息，有什么可以帮助您的吗？`, 'ai');
        }
    }
}