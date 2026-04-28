# Multi-OCR

多图文本识别 Web 应用，支持本地 OCR 引擎和 AI 视觉模型两种识别模式。

## 功能

- **多图管理** — 上传多张图片，拖拽排序，批量识别
- **双模式识别** — 本地 OCR（4 引擎融合）与 AI 视觉模型自由切换
- **可编辑文本** — 识别结果支持在线编辑，自动保存
- **框选识别** — 支持对图片局部区域进行 OCR 识别
- **灵活配置** — 前端配置面板，可切换模型和 API 密钥，支持多种视觉模型
- **文本导出** — 一键导出所有识别文本
- **数据持久化** — SQLite 存储，刷新页面不丢失

## 识别模式

| 模式 | 引擎 | 特点 |
|------|------|------|
| 本地识别 | PaddleOCR-ch + PaddleOCR-en + RapidOCR + Tesseract | 离线可用，多引擎融合投票 |
| AI 识别 | GLM-4V / Qwen-VL / GPT-4o 等视觉模型 | 英文识别更准确，需 API 密钥 |

## 快速开始

### 环境要求

- Python 3.8+
- Go 1.21+
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

### 安装依赖

```bash
pip install paddleocr rapidocr_onnxruntime opencv-python Pillow
# Tesseract (Ubuntu)
sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

### 启动服务

```bash
# 1. 启动 OCR 服务（端口 8081）
python ocr_server.py

# 2. 启动 Web 应用（端口 8082）
cd go-app2 && go build -o app && ./app
```

浏览器访问 http://localhost:8082

### AI 识别配置

两种方式设置 API 密钥：

1. **前端配置面板** — 点击工具栏齿轮图标，填写 API 地址、模型名称和密钥
2. **环境变量** — 启动前设置 `VISION_API_KEY`

```bash
export VISION_API_KEY=your_api_key_here
python ocr_server.py
```

### 支持的 AI 模型预设

| 模型 | 提供商 | 说明 |
|------|--------|------|
| glm-4v-flash | 智谱 | 免费，默认 |
| glm-4v-plus | 智谱 | 付费，精度更高 |
| qwen-vl-max | 通义千问 | 付费 |
| qwen-vl-plus | 通义千问 | 付费 |
| gpt-4o | OpenAI | 付费 |

也可自定义任何兼容 OpenAI Chat Completions 格式的 API 端点。

## 项目结构

```
.
├── ocr_server.py          # OCR 服务，本地引擎 + AI 视觉模型
├── go-app2/
│   ├── main.go            # Web 应用（Go，嵌入 HTML/JS/CSS）
│   ├── go.mod
│   └── go.sum
├── .env.example           # 环境变量示例
├── .gitignore
└── README.md
```

## API

### OCR 服务 (端口 8081)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /ocr | 识别图片 |
| GET | /health | 健康检查 |

**POST /ocr** 请求体：
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "mode": "ai",
  "ai_config": {
    "model": "glm-4v-flash",
    "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_key": "your_key"
  }
}
```

### Web 应用 (端口 8082)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | Web 页面 |
| POST | /api/upload | 上传图片 |
| GET | /api/images | 图片列表 |
| POST | /api/ocr/:id | 识别指定图片 |
| DELETE | /api/images/:id | 删除图片 |
| PUT | /api/text/:id | 保存编辑文本 |
| POST | /api/reorder | 调整图片顺序 |
| POST | /api/export | 导出文本 |
| GET | /api/health | OCR 服务状态 |

## License

MIT
