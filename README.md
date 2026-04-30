# Multi-OCR

多图文本/表格识别 Web 应用，基于 AI 视觉模型，支持两阶段流水线识别。

## 功能

- **多图管理** — 上传多张图片，拖拽排序，批量识别
- **双模式识别** — 文本模式（纯 OCR）与表格模式（提取 + 结构化）分离存储
- **两阶段流水线** — Stage 1 视觉模型提取 → Stage 2 AI 结构化（表格模式）
- **并行加速** — 多段图片并行请求，Stage 2 智能跳过
- **多模型支持** — GLM-4V、MiniMax VLM、Qwen-VL、GPT-4o、Claude 等
- **可编辑文本** — 识别结果在线编辑，自动保存
- **导出** — 文本模式导出 txt，表格模式导出 xlsx
- **数据持久化** — SQLite 存储，刷新不丢失

## 架构

```
┌─────────┐     ┌──────────────┐     ┌────────────┐
│ Browser │────▶│ Go App :8082 │────▶│ OCR :8081  │
│         │◀────│ (Frontend +  │◀────│ (Python)   │
└─────────┘     │  Backend)    │     └─────┬──────┘
                └──────────────┘           │
                                           ▼
                                  ┌────────────────┐
                                  │ Stage 1: VLM   │ (MiniMax / GLM-4V)
                                  │ Stage 2: LLM   │ (Claude / GLM-4)
                                  └────────────────┘
```

### 表格模式两阶段流水线

1. **Stage 1** — 视觉模型（MiniMax VLM / GLM-4V）从图片中提取表格原始文本
2. **Stage 2** — 语言模型（Claude / GLM-4）将原始文本结构化为标准 Markdown 表格
3. **智能跳过** — 若 Stage 1 输出已是单个完整表格，自动跳过 Stage 2

## 快速开始

### 环境要求

- Python 3.8+
- Go 1.21+

### 安装依赖

```bash
pip install opencv-python Pillow requests
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

点击工具栏齿轮图标，选择模型预设并填写 API 密钥。配置自动保存到服务端。

## 支持的模型

### 视觉识别模型（Stage 1）

| 模型 | 提供商 | 说明 |
|------|--------|------|
| minimax-vlm | MiniMax | Token Plan 免费，推荐 |
| glm-4v-flash | 智谱 | 免费 |
| glm-4v-plus | 智谱 | 付费，精度更高 |
| qwen-vl-max | 通义千问 | 付费 |
| gpt-4o | OpenAI | 付费 |

### 表格结构化模型（Stage 2）

| 模型 | 提供商 | 说明 |
|------|--------|------|
| claude-sonnet-4 | 智谱中转 | 推荐 |
| claude-haiku-4.5 | 智谱中转 | 更快 |
| MiniMax-M2.7 | MiniMax | Token Plan |
| glm-4-flash | 智谱 | 免费 |

## 项目结构

```
.
├── ocr_server.py          # OCR 服务（Python），视觉模型 + 结构化
├── go-app2/
│   ├── main.go            # Web 应用（Go，嵌入 HTML/JS/CSS）
│   ├── config.json        # 服务端配置（gitignored，需自行创建）
│   ├── go.mod
│   └── go.sum
├── go-app/                # 图片区域选择工具（独立）
├── .gitignore
└── README.md
```

## API

### OCR 服务 (端口 8081)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /ocr | 识别图片 |

**POST /ocr** 请求体：
```json
{
  "image_base64": "base64编码的图片数据",
  "recog_type": "table",
  "ai_config": {
    "model": "minimax-vlm",
    "api_key": "your_key",
    "api_url": "https://api.minimaxi.com",
    "struct_api_key": "struct_key",
    "struct_api_url": "https://open.bigmodel.cn/api/anthropic",
    "struct_model": "claude-sonnet-4-20250514"
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
| GET | /api/export | 导出文本/xlsx |
| GET | /api/export/xlsx | 导出 Excel |

## License

MIT
