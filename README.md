# Multi-Vision

多图 AI 视觉识别 Web 应用，支持文本/表格两种模式、四点透视框选、两阶段流水线处理、内容级去重。

## 架构

```
┌─────────┐     ┌───────────────┐     ┌──────────────┐
│ Browser │────▶│ Go App :29817│────▶│ Vision :29818│
│         │◀────│ (Frontend +   │◀────│ (Python)     │
└─────────┘     │  Backend)     │     └──────┬───────┘
                └───────┬───────┘            │
                        │                    ▼
                        │           ┌──────────────────┐
                        ▼           │ Stage 1: VLM     │ 视觉识别
                  SQLite DB        │ Stage 2: LLM     │ 结构化/去重
                  (图片+结果)       └──────────────────┘
```

### Go + Python 分工

- **Go App**（go-app2/main.go）：Web 服务器，嵌入式前端（HTML/JS/CSS），处理路由、上传、SQLite、导出、配置管理。单文件编译部署。
- **Python 服务**（ocr_server.py）：AI 模型调用层。OpenAI / Anthropic / MiniMax 多格式 API 支持，OpenCV 透视矫正、图片预处理。

### 两种模式

| | 文本模式 | 表格模式 |
|---|---|---|
| 流程 | 单阶段：VLM 直接提取 | 两阶段：VLM 提取 → LLM 结构化 |
| 输出 | 纯文本（按语义分行） | Markdown 表格 |
| 导出 | txt | xlsx |
| 编辑 | contenteditable 整段编辑 | contenteditable 单元格编辑 |

### 两阶段流水线（表格模式）

```
图片 → 缩放(1200px) → 分段(300px重叠) → 并行VLM → 合并去重 → [完整表格?] → 是 → 直接输出
                                                            ↓ 否
                                                    LLM结构化 → Markdown表格
```

1. **Stage 1**（视觉模型）：图片分段后并行调用 VLM 提取文字
2. **智能跳过**：检测输出是否已是完整表格，是则跳过 Stage 2
3. **Stage 2**（语言模型）：将不规范的原始文本整理为结构化 Markdown 表格

### 四点透视框选

像扫描全能王一样拖动四个角对齐文档边缘，适用于倾斜拍摄的文档：

- 前端：Canvas 多边形绘制 + 角点拖拽交互
- 后端：`cv2.getPerspectiveTransform` + `cv2.warpPerspective` 透视矫正
- 矫正后送入 VLM 识别，减少因倾斜导致的误读

### 大图处理与去重

1. **分段**：1200px 高度切割，300px 重叠区域
2. **并行**：ThreadPoolExecutor 并行调用 VLM
3. **多窗口重叠检测**：1-5行滑动窗口匹配，优于单行 fuzzy 匹配
4. **内容级去重**：标准化签名比对（去除标点差异），按题号索引精确定位
5. **尾部碎片清理**：反向扫描不完整行（无句末标点）并移除
6. **孤立答案清理**：直接使用行索引集合精准移除

## 功能特性

### 前端
- 左右分栏：左侧图片列表，右侧文本结果
- Canvas 四点透视框选（角点拖拽、滚轮缩放、空格+拖拽平移）
- 框选区域持久化（刷新不丢失）
- CapCut 风格拖拽排序
- 每张图片独立"开始识别"按钮
- 表格可编辑单元格，自动保存
- 多模型预设一键切换，每模型独立 API Key 存储

### 后端
- 按模型预设的每模型独立密钥存储
- 透视矫正端点 `POST /ocr/region`
- 并发请求处理（ThreadingHTTPServer）
- 多种 AI 模型格式支持（OpenAI / Anthropic / MiniMax）

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
# 1. 启动 Vision 服务（端口 29818）
python ocr_server.py

# 2. 启动 Web 应用（端口 29817）
cd go-app2 && go build -o app && ./app
```

浏览器访问 http://localhost:29817

### AI 识别配置

点击工具栏齿轮图标，选择模型预设并填写 API 密钥。密钥按模型独立存储，切换预设自动加载已保存的密钥。

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
| claude-sonnet-4 | Anthropic | 推荐 |
| claude-haiku-4.5 | Anthropic | 更快 |
| MiniMax-M2.7 | MiniMax | Token Plan |
| glm-4-flash | 智谱 | 免费 |

## 项目结构

```
.
├── ocr_server.py          # Vision 服务（Python）
│                          # - 视觉模型调用（GLM / MiniMax）
│                          # - 透视矫正（cv2 warpPerspective）
│                          # - 多窗口重叠检测 + 内容级去重
│                          # - Stage 2 表格结构化
├── go-app2/
│   ├── main.go            # Web 应用（Go，嵌入 HTML/JS/CSS）
│   │                      # - 路由、上传、SQLite、导出
│   │                      # - 四点透视框选交互
│   │                      # - 每模型独立密钥存储
│   ├── config.json        # 服务端配置（gitignored）
│   ├── go.mod
│   └── go.sum
├── multi_ocr.db           # SQLite 数据库（gitignored）
├── .gitignore
└── README.md
```

## API

### Vision 服务 (端口 29818)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /ocr | 全图识别 |
| POST | /ocr/region | 四点透视框选识别 |

**POST /ocr/region** 请求体：
```json
{
  "image_base64": "完整图片base64",
  "regions": [{ "points": [{ "x": 100, "y": 50 }, { "x": 500, "y": 50 }, { "x": 500, "y": 300 }, { "x": 100, "y": 300 }] }],
  "recog_type": "text",
  "ai_config": { "model": "minimax-vlm", "api_key": "your_key", "api_url": "https://api.minimaxi.com" }
}
```

### Web 应用 (端口 29817)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | Web 页面 |
| POST | /api/upload | 上传图片 |
| GET | /api/images | 图片列表（含 text / table_text） |
| POST | /api/ocr/region | 代理转发透视框选到 Vision 服务 |
| POST | /api/ocr/:id | 识别指定图片 |
| DELETE | /api/images/:id | 删除图片 |
| PUT | /api/text/:id | 保存编辑文本 |
| PUT | /api/regions/:id | 保存框选区域 |
| PUT | /api/table_text/:id | 保存表格文本 |
| POST | /api/reorder | 调整图片顺序 |
| GET | /api/export | 导出文本 / xlsx |
| GET | /api/config | 获取配置 |
| POST | /api/config | 保存配置 |

### 数据模型 (ImageItem)

```json
{
  "id": 1,
  "name": "photo.jpg",
  "data": "data:image/jpeg;base64,...",
  "text": "OCR 文本识别结果",
  "table_text": "OCR 表格识别结果（Markdown）",
  "status": "done",
  "table_status": "ready",
  "recog_type": "text",
  "order": 1,
  "regions": "[{\"points\":[{\"x\":100,\"y\":50},...] }]"
}
```

## Version History

- **v1.4.0** — 四点透视框选、多窗口重叠检测、内容级去重、每模型密钥存储
- **v1.3.0** — 框选区域持久化、文本保存到数据库、开始识别按钮
- **v1.2.0** — 两阶段流水线、MiniMax VLM、表格模式
- **v1.1.0** — 大图分段、并发请求、全屏模态框
- **v1.0.0** — 基础多图识别

## License

MIT
