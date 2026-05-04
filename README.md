# Multi-Vision

多图 AI 视觉识别 Web 应用，支持文本/表格两种模式，两阶段流水线处理。

## 设计大局

### 架构

```
┌─────────┐     ┌──────────────┐     ┌────────────┐
│ Browser │────▶│ Go App :8082 │────▶│ Vision :8081│
│         │◀────│ (Frontend +  │◀────│ (Python)   │
└─────────┘     │  Backend)    │     └─────┬──────┘
                └──────┬───────┘           │
                       │                   ▼
                       │          ┌─────────────────┐
                       ▼          │ Stage 1: VLM    │ 视觉识别
                  SQLite DB      │ Stage 2: LLM    │ 结构化/去重
                  (图片+结果)     └─────────────────┘
```

### Go + Python 分工

- **Go App**（go-app2/main.go）：Web 服务器，嵌入式前端（HTML/JS/CSS 全在 main.go），处理路由、上传、SQLite、导出。选 Go 是因为嵌入式前端简单、编译为单文件部署方便。
- **Python 服务**（ocr_server.py）：AI 模型调用层。选 Python 是因为调用 AI SDK（OpenAI 格式、Anthropic 格式、MiniMax 自有格式）更方便，OpenCV 图片处理生态成熟。

### 两种模式

| | 文本模式 | 表格模式 |
|---|---|---|
| 流程 | 单阶段：VLM 直接提取 | 两阶段：VLM 提取 → LLM 结构化 |
| 输出 | 纯文本（按语义分行） | Markdown 表格 |
| 导出 | txt | xlsx |
| 编辑 | contenteditable 整段编辑 | contenteditable 单元格编辑 |
| 存储 | `text` 字段 | `table_text` 字段 |

### 两阶段流水线（表格模式）

```
图片 → 大图分段(1800px/200px重叠) → 并行调用VLM → 合并去重 → [已是完整表格?] → 是 → 直接输出
                                                          ↓ 否
                                                  LLM结构化 → Markdown表格
```

1. **Stage 1**（视觉模型）：从图片中提取原始文字，prompt 要求输出 Markdown 表格格式
2. **智能跳过**：检测输出是否已是单个完整表格（无重复表头），是则跳过 Stage 2
3. **Stage 2**（语言模型）：将多段/不规范的原始文本整理为结构化 Markdown 表格

为什么需要两阶段？视觉模型擅长"读图"但不擅长把零散文字组织成规整表格；LLM 擅长文本整理但无法直接看图。

### 大图处理策略

1. **缩放**：长边限制 2000px（平衡质量与分段数）
2. **分段**：按 1800px 高度切割，200px 重叠区域
3. **并行**：ThreadPoolExecutor 并行处理各段
4. **合并**：fuzzy 匹配（ratio ≥ 0.6）找到重叠行，去重拼接
5. **清理**：移除截断碎片（连续短行）、题目级去重（相同题号只保留首次）

### 数据存储

单 SQLite 文件（multi_ocr.db），一张表存所有数据：

| 字段 | 用途 |
|------|------|
| data | base64 data URL（图片） |
| text / table_text | 文本/表格识别结果 |
| status / table_status | 识别状态（ready/done/error） |
| recog_type | 模式（text/table） |
| sort_order | 拖拽排序序号 |

### 前端特性

- 左右分栏：左侧图片列表，右侧文本结果
- CapCut 风格拖拽排序
- Canvas 框选识别（区域选择）
- 表格可编辑单元格，自动保存
- 多模型预设一键切换
- 待识别图片显示"开始识别"按钮

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
# 1. 启动 Vision 服务（端口 8081）
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
├── ocr_server.py          # Vision 服务（Python），视觉模型调用 + 结构化 + 去重
├── go-app2/
│   ├── main.go            # Web 应用（Go，嵌入 HTML/JS/CSS）
│   ├── config.json        # 服务端配置（gitignored，需自行创建）
│   ├── go.mod
│   └── go.sum
├── multi_ocr.db           # SQLite 数据库（gitignored）
├── .gitignore
└── README.md
```

## API

### Vision 服务 (端口 8081)

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
