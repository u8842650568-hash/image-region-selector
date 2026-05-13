# Multi-Vision

多图 AI 视觉识别 Web 应用，支持多模型 OCR、四点透视框选、大图自动分段并行识别、内容去重。

单 Go 二进制，无 Python 依赖，直接调用 AI Vision API。

## 架构

```
┌─────────┐     ┌───────────────┐     ┌──────────────────┐
│ Browser │────▶│ Go App :29817│────▶│ AI Vision API    │
│ (前端)   │◀────│ (嵌入式 Web)  │◀────│ (GLM / MiniMax)  │
└─────────┘     └───────┬───────┘     └──────────────────┘
                        │
                        ▼
                  SQLite DB
                  (图片 + 识别结果)
```

### 核心模块

- **main.go**：Web 服务器，嵌入式前端（HTML/JS/CSS），路由、上传、SQLite、导出、配置管理
- **vision.go**：AI Vision API 调用层，图片压缩、透视矫正、标点归一化、断行合并

## 功能特性

- **多图上传**：左右分栏，左侧图片列表 + 右侧识别结果
- **四点透视框选**：像扫描全能王一样拖动四角对齐文档边缘，自动透视矫正后识别
- **大图分段**：自动按 1200px 分段 + 300px 重叠区域，并行调用 API
- **内容去重**：多窗口滑动匹配 + 标准化签名比对，自动移除重复行
- **多模型切换**：GLM-4V Flash、GLM-4.6V、MiniMax VLM 等预设，每模型独立 API Key 存储
- **CapCut 风格拖拽排序**
- **导出**：TXT / XLSX

## 快速开始

### 环境要求

- Go 1.21+

### 构建运行

```bash
cd go-app2
go build -o app
./app
```

浏览器访问 http://localhost:29817

### AI 配置

点击工具栏齿轮图标，选择模型预设并填写 API 密钥。密钥按模型独立存储，切换预设自动加载。

## 支持的模型

| 模型 | 提供商 | 说明 |
|------|--------|------|
| glm-4v-flash | 智谱 | 免费，Coding Plan 专用接口 |
| glm-4.6v | 智谱 | 专业 OCR，Coding Plan 专用接口 |
| minimax-vlm | MiniMax | Token Plan |

> 智谱 Coding Plan 套餐密钥需使用 `/api/coding/paas/v4/` 专用接口。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | Web 页面 |
| POST | /api/upload | 上传图片 |
| GET | /api/images | 图片列表（含识别结果） |
| POST | /api/ocr/region | 透视框选识别 |
| POST | /api/ocr/:id | 识别指定图片 |
| DELETE | /api/images/:id | 删除图片 |
| PUT | /api/text/:id | 保存编辑文本 |
| PUT | /api/regions/:id | 保存框选区域 |
| POST | /api/reorder | 调整图片顺序 |
| GET | /api/export | 导出 TXT / XLSX |
| GET | /api/config | 获取配置 |
| PUT | /api/config | 保存配置 |

## 项目结构

```
.
├── go-app2/
│   ├── main.go          # Web 服务器（嵌入式前端 + 后端逻辑）
│   ├── vision.go        # AI Vision API 调用（压缩/矫正/去重）
│   ├── go.mod
│   └── go.sum
├── .gitignore
├── LICENSE              # Apache 2.0
└── README.md
```

## License

Apache License 2.0
