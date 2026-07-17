# 多模态试卷智能解析 Agent

基于 ReAct Agent 的多模态文档解析系统。上传试卷图片，自动 OCR 识别、向量入库、智能问答。

## 快速开始

### 环境

- Python 3.10+
- Windows / macOS / Linux

### 安装

```bash
git clone https://github.com/suyuemian/agent-paper-parser.git
cd agent4

# 安装依赖（清华源加速，可选）
pip install -r requirements.txt
# 国内用户推荐：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 启动

```bash
# Demo 模式 —— 无需 API Key，OCR + 文本检索可用
python run.py --demo

# 完整模式 —— 配置 API Key 后，全部功能可用
cp .env.example .env   # 编辑 .env 填入 Key
python run.py

# 生产模式 —— 关闭热重载
python run.py --prod
```

浏览器打开 **http://localhost:8000/**，上传图片即可体验。

API 文档：http://localhost:8000/docs

## 项目结构

```
agent4/
├── run.py                  # 启动入口 (支持 --demo / --prod)
├── requirements.txt        # Python 依赖（已锁定关键版本）
├── .env.example            # 环境变量模板
├── test_e2e.py             # 端到端测试脚本
├── backend/
│   ├── main.py             # FastAPI 应用 + 模型预加载
│   ├── config.py           # 全局配置
│   ├── modules/
│   │   ├── image_processor.py   # 图像预处理（CLAHE/二值化/霍夫校正/去噪）
│   │   ├── ocr_engine.py        # PaddleOCR 封装
│   │   ├── vector_store.py      # 双向量存储（BGE + CLIP + Chroma + BM25）
│   │   ├── agent_core.py        # ReAct Agent 引擎（Tool/Memory/降级）
│   │   └── knowledge.py         # 知识点抽取 + 错题分析 + 练习生成
│   └── routes/
│       └── api.py               # REST API（6 个端点 + Tool 注册）
└── frontend/
    └── index.html               # Web 界面（单文件，纯 HTML/JS）
```

## 功能矩阵

| 功能 | 需要 API Key? | 说明 |
|------|:---:|------|
| 图像预处理 | 否 | CLAHE 光照均衡、霍夫变换校正、中值滤波去噪 |
| OCR 识别 | 否 | PaddleOCR，印刷体中英文，置信度标记 |
| 文本向量检索 | 否 | Chroma + BGE 语义检索 + RRF 融合 + BM25 降级 |
| 图片向量检索 | 否 | CLIP 以图搜图（首次下载 ~600MB） |
| Agent 智能问答 | **是** | ReAct 循环，自主决策工具调用 |
| 知识点抽取 | **是** | Qwen-VL 自动标注题目知识点 |
| 错题分析 | **是** | 四种错误类型归因 + 学习建议 |
| 练习生成 | **是** | 基于薄弱知识点的变式题生成 |

API Key 获取：https://dashscope.aliyun.com（阿里云灵积，新用户有免费额度）

## 技术架构

```
┌──────────────────────────────────────────┐
│              用户交互层                    │
│       Web 界面 (HTML/JS 单文件)            │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│          多模态 Agent 引擎 (ReAct)         │
│  Think → Act → Observe → Repeat          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ 意图识别 │ │ 任务调度 │ │ 工具调用 │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│              核心处理层                    │
│  ┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │图像预处理│ │ OCR │ │知识抽取│ │答疑 │ │
│  └────────┘ └──────┘ └──────┘ └──────┘ │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│              数据存储层                    │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │ Chroma 向量库│  │  SQLite / 文件系统│  │
│  │ (BGE + CLIP) │  │  (图片 + 元数据) │  │
│  └──────────────┘  └──────────────────┘  │
└──────────────────────────────────────────┘
```

## 设计决策

### 为什么是 ReAct 而不是 Multi-Agent？

本项目任务流程相对固定（识别→检索→回答，3-5 步），不需要多个 Agent 协作。单 Agent 多工具足够，且调试更简单。选择了 LangChain ReActAgent 实现。

### 为什么是 Chroma 而不是 Milvus？

作为个人项目，Chroma 的 Python 原生体验、零配置部署比 Milvus 的分布式能力更重要。轻松切换——Chroma 和 Milvus 的 API 非常相似。

### Tool 粒度怎么定？

每个 Tool 只做一件事（如 `ocr_recognize` 只做识别，`vector_search` 只做检索）。避免"大而全"的 Tool 会让 Agent 失去对任务流程的掌控。

### 为什么用 Qwen-VL 而不是 GPT-4o？

阿里云 DashScope 国内访问稳定、无网络问题、中文视觉理解好、新用户有免费额度。适合个人开发者。
当然，这取决于个人选择。

## 常见问题

### 启动后首次上传很慢？

首次运行会自动下载模型：
- PaddleOCR 检测+识别模型 (~200MB)
- MiniLM 文本向量模型 (~80MB)  
- CLIP 图像向量模型 (~600MB，仅图片检索需要)

下载后缓存到本地，后续启动秒开。服务启动时会预加载文本嵌入模型。

### `import cv2` 报错怎么办？

PaddleOCR 安装 `opencv-contrib-python`，可能与系统中其他 opencv 版本冲突：

```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-contrib-python==4.10.0.84
```

### PaddleOCR 初始化报 `Unknown argument`？

PaddleOCR 3.x API 相较 2.x 有较大变化。本项目已验证兼容 **PaddlePaddle 3.1.1 + PaddleOCR 3.7**。如果遇到 API 错误，请检查版本：

```bash
pip show paddlepaddle paddleocr  # 确认 paddlepaddle==3.1.1, paddleocr>=3.7
```

### 向量模型下载失败或很慢？

国内用户已默认配置 HuggingFace 镜像 (`hf-mirror.com`)。如果仍有问题：

```bash
# 手动设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Windows 终端输出乱码？

Windows 终端默认使用 GBK 编码，日志中的中文可能显示为乱码。这不影响功能，仅影响日志可读性。在 VS Code 终端或 Windows Terminal 中运行可解决。

### 没有 API Key 能用什么？

`python run.py --demo` 启动后，OCR 识别和文本检索完全可用。只有 Agent 问答、知识点抽取、错题分析这三个 LLM 功能需要 Key。

### Mac / Linux 兼容吗？

代码已处理跨平台兼容性（使用 pathlib 处理路径，无硬编码 Windows 路径）。如果在非 Windows 系统遇到问题，请提 Issue。

## License

MIT
