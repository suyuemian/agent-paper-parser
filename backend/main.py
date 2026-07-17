"""
FastAPI 主入口
多模态古籍/试卷智能解析Agent - Web API服务
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import UPLOADS_DIR, DASHSCOPE_API_KEY
from backend.routes.api import router as api_router

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# API Key 可用性
HAS_API_KEY = bool(DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your-api-key-here")

# 创建 FastAPI 应用
app = FastAPI(
    title="多模态试卷智能解析Agent",
    description="基于 ReAct Agent 的试卷OCR识别、知识点抽取与智能答疑系统",
    version="0.1.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router, prefix="/api/v1")

# 确保上传目录存在
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 前端静态文件
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    """前端页面"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "多模态试卷智能解析Agent", "status": "running"}


@app.get("/health")
async def health():
    """健康检查 + 服务能力"""
    return {
        "status": "ok",
        "has_api_key": HAS_API_KEY,
        "features": {
            "ocr": True,
            "vector_search": True,
            "agent_qa": HAS_API_KEY,
            "knowledge_extraction": HAS_API_KEY,
            "error_analysis": HAS_API_KEY
        }
    }


@app.on_event("startup")
async def startup():
    """启动时初始化"""
    logger.info("Agent 服务启动中...")

    # 预加载文本嵌入模型
    try:
        from backend.modules.vector_store import vector_store
        _ = vector_store.text_embedder._load()
        logger.info("文本嵌入模型就绪")
    except Exception as e:
        logger.warning(f"文本嵌入模型预加载失败（将在首次请求时重试）: {e}")

    if HAS_API_KEY:
        logger.info("API Key 已配置，全部功能可用")
    else:
        logger.info("API Key 未配置，Agent 问答功能不可用")
        logger.info("OCR 识别和文本检索仍可正常使用")

    logger.info(f"前端页面: http://localhost:8000/")
    logger.info(f"API 文档: http://localhost:8000/docs")
    logger.info("Agent 服务就绪")
