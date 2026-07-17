"""
项目配置模块
从 .env 文件和环境变量中加载配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# === API 配置 ===
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# === Qwen-VL 模型配置 ===
QWEN_VL_MODEL = "qwen-vl-plus"  # DashScope 上的模型名

# === 路径配置 ===
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# 确保目录存在
for d in [DATA_DIR, SAMPLES_DIR, UPLOADS_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === OCR 配置 ===
OCR_CONFIDENCE_THRESHOLD = 0.6   # 低置信度阈值
OCR_USE_ANGLE_CLS = True         # 是否使用文本方向分类

# === 向量检索配置 ===
VECTOR_TOP_K = 5                 # 默认检索返回数
# 生产环境推荐: "BAAI/bge-m3" (中文效果好，约2GB)
# 演示环境用轻量模型: "all-MiniLM-L6-v2" (80MB，秒下)
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# === Agent 配置 ===
AGENT_MAX_STEPS = 5              # ReAct 最大循环步数
AGENT_MAX_RETRIES = 2            # LLM 输出格式错误时最大重试次数
AGENT_MEMORY_ROUNDS = 5          # 短期记忆保留对话轮数
