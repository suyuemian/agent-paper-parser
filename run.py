"""
项目启动脚本

用法:
    python run.py               # 完整模式（需要 .env 中的 API Key）
    python run.py --demo        # Demo 模式（无需 Key，OCR + 检索可用）
    python run.py --prod        # 生产模式（关闭热重载）

首次运行会自动下载模型（PaddleOCR ~200MB, MiniLM ~80MB），请耐心等待。
之后使用缓存，秒启动。
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def check_env(demo_mode=False):
    """检查环境配置"""
    from backend.config import DASHSCOPE_API_KEY

    has_key = DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your-api-key-here"

    if not has_key:
        if demo_mode:
            print("[INFO] Demo 模式启动，无需 API Key")
            print("       OCR 识别和文本检索功能可用")
            print("       Agent 智能问答需要配置 API Key，见 .env.example\n")
        else:
            print("=" * 60)
            print()
            print("  [WARN] 未检测到 DashScope API Key")
            print()
            print("  完整功能需要 API Key，获取方式:")
            print("    1. 访问 https://dashscope.aliyun.com")
            print("    2. 注册并开通模型服务，获取 Key")
            print("    3. 编辑 .env 文件: DASHSCOPE_API_KEY=你的Key")
            print()
            print("  没有 Key？试试 Demo 模式:")
            print("    python run.py --demo")
            print()
            print("=" * 60)
            print()

    return has_key


def main():
    parser = argparse.ArgumentParser(description="多模态试卷智能解析Agent")
    parser.add_argument("--demo", action="store_true",
                        help="Demo 模式，无需 API Key（仅 OCR + 检索）")
    parser.add_argument("--prod", action="store_true",
                        help="生产模式，关闭热重载")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口 (默认 8000)")
    args = parser.parse_args()

    has_key = check_env(demo_mode=args.demo)

    import uvicorn

    mode_label = "Demo" if args.demo else ("生产" if args.prod else "开发")
    print(f">>> 启动多模态试卷智能解析Agent [{mode_label}模式]")
    print(f"    前端页面: http://localhost:{args.port}/")
    print(f"    API 文档: http://localhost:{args.port}/docs")
    print(f"    按 Ctrl+C 停止\n")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=not args.prod and not args.demo
    )


if __name__ == "__main__":
    main()
