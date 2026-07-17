"""
端到端测试脚本
不需要启动 Web 服务，直接在命令行跑通完整流程：
  生成测试图片 -> 图像预处理 -> OCR识别 -> 向量索引 -> Agent问答

用法：
  python test_e2e.py
"""
import os
import sys
from pathlib import Path

# 国内用户使用 HuggingFace 镜像，加速模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def create_test_image():
    """用 Pillow 生成一张含中文文字的测试试卷图片"""
    from PIL import Image, ImageDraw, ImageFont
    import os

    img = Image.new("RGB", (800, 500), color=(255, 255, 245))
    draw = ImageDraw.Draw(img)

    # 尝试使用系统字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 28)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # 画标题
    draw.text((50, 30), "高三数学模拟试卷（一）", fill=(0, 0, 0), font=font)

    # 画题目
    questions = [
        "1. 已知函数 f(x) = x^2 + 2x + 3，求 f(x) 在区间 [0, 5] 上的最大值。",
        "2. 在三角形ABC中，角A=60度，边a=6，边b=8，求边c的长度。",
        "3. 计算定积分: ∫(0 to 1) x^2 dx 的值。",
        "4. 已知等差数列{an}的首项为2，公差为3，求前10项的和。",
        "5. 解不等式: |2x - 1| < 5",
    ]
    y = 90
    for q in questions:
        draw.text((50, y), q, fill=(30, 30, 30), font=font)
        y += 55

    save_path = ROOT / "data" / "samples" / "test_paper.png"
    img.save(str(save_path))
    print(f"[1] 测试图片已生成: {save_path}")
    return str(save_path)


def test_image_preprocessing(image_path):
    """测试图像预处理"""
    from backend.modules.image_processor import ImagePreprocessor

    processor = ImagePreprocessor()
    img = processor.process(image_path)
    print(f"[2] 图像预处理完成, 尺寸: {img.shape}")
    return img


def test_ocr(img, image_path):
    """测试 OCR 识别"""
    from backend.modules.ocr_engine import ocr_engine

    result = ocr_engine.recognize(img, image_path)
    print(f"[3] OCR 识别完成: {len(result.boxes)} 个文本块")
    for box in result.boxes:
        flag = " [!低置信度]" if box.is_low_confidence else ""
        print(f"    [{box.confidence:.2f}] {box.text[:60]}{flag}")
    return result


def test_vector_index(ocr_result, image_path):
    """测试向量索引"""
    from backend.modules.vector_store import vector_store

    doc_id = "test_paper_001"
    chunks = [b.text for b in ocr_result.boxes if b.text.strip()]
    metadatas = [
        {"page": 1, "confidence": b.confidence}
        for b in ocr_result.boxes if b.text.strip()
    ]

    # 注：image_path 可选，跳过 CLIP 图像向量（首次需下载 ~600MB）
    # CLIP 模型会在首次调用 image_search 时自动下载
    vector_store.index_document(
        document_id=doc_id,
        text_chunks=chunks,
        chunk_metadatas=metadatas,
        image_path=None  # 跳过图像向量，加速演示
    )
    print(f"[4] 向量索引完成: {len(chunks)} 个文本块已入库")


def test_search():
    """测试向量检索"""
    from backend.modules.vector_store import vector_store

    query = "二次函数最大值"
    results = vector_store.search_hybrid(query, top_k=3)
    print(f"[5] 检索测试: '{query}' -> {len(results)} 条结果")
    for r in results:
        print(f"    [{r.score:.2f}] [{r.source}] {r.text[:80]}")


def test_agent(image_path):
    """测试 Agent 问答"""
    from backend.modules.agent_core import ReActAgent, Tool, get_agent
    from backend.modules.vector_store import vector_store
    from backend.modules.knowledge import knowledge_extractor
    from backend.modules.ocr_engine import ocr_engine
    from backend.modules.image_processor import ImagePreprocessor

    ip = ImagePreprocessor()

    agent = get_agent()

    # 注册基础工具（简化版，不依赖API路由）
    def _ocr_tool(img_path: str) -> str:
        img = ip.process(img_path)
        result = ocr_engine.recognize(img, img_path)
        agent.memory.set_working_context("current_ocr", result.full_text)
        return f"识别到 {len(result.boxes)} 个文本块:\n{result.full_text}"

    agent.register_tool(Tool(
        name="ocr_recognize",
        description="对试卷图片进行OCR文字识别",
        parameters={
            "type": "object",
            "properties": {"image_path": {"type": "string", "description": "图片路径"}},
            "required": ["image_path"]
        },
        func=_ocr_tool
    ))

    def _search_tool(query: str, top_k: int = 3) -> str:
        results = vector_store.search_hybrid(query, top_k=top_k)
        if not results:
            return "未找到相关内容"
        return "\n".join(f"[{i+1}] {r.text[:200]}" for i, r in enumerate(results))

    agent.register_tool(Tool(
        name="vector_search",
        description="在试卷知识库中搜索相关内容",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索内容"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3}
            },
            "required": ["query"]
        },
        func=_search_tool
    ))

    def _extract_tool(question_text: str, subject: str = "数学") -> str:
        points = knowledge_extractor.extract(question_text, subject)
        if not points:
            return "未识别出知识点"
        return "\n".join(f"- {p.name} ({p.category})" for p in points)

    agent.register_tool(Tool(
        name="extract_knowledge",
        description="从题目中提取知识点",
        parameters={
            "type": "object",
            "properties": {
                "question_text": {"type": "string", "description": "题目文本"},
                "subject": {"type": "string", "description": "学科", "default": "数学"}
            },
            "required": ["question_text"]
        },
        func=_extract_tool
    ))

    print(f"\n[6] Agent 问答测试 (已注册 {len(agent.tools)} 个工具):")
    print("-" * 60)

    # 第一问：识别试卷
    print("\n>>> 用户: 帮我识别这张试卷的内容")
    answer = agent.run("帮我识别这张试卷的内容", image_path)
    print(f">>> Agent: {answer[:500]}")

    # 第二问：基于已识别的结果提问（不需要再传图片）
    print("\n>>> 用户: 第1题关于二次函数的题目，涉及什么知识点？")
    answer = agent.run("第1题关于二次函数的题目，涉及什么知识点？")
    print(f">>> Agent: {answer[:500]}")

    print("\n" + "=" * 60)
    print("端到端测试完成!")


def main():
    print("=" * 60)
    print("多模态试卷智能解析Agent - 端到端测试")
    print("=" * 60)
    print()

    # Step 1: 生成测试图片
    image_path = create_test_image()

    # Step 2: 图像预处理
    img = test_image_preprocessing(image_path)

    # Step 3: OCR 识别
    ocr_result = test_ocr(img, image_path)

    # Step 4: 向量索引
    test_vector_index(ocr_result, image_path)

    # Step 5: 检索测试
    test_search()

    # Step 6: Agent 问答
    test_agent(image_path)


if __name__ == "__main__":
    main()
