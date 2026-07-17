"""
API 路由

端点：
  POST /api/v1/upload     - 上传试卷图片，执行 OCR + 索引
  POST /api/v1/ask        - 向 Agent 提问
  GET  /api/v1/search     - 检索知识库
  POST /api/v1/analyze    - 错题分析
  POST /api/v1/practice   - 生成练习题
"""
import json
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import UPLOADS_DIR, DASHSCOPE_API_KEY

# 检查 API Key 是否可用
HAS_API_KEY = bool(DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your-api-key-here")

API_KEY_REQUIRED_MSG = (
    "此功能需要配置 DashScope API Key。"
    "获取方式：https://dashscope.aliyun.com"
)
from backend.modules.image_processor import ImagePreprocessor
from backend.modules.ocr_engine import ocr_engine, OCRResult
from backend.modules.vector_store import vector_store, SearchResult
from backend.modules.agent_core import ReActAgent, Tool, get_agent
from backend.modules.knowledge import (
    knowledge_extractor, error_analyzer, practice_generator,
    KnowledgePoint, ErrorAnalysis, PracticeQuestion
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 图像预处理器
image_processor = ImagePreprocessor()


# ============================================================
# 辅助：初始化 Agent Tools
# ============================================================

def setup_agent_tools(agent: ReActAgent):
    """为 Agent 注册所有工具"""
    if agent.tools:
        return  # 已注册

    # Tool 1: OCR 识别
    def _ocr_recognize(image_path: str) -> str:
        """对上传图像做 OCR 识别"""
        img = image_processor.process(image_path)
        result = ocr_engine.recognize(img, image_path)
        # 存入工作记忆
        agent.memory.set_working_context("current_ocr", result.full_text)
        agent.memory.set_working_context("ocr_boxes", [
            {"text": b.text, "confidence": b.confidence, "is_low": b.is_low_confidence}
            for b in result.boxes
        ])
        low_count = len(result.get_low_confidence_boxes())
        summary = f"识别到 {len(result.boxes)} 个文本块。"
        if low_count:
            summary += f" 其中 {low_count} 个文本块置信度较低，建议人工确认。"
        summary += f"\n\n全文:\n{result.full_text}"
        return summary

    agent.register_tool(Tool(
        name="ocr_recognize",
        description="对上传的试卷图片进行OCR文字识别，返回识别到的全部文本内容。当用户上传了图片时应该首先调用此工具。",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图像文件的路径"
                }
            },
            "required": ["image_path"]
        },
        func=_ocr_recognize
    ))

    # Tool 2: 向量检索
    def _vector_search(query: str, top_k: int = 5) -> str:
        """在知识库中语义检索"""
        results = vector_store.search_hybrid(query, top_k=top_k)
        if not results:
            return "未找到相关内容。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] (相关度: {r.score:.2f}, 来源: {r.source})\n{r.text[:300]}")
        return "\n\n".join(lines)

    agent.register_tool(Tool(
        name="vector_search",
        description="在已经索引的试卷知识库中进行语义搜索，检索与查询相关的题目、知识点和解答。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        func=_vector_search
    ))

    # Tool 3: 以图搜图
    def _image_search(image_path: str, top_k: int = 5) -> str:
        """以图搜图"""
        results = vector_store.search_image(image_path, top_k=top_k)
        if not results:
            return "未找到相似图片。"
        lines = [f"找到 {len(results)} 个相似结果:"]
        for i, r in enumerate(results, 1):
            doc_id = r.metadata.get("document_id", "unknown")
            lines.append(f"[{i}] 文档: {doc_id}, 相似度: {r.score:.2f}")
        return "\n".join(lines)

    agent.register_tool(Tool(
        name="image_search",
        description="以图片搜索相似试卷图片。当用户想找'和这道题长得像的题目'时使用。",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "查询图片路径"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5",
                    "default": 5
                }
            },
            "required": ["image_path"]
        },
        func=_image_search
    ))

    # Tool 4: 知识点抽取
    def _extract_knowledge(question_text: str, subject: str = "数学") -> str:
        """抽取知识点"""
        points = knowledge_extractor.extract(question_text, subject)
        if not points:
            return "未能识别出知识点，请确认题目内容是否完整。"
        lines = ["识别到以下知识点:"]
        for p in points:
            lines.append(f"- {p.name} ({p.category}) [置信度: {p.confidence:.0%}]")
        return "\n".join(lines)

    agent.register_tool(Tool(
        name="extract_knowledge",
        description="从题目文本中自动识别和抽取涉及的知识点。用于标注试卷中每道题考了什么。",
        parameters={
            "type": "object",
            "properties": {
                "question_text": {
                    "type": "string",
                    "description": "题目文本内容"
                },
                "subject": {
                    "type": "string",
                    "description": "学科，如'数学'、'语文'、'英语'",
                    "default": "数学"
                }
            },
            "required": ["question_text"]
        },
        func=_extract_knowledge
    ))

    # Tool 5: 错题分析
    def _analyze_error(
        question: str,
        student_answer: str,
        correct_answer: str
    ) -> str:
        """分析错题"""
        analysis = error_analyzer.analyze(question, student_answer, correct_answer)
        return (
            f"错误类型: {analysis.error_type}\n"
            f"薄弱知识点: {analysis.knowledge_gap}\n"
            f"学习建议: {analysis.suggestion}\n"
            f"题目难度: {analysis.difficulty}"
        )

    agent.register_tool(Tool(
        name="analyze_error",
        description="分析学生错题，判断错误类型（概念性/计算性/审题性/方法性）并给出学习建议。",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "题目原文"
                },
                "student_answer": {
                    "type": "string",
                    "description": "学生的错误答案"
                },
                "correct_answer": {
                    "type": "string",
                    "description": "标准答案"
                }
            },
            "required": ["question", "student_answer", "correct_answer"]
        },
        func=_analyze_error
    ))

    # Tool 6: 生成练习
    def _generate_practice(
        knowledge_points: str,
        error_type: str = "",
        count: int = 3
    ) -> str:
        """生成练习"""
        # knowledge_points 可能是 JSON 数组或逗号分隔的字符串
        try:
            points = json.loads(knowledge_points)
        except json.JSONDecodeError:
            points = [k.strip() for k in knowledge_points.split(",")]

        questions = practice_generator.generate(points, error_type, count)
        if not questions:
            return "未能生成练习题，请稍后重试。"
        lines = [f"生成 {len(questions)} 道练习题:"]
        for i, q in enumerate(questions, 1):
            lines.append(f"\n第{i}题 ({q.difficulty}):\n{q.question}\n答案: {q.answer}")
        return "\n".join(lines)

    agent.register_tool(Tool(
        name="generate_practice",
        description="根据薄弱知识点生成针对性练习题。用于错题整理后做专项训练。",
        parameters={
            "type": "object",
            "properties": {
                "knowledge_points": {
                    "type": "string",
                    "description": "知识点列表，JSON数组或逗号分隔，如'[\"二次函数\",\"最值问题\"]'或'二次函数,最值问题'"
                },
                "error_type": {
                    "type": "string",
                    "description": "错误类型，如'概念性错误'",
                    "default": ""
                },
                "count": {
                    "type": "integer",
                    "description": "生成题目数量，默认3",
                    "default": 3
                }
            },
            "required": ["knowledge_points"]
        },
        func=_generate_practice
    ))

    logger.info(f"Agent 工具注册完成，共 {len(agent.tools)} 个工具")


# 为 practice_generator 注入 vector_store
practice_generator.vector_store = vector_store


# ============================================================
# API 端点
# ============================================================

@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    """
    上传试卷图片，执行 OCR 识别并建立索引

    Returns:
        {
            "document_id": str,
            "full_text": str,
            "box_count": int,
            "low_confidence_count": int
        }
    """
    # 校验文件类型
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"不支持的文件类型: {file.content_type}")

    # 保存文件
    doc_id = document_id or uuid.uuid4().hex[:12]
    file_ext = Path(file.filename).suffix or ".jpg"
    save_path = UPLOADS_DIR / f"{doc_id}{file_ext}"
    content = await file.read()
    save_path.write_bytes(content)

    logger.info(f"上传文件: {file.filename} -> {save_path}")

    # 预处理 + OCR
    img = image_processor.process(str(save_path))
    result = ocr_engine.recognize(img, str(save_path))

    # 索引到向量库
    if result.boxes:
        chunks = [b.text for b in result.boxes if b.text.strip()]
        metadatas = [
            {
                "page": 1,
                "bbox": str(b.bbox),
                "confidence": b.confidence,
                "is_low_confidence": b.is_low_confidence
            }
            for b in result.boxes if b.text.strip()
        ]
        if chunks:
            # 仅索引文本，不触发 CLIP 图像模型下载
            # 需要图像检索时，调用 /api/v1/documents/{id}/index-image
            vector_store.index_document(
                document_id=doc_id,
                text_chunks=chunks,
                chunk_metadatas=metadatas,
                image_path=None
            )

    return {
        "document_id": doc_id,
        "image_path": str(save_path),
        "full_text": result.full_text,
        "box_count": len(result.boxes),
        "low_confidence_count": len(result.get_low_confidence_boxes()),
        "boxes": [
            {
                "text": b.text,
                "confidence": b.confidence,
                "is_low_confidence": b.is_low_confidence,
                "bbox": b.bbox
            }
            for b in result.boxes
        ]
    }


@router.post("/ask")
async def ask_agent(
    question: str = Form(...),
    image_path: Optional[str] = Form(None),
    session_id: Optional[str] = Form("default")
):
    """
    向 Agent 提问

    Args:
        question: 用户问题
        image_path: 可选，关联的图片路径
        session_id: 会话ID（用于区分不同用户的对话历史）

    Returns:
        {"answer": str, "session_id": str}
    """
    agent = get_agent()
    if agent is None:
        return {
            "answer": "Agent 问答功能需要配置 DashScope API Key。\n\n"
                      "获取方式：访问 https://dashscope.aliyun.com 注册并获取 Key，\n"
                      "然后编辑 .env 文件：DASHSCOPE_API_KEY=你的Key\n\n"
                      "OCR 识别和文本检索功能不受影响。",
            "session_id": session_id
        }
    setup_agent_tools(agent)

    # 如果有 image_path，检查文件是否存在
    img = image_path if image_path and Path(image_path).exists() else None

    answer = agent.run(question, img)

    return {
        "answer": answer,
        "session_id": session_id
    }


@router.get("/search")
async def search_knowledge(
    q: str,
    top_k: int = 5
):
    """
    检索知识库

    Args:
        q: 查询文本
        top_k: 返回数量

    Returns:
        {"results": [...]}
    """
    results = vector_store.search_hybrid(q, top_k=top_k)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "text": r.text,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata
            }
            for r in results
        ]
    }


@router.post("/analyze")
async def analyze_wrong_question(
    question: str = Form(...),
    student_answer: str = Form(...),
    correct_answer: str = Form(...)
):
    """
    错题分析

    Returns:
        ErrorAnalysis
    """
    if not HAS_API_KEY:
        return {"error": API_KEY_REQUIRED_MSG}
    result = error_analyzer.analyze(question, student_answer, correct_answer)
    return {
        "error_type": result.error_type,
        "knowledge_gap": result.knowledge_gap,
        "suggestion": result.suggestion,
        "difficulty": result.difficulty
    }


@router.post("/practice")
async def generate_practice(
    knowledge_points: str = Form(...),
    error_type: str = Form(""),
    count: int = Form(3)
):
    """
    生成专项练习

    Args:
        knowledge_points: 知识点（逗号分隔或JSON数组）
        error_type: 错误类型
        count: 题目数量

    Returns:
        {"questions": [...]}
    """
    if not HAS_API_KEY:
        return {"error": API_KEY_REQUIRED_MSG}
    try:
        points = json.loads(knowledge_points) if knowledge_points.startswith("[") else [k.strip() for k in knowledge_points.split(",")]
    except json.JSONDecodeError:
        points = [k.strip() for k in knowledge_points.split(",")]

    questions = practice_generator.generate(points, error_type, count)
    return {
        "count": len(questions),
        "questions": [
            {
                "question": q.question,
                "answer": q.answer,
                "difficulty": q.difficulty,
                "knowledge_points": q.knowledge_points
            }
            for q in questions
        ]
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """删除已索引的文档"""
    vector_store.delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}


@router.get("/documents")
async def list_documents():
    """列出所有已索引的文档"""
    return {"documents": vector_store.get_document_ids()}
