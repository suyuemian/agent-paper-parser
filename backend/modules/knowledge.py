"""
知识点抽取与错题分析模块

功能：
  1. 知识点自动标注（基于 LLM 信息抽取）
  2. 错题归因分析（概念性/计算性/审题性/方法性错误）
  3. 专项练习推荐（基于向量检索的相似题匹配）
"""
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from backend.config import DASHSCOPE_API_KEY, QWEN_VL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class KnowledgePoint:
    """知识点"""
    name: str               # 知识点名称，如"二次函数最值"
    category: str           # 学科分类，如"数学/代数/函数"
    confidence: float       # 置信度 [0, 1]


@dataclass
class ErrorAnalysis:
    """错题分析结果"""
    error_type: str         # "概念性错误" | "计算性错误" | "审题性错误" | "方法性错误"
    knowledge_gap: str      # 薄弱知识点
    suggestion: str         # 学习建议
    difficulty: str         # 题目难度 "easy" | "medium" | "hard"


@dataclass
class PracticeQuestion:
    """练习题"""
    question: str
    answer: str
    difficulty: str
    knowledge_points: List[str]


class KnowledgeExtractor:
    """知识点抽取器"""

    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY

    def extract(
        self,
        question_text: str,
        subject: str = "数学"
    ) -> List[KnowledgePoint]:
        """
        从题目文本中抽取知识点

        Args:
            question_text: 题目文本
            subject: 学科（数学/语文/英语等）

        Returns:
            知识点列表
        """
        prompt = f"""你是一个{subject}教育专家。请分析以下题目，提取涉及的知识点。

题目:
{question_text}

请以JSON数组格式返回知识点列表，每个知识点包含:
- name: 知识点名称（如"二次函数最值问题"）
- category: 知识分类路径（如"代数/函数/二次函数"）
- confidence: 置信度(0-1)

只返回JSON，不要其他内容。示例:
[{{"name": "勾股定理", "category": "几何/三角形", "confidence": 0.95}}]"""

        response = self._call_llm(prompt)

        try:
            # 尝试解析 JSON
            data = json.loads(response)
            return [
                KnowledgePoint(
                    name=item["name"],
                    category=item["category"],
                    confidence=item["confidence"]
                )
                for item in data
            ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"知识点解析失败: {e}, 原始回复: {response[:200]}")
            return []

    def _call_llm(self, prompt: str) -> str:
        """调用 Qwen-VL 纯文本模式"""
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.api_key

        response = Generation.call(
            model="qwen-plus",  # 纯文本任务用 qwen-plus 更高效
            prompt=prompt,
            result_format="message"
        )

        if response.output and response.output.choices:
            return response.output.choices[0].message.content
        return "[]"


class ErrorAnalyzer:
    """错题归因分析器"""

    ERROR_TYPES = ["概念性错误", "计算性错误", "审题性错误", "方法性错误"]

    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY

    def analyze(
        self,
        question: str,
        student_answer: str,
        correct_answer: str
    ) -> ErrorAnalysis:
        """
        分析错题的错误原因

        Args:
            question: 题目原文
            student_answer: 学生的错误答案
            correct_answer: 标准答案

        Returns:
            ErrorAnalysis: 错误分析结果
        """
        prompt = f"""你是一位有经验的数学老师。请分析以下学生的错题，判断错误类型。

题目:
{question}

学生答案:
{student_answer}

标准答案:
{correct_answer}

请判断错误属于以下哪种类型：
1. 概念性错误 - 学生对知识点本身理解有误
2. 计算性错误 - 计算过程出错，但思路正确
3. 审题性错误 - 没有正确理解题目要求
4. 方法性错误 - 解题方法选择不当

请以JSON格式返回:
{{
    "error_type": "错误类型",
    "knowledge_gap": "具体薄弱的知识点",
    "suggestion": "针对性的学习建议",
    "difficulty": "easy/medium/hard"
}}

只返回JSON，不要其他内容。"""

        response = self._call_llm(prompt)

        try:
            data = json.loads(response)
            return ErrorAnalysis(
                error_type=data.get("error_type", "未知"),
                knowledge_gap=data.get("knowledge_gap", ""),
                suggestion=data.get("suggestion", ""),
                difficulty=data.get("difficulty", "medium")
            )
        except json.JSONDecodeError:
            return ErrorAnalysis(
                error_type="未知",
                knowledge_gap="",
                suggestion="请重新分析",
                difficulty="medium"
            )

    def _call_llm(self, prompt: str) -> str:
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.api_key

        response = Generation.call(
            model="qwen-plus",
            prompt=prompt,
            result_format="message"
        )

        if response.output and response.output.choices:
            return response.output.choices[0].message.content
        return "{}"


class PracticeGenerator:
    """专项练习生成器"""

    def __init__(self, vector_store=None):
        self.api_key = DASHSCOPE_API_KEY
        self.vector_store = vector_store  # 延迟注入

    def generate(
        self,
        knowledge_points: List[str],
        error_type: str = "",
        count: int = 3
    ) -> List[PracticeQuestion]:
        """
        生成针对性练习题

        策略：
        1. 先从知识库检索相似题目
        2. 用 LLM 改编生成变式题
        """
        # Step 1: 向量检索相似题
        similar_questions = []
        if self.vector_store:
            query = " ".join(knowledge_points)
            results = self.vector_store.search_text(query, top_k=count)
            similar_questions = [r.text for r in results]

        # Step 2: LLM 生成练习题
        prompt = f"""你是一位数学老师。请为以下知识点生成{count}道练习题。

知识点: {', '.join(knowledge_points)}
错误类型: {error_type or '不限'}
难度: 由易到难

参考题目:
{chr(10).join(f'- {q}' for q in similar_questions) if similar_questions else '无'}

请以JSON数组格式返回，每道题包含:
- question: 题目
- answer: 答案
- difficulty: easy/medium/hard
- knowledge_points: 知识点列表

只返回JSON，不要其他内容。"""

        response = self._call_llm(prompt)

        try:
            data = json.loads(response)
            return [
                PracticeQuestion(
                    question=item["question"],
                    answer=item["answer"],
                    difficulty=item.get("difficulty", "medium"),
                    knowledge_points=item.get("knowledge_points", knowledge_points)
                )
                for item in data
            ]
        except json.JSONDecodeError:
            return []

    def _call_llm(self, prompt: str) -> str:
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.api_key

        response = Generation.call(
            model="qwen-plus",
            prompt=prompt,
            result_format="message"
        )

        if response.output and response.output.choices:
            return response.output.choices[0].message.content
        return "[]"


# 全局单例
knowledge_extractor = KnowledgeExtractor()
error_analyzer = ErrorAnalyzer()
practice_generator = PracticeGenerator()
