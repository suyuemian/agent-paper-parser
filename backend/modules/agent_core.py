"""
Agent 核心引擎

基于 ReAct 模式的多模态智能体：
  - Think -> Act -> Observe -> Repeat
  - 6 个自定义 Tool
  - 3 层 Memory 架构
  - 容错降级机制
"""
import json
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from backend.config import (
    DASHSCOPE_API_KEY, QWEN_VL_MODEL,
    AGENT_MAX_STEPS, AGENT_MAX_RETRIES, AGENT_MEMORY_ROUNDS
)

logger = logging.getLogger(__name__)


# ============================================================
# Tool 定义
# ============================================================

@dataclass
class Tool:
    """Agent 可调用的工具"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式的参数定义
    func: Callable              # 实际执行函数

    def to_openai_tool(self) -> Dict[str, Any]:
        """转为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        try:
            result = self.func(**kwargs)
            # 如果结果是 dict/list，序列化为 JSON 字符串
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {e}"


# ============================================================
# Memory 管理
# ============================================================

@dataclass
class ConversationTurn:
    """单轮对话记录"""
    role: str           # "user" | "assistant" | "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    """
    Agent 三层记忆架构：
    1. 短期记忆：最近 N 轮对话（保留在上下文中）
    2. 工作记忆：当前文档的 OCR 全文
    3. 长期记忆：用户知识库（通过 vector_store 检索加载）
    """

    def __init__(self, max_rounds: int = AGENT_MEMORY_ROUNDS):
        self.max_rounds = max_rounds
        self.short_term: List[ConversationTurn] = []
        self.working: Dict[str, Any] = {}  # 工作区（当前文档上下文）

    def add_turn(self, role: str, content: str, **meta):
        self.short_term.append(ConversationTurn(role=role, content=content, metadata=meta))
        # 只保留最近 N 轮
        if len(self.short_term) > self.max_rounds * 2:  # 每轮=user+assistant，所以乘2
            self.short_term = self.short_term[-self.max_rounds * 2:]

    def set_working_context(self, key: str, value: Any):
        """设置工作记忆"""
        self.working[key] = value

    def get_working_context(self, key: str) -> Any:
        return self.working.get(key)

    def get_conversation_history(self) -> str:
        """获取格式化的对话历史"""
        if not self.short_term:
            return "（无对话历史）"

        lines = []
        for turn in self.short_term[-self.max_rounds * 2:]:
            role_name = {"user": "用户", "assistant": "助手", "tool": "工具"}
            lines.append(f"[{role_name.get(turn.role, turn.role)}]: {turn.content}")
        return "\n".join(lines)

    def clear(self):
        self.short_term.clear()
        self.working.clear()


# ============================================================
# Agent 核心
# ============================================================

# System Prompt 模板
SYSTEM_PROMPT = """你是一个多模态试卷智能解析助手。你可以：

1. **识别试卷内容**：调用 ocr_recognize 工具识别用户上传的试卷图片
2. **检索知识**：调用 vector_search 在知识库中检索相关内容
3. **以图搜题**：调用 image_search 查找相似的题目图片
4. **分析知识点**：调用 extract_knowledge 提取题目涉及的知识点
5. **分析错题**：调用 analyze_error 分析学生错题的错误原因
6. **生成练习**：调用 generate_practice 生成针对性练习题

## 工作流程

当用户上传试卷图片时，按以下步骤处理：
1. 先用 ocr_recognize 识别图片内容
2. 用 extract_knowledge 标注知识点
3. 如果用户有疑问，用 vector_search 检索相关知识后回答
4. 如果是错题分析需求，用 analyze_error 分析错误原因

## 重要规则
- 每次只调用一个工具，调用后等待结果再决定下一步
- 如果工具执行失败，尝试换一种方式或告知用户
- 如果 OCR 识别结果中有低置信度标记，提醒用户手动确认
- 回答要分步骤、有逻辑，像老师在教学生
- 用中文回答

## 当前上下文
{working_context}

## 对话历史
{conversation_history}"""


class ReActAgent:
    """ReAct Agent 引擎"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY")

        self.model = QWEN_VL_MODEL
        self.max_steps = AGENT_MAX_STEPS
        self.max_retries = AGENT_MAX_RETRIES
        self.memory = AgentMemory()
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    # ---------- LLM 调用 ----------

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        调用 Qwen-VL API (DashScope)

        Returns:
            {
                "type": "text" | "tool_call",
                "content": str,          # type=text 时
                "tool_name": str,        # type=tool_call 时
                "tool_args": dict        # type=tool_call 时
            }
        """
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = self.api_key

        try:
            # 构建 DashScope 格式的消息
            dashscope_messages = self._format_messages(messages)

            if tools:
                # 带工具定义的调用
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=dashscope_messages,
                    tools=tools
                )
            else:
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=dashscope_messages
                )

            # 解析响应
            output = response.output
            if output is None:
                raise RuntimeError(f"API 返回为空: {response}")

            if hasattr(output, 'choices') and output.choices:
                choice = output.choices[0]
                message = choice.message

                # 检查是否是工具调用
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    return {
                        "type": "tool_call",
                        "tool_name": tool_call["function"]["name"],
                        "tool_args": json.loads(tool_call["function"]["arguments"])
                    }

                # 普通文本回复
                content_text = ""
                if hasattr(message, 'content'):
                    content_list = message.content if isinstance(message.content, list) else [message.content]
                    for item in content_list:
                        if isinstance(item, dict):
                            content_text += item.get("text", "")
                        elif isinstance(item, str):
                            content_text += item

                return {"type": "text", "content": content_text.strip()}

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析工具参数失败 (重试 {retry_count+1}/{self.max_retries}): {e}")
            if retry_count < self.max_retries:
                return self._call_llm(messages, tools, retry_count + 1)
            return {"type": "text", "content": "抱歉，工具调用参数解析失败，请换一种方式提问。"}

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"type": "text", "content": f"抱歉，服务暂时不可用: {e}"}

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """将标准消息格式转为 DashScope 格式"""
        formatted = []
        for msg in messages:
            content_list = []
            role = msg["role"]

            if isinstance(msg.get("content"), str):
                content_list.append({"text": msg["content"]})
            elif isinstance(msg.get("content"), list):
                content_list = msg["content"]

            formatted.append({
                "role": role,
                "content": content_list
            })
        return formatted

    # ---------- Agent 循环 ----------

    def run(
        self,
        user_input: str,
        image_path: Optional[str] = None
    ) -> str:
        """
        执行 Agent 主循环

        Args:
            user_input: 用户文本输入
            image_path: 用户上传的图片路径（可选）

        Returns:
            Agent 的最终回答
        """
        # 初始化上下文
        self.memory.add_turn("user", user_input)

        # 构建初始消息
        messages = self._build_initial_messages(user_input, image_path)

        # 主循环
        for step in range(self.max_steps):
            logger.info(f"--- Agent Step {step + 1}/{self.max_steps} ---")

            # Think & Act: 调用 LLM
            tool_defs = [t.to_openai_tool() for t in self.tools.values()] if self.tools else None
            response = self._call_llm(messages, tool_defs)

            if response["type"] == "text":
                # LLM 直接回答 -> 结束
                self.memory.add_turn("assistant", response["content"])
                return response["content"]

            elif response["type"] == "tool_call":
                tool_name = response["tool_name"]
                tool_args = response["tool_args"]
                logger.info(f"  Tool: {tool_name}({tool_args})")

                # 添加助手消息（含工具调用）
                messages.append({
                    "role": "assistant",
                    "content": f"调用工具 {tool_name}"
                })

                # Observe: 执行工具
                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = f"未知工具: {tool_name}"
                else:
                    observation = tool.execute(**tool_args)

                logger.info(f"  Observation: {observation[:200]}...")

                # 添加工具结果
                messages.append({
                    "role": "user",
                    "content": f"工具 {tool_name} 返回结果:\n{observation}"
                })
                self.memory.add_turn("tool", f"{tool_name}: {observation}")

        # 达到最大步数
        return "抱歉，处理步骤过多，请简化您的问题重试。"

    def _build_initial_messages(
        self,
        user_input: str,
        image_path: Optional[str] = None
    ) -> List[Dict]:
        """构建初始消息列表"""

        # 系统消息
        working_ctx = json.dumps(self.memory.working, ensure_ascii=False) if self.memory.working else "无"
        conv_history = self.memory.get_conversation_history()

        system_content = SYSTEM_PROMPT.format(
            working_context=working_ctx,
            conversation_history=conv_history
        )

        messages = [{"role": "system", "content": system_content}]

        # 用户消息
        if image_path and Path(image_path).exists():
            # 多模态消息：文本 + 图片
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()

            messages.append({
                "role": "user",
                "content": [
                    {"text": user_input},
                    {"image": f"data:image/jpeg;base64,{img_base64}"}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_input})

        return messages


# 全局 Agent 实例（延迟初始化）
_agent_instance: Optional[ReActAgent] = None


def get_agent() -> Optional[ReActAgent]:
    """获取全局 Agent 单例。无 API Key 时返回 None。"""
    global _agent_instance
    if _agent_instance is None:
        if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your-api-key-here":
            logger.warning("API Key 未配置，Agent 不可用")
            return None
        _agent_instance = ReActAgent()
    return _agent_instance
