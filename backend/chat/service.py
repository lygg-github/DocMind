# ========== 导入部分 ==========
import asyncio  # 异步并发支持
import json  # JSON 序列化，用于 SSE 流式响应

# LangChain 消息类型
from langchain_core.messages import (
    AIMessage,        # AI 消息
    AIMessageChunk,   # AI 流式消息块
    HumanMessage,     # 用户消息
    SystemMessage,    # 系统消息
)

# 导入聊天组件
from backend.chat.runtime import agent, fast_model  # LangGraph Agent 和快速模型
from backend.chat.storage import ConversationStorage  # 对话存储服务
from backend.chat.rag_context import get_last_rag_context  # 获取 RAG 上下文
from backend.chat.streaming import set_rag_step_queue  # 设置 RAG 步骤队列
from backend.tools import reset_knowledge_tool_calls  # 重置知识工具调用

# ========== 全局变量 ==========
# 对话存储实例
storage = ConversationStorage()

# 上下文窗口大小：保留最近 6 轮对话
CONTEXT_WINDOW_MESSAGES = 6


# ========== 辅助函数 ==========
def _build_context_messages(
    messages: list,
    persistent_note: str,
    user_text: str,
) -> list:
    """
    构建上下文消息列表
    
    Args:
        messages: 历史消息列表
        persistent_note: 持久化笔记（工作记忆）
        user_text: 当前用户输入
    
    Returns:
        完整的上下文消息列表
    
    设计：
    - 保留最近 6 轮对话（短记忆）
    - 添加持久化笔记（长记忆）
    - 追加当前用户输入
    """
    # 截取最近 6 轮对话（防止超出上下文窗口）
    short_term = messages[-CONTEXT_WINDOW_MESSAGES:] if len(messages) > CONTEXT_WINDOW_MESSAGES else messages
    
    # 构建上下文消息列表
    context_messages: list = []
    
    # 如果有持久化笔记，添加系统消息
    if persistent_note:
        context_messages.append(
            SystemMessage(
                content=(
                    "【对话持久化笔记（你的工作记忆）】\n"
                    f"{persistent_note}\n"
                    "请参考以上笔记保持对话连贯性，避免重复回答已解决的问题。"
                )
            )
        )
    
    # 添加短记忆（最近 6 轮）
    context_messages.extend(short_term)
    
    # 添加当前用户输入
    context_messages.append(HumanMessage(content=user_text))
    
    return context_messages


async def update_persistent_note(
    current_note: str,
    user_text: str,
    ai_response: str,
) -> str:
    """
    异步更新持久化笔记
    
    Args:
        current_note: 当前笔记
        user_text: 用户输入
        ai_response: AI 回复
    
    Returns:
        更新后的笔记
    
    设计：使用线程池执行同步任务，避免阻塞事件循环
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _update_persistent_note_sync(current_note, user_text, ai_response),
    )


def _generate_session_title_sync(user_text: str) -> str:
    """
    同步生成会话标题
    
    Args:
        user_text: 用户首次输入
    
    Returns:
        会话标题（10 字以内）
    
    原理：使用快速模型总结用户问题生成标题
    """
    try:
        # 构建提示词
        prompt = (
            "请根据用户的首次提问，生成一个简短的对话标题（控制在 10 个字以内，不要标点）。\n"
            f"用户提问：{user_text}"
        )
        # 调用快速模型
        res = fast_model.invoke([SystemMessage(content=prompt)])
        # 清理标题（去除引号和句号）
        title = (res.content or "").strip().strip('"').strip("。")
        return title or "新会话"
    except Exception as e:
        print(f"Title generation error: {e}")
        return "新会话"


async def generate_session_title(user_text: str) -> str:
    """
    异步生成会话标题
    
    Args:
        user_text: 用户首次输入
    
    Returns:
        会话标题
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _generate_session_title_sync(user_text))


def _update_persistent_note_sync(current_note: str, user_text: str, ai_response: str) -> str:
    """
    同步更新持久化笔记（Context Manager Agent）
    
    Args:
        current_note: 当前笔记
        user_text: 用户输入
        ai_response: AI 回复
    
    Returns:
        更新后的笔记
    
    核心逻辑：
    1. 智能合并新旧信息（不是简单拼接）
    2. 过滤噪音，控制在 500 字以内
    3. 冲突时保留最可靠或最新版本
    """
    try:
        # 构建提示词（Context Manager Agent）
        prompt = (
            "你是一个【Context Manager Agent】(上下文管理器)，负责维护多轮对话中的「持久化笔记」。\n"
            "笔记是模型在有限上下文窗口下的长效工作记忆，记录已解决的问题与关键事实。\n\n"
            "更新规则：\n"
            "1. 将新信息与现有笔记智能合并，不要简单拼接。\n"
            "2. 过滤噪音，控制在 500 字以内，用简明条目输出。\n"
            "3. 若信息冲突，保留最可靠或最新版本。\n\n"
            f"▼ 现有笔记：\n{current_note if current_note else '无'}\n\n"
            f"▼ 最新一轮对话：\n用户：{user_text}\nAI：{ai_response}\n\n"
            "请直接输出更新后的笔记（纯文本，不要解释或 Markdown 代码块）："
        )
        # 调用快速模型更新笔记
        res = fast_model.invoke([SystemMessage(content=prompt)])
        return (res.content or "").strip()
    except Exception as e:
        print(f"Context Manager Error: {e}")
        return current_note


# ========== 核心函数：同步聊天 ==========
def chat_with_agent(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    """
    同步聊天函数（非流式）
    
    Args:
        user_text: 用户输入
        user_id: 用户 ID
        session_id: 会话 ID
    
    Returns:
        包含回复和 RAG 轨迹的字典
    
    执行流程：
    1. 加载历史对话
    2. 构建上下文
    3. 调用 Agent
    4. 保存对话
    5. 更新持久化笔记
    """
    # ========== 加载历史对话 ==========
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    # ========== 重置 RAG 上下文 ==========
    get_last_rag_context(clear=True)
    reset_knowledge_tool_calls()

    # ========== 构建上下文消息 ==========
    context_messages = _build_context_messages(messages, persistent_note, user_text)
    
    # ========== 保存用户输入 ==========
    messages.append(HumanMessage(content=user_text))
    storage.save(user_id, session_id, messages)

    # ========== 调用 Agent ==========
    result = agent.invoke(
        {"messages": context_messages},
        config={"recursion_limit": 8},  # 递归限制（防止无限循环）
    )

    # ========== 提取回复内容 ==========
    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    # ========== 保存 AI 回复 ==========
    messages.append(AIMessage(content=response_content))

    # ========== 获取 RAG 轨迹 ==========
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # ========== 更新元数据 ==========
    save_meta = dict(metadata)
    if is_first_message:
        save_meta["title"] = _generate_session_title_sync(user_text)  # 生成标题
    save_meta["persistent_note"] = _update_persistent_note_sync(
        persistent_note, user_text, response_content  # 更新笔记
    )

    # ========== 保存对话（含 RAG 轨迹） ==========
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(
        user_id,
        session_id,
        messages,
        metadata=save_meta,
        extra_message_data=extra_message_data,
    )

    return {
        "response": response_content,  # AI 回复
        "rag_trace": rag_trace,        # RAG 检索轨迹
    }


# ========== 核心函数：流式聊天 ==========
async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    """
    异步流式聊天函数（SSE 格式）
    
    Args:
        user_text: 用户输入
        user_id: 用户 ID
        session_id: 会话 ID
    
    Yields:
        SSE 格式的事件数据
    
    事件类型：
    - content: AI 回复内容（流式）
    - rag_step: RAG 检索步骤
    - session_title: 会话标题
    - trace: RAG 检索轨迹
    - [DONE]: 结束标记
    
    执行流程：
    1. 加载历史对话
    2. 启动标题生成任务（首次消息）
    3. 启动 Agent 工作协程
    4. 循环输出事件
    5. 保存对话和更新笔记
    """
    # ========== 加载历史对话 ==========
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    is_first_message = len(messages) == 0

    # ========== 重置 RAG 上下文 ==========
    get_last_rag_context(clear=True)
    reset_knowledge_tool_calls()

    # ========== 创建输出队列 ==========
    output_queue = asyncio.Queue()

    # ========== RAG 步骤代理 ==========
    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    # ========== 构建上下文消息 ==========
    context_messages = _build_context_messages(messages, persistent_note, user_text)
    
    # ========== 保存用户输入 ==========
    messages.append(HumanMessage(content=user_text))
    storage.save(user_id, session_id, messages)

    # ========== 启动标题生成任务（首次消息） ==========
    title_task = None
    if is_first_message:
        def _on_title_done(fut):
            try:
                title = fut.result()
                output_queue.put_nowait(
                    {"type": "session_title", "title": title, "session_id": session_id}
                )
            except Exception as e:
                print(f"Title task error: {e}")

        title_task = asyncio.create_task(generate_session_title(user_text))
        title_task.add_done_callback(_on_title_done)

    # ========== 完整回复累积 ==========
    full_response = ""

    # ========== Agent 工作协程 ==========
    async def _agent_worker():
        nonlocal full_response
        try:
            # 流式调用 Agent
            async for msg, _metadata in agent.astream(
                {"messages": context_messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                # 跳过非 AI 消息块
                if not isinstance(msg, AIMessageChunk):
                    continue
                # 跳过工具调用块
                if getattr(msg, "tool_call_chunks", None):
                    continue

                # 提取内容
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                # 发送内容到输出队列
                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            await output_queue.put(None)  # 结束标记

    agent_task = asyncio.create_task(_agent_worker())

    # ========== 主循环：输出事件 ==========
    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()

    # ========== 获取 RAG 轨迹 ==========
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # ========== 输出 RAG 轨迹 ==========
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # ========== 输出结束标记 ==========
    yield "data: [DONE]\n\n"

    # ========== 保存对话 ==========
    save_meta = dict(metadata)
    if is_first_message and title_task is not None:
        try:
            save_meta["title"] = await title_task
        except Exception:
            pass

    try:
        save_meta["persistent_note"] = await update_persistent_note(
            persistent_note, user_text, full_response
        )
    except Exception as e:
        print(f"Update persistent note error: {e}")

    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(
        user_id,
        session_id,
        messages,
        metadata=save_meta,
        extra_message_data=extra_message_data,
    )
