# 导入JSON模块（用于序列化SSE事件）
import json
# 导入正则表达式（用于解析错误码）
import re

# FastAPI核心组件
from fastapi import APIRouter, Depends, HTTPException
# StreamingResponse用于返回Server-Sent Events流式响应
from fastapi.responses import StreamingResponse

# 从chat服务模块导入同步和流式对话函数
from backend.chat import chat_with_agent, chat_with_agent_stream
# 导入User模型
from backend.db.models import User
# 导入获取当前用户的依赖函数
from backend.infra.auth import get_current_user
# 导入请求和响应的Pydantic模型
from backend.schemas import ChatRequest, ChatResponse

# 创建路由分组，tags=["chat"]用于API文档分组
router = APIRouter(tags=["chat"])


# ========== 同步对话接口 ==========
# POST /chat
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        # 获取session_id，默认值为"default_session"
        session_id = request.session_id or "default_session"
        # 调用chat核心服务获取回复（同步模式）
        resp = chat_with_agent(request.message, current_user.username, session_id)
        # 如果返回的是字典，直接用ChatResponse包装
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        # 否则用response字段包装
        return ChatResponse(response=resp)
    except Exception as e:
        # 异常处理：解析错误信息
        message = str(e)
        # 尝试从错误信息中提取HTTP错误码（如OpenAI的"Error code: 429"）
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            # 429: 限流错误
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            # 401/403: 认证/授权错误
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            # 其他HTTP错误码
            raise HTTPException(status_code=code, detail=message)
        # 未识别的异常，返回500
        raise HTTPException(status_code=500, detail=message)


# ========== 流式对话接口（SSE） ==========
# POST /chat/stream
@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    # 定义异步事件生成器（用于SSE）
    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            # 异步迭代chat服务的流式输出
            async for chunk in chat_with_agent_stream(
                request.message,
                current_user.username,
                session_id,
            ):
                # yield每个chunk给客户端（SSE格式）
                yield chunk
        except Exception as e:
            # 异常时yield一个错误事件
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    # 返回StreamingResponse（SSE流式响应）
    return StreamingResponse(
        event_generator(),                   # 事件生成器
        media_type="text/event-stream",      # MIME类型为SSE
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",  # 禁用缓存
            "Connection": "keep-alive",                            # 保持长连接
            "X-Accel-Buffering": "no",                             # 禁用nginx缓冲
        },
    )
