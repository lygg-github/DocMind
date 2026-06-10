# FastAPI核心组件
from fastapi import APIRouter, Depends, HTTPException

# 导入会话存储模块
from backend.chat import storage
# 导入User模型
from backend.db.models import User
# 导入获取当前用户依赖
from backend.infra.auth import get_current_user
# 导入Pydantic响应模型
from backend.schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

# 创建路由分组
router = APIRouter(tags=["sessions"])


# ========== 获取会话消息列表接口 ==========
# GET /sessions/{session_id}
@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        # 从存储中获取会话消息，并转换为MessageInfo对象
        messages = [
            MessageInfo(
                type=msg["type"],                    # 消息类型（user/assistant）
                content=msg["content"],              # 消息内容
                timestamp=msg["timestamp"],           # 时间戳
                rag_trace=msg.get("rag_trace"),       # RAG检索轨迹（可选）
            )
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 获取用户所有会话列表接口 ==========
# GET /sessions
@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    try:
        # 获取用户的所有会话信息
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        # 按更新时间倒序
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 删除会话接口 ==========
# DELETE /sessions/{session_id}
@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    try:
        # 删除会话
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
