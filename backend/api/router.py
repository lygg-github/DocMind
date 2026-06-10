# 导入FastAPI的APIRouter类
from fastapi import APIRouter

# 导入各个子路由模块
from backend.api.routes import auth, chat, documents, sessions

# ========== 创建主路由 ==========
# 创建顶级APIRouter实例
router = APIRouter()
# 注册认证路由
router.include_router(auth.router)
# 注册会话路由
router.include_router(sessions.router)
# 注册对话路由
router.include_router(chat.router)
# 注册文档管理路由
router.include_router(documents.router)
