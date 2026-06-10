# 从backend.api.router模块导入主router
from backend.api.router import router

# 公开的API（外部可导入router）
__all__ = ["router"]
