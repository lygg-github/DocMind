# 从FastAPI导入核心组件：APIRouter用于创建路由分组，Depends用于依赖注入
from fastapi import APIRouter, Depends, HTTPException
# 从SQLAlchemy导入Session类（数据库会话类型）
from sqlalchemy.orm import Session

# 导入User数据库模型
from backend.db.models import User
# 导入认证相关函数
from backend.infra.auth import (
    authenticate_user,      # 验证用户登录
    create_access_token,    # 创建JWT访问令牌
    get_current_user,       # 获取当前登录用户
    get_db,                 # 获取数据库会话
    get_password_hash,      # 密码哈希加密
    resolve_role,           # 解析用户角色（处理admin_code）
)
# 导入认证相关的数据模式（Pydantic模型）
from backend.schemas import AuthResponse, CurrentUserResponse, LoginRequest, RegisterRequest

# 创建APIRouter实例，tags=["auth"]用于API文档分组
router = APIRouter(tags=["auth"])


# ========== 用户注册接口 ==========
# POST /auth/register，response_model指定返回类型为AuthResponse
@router.post("/auth/register", response_model=AuthResponse)
# 异步函数：request是注册请求体，db是数据库会话（通过Depends自动注入）
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # 获取用户名（去除首尾空白）
    username = (request.username or "").strip()
    # 获取密码（去除首尾空白）
    password = (request.password or "").strip()
    # 校验必填字段
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    # 检查用户名是否已存在
    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    # 解析用户角色（如果传admin_code则可能提升为admin）
    role = resolve_role(request.role, request.admin_code)
    # 创建User对象：密码经过哈希加密后存储
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    # 添加到数据库会话
    db.add(user)
    # 提交事务
    db.commit()

    # 生成JWT访问令牌
    token = create_access_token(username=username, role=role)
    # 返回认证响应（token、用户名、角色）
    return AuthResponse(access_token=token, username=username, role=role)


# ========== 用户登录接口 ==========
# POST /auth/login
@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    # 验证用户凭据（用户名和密码）
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    # 登录成功，生成token
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


# ========== 获取当前用户信息接口 ==========
# GET /auth/me，Depends(get_current_user)会自动从JWT token解析当前用户
@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    # 返回当前登录用户的信息
    return CurrentUserResponse(username=current_user.username, role=current_user.role)
