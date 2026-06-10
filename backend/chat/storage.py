# ========== 导入部分 ==========
from datetime import datetime  # 日期时间处理，用于记录消息时间戳

# LangChain 消息类型
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 导入数据库模型
from backend.db.models import ChatMessage, ChatSession, User
# 导入 Redis 缓存客户端
from backend.infra.cache import cache
# 导入数据库会话工厂
from backend.infra.database import SessionLocal


# ========== 核心类 ==========
class ConversationStorage:
    """
    对话存储（PostgreSQL + Redis）
    
    设计目的：
    - 持久化存储用户对话（PostgreSQL）
    - 缓存热点对话（Redis）
    - 支持 LangChain 消息格式转换
    
    存储结构：
    - ChatSession：会话表（存储元数据：标题、持久化笔记等）
    - ChatMessage：消息表（存储具体对话内容）
    - Redis：缓存会话列表和消息列表
    """

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        """
        生成消息缓存键
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
        
        Returns:
            Redis 键名，格式：chat_messages:{user_id}:{session_id}
        """
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        """
        生成会话列表缓存键
        
        Args:
            user_id: 用户 ID
        
        Returns:
            Redis 键名，格式：chat_sessions:{user_id}
        """
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        """
        将数据库记录转换为 LangChain 消息对象
        
        Args:
            records: 数据库记录列表
        
        Returns:
            LangChain 消息对象列表
        
        消息类型映射：
        - human → HumanMessage
        - ai → AIMessage
        - system → SystemMessage
        """
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict = None,
        extra_message_data: list = None,
    ):
        """
        保存对话到数据库和缓存
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            messages: LangChain 消息对象列表
            metadata: 会话元数据（标题、持久化笔记等）
            extra_message_data: 每条消息的额外数据（如 RAG 轨迹）
        
        执行流程：
        1. 确保用户存在
        2. 确保会话存在（或更新元数据）
        3. 删除旧消息
        4. 插入新消息
        5. 写入 Redis 缓存
        """
        db = SessionLocal()
        try:
            # ========== 确保用户存在 ==========
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            # ========== 确保会话存在 ==========
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                # 会话不存在：创建新会话
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()  # 获取 session.id
            elif metadata is not None:
                # 会话已存在：合并元数据
                existing_meta = session.metadata_json or {}
                session.metadata_json = {**existing_meta, **metadata}

            # ========== 删除旧消息（全量替换） ==========
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            # ========== 序列化消息 ==========
            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                # 获取额外数据（如 RAG 轨迹）
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")

                # 写入数据库
                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                
                # 序列化（用于缓存）
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            # 更新会话时间
            session.updated_at = now
            db.commit()

            # ========== 写入 Redis 缓存 ==========
            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            # 删除会话列表缓存（强制刷新）
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        """
        加载对话消息
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
        
        Returns:
            LangChain 消息对象列表
        
        查询策略：
        1. 先查 Redis 缓存
        2. 缓存未命中再查数据库
        3. 查询结果回填缓存
        """
        # 先查缓存
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        # 缓存未命中：查数据库
        records = self.get_session_messages(user_id, session_id)
        # 回填缓存
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def load_with_meta(self, user_id: str, session_id: str) -> tuple[list, dict]:
        """
        加载对话消息及会话元数据
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
        
        Returns:
            (消息列表，元数据字典)
        
        元数据包含：
        - title: 会话标题
        - persistent_note: 持久化笔记
        """
        # 加载消息
        messages = self.load(user_id, session_id)
        
        # 加载元数据
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return messages, {}
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return messages, {}
            return messages, dict(session.metadata_json or {})
        finally:
            db.close()

    def list_sessions(self, user_id: str) -> list:
        """
        列出用户的所有会话 ID
        
        Args:
            user_id: 用户 ID
        
        Returns:
            会话 ID 列表
        """
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        """
        列出用户的会话详细信息
        
        Args:
            user_id: 用户 ID
        
        Returns:
            会话信息列表，每个元素包含：
            - session_id: 会话 ID
            - title: 会话标题
            - updated_at: 更新时间
            - message_count: 消息数量
        
        查询策略：
        1. 先查 Redis 缓存
        2. 缓存未命中再查数据库
        3. 查询结果回填缓存
        """
        # 先查缓存
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            # 查询会话（按更新时间倒序）
            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                # 统计消息数量
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                meta = s.metadata_json or {}
                result.append(
                    {
                        "session_id": s.session_id,
                        "title": meta.get("title") or s.session_id,  # 使用元数据标题或会话 ID
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            # 回填缓存
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        """
        获取会话的原始消息记录（字典格式）
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
        
        Returns:
            消息字典列表
        
        查询策略：
        1. 先查 Redis 缓存
        2. 缓存未命中再查数据库
        3. 查询结果回填缓存
        """
        # 先查缓存
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            # 查询消息（按时间正序）
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            # 回填缓存
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
        
        Returns:
            是否删除成功
        
        执行流程：
        1. 验证用户和会话存在
        2. 从数据库删除（级联删除消息）
        3. 从缓存删除
        """
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            # 删除会话（ChatMessage 表有外键级联删除）
            db.delete(session)
            db.commit()
            
            # 从缓存删除
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()
