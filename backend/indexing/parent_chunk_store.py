"""父级分块文档存储（用于 Auto-merging Retriever）"""

# ========== 导入部分 ==========
from datetime import datetime  # 日期时间处理，用于记录更新时间
from typing import List  # 类型提示

# 导入 Redis 缓存客户端
from backend.infra.cache import cache
# 导入数据库会话工厂
from backend.infra.database import SessionLocal
# 导入父级分块数据模型（ORM 模型）
from backend.db.models import ParentChunk


# ========== 核心类 ==========
class ParentChunkStore:
    """
    基于 PostgreSQL + Redis 的父级分块存储
    
    设计目的：
    - 配合 Auto-merging Retriever 使用
    - 存储 L1/L2 级别的父分块（L3 叶子块存 Milvus）
    - 使用 Redis 缓存加速查询
    
    存储策略：
    - PostgreSQL：持久化存储
    - Redis：缓存层，热点数据加速
    """

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        """
        将 ORM 模型对象转换为字典
        
        Args:
            item: ParentChunk ORM 对象
        
        Returns:
            包含所有字段的字典
        """
        return {
            "text": item.text,                    # 分块文本内容
            "filename": item.filename,            # 文件名
            "file_type": item.file_type,          # 文件类型
            "file_path": item.file_path,          # 文件路径
            "page_number": item.page_number,      # 页码
            "chunk_id": item.chunk_id,            # 分块唯一ID
            "parent_chunk_id": item.parent_chunk_id,  # 父块ID
            "root_chunk_id": item.root_chunk_id,      # 根块ID
            "chunk_level": item.chunk_level,          # 分块级别（1/2）
            "chunk_idx": item.chunk_idx,              # 全局分块索引
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        """
        生成 Redis 缓存键
        
        Args:
            chunk_id: 分块ID
        
        Returns:
            Redis 键名，格式：parent_chunk:{chunk_id}
        """
        return f"parent_chunk:{chunk_id}"

    def upsert_documents(self, docs: List[dict]) -> int:
        """
        写入/更新父级分块
        
        Args:
            docs: 父级分块列表，每个元素是包含字段的字典
        
        Returns:
            写入/更新的条数
        
        执行流程：
        1. 遍历每个文档
        2. 查询数据库判断是更新还是新增
        3. 写入 PostgreSQL
        4. 写入 Redis 缓存
        5. 提交事务
        """
        # 空列表检查
        if not docs:
            return 0

        # 创建数据库会话
        db = SessionLocal()
        # 计数器：记录写入/更新的条数
        upserted = 0
        
        try:
            # 遍历每个文档
            for doc in docs:
                # 获取并清理 chunk_id
                chunk_id = (doc.get("chunk_id") or "").strip()
                # chunk_id 为空则跳过
                if not chunk_id:
                    continue

                # ========== 查询现有记录 ==========
                # 根据 chunk_id 查询数据库中是否已存在
                record = db.query(ParentChunk).filter(ParentChunk.chunk_id == chunk_id).first()
                
                # ========== 构建数据库载荷 ==========
                payload = {
                    "text": doc.get("text", ""),                              # 分块文本
                    "filename": doc.get("filename", ""),                      # 文件名
                    "file_type": doc.get("file_type", ""),                    # 文件类型
                    "file_path": doc.get("file_path", ""),                    # 文件路径
                    "page_number": int(doc.get("page_number", 0) or 0),       # 页码（转整数）
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),        # 父块ID
                    "root_chunk_id": doc.get("root_chunk_id", ""),            # 根块ID
                    "chunk_level": int(doc.get("chunk_level", 0) or 0),       # 分块级别（转整数）
                    "chunk_idx": int(doc.get("chunk_idx", 0) or 0),           # 全局索引（转整数）
                    "updated_at": datetime.utcnow(),                          # 更新时间（UTC）
                }
                
                # ========== 构建缓存载荷 ==========
                # 缓存不需要 updated_at，但需要 chunk_id
                cache_payload = {
                    "chunk_id": chunk_id,                    # 分块ID（缓存键）
                    "text": payload["text"],                 # 分块文本
                    "filename": payload["filename"],         # 文件名
                    "file_type": payload["file_type"],       # 文件类型
                    "file_path": payload["file_path"],       # 文件路径
                    "page_number": payload["page_number"],   # 页码
                    "parent_chunk_id": payload["parent_chunk_id"],  # 父块ID
                    "root_chunk_id": payload["root_chunk_id"],      # 根块ID
                    "chunk_level": payload["chunk_level"],           # 分块级别
                    "chunk_idx": payload["chunk_idx"],               # 全局索引
                }
                
                # ========== 写入数据库 ==========
                if record:
                    # 记录已存在：更新字段
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    # 记录不存在：新增记录
                    db.add(ParentChunk(chunk_id=chunk_id, **payload))

                # ========== 写入 Redis 缓存 ==========
                cache.set_json(self._cache_key(chunk_id), cache_payload)
                
                # 计数器 +1
                upserted += 1

            # 提交事务
            db.commit()
        finally:
            # 确保关闭数据库连接
            db.close()

        return upserted

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """
        根据 chunk_id 批量获取父级分块
        
        Args:
            chunk_ids: chunk_id 列表
        
        Returns:
            分块数据列表（保持输入顺序）
        
        查询策略：
        1. 先查 Redis 缓存
        2. 缓存未命中再查 PostgreSQL
        3. 查询结果写入缓存
        """
        # 空列表检查
        if not chunk_ids:
            return []

        # 结果字典：保持顺序
        ordered_results = {}
        # 缓存未命中的 ID 列表
        missing_ids = []
        
        # ========== 第一轮：查缓存 ==========
        for chunk_id in chunk_ids:
            # 清理 chunk_id
            key = (chunk_id or "").strip()
            if not key:
                continue
            
            # 尝试从 Redis 获取
            cached = cache.get_json(self._cache_key(key))
            if cached:
                # 缓存命中
                ordered_results[key] = cached
            else:
                # 缓存未命中，加入待查询列表
                missing_ids.append(key)

        # ========== 第二轮：查数据库 ==========
        if missing_ids:
            db = SessionLocal()
            try:
                # 批量查询未命中的 ID
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                
                for row in rows:
                    # ORM 对象转字典
                    payload = self._to_dict(row)
                    # 存入结果
                    ordered_results[row.chunk_id] = payload
                    # 写入缓存（回填）
                    cache.set_json(self._cache_key(row.chunk_id), payload)
            finally:
                db.close()

        # ========== 返回结果（保持顺序） ==========
        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除父级分块
        
        Args:
            filename: 文件名
        
        Returns:
            删除的条数
        
        执行流程：
        1. 查询该文件的所有父级分块
        2. 从数据库删除
        3. 从缓存删除
        """
        # 空文件名检查
        if not filename:
            return 0

        db = SessionLocal()
        try:
            # 查询该文件的所有父级分块
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            
            # 提取所有 chunk_id（用于删除缓存）
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            
            if deleted > 0:
                # 从数据库删除
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                
                # 从缓存删除
                for chunk_id in chunk_ids:
                    cache.delete(self._cache_key(chunk_id))
            
            return deleted
        finally:
            db.close()
