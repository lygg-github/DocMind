"""文档向量化并写入 Milvus - 支持密集+稀疏向量"""

# ========== 导入部分 ==========
import os  # 操作系统接口，用于读取环境变量

# 导入向量化服务：
# - EmbeddingService: 向量化服务类（用于类型提示）
# - _default_embedding_service: 全局单例实例（默认使用）
from backend.indexing.embedding import EmbeddingService, embedding_service as _default_embedding_service

# 导入 Milvus 客户端：
# - MilvusStore: Milvus 访问类（用于类型提示）
# - get_milvus_store: 获取全局单例实例
from backend.indexing.milvus_client import MilvusStore, get_milvus_store


# ========== 核心类 ==========
class MilvusWriter:
    """
    文档向量化并写入 Milvus 服务
    
    核心功能：
    1. 将文档分块向量化（稠密+BGE-M3 + 稀疏+BM25）
    2. 批量写入 Milvus 向量库
    3. 支持进度回调
    """

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusStore = None):
        """
        初始化 MilvusWriter
        
        Args:
            embedding_service: 向量化服务实例（可选，默认为全局单例）
            milvus_manager: Milvus 客户端实例（可选，默认为全局单例）
        
        设计：支持依赖注入，便于测试时替换 mock 对象
        """
        # 使用传入的实例或默认单例
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or get_milvus_store()

    def write_documents(self, documents: list[dict], batch_size: int = 50, progress_callback=None):
        """
        将文档批量向量化并写入 Milvus

        Args:
            documents: 文档分块列表，每个分块是包含字段的字典
            batch_size: 每批次处理的文档数，默认 50
            progress_callback: 进度回调函数，接收 (processed_count, total_count)

        执行流程：
        1. 增量更新 BM25 统计
        2. 批量向量化（稠密+稀疏）
        3. 批量插入 Milvus
        """
        # 空列表检查：如果没有文档，直接返回
        if not documents:
            return

        # 获取稠密向量维度：从环境变量读取，默认 1024（BGE-M3 模型输出维度）
        dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        
        # 提取所有文档的文本内容
        all_texts = [doc["text"] for doc in documents]
        
        # ========== 关键步骤：增量更新 BM25 统计 ==========
        # 在向量化之前先更新 BM25 的词表和文档频率统计
        # 这样后续生成的稀疏向量才能基于最新的统计数据
        self.embedding_service.increment_add_documents(all_texts)

        # 获取文档总数
        total = len(documents)
        
        # ========== 使用 session 复用连接 ==========
        # 在同一业务流内复用连接，避免频繁创建/销毁连接
        with self.milvus_manager.session() as client:
            # 确保集合存在（不存在则创建）
            MilvusStore.ensure_collection(client, self.milvus_manager.collection_name, dense_dim)

            # ========== 批量处理循环 ==========
            # 按 batch_size 分批处理，避免单次处理过多数据
            for i in range(0, total, batch_size):
                # 截取当前批次的文档
                batch = documents[i : i + batch_size]
                
                # 提取当前批次的文本
                texts = [doc["text"] for doc in batch]
                
                # ========== 向量化：同时获取稠密和稀疏向量 ==========
                dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

                # ========== 构建插入数据 ==========
                # 将向量和元数据组合成 Milvus 可接受的格式
                # 防御性截断：确保 text 不超 Milvus VARCHAR(2000) 限制
                _truncated = 0
                _safe_batch = []
                for _doc in batch:
                    _text = _doc.get("text", "")
                    if len(_text) > 2000:
                        _truncated += 1
                        _doc = {**_doc, "text": _text[:2000]}
                    _safe_batch.append(_doc)
                if _truncated:
                    import logging, sys
                    logging.getLogger("backend.indexing.milvus_writer").warning(
                        f"[MilvusWriter] 截断了 {_truncated}/{len(batch)} 个超长文本（>2000字符）"
                    )
                    print(f"!!! [MilvusWriter] TRUNCATED {_truncated}/{len(batch)} chunks >2000 chars !!!", file=sys.stderr)
                batch = _safe_batch
                insert_data = [
                    {
                        "dense_embedding": dense_emb,  # BGE-M3 稠密向量
                        "sparse_embedding": sparse_emb,  # BM25 稀疏向量
                        "text": doc["text"][:2000],       # 分块文本内容（安全截断，防御性编程）
                        "filename": doc["filename"],      # 文件名
                        "file_type": doc["file_type"],    # 文件类型（PDF/Word/Excel）
                        "file_path": doc.get("file_path", ""),      # 文件路径（可选）
                        "page_number": doc.get("page_number", 0),    # 页码（可选）
                        "chunk_idx": doc.get("chunk_idx", 0),        # 全局分块索引（可选）
                        "chunk_id": doc.get("chunk_id", ""),          # 分块唯一ID（可选）
                        "parent_chunk_id": doc.get("parent_chunk_id", ""),  # 父块ID（用于合并）
                        "root_chunk_id": doc.get("root_chunk_id", ""),      # 根块ID（用于合并）
                        "chunk_level": doc.get("chunk_level", 0),    # 分块级别（1/2/3）
                    }
                    # 同时遍历文档、稠密向量、稀疏向量
                    for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
                ]

                # ========== 插入 Milvus ==========
                client.insert(self.milvus_manager.collection_name, insert_data)

                # ========== 进度回调（可选） ==========
                if progress_callback:
                    # 计算已处理数量（不超过总数）
                    processed = min(i + batch_size, total)
                    # 调用回调函数
                    progress_callback(processed, total)
