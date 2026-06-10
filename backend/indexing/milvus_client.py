"""Milvus 访问层：无状态 Store + 短生命周期 gRPC 连接（避免长期持有失效 channel）。"""

# ========== 导入部分 ==========
from __future__ import annotations  # 支持 Python 3.7+ 的类型注解语法

import os  # 操作系统接口，用于读取环境变量
from contextlib import contextmanager  # 上下文管理器装饰器
from dataclasses import dataclass  # 数据类装饰器
from typing import Callable, Iterator, TypeVar  # 类型提示

# 从 pymilvus 导入核心类：
# - AnnSearchRequest: 近似最近邻搜索请求对象
# - DataType: 数据类型枚举
# - MilvusClient: Milvus 客户端
# - RRFRanker: 倒数排名融合器（用于混合检索融合）
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

# ========== 常量定义 ==========
QUERY_MAX_LIMIT = 16384  # 查询最大限制（Milvus 单次查询上限）
T = TypeVar("T")  # 泛型类型变量，用于 _run 方法的类型提示


# ========== 配置类 ==========
@dataclass(frozen=True)  # 冻结的数据类，不可修改
class MilvusSettings:
    """Milvus 连接配置类"""
    host: str              # Milvus 服务器主机地址
    port: str              # Milvus 服务器端口
    collection_name: str   # 集合名称（表名）
    uri: str               # 完整连接地址（http://host:port）
    timeout: float         # 连接超时时间（秒）

    @classmethod
    def from_env(cls) -> MilvusSettings:
        """
        从环境变量加载配置
        
        环境变量映射：
        - MILVUS_HOST → host（默认 localhost）
        - MILVUS_PORT → port（默认 19530）
        - MILVUS_COLLECTION → collection_name（默认 embeddings_collection）
        - MILVUS_TIMEOUT → timeout（默认 30秒）
        """
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        timeout = float(os.getenv("MILVUS_TIMEOUT", "30"))
        return cls(
            host=host,
            port=port,
            collection_name=collection,
            uri=f"http://{host}:{port}",  # 拼接完整连接地址
            timeout=timeout,
        )


# ========== 连接管理 ==========
@contextmanager
def milvus_client_session(settings: MilvusSettings | None = None) -> Iterator[MilvusClient]:
    """
    一次 RPC 会话上下文管理器
    
    设计原则：
    - 短生命周期连接：创建连接 → 执行操作 → 关闭连接
    - 不缓存 gRPC channel：避免长期持有导致连接失效
    - 自动资源清理：使用 finally 确保连接关闭
    
    Args:
        settings: Milvus 配置，为 None 时从环境变量加载
    
    Yields:
        MilvusClient: 初始化好的 Milvus 客户端
    """
    # 获取配置：优先使用传入的 settings，否则从环境变量加载
    cfg = settings or MilvusSettings.from_env()
    
    # 创建 Milvus 客户端连接
    client = MilvusClient(uri=cfg.uri, timeout=cfg.timeout)
    
    try:
        # 将客户端对象返回给调用者
        yield client
    finally:
        # 无论是否发生异常，都确保关闭连接
        client.close()


# ========== 辅助函数 ==========
def _normalize_filter(filter_expr: str) -> str:
    """
    规范化过滤表达式
    
    处理空字符串情况：当过滤表达式为空时，返回 "id >= 0" 作为默认值
    （确保查询不会因为空过滤条件而失败）
    
    Args:
        filter_expr: 原始过滤表达式
    
    Returns:
        规范化后的过滤表达式
    """
    return filter_expr.strip() if filter_expr.strip() else "id >= 0"


# ========== 核心类 ==========
class MilvusStore:
    """
    Milvus 集合读写类
    
    设计特点：
    - 无状态设计：不持有连接对象
    - 连接复用：通过 _run 方法统一管理连接生命周期
    - 支持混合检索：稠密向量 + 稀疏向量
    """

    def __init__(self, settings: MilvusSettings | None = None):
        """
        初始化 MilvusStore
        
        Args:
            settings: Milvus 配置，为 None 时从环境变量加载
        """
        self._settings = settings or MilvusSettings.from_env()

    @property
    def collection_name(self) -> str:
        """获取集合名称（只读属性）"""
        return self._settings.collection_name

    def _run(self, operation: Callable[[MilvusClient], T]) -> T:
        """
        执行 Milvus 操作的统一封装
        
        为所有数据库操作提供统一的连接管理：
        1. 创建连接
        2. 执行操作（通过传入的 lambda 函数）
        3. 关闭连接
        4. 返回操作结果
        
        Args:
            operation: 接收 MilvusClient 并返回结果的函数
        
        Returns:
            操作执行结果
        """
        with milvus_client_session(self._settings) as client:
            return operation(client)

    @contextmanager
    def session(self) -> Iterator[MilvusClient]:
        """
        同一业务流内复用连接的上下文管理器
        
        使用场景：单次上传大量数据时，复用连接避免频繁创建/销毁
        
        Yields:
            MilvusClient: 可复用的客户端连接
        """
        with milvus_client_session(self._settings) as client:
            yield client

    @staticmethod
    def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int) -> None:
        """
        确保集合存在，不存在则创建
        
        创建集合时包含：
        1. 定义 schema（字段结构）
        2. 配置索引（稠密向量用 HNSW，稀疏向量用倒排索引）
        
        Args:
            client: MilvusClient 实例
            collection_name: 集合名称
            dense_dim: 稠密向量维度（默认 1024，BGE-M3 模型输出）
        """
        # 如果集合已存在，直接返回
        if client.has_collection(collection_name):
            return

        # ========== 创建 Schema ==========
        # auto_id=True: 主键自动生成
        # enable_dynamic_field=True: 允许动态字段
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        
        # 主键字段
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        
        # 向量字段
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)  # BGE-M3 稠密向量
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)         # BM25 稀疏向量
        
        # 元数据字段
        schema.add_field("text", DataType.VARCHAR, max_length=65535)      # 分块文本内容（足够容纳任何单块文本）
        schema.add_field("filename", DataType.VARCHAR, max_length=255)    # 文件名
        schema.add_field("file_type", DataType.VARCHAR, max_length=50)    # 文件类型（PDF/Word/Excel）
        schema.add_field("file_path", DataType.VARCHAR, max_length=1024)  # 文件路径
        schema.add_field("page_number", DataType.INT64)                   # 页码
        schema.add_field("chunk_idx", DataType.INT64)                     # 全局分块索引
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)    # 分块唯一ID
        schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)  # 父块ID（用于合并）
        schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)    # 根块ID（用于合并）
        schema.add_field("chunk_level", DataType.INT64)                   # 分块级别（1/2/3）

        # ========== 创建索引配置 ==========
        index_params = client.prepare_index_params()
        
        # 稠密向量索引：HNSW（层次导航小世界图）
        index_params.add_index(
            field_name="dense_embedding",
            index_type="HNSW",           # 索引类型：HNSW
            metric_type="IP",            # 相似度度量：内积（Inner Product）
            params={"M": 16, "efConstruction": 256},  # M: 每层最大连接数；efConstruction: 构建时搜索范围
        )
        
        # 稀疏向量索引：SPARSE_INVERTED_INDEX
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",  # 稀疏倒排索引
            metric_type="IP",                    # 相似度度量：内积
            params={"drop_ratio_build": 0.2},    # 构建时丢弃比例
        )
        
        # ========== 创建集合 ==========
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def init_collection(self, dense_dim: int | None = None) -> None:
        """
        初始化集合（对外接口）
        
        Args:
            dense_dim: 稠密向量维度，为 None 时从环境变量获取（默认 1024）
        """
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        # 定义内部操作函数
        def _init(client: MilvusClient) -> None:
            self.ensure_collection(client, self.collection_name, dense_dim)

        # 通过 _run 执行
        self._run(_init)

    def insert(self, data: list[dict]):
        """
        插入数据
        
        Args:
            data: 待插入的数据列表，每个元素是包含字段的字典
        
        Returns:
            插入结果（包含插入的 ID 列表）
        """
        return self._run(lambda client: client.insert(self.collection_name, data))

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        """
        条件查询（非向量检索）
        
        Args:
            filter_expr: 过滤表达式，如 "filename = 'report.pdf'"
            output_fields: 需要返回的字段列表，默认 ["filename", "file_type"]
            limit: 返回数量限制，默认 10000
            offset: 偏移量，用于分页
        
        Returns:
            查询结果列表
        """
        expr = _normalize_filter(filter_expr)
        fields = output_fields or ["filename", "file_type"]

        def _query(client: MilvusClient):
            return client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=fields,
                limit=min(limit, QUERY_MAX_LIMIT),  # 限制最大返回数
                offset=offset,
            )

        return self._run(_query)

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """
        分页拉取所有符合条件的数据
        
        特点：单次 session 内完成，避免每页新建连接
        
        Args:
            filter_expr: 过滤表达式
            output_fields: 需要返回的字段列表
        
        Returns:
            所有符合条件的数据列表
        """
        fields = output_fields or ["filename", "file_type"]
        expr = _normalize_filter(filter_expr)

        def _query_all(client: MilvusClient) -> list:
            out: list = []
            offset = 0
            while True:
                batch = client.query(
                    collection_name=self.collection_name,
                    filter=expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
                if not batch:  # 没有更多数据
                    break
                out.extend(batch)
                if len(batch) < QUERY_MAX_LIMIT:  # 最后一页
                    break
                offset += len(batch)  # 移动到下一页
            return out

        return self._run(_query_all)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        根据 chunk_id 批量获取分块数据
        
        Args:
            chunk_ids: chunk_id 列表
        
        Returns:
            分块数据列表
        """
        # 过滤空字符串
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        
        # 构建 IN 查询条件：chunk_id in ["id1", "id2", ...]
        quoted_ids = ", ".join(f'"{item}"' for item in ids)
        return self.query(
            filter_expr=f"chunk_id in [{quoted_ids}]",
            output_fields=[
                "text", "filename", "file_type", "page_number",
                "chunk_id", "parent_chunk_id", "root_chunk_id",
                "chunk_level", "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        混合检索：稠密向量 + 稀疏向量
        
        算法流程：
        1. 分别进行稠密向量检索和稀疏向量检索
        2. 使用 RRFRanker 融合两路召回结果
        3. 返回 top_k 个结果
        
        Args:
            dense_embedding: BGE-M3 生成的稠密向量（1024维）
            sparse_embedding: BM25 生成的稀疏向量（{词索引: 得分}）
            top_k: 返回结果数量，默认 5
            rrf_k: RRF 融合参数，默认 60（公式：score = 1/(k + rank)）
            filter_expr: 过滤表达式
        
        Returns:
            格式化后的检索结果列表
        """
        # 需要返回的字段
        output_fields = [
            "text", "filename", "file_type", "page_number",
            "chunk_id", "parent_chunk_id", "root_chunk_id",
            "chunk_level", "chunk_idx",
        ]
        
        # ========== 稠密向量检索请求 ==========
        dense_search = AnnSearchRequest(
            data=[dense_embedding],                     # 查询向量
            anns_field="dense_embedding",               # 向量字段名
            param={"metric_type": "IP", "params": {"ef": 64}},  # 内积相似度，搜索范围 64
            limit=top_k * 2,                            # 召回 2*top_k 供后续融合
            expr=filter_expr,                           # 过滤条件
        )
        
        # ========== 稀疏向量检索请求 ==========
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],                    # 稀疏向量
            anns_field="sparse_embedding",              # 稀疏向量字段名
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,                            # 召回 2*top_k
            expr=filter_expr,                           # 过滤条件
        )
        
        # ========== RRFRanker 融合器 ==========
        # RRF（Reciprocal Rank Fusion）公式：score = 1/(k + rank)
        # k 值越大，排名越重要
        reranker = RRFRanker(k=rrf_k)

        # 定义检索操作
        def _search(client: MilvusClient):
            return client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],  # 两路检索请求
                ranker=reranker,                     # 融合器
                limit=top_k,                         # 最终返回数量
                output_fields=output_fields,         # 返回字段
            )

        # 执行检索
        results = self._run(_search)
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),  # 相似度得分
                })
        return formatted_results

    def dense_retrieve(
        self,
        dense_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        """
        稠密向量检索（降级时使用）
        
        当混合检索失败时，降级到仅稠密向量检索
        
        Args:
            dense_embedding: BGE-M3 生成的稠密向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式
        
        Returns:
            格式化后的检索结果列表
        """
        def _search(client: MilvusClient):
            return client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text", "filename", "file_type", "page_number",
                    "chunk_id", "parent_chunk_id", "root_chunk_id",
                    "chunk_level", "chunk_idx",
                ],
                filter=filter_expr,
            )

        results = self._run(_search)
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "score": hit.get("distance", 0.0),
                })
        return formatted_results

    def delete(self, filter_expr: str):
        """
        删除符合条件的数据
        
        Args:
            filter_expr: 过滤表达式
        
        Returns:
            删除结果
        """
        return self._run(
            lambda client: client.delete(collection_name=self.collection_name, filter=filter_expr)
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self._run(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self) -> None:
        """删除集合（谨慎使用！）"""
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run(_drop)


# ========== 全局单例 ==========
# 兼容旧名；全项目共用同一无状态 Store 实例即可（不缓存连接）
MilvusManager = MilvusStore

# 全局单例实例（懒加载）
_store: MilvusStore | None = None


def get_milvus_store() -> MilvusStore:
    """
    获取全局 MilvusStore 单例
    
    使用懒加载模式：首次调用时创建实例
    
    Returns:
        MilvusStore 单例实例
    """
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
