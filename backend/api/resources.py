# 导入os模块（用于目录操作）
import os
# 导入Path（用于路径处理）
from pathlib import Path

# 从indexing模块导入核心服务
from backend.indexing import (
    DocumentLoader,            # 文档加载器（解析+三级分块）
    MilvusWriter,              # Milvus写入器（向量化+入库）
    ParentChunkStore,          # 父级分块存储（PostgreSQL+Redis）
    embedding_service,         # Embedding服务（稠密+稀疏向量）
)
# 导入Milvus管理器工厂函数
from backend.indexing.milvus_client import get_milvus_store

# ========== 路径配置 ==========
# BASE_DIR: backend目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR: 项目根目录下的data目录
DATA_DIR = BASE_DIR.parent / "data"
# UPLOAD_DIR: 上传文件保存目录
UPLOAD_DIR = DATA_DIR / "documents"

# ========== 全局共享资源实例（单例） ==========
# 文档加载器实例
loader = DocumentLoader()
# 父级分块存储实例
parent_chunk_store = ParentChunkStore()
# Milvus管理器实例
milvus_manager = get_milvus_store()
# Milvus写入器实例（传入embedding服务和milvus管理器）
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)


def remove_bm25_stats_for_filename(filename: str) -> None:
    """
    删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。

    Args:
        filename: 要清理的文件名
    """
    # 查询该文件的所有chunks的text字段
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',  # 过滤表达式
        output_fields=["text"],                    # 只查询text字段
    )
    # 提取所有text内容
    texts = [r.get("text") or "" for r in rows]
    # 调用embedding服务的增量删除方法，扣减BM25统计
    embedding_service.increment_remove_documents(texts)


def is_supported_document(filename: str) -> bool:
    """
    检查文件类型是否支持（PDF/Word/Excel/HTML）

    Args:
        filename: 文件名

    Returns:
        True表示支持，False表示不支持
    """
    # 转小写便于比较
    file_lower = filename.lower()
    # 支持的文件后缀：pdf/docx/doc/xlsx/xls/html/htm
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
        or file_lower.endswith((".html", ".htm"))
    )


async def save_upload_file(file, file_path: Path) -> None:
    """
    异步保存上传文件到指定路径（流式写入，避免内存溢出）

    Args:
        file: FastAPI的UploadFile对象
        file_path: 目标保存路径
    """
    # 以二进制写入模式打开文件
    with open(file_path, "wb") as f:
        # 循环读取并写入（每次1MB）
        while True:
            # 异步读取1MB数据块
            chunk = await file.read(1024 * 1024)
            # 读取完毕则退出循环
            if not chunk:
                break
            # 写入文件
            f.write(chunk)


def ensure_upload_dir() -> None:
    """确保上传目录存在（不存在则创建）"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)  # exist_ok=True避免目录已存在时报错
