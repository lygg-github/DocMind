# 导入os（用于路径操作）
import os

# FastAPI核心组件
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

# 从resources模块导入共享资源
from backend.api.resources import (
    UPLOAD_DIR,                          # 上传目录路径
    ensure_upload_dir,                   # 确保上传目录存在
    is_supported_document,               # 检查文件类型是否支持
    loader,                              # 文档加载器（解析+三级分块）
    milvus_manager,                      # Milvus管理器
    milvus_writer,                       # Milvus写入器（向量化+入库）
    parent_chunk_store,                  # 父级分块存储
    remove_bm25_stats_for_filename,      # 删除BM25统计
    save_upload_file,                    # 异步保存上传文件
)
# 导入User模型
from backend.db.models import User
# 导入管理员权限验证
from backend.infra.auth import require_admin
# 导入任务管理器
from backend.jobs import DELETE_STEPS, delete_job_manager, upload_job_manager
# 导入Pydantic响应模型
from backend.schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)

# 创建路由分组
router = APIRouter(tags=["documents"])


# ========== 后台任务：处理文档上传 ==========
def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """后台异步处理文档上传：清理旧版→解析分块→写入父级块→向量化入库"""
    # 初始化失败步骤标记
    failed_step = "cleanup"
    try:
        # 步骤1：完成upload步骤（文件已由接口层保存）
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        # ========== 步骤2：清理同名旧文档 ==========
        failed_step = "cleanup"
        # 更新任务进度
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        # 初始化Milvus集合
        milvus_manager.init_collection()
        # 删除表达式：按文件名过滤
        delete_expr = f'filename == "{filename}"'
        # 删除BM25统计（增量持久化）
        try:
            remove_bm25_stats_for_filename(filename)
        except Exception:
            pass  # 旧文档可能不存在
        # 删除Milvus中的旧向量
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        # 删除PostgreSQL中的父级分块
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass
        # 完成清理步骤
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        # ========== 步骤3：解析文档并三级分块 ==========
        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行三级分块")
        # 调用DocumentLoader解析文档，返回所有分块
        new_docs = loader.load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        # 分离父级分块（L1/L2）和叶子分块（L3）
        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        # 完成解析步骤
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        # ========== 步骤4：写入父级分块到PostgreSQL+Redis ==========
        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        # ========== 步骤5：向量化并写入Milvus ==========
        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        # 初始化进度
        upload_job_manager.update_step(
            job_id,
            "vector_store",
            0,
            "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0,
        )

        # 定义进度回调函数（用于实时更新任务进度）
        def _on_vector_progress(processed: int, total: int) -> None:
            # 计算百分比
            percent = round(processed * 100 / total) if total else 100
            # 更新任务进度
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        # 执行向量化+入库（带进度回调）
        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
        # 标记任务完成
        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as e:
        # 任务失败，标记失败状态
        upload_job_manager.fail_job(job_id, failed_step, str(e))


# ========== 后台任务：处理文档删除 ==========
def _process_delete_job(job_id: str, filename: str) -> None:
    """后台异步处理文档删除：清理BM25→删除Milvus→删除父级分块"""
    failed_step = "prepare"
    try:
        # ========== 步骤1：准备删除 ==========
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        # ========== 步骤2：清理BM25统计 ==========
        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        remove_bm25_stats_for_filename(filename)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        # ========== 步骤3：删除Milvus向量数据 ==========
        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(delete_expr)
        # 提取删除数量
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        # ========== 步骤4：删除PostgreSQL父级分块 ==========
        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 PostgreSQL 父级分块")
        parent_chunk_store.delete_by_filename(filename)#这个其实删除了redis中的父类缓存
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        # 标记任务完成
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as e:
        # 任务失败
        delete_job_manager.fail_job(job_id, failed_step, str(e))


# ========== 获取文档列表接口 ==========
# GET /documents（管理员权限）
@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    try:
        # 初始化Milvus集合
        milvus_manager.init_collection()
        # 查询所有文档的filename和file_type（limit=10000）
        results = milvus_manager.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        # 统计每个文件的分块数量
        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        # 构造响应
        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


# ========== 异步上传文档接口 ==========
# POST /documents/upload/async（管理员权限）
@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,    # FastAPI后台任务管理器
    file: UploadFile = File(...),         # 上传的文件
    _: User = Depends(require_admin),     # 管理员权限验证
):
    # 获取文件名
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 检查文件类型
    if not is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

    # 确保上传目录存在
    ensure_upload_dir()
    # 创建上传任务
    job = upload_job_manager.create_job(filename)
    # 构造文件保存路径
    file_path = UPLOAD_DIR / filename

    try:
        # 同步阶段：保存文件
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "正在保存文件到服务器")
        await save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        # 保存失败，标记任务失败
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 添加后台任务：解析+向量化入库
    background_tasks.add_task(_process_upload_job, job["job_id"], str(file_path), filename)
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


# ========== 获取上传任务进度接口 ==========
# GET /documents/upload/jobs/{job_id}
@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    # 查询任务状态
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


# ========== 获取所有上传任务列表接口 ==========
# GET /documents/upload/jobs
@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    # 获取所有任务
    jobs = upload_job_manager.list_jobs()
    # 按创建时间倒序
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


# ========== 异步删除文档接口 ==========
# DELETE /documents/delete/async/{filename}
@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
):
    # 创建删除任务
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    # 添加后台任务
    background_tasks.add_task(_process_delete_job, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


# ========== 获取删除任务进度接口 ==========
# GET /documents/delete/jobs/{job_id}
@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


# ========== 同步上传文档接口（兼容旧版） ==========
# POST /documents/upload
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    try:
        # 文件名校验
        filename = file.filename or ""
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not is_supported_document(filename):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        # 确保上传目录存在
        ensure_upload_dir()
        # 初始化Milvus
        milvus_manager.init_collection()

        # 删除同名旧文档
        delete_expr = f'filename == "{filename}"'
        try:
            remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        # 保存文件
        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 解析文档
        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        # 分离父级分块和叶子分块
        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        # 写入父级分块
        parent_chunk_store.upsert_documents(parent_docs)
        # 向量化入库
        milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


# ========== 同步删除文档接口（兼容旧版） ==========
# DELETE /documents/{filename}
@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    try:
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        # 删除BM25统计
        remove_bm25_stats_for_filename(filename)
        # 删除Milvus向量
        result = milvus_manager.delete(delete_expr)
        # 删除父级分块
        parent_chunk_store.delete_by_filename(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
