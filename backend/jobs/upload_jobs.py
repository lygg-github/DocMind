# ========== 模块说明 ==========
"""上传任务进度管理。

轻量版先使用进程内存保存任务状态，适合当前单进程开发部署。
如果后续要支持多进程或服务重启恢复，可以把同样的数据结构迁移到 Redis/PostgreSQL。
"""

# 导入未来注解支持（Python 3.7+ 兼容）
from __future__ import annotations

# deepcopy用于返回深拷贝对象，防止外部修改内部状态
from copy import deepcopy
# UTC时间和datetime用于生成时间戳
from datetime import UTC, datetime
# Lock用于线程安全保护
from threading import Lock
# Literal用于字面量类型提示
from typing import Literal
# uuid4用于生成唯一任务ID
from uuid import uuid4


# ========== 类型定义 ==========
# 步骤状态类型：pending(等待), running(执行中), completed(完成), failed(失败)
StepStatus = Literal["pending", "running", "completed", "failed"]
# 任务状态类型：与步骤状态一致
JobStatus = Literal["pending", "running", "completed", "failed"]


# ========== 默认步骤配置 ==========
# 上传任务的默认步骤序列
DEFAULT_STEPS = [
    ("upload", "文档上传"),           # 步骤1：上传文件到服务器
    ("cleanup", "清理旧版本"),        # 步骤2：删除同名旧文档
    ("parse", "解析与分块"),          # 步骤3：解析文档并三级分块
    ("parent_store", "父级分块入库"),  # 步骤4：L1/L2分块写入PostgreSQL
    ("vector_store", "向量化入库"),    # 步骤5：L3分块向量化写入Milvus
]

# 删除任务的步骤序列
DELETE_STEPS = [
    ("prepare", "准备删除"),          # 步骤1：初始化Milvus集合
    ("bm25", "同步 BM25 统计"),       # 步骤2：扣减BM25词频统计
    ("milvus", "删除向量数据"),        # 步骤3：删除Milvus中的向量
    ("parent_store", "删除父级分块"),  # 步骤4：删除PostgreSQL中的父级分块
]


# ========== 辅助函数 ==========
def _now_iso() -> str:
    """获取当前UTC时间的ISO格式字符串（用于时间戳）"""
    return datetime.now(UTC).isoformat()


# ========== 核心类定义 ==========
class UploadJobManager:
    """线程安全的上传任务状态容器。
    
    负责管理文档上传和删除任务的进度追踪，提供线程安全的状态更新和查询接口。
    """

    def __init__(self):
        """初始化任务管理器"""
        # _jobs字典：key=job_id, value=任务状态字典（内存存储）
        self._jobs: dict[str, dict] = {}
        # _lock互斥锁：保证多线程环境下的数据一致性
        self._lock = Lock()

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "等待上传",
        completion_step: str = "vector_store",
    ) -> dict:
        """
        创建新任务
        
        Args:
            filename: 文件名
            steps: 步骤列表，默认为DEFAULT_STEPS
            current_step: 当前步骤，默认"upload"
            message: 初始消息
            completion_step: 完成标记步骤（区分上传和删除任务）
        
        Returns:
            任务状态字典（深拷贝，防止外部修改）
        """
        # 使用传入的步骤列表或默认步骤
        steps = steps or DEFAULT_STEPS
        # 生成唯一job_id（uuid4十六进制字符串）
        job_id = uuid4().hex
        # 获取当前时间
        now = _now_iso()
        
        # 构建任务状态字典
        job = {
            "job_id": job_id,           # 任务唯一标识
            "filename": filename,       # 关联的文件名
            "status": "pending",        # 任务状态：pending/running/completed/failed
            "current_step": current_step, # 当前执行步骤
            "message": message,         # 当前消息
            # 完成节点用于区分上传和删除，避免complete_job写死最后一步
            "completion_step": completion_step,
            "total_chunks": 0,          # 总分块数（向量化时使用）
            "processed_chunks": 0,      # 已处理分块数
            "error": None,              # 错误信息（失败时填充）
            "created_at": now,          # 创建时间
            "updated_at": now,          # 更新时间
            # 各步骤详细状态列表
            "steps": [
                {
                    "key": key,           # 步骤标识
                    "label": label,       # 步骤显示名称
                    "percent": 0,         # 进度百分比(0-100)
                    "status": "pending",  # 步骤状态
                    "message": "",        # 步骤消息
                }
                for key, label in steps
            ],
        }
        
        # 线程安全：获取锁后写入
        with self._lock:
            self._jobs[job_id] = job
            # 返回深拷贝，防止外部修改内部状态
            return deepcopy(job)

    def get_job(self, job_id: str) -> dict | None:
        """
        根据job_id查询任务状态
        
        Args:
            job_id: 任务ID
        
        Returns:
            任务状态字典（深拷贝）或None（任务不存在）
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: StepStatus = "running",
        message: str = "",
        *,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> dict | None:
        """
        更新指定步骤的进度和状态
        
        Args:
            job_id: 任务ID
            step_key: 步骤标识（如"parse"）
            percent: 进度百分比（自动限制在0-100）
            status: 步骤状态
            message: 进度消息
            total_chunks: 总分块数（可选，向量化时使用）
            processed_chunks: 已处理分块数（可选）
        
        Returns:
            更新后的任务状态字典或None
        """
        # 限制百分比在0-100之间
        percent = max(0, min(100, int(percent)))
        
        with self._lock:
            # 获取任务
            job = self._jobs.get(job_id)
            if not job:
                return None

            # 查找指定步骤
            step = self._find_step(job, step_key)
            if not step:
                return None

            # 更新步骤信息
            step["percent"] = percent
            step["status"] = status
            step["message"] = message
            
            # 更新任务级别状态
            job["status"] = "failed" if status == "failed" else "running"
            job["current_step"] = step_key
            job["message"] = message
            job["updated_at"] = _now_iso()

            # 更新分块统计（可选）
            if total_chunks is not None:
                job["total_chunks"] = int(total_chunks)
            if processed_chunks is not None:
                job["processed_chunks"] = int(processed_chunks)

            return deepcopy(job)

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> dict | None:
        """
        标记步骤完成（进度100%，状态completed）
        
        Args:
            job_id: 任务ID
            step_key: 步骤标识
            message: 完成消息
        
        Returns:
            更新后的任务状态字典或None
        """
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id: str, message: str = "文档入库完成") -> dict | None:
        """
        标记整个任务完成
        
        Args:
            job_id: 任务ID
            message: 完成消息
        
        Returns:
            更新后的任务状态字典或None
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            # 将所有未失败的步骤标记为完成
            for step in job["steps"]:
                if step["status"] != "failed":
                    step["percent"] = 100
                    step["status"] = "completed"
            
            # 更新任务级别状态
            job["status"] = "completed"
            job["current_step"] = job.get("completion_step") or job["current_step"]
            job["message"] = message
            job["error"] = None
            job["updated_at"] = _now_iso()
            
            return deepcopy(job)

    def fail_job(self, job_id: str, step_key: str, error: str) -> dict | None:
        """
        标记任务失败
        
        Args:
            job_id: 任务ID
            step_key: 失败的步骤标识
            error: 错误信息
        
        Returns:
            更新后的任务状态字典或None
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            # 查找并标记失败步骤
            step = self._find_step(job, step_key)
            if step:
                step["status"] = "failed"
                step["message"] = error
            
            # 更新任务级别状态
            job["status"] = "failed"
            job["current_step"] = step_key
            job["message"] = error
            job["error"] = error
            job["updated_at"] = _now_iso()
            
            return deepcopy(job)

    def list_jobs(self) -> list[dict]:
        """
        获取所有任务列表
        
        Returns:
            任务状态字典列表（深拷贝）
        """
        with self._lock:
            return [deepcopy(job) for job in self._jobs.values()]

    @staticmethod
    def _find_step(job: dict, step_key: str) -> dict | None:
        """
        在任务中查找指定步骤（静态方法）
        
        Args:
            job: 任务状态字典
            step_key: 步骤标识
        
        Returns:
            步骤字典或None
        """
        for step in job["steps"]:
            if step["key"] == step_key:
                return step
        return None


# ========== 全局单例实例 ==========
# 上传任务管理器实例
upload_job_manager = UploadJobManager()
# 删除任务管理器实例（独立于上传任务）
delete_job_manager = UploadJobManager()
