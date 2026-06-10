# ========== 导入部分 ==========
from collections import defaultdict  # 用于分组聚合
from typing import List, Tuple, Dict, Any, Optional  # 类型提示
import os  # 环境变量读取
import json  # JSON 处理
import requests  # HTTP 请求（用于 Rerank API）

# 导入核心检索依赖
from backend.indexing.milvus_client import get_milvus_store  # Milvus 客户端
from backend.indexing.embedding import embedding_service as _embedding_service  # 向量化服务
from backend.indexing.parent_chunk_store import ParentChunkStore  # 父块存储
from langchain.chat_models import init_chat_model  # LangChain 模型初始化


# ========== 环境变量配置 ==========
ARK_API_KEY = os.getenv("ARK_API_KEY")          # API Key
MODEL = os.getenv("MODEL")                      # 主模型
BASE_URL = os.getenv("BASE_URL")                # API 基础URL
RERANK_MODEL = os.getenv("RERANK_MODEL")        # 精排模型
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")  # 精排服务地址
RERANK_API_KEY = os.getenv("RERANK_API_KEY")    # 精排 API Key
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"  # 是否启用自动合并
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))  # 合并阈值
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))  # 叶子块检索层级


def _read_positive_int_env(name: str, default: int) -> int:
    """
    安全读取正整数环境变量
    
    Args:
        name: 环境变量名
        default: 默认值
    
    Returns:
        正整数值（至少为1）
    """
    try:
        return max(int(os.getenv(name, str(default))), 1)
    except ValueError:
        return default


# 检索候选池乘数（用于计算 candidate_k = top_k × multiplier）
RETRIEVAL_CANDIDATE_MULTIPLIER = _read_positive_int_env("RETRIEVAL_CANDIDATE_MULTIPLIER", 3)
_RETRIEVAL_CANDIDATE_K_RAW = os.getenv("RETRIEVAL_CANDIDATE_K", "").strip()  # 直接配置的候选池大小


def _read_float_env(name: str, default: float) -> float:
    """
    安全读取浮点环境变量
    
    Args:
        name: 环境变量名
        default: 默认值
    
    Returns:
        浮点数值
    """
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# 精排最低分数阈值（低于此分数的文档被过滤）
RERANK_MIN_SCORE = _read_float_env("RERANK_MIN_SCORE", 0.0)

# 检索轨迹字段列表（用于记录检索过程信息）
RETRIEVAL_TRACE_FIELDS = (
    "retrieval_pipeline",          # 检索流水线名称
    "retrieval_mode",              # 检索模式（hybrid/dense_fallback/failed）
    "candidate_k",                 # 候选池大小
    "candidate_k_source",          # 候选池配置来源（env/multiplier）
    "candidate_k_config_error",    # 配置错误信息
    "retrieval_candidate_multiplier",  # 候选池乘数
    "retrieval_top_k",             # 最终返回数量
    "leaf_retrieve_level",         # 叶子块层级
    "recall_count",                # 召回数量
    "post_merge_candidate_count",  # 合并后候选数量
    "candidate_count",             # 精排候选数量
    "auto_merge_enabled",          # 是否启用自动合并
    "auto_merge_applied",          # 是否应用了合并
    "auto_merge_threshold",        # 合并阈值
    "auto_merge_replaced_chunks",  # 被替换的块数量
    "auto_merge_steps",            # 合并步骤数
    "rerank_enabled",              # 是否启用精排
    "rerank_applied",              # 是否应用了精排
    "rerank_model",                # 精排模型
    "rerank_endpoint",             # 精排服务地址
    "rerank_error",                # 精排错误信息
    "rerank_min_score",            # 精排最低分数
    "post_rerank_count",           # 精排后数量
    "post_threshold_count",        # 阈值过滤后数量
    "retrieval_empty",             # 是否为空结果
)


# ========== 全局依赖初始化 ==========
# 与 API 共用 embedding_service，保证 BM25 状态一致
_milvus_manager = get_milvus_store()  # Milvus 向量数据库管理器
_parent_chunk_store = ParentChunkStore()  # 父块存储（PostgreSQL + Redis）

_stepback_model = None  # Step-back 查询扩展模型（懒加载）


def resolve_candidate_k(top_k: int) -> Tuple[int, Dict[str, Any]]:
    """
    解析 Milvus 候选池大小
    
    优先级：RETRIEVAL_CANDIDATE_K 环境变量 > top_k × multiplier
    
    Args:
        top_k: 最终返回数量
    
    Returns:
        (candidate_k, config_info): 候选池大小和配置信息
    """
    if _RETRIEVAL_CANDIDATE_K_RAW:
        try:
            candidate_k = max(int(_RETRIEVAL_CANDIDATE_K_RAW), top_k)
        except ValueError:
            # 配置值无效，回退到 multiplier 计算
            candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
            return candidate_k, {
                "candidate_k_source": "multiplier",
                "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
                "candidate_k_config_error": "invalid RETRIEVAL_CANDIDATE_K",
            }
        return candidate_k, {
            "candidate_k_source": "env",
            "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
        }
    # 使用 multiplier 计算
    candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
    return candidate_k, {
        "candidate_k_source": "multiplier",
        "retrieval_candidate_multiplier": RETRIEVAL_CANDIDATE_MULTIPLIER,
    }


def retrieval_trace_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    从检索元数据中提取应写入 rag_trace 的字段
    
    Args:
        meta: 检索元数据
    
    Returns:
        过滤后的字段字典
    """
    return {key: meta[key] for key in RETRIEVAL_TRACE_FIELDS if key in meta and meta[key] is not None}


def merge_retrieval_trace(accumulated: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多路检索轨迹（用于扩展召回场景）
    
    规则：
    - 计数类字段累加（如 recall_count）
    - 配置类字段保留首次出现的值
    
    Args:
        accumulated: 已累积的轨迹
        meta: 新传入的轨迹
    
    Returns:
        合并后的轨迹
    """
    incoming = retrieval_trace_fields(meta)
    if not accumulated:
        return incoming
    
    # 需要累加的字段
    additive = {
        "recall_count",
        "post_merge_candidate_count",
        "auto_merge_replaced_chunks",
        "auto_merge_steps",
    }
    
    merged = dict(accumulated)
    for key, value in incoming.items():
        if key in additive:
            # 累加计数
            merged[key] = int(merged.get(key) or 0) + int(value or 0)
        elif key == "auto_merge_applied":
            # 只要有一次应用了就为 True
            merged[key] = bool(merged.get(key)) or bool(value)
        else:
            # 保留首次值
            merged.setdefault(key, value)
    return merged


def _get_rerank_endpoint() -> str:
    """
    获取精排服务端点 URL
    
    Returns:
        完整的精排 API 地址
    """
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    # 确保 URL 以 /v1/rerank 结尾
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _effective_score(doc: dict) -> Optional[float]:
    """
    获取文档的有效分数（用于排序）
    
    优先级：rerank_score（精排分）> score（召回分）
    
    Args:
        doc: 文档对象
    
    Returns:
        有效分数或 None
    """
    rerank_score = doc.get("rerank_score")
    if rerank_score is not None:
        return float(rerank_score)
    score = doc.get("score")
    if score is not None:
        return float(score)
    return None


def _meets_rerank_min_score(doc: dict) -> bool:
    """
    检查文档是否满足最低精排分数要求
    
    Args:
        doc: 文档对象
    
    Returns:
        True: 满足要求，False: 不满足
    """
    score = _effective_score(doc)
    if score is None:
        return RERANK_MIN_SCORE <= 0
    return score >= RERANK_MIN_SCORE


def _merge_rank_score_into(target: dict, source: dict) -> None:
    """
    将源文档的分数合并到目标文档
    
    规则：保留较高的分数
    
    Args:
        target: 目标文档（被合并到的文档）
        source: 源文档（提供分数的文档）
    """
    incoming = _effective_score(source)
    if incoming is None:
        return
    
    # 判断是否使用精排分
    uses_rerank = source.get("rerank_score") is not None or target.get("rerank_score") is not None
    
    if uses_rerank:
        existing = target.get("rerank_score")
        if existing is None:
            target["rerank_score"] = incoming
        else:
            target["rerank_score"] = max(float(existing), incoming)
        return
    
    # 使用召回分
    existing = target.get("score")
    if existing is None:
        target["score"] = incoming
    else:
        target["score"] = max(float(existing), incoming)


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    """
    将子块合并到父块级别
    
    当某个父块的子块数量 >= threshold 时，用父块替换所有子块
    
    Args:
        docs: 文档列表
        threshold: 触发合并的子块数量阈值
    
    Returns:
        (merged_docs, merged_count): 合并后的文档列表和被替换的块数量
    """
    # 按父块 ID 分组
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    # 筛选需要合并的父块（子块数量 >= threshold）
    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    # 从父块存储获取父块内容
    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    parent_slot: Dict[str, int] = {}  # 记录父块在 merged_docs 中的位置
    merged_count = 0
    
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        # 没有父块或父块不在合并列表中，直接保留
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue

        # 父块已存在，合并分数
        if parent_id in parent_slot:
            existing = merged_docs[parent_slot[parent_id]]
            _merge_rank_score_into(existing, doc)
            merged_count += 1
            continue

        # 创建新的父块条目
        parent_doc = dict(parent_map[parent_id])
        _merge_rank_score_into(parent_doc, doc)
        parent_doc["merged_from_children"] = True  # 标记为合并生成
        parent_doc["merged_child_count"] = len(groups[parent_id])  # 记录合并了多少子块
        parent_slot[parent_id] = len(merged_docs)
        merged_docs.append(parent_doc)
        merged_count += 1

    return merged_docs, merged_count


def _empty_merge_meta() -> Dict[str, Any]:
    """
    创建空的合并元数据
    
    Returns:
        初始合并元数据
    """
    return {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
        "post_merge_candidate_count": 0,
    }


def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    """
    在完整召回候选上执行 L3→L2→L1 层级合并
    
    不改变顺序，精排由后续步骤负责
    
    Args:
        docs: 召回候选列表
    
    Returns:
        (merged_docs, meta): 合并后的文档列表和合并元数据
    """
    meta = _empty_merge_meta()
    meta["post_merge_candidate_count"] = len(docs)
    
    # 如果未启用自动合并或没有文档，直接返回
    if not AUTO_MERGE_ENABLED or not docs:
        return docs, meta

    # L3 → L2 合并
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    # L2 → L1 合并
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    # 更新合并元数据
    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    meta.update({
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
        "post_merge_candidate_count": len(merged_docs),
    })
    return merged_docs, meta


def _sort_by_rank_score(docs: List[dict]) -> List[dict]:
    """
    按有效分数降序排序文档
    
    Args:
        docs: 文档列表
    
    Returns:
        排序后的文档列表
    """
    return sorted(docs, key=lambda item: _effective_score(item) or 0.0, reverse=True)


def dedupe_documents(docs: List[dict]) -> List[dict]:
    """
    按 chunk_id 去重，重复项保留更高的 rank 分
    
    优先级：rerank_score > score
    
    Args:
        docs: 文档列表
    
    Returns:
        去重后的文档列表（保持原始顺序）
    """
    by_key: Dict[str, dict] = {}
    order: List[str] = []
    
    for item in docs:
        # 优先使用 chunk_id，否则用 filename|page_number|text 作为唯一标识
        chunk_id = (item.get("chunk_id") or "").strip()
        key = chunk_id or f"{item.get('filename')}|{item.get('page_number')}|{item.get('text')}"
        
        if key not in by_key:
            by_key[key] = item
            order.append(key)
            continue
        
        # 合并分数到已存在的文档
        _merge_rank_score_into(by_key[key], item)
    
    # 保持原始顺序返回
    return [by_key[key] for key in order]


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    """
    调用 Jina Rerank API 对文档进行精排
    
    Args:
        query: 用户查询
        docs: 待精排的文档列表
        top_k: 返回数量
    
    Returns:
        (reranked_docs, meta): 精排后的文档列表和精排元数据
    """
    # 为每个文档添加原始排名（用于精排失败时的回退排序）
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    
    # 如果没有文档或未启用精排，直接返回排序后的结果
    if not docs_with_rank or not meta["rerank_enabled"]:
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta

    # 构建精排请求
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        
        # 检查 HTTP 错误
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            return _sort_by_rank_score(docs_with_rank)[:top_k], meta

        # 解析精排结果
        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            return reranked[:top_k], meta

        # 精排结果为空
        meta["rerank_error"] = "empty_rerank_results"
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta
    
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # 异常时回退到排序结果
        meta["rerank_error"] = str(e)
        return _sort_by_rank_score(docs_with_rank)[:top_k], meta


def _get_stepback_model():
    """
    获取 Step-back 查询扩展模型（懒加载单例）
    
    Returns:
        模型实例或 None（环境变量未配置时）
    """
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,  # 低温度保证确定性
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    """
    将具体问题抽象为更高层次的"退步问题"
    
    用于探寻背后的通用原理或核心概念
    
    Args:
        query: 用户原始问题
    
    Returns:
        退步问题（字符串）
    """
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的'退步问题'，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    """
    回答退步问题，提供通用原理/背景知识
    
    Args:
        step_back_question: 退步问题
    
    Returns:
        退步问题的回答（控制在120字以内）
    """
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    """
    根据用户问题生成"假设性文档"
    
    用于帮助检索相关信息，文档可以包含合理推测
    
    Args:
        query: 用户问题
    
    Returns:
        假设性文档内容
    """
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段'假设性文档'，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    """
    执行 Step-back 查询扩展
    
    将原始问题扩展为包含退步问题和答案的查询
    
    Args:
        query: 用户原始问题
    
    Returns:
        扩展结果，包含 step_back_question, step_back_answer, expanded_query
    """
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def _finalize_retrieval(
    query: str,
    retrieved: List[dict],
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    检索流水线最终处理：召回候选 → Auto-merge → Rerank → 阈值过滤
    
    Args:
        query: 用户查询
        retrieved: 召回的原始文档
        top_k: 最终返回数量
        retrieval_mode: 检索模式
        candidate_k: 候选池大小
        candidate_config: 候选池配置信息
    
    Returns:
        {"docs": 最终文档列表, "meta": 检索元数据}
    """
    # Step 1: Auto-merge（自动合并父块）
    candidates, merge_meta = _auto_merge_candidates(retrieved)
    
    # Step 2: Rerank（精排）
    reranked_docs, rerank_meta = _rerank_documents(query=query, docs=candidates, top_k=top_k)
    
    # Step 3: 阈值过滤
    post_rerank_count = len(reranked_docs)
    final_docs = [d for d in reranked_docs if _meets_rerank_min_score(d)]
    
    # 构建完整元数据
    meta = {
        **rerank_meta,
        **merge_meta,
        **candidate_config,
        "retrieval_mode": retrieval_mode,
        "retrieval_pipeline": "recall_merge_rerank",
        "candidate_k": candidate_k,
        "retrieval_top_k": top_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "recall_count": len(retrieved),
        "rerank_min_score": RERANK_MIN_SCORE,
        "post_rerank_count": post_rerank_count,
        "post_threshold_count": len(final_docs),
        "retrieval_empty": len(final_docs) == 0,
    }
    
    return {"docs": final_docs, "meta": meta}


def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    核心检索函数：执行混合检索（稠密+BGE-M3 + 稀疏+BM25）
    
    支持三级降级回退：
    1. Hybrid（混合检索）
    2. Dense-only（仅稠密检索）
    3. 空结果兜底
    
    Args:
        query: 用户查询
        top_k: 最终返回数量（默认5）
    
    Returns:
        {"docs": 检索到的文档列表, "meta": 检索元数据}
    """
    # 解析候选池大小
    candidate_k, candidate_config = resolve_candidate_k(top_k)
    
    # 构建过滤条件：只检索叶子层（L3）
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    
    try:
        # ========== 第一级：混合检索（Hybrid） ==========
        # 向量化：稠密 + 稀疏
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        # 执行混合检索
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        
        return _finalize_retrieval(
            query=query,
            retrieved=retrieved,
            top_k=top_k,
            retrieval_mode="hybrid",
            candidate_k=candidate_k,
            candidate_config=candidate_config,
        )
    
    except Exception:
        try:
            # ========== 第二级：仅稠密检索（Dense-only 降级） ==========
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            
            return _finalize_retrieval(
                query=query,
                retrieved=retrieved,
                top_k=top_k,
                retrieval_mode="dense_fallback",
                candidate_k=candidate_k,
                candidate_config=candidate_config,
            )
        
        except Exception:
            # ========== 第三级：空结果兜底 ==========
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "retrieval_pipeline": "recall_merge_rerank",
                    "candidate_k": candidate_k,
                    **candidate_config,
                    "retrieval_top_k": top_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "recall_count": 0,
                    **_empty_merge_meta(),
                    "candidate_count": 0,
                    "rerank_min_score": RERANK_MIN_SCORE,
                    "post_rerank_count": 0,
                    "post_threshold_count": 0,
                    "retrieval_empty": True,
                },
            }
