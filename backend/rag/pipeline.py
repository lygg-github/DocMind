# ========== 导入部分 ==========
# 类型提示：Annotated用于带元数据的类型，Literal用于字面量类型，TypedDict用于类型化字典
from typing import Annotated, Literal, TypedDict, List, Optional  
import operator  # operator.add用于状态合并（reducer）
import os  # 环境变量读取
from langchain.chat_models import init_chat_model  # LangChain 模型初始化工厂函数
from langgraph.graph import StateGraph, END  # StateGraph是状态图核心类，END是终止节点
from langgraph.types import Send  # Send用于并行分发到子图
from pydantic import BaseModel, Field  # Pydantic数据验证模型

# 导入RAG核心工具函数
from backend.rag.utils import (
    retrieve_documents,        # 核心检索函数：混合检索+三级降级回退
    step_back_expand,          # Step-back查询扩展：生成退步问题和答案
    generate_hypothetical_document,  # HyDE：生成假设性文档
    dedupe_documents,          # 文档去重：按chunk_id去重，保留最高分
    retrieval_trace_fields,    # 提取检索轨迹字段
    merge_retrieval_trace,     # 合并多路检索轨迹
)
# 导入流式相关模块：emit_rag_step用于输出检索步骤，set_sub_agent_group用于子Agent分组
from backend.chat.streaming import emit_rag_step, set_sub_agent_group, clear_sub_agent_group


# ========== 环境变量配置 ==========
API_KEY = os.getenv("ARK_API_KEY")      # API密钥（用于调用LLM）
MODEL = os.getenv("MODEL")              # 主模型名称
BASE_URL = os.getenv("BASE_URL")        # API基础URL（支持自定义部署）
GRADE_MODEL = os.getenv("GRADE_MODEL", "gpt-4.1")  # 文档相关性评分模型（默认gpt-4.1）
FAST_MODEL = os.getenv("FAST_MODEL") or MODEL       # 快速模型（用于复杂度分类，可独立配置）


# ========== 模型实例（模块级单例，懒加载） ==========
_grader_model = None      # 文档相关性评分模型（用于判断检索结果是否相关）
_router_model = None      # 查询重写策略路由模型（选择step_back/hyde/complex）
_complexity_model = None  # 问题复杂度分类模型（判断simple/complex）


def _get_grader_model():
    """
    获取文档相关性评分模型（懒加载单例）
    
    Returns:
        评分模型实例或None（环境变量未配置时）
    
    用途：评估检索到的文档与用户问题的相关性，决定是否需要重写查询
    """
    global _grader_model
    # 检查环境变量是否配置
    if not API_KEY or not GRADE_MODEL:
        return None
    # 懒加载：首次调用时初始化
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,      # temperature=0确保确定性输出（是/否判断）
            stream_usage=True,  # 启用流式使用追踪
        )
    return _grader_model


def _get_router_model():
    """
    获取查询重写策略路由模型（懒加载单例）
    
    Returns:
        路由模型实例或None（环境变量未配置时）
    
    用途：根据用户问题选择最合适的查询扩展策略
    """
    global _router_model
    if not API_KEY or not MODEL:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,      # 确定性输出（策略选择）
            stream_usage=True,
        )
    return _router_model


def _get_complexity_model():
    """
    获取问题复杂度分类模型（懒加载单例）
    
    Returns:
        复杂度模型实例或None（环境变量未配置时）
    
    用途：判断问题复杂度，决定是否分解为子问题并行检索
    """
    global _complexity_model
    if not API_KEY or not FAST_MODEL:
        return None
    if _complexity_model is None:
        _complexity_model = init_chat_model(
            model=FAST_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,      # 确定性输出（分类判断）
            stream_usage=True,
        )
    return _complexity_model


# ========== 提示词定义 ==========
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


# ========== Pydantic数据模型（用于结构化输出） ==========
class GradeDocuments(BaseModel):
    """文档相关性评分结果模型：用于LLM结构化输出"""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """查询扩展策略选择模型：用于LLM选择step_back/hyde/complex"""
    strategy: Literal["step_back", "hyde", "complex"]


class ComplexityResult(BaseModel):
    """问题复杂度分类结果模型：用于LLM判断simple/complex"""
    complexity: Literal["simple", "complex"] = Field(
        description="问题复杂度：'simple'为简单问题，'complex'为复杂问题"
    )
    reason: str = Field(default="", description="分类理由")


class SubQuestions(BaseModel):
    """复杂问题分解后的子问题列表模型"""
    sub_questions: List[str] = Field(
        description="2-4个独立子问题，每个聚焦原问题的一个方面",
        min_length=1,
        max_length=4,
    )


# ========== 状态定义（RAG流程的全局状态） ==========
class RAGState(TypedDict):
    """
    RAG检索流程的状态定义（TypedDict用于类型安全的字典）
    
    核心字段：
    - question: 用户原始问题（始终不变）
    - query: 当前用于检索的查询（可能被扩展）
    - context: 检索到的上下文（格式化后供LLM使用）
    - docs: 检索到的原始文档列表
    - route: 当前路由方向（用于条件分支）
    - rag_trace: 检索轨迹（用于追踪、调试和前端展示）
    
    查询扩展字段：
    - expansion_type: 扩展策略类型（step_back/hyde/complex）
    - expanded_query: 扩展后的查询字符串
    - step_back_question: 退步问题（更抽象的通用问题）
    - step_back_answer: 退步问题的回答
    - hypothetical_doc: HyDE生成的假设性文档
    
    复杂度路由字段：
    - complexity: 问题复杂度（simple/complex）
    - complexity_reason: 复杂度分类理由
    - sub_questions: 分解后的子问题列表
    - is_sub_agent: 是否为子Agent调用（用于区分主流程和子流程）
    - sub_results: 子Agent返回结果（使用operator.add合并多个子结果）
    """
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]
    # 复杂度路由新增字段
    complexity: Optional[str]
    complexity_reason: Optional[str]
    sub_questions: Optional[List[str]]
    is_sub_agent: bool
    # Annotated[List[dict], operator.add]表示多个子Agent的结果会自动合并
    sub_results: Annotated[List[dict], operator.add]  


# ========== 辅助函数 ==========
def _format_docs(docs: List[dict]) -> str:
    """
    将文档列表格式化为可读字符串（供LLM使用的context格式）
    
    Args:
        docs: 文档列表，每个文档需包含filename, page_number, text字段
    
    Returns:
        格式化后的字符串，每条文档格式：[序号] 文件名 (页码):\n内容
    """
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):  # 从1开始编号
        source = doc.get("filename", "Unknown")  # 文件名（来源）
        page = doc.get("page_number", "N/A")    # 页码
        text = doc.get("text", "")              # 文档内容
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    # 用分隔线连接各文档
    return "\n\n---\n\n".join(chunks)


# ========== 核心节点函数（状态图节点） ==========
def retrieve_initial(state: RAGState) -> RAGState:
    """
    初始检索节点：执行第一次混合检索
    
    Args:
        state: 当前RAG状态
    
    Returns:
        更新后的状态，包含检索结果和轨迹
    
    执行流程：
    1. 获取用户问题
    2. 调用retrieve_documents执行混合检索（稠密+BGE-M3 + 稀疏+BM25）
    3. 格式化上下文
    4. 记录检索轨迹
    """
    # 获取用户原始问题
    query = state["question"]
    # 输出检索步骤（用于流式展示）
    emit_rag_step("🔍", "正在检索知识库...", f"查询: {query[:50]}")  # 截断显示前50字符
    
    # ========== 核心检索调用 ==========
    # retrieve_documents包含：混合检索、Auto-merge、Rerank、阈值过滤 
    retrieved = retrieve_documents(query, top_k=5)
    results = retrieved.get("docs", [])           # 检索到的文档列表
    retrieve_meta = retrieved.get("meta", {})     # 检索元数据（模式、召回数等）
    context = _format_docs(results)              # 格式化上下文
    
    # ========== 输出检索过程信息（用于前端展示） ==========
    # 三级分块检索信息
    emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    # Auto-merging合并信息
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    # 检索完成信息
    emit_rag_step("✅", f"检索完成，找到 {len(results)} 个片段", f"模式: {retrieve_meta.get('retrieval_mode', 'hybrid')}")
    
    # 如果没有检索结果，提示强制step-back
    if not results:
        emit_rag_step("⚠️", "无可用片段，跳过评估并强制step-back扩展检索")
    
    # ========== 构建检索轨迹（用于追踪和前端展示） ==========
    rag_trace = {
        "tool_used": True,                          # 标记使用了检索工具
        "tool_name": "search_knowledge_base",       # 工具名称
        "query": query,                             # 原始查询
        "expanded_query": query,                    # 扩展后的查询（初始阶段等于原始查询）
        "retrieved_chunks": results,                # 检索到的文档
        "initial_retrieved_chunks": results,        # 初始检索结果（保留用于对比）
        "retrieval_stage": "initial",               # 检索阶段标记
        **retrieval_trace_fields(retrieve_meta),    # 展开检索元数据字段
    }
    
    # 返回更新后的状态
    return {
        "query": query,
        "docs": results,# 检索到的文档列表，分有没有两种情况
        "context": context,
        "rag_trace": rag_trace,
    }


def _route_after_initial(state: RAGState) -> Literal["grade_documents", "rewrite_question"]:
    """
    初始检索后的路由判断（条件分支函数）
    
    Args:
        state: 当前状态
    
    Returns:
        "grade_documents": 有检索结果，进行相关性评估
        "rewrite_question": 无检索结果，直接重写查询
    """
    if not state.get("docs"):
        return "rewrite_question"  # 无结果→重写查询
    return "grade_documents"       # 有结果→评估相关性


def grade_documents_node(state: RAGState) -> RAGState:
    """
    文档相关性评估节点：判断检索结果是否与问题相关
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态，包含评估结果和路由方向
    
    执行流程：
    1. 获取评分模型
    2. 如果模型不可用，直接路由到重写
    3. 使用GRADE_PROMPT评估文档相关性
    4. 根据评分决定路由方向（generate_answer或rewrite_question）
    """
    # 获取评分模型（懒加载）
    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估文档相关性...")
    
    # 如果评分模型不可用，跳过评估，直接路由到重写
    if not grader:
        grade_update = {
            "grade_score": "unknown",
            "grade_route": "rewrite_question",
            "rewrite_needed": True,
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return {"route": "rewrite_question", "rag_trace": rag_trace}
    
    # 准备评估数据
    question = state["question"]
    context = state.get("context", "")
    # 格式化提示词
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    # ========== 调用评分模型（结构化输出） ==========
    response = grader.with_structured_output(GradeDocuments, method="function_calling").invoke(
        [{"role": "user", "content": prompt}]
    )
    score = (response.binary_score or "").strip().lower()  # 标准化评分
    
    # 根据评分决定路由方向
    route = "generate_answer" if score == "yes" else "rewrite_question"
    
    # 输出评估结果
    if route == "generate_answer":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询", f"评分: {score}")
    
    # 更新检索轨迹
    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
    }
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    """
    查询重写节点：选择并执行查询扩展策略
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态，包含扩展后的查询和策略信息
    
    支持的策略：
    - step_back: 退步扩展（处理包含具体细节的问题，先理解通用概念）
    - hyde: 假设性文档扩展（处理模糊、概念性问题）
    - complex: 复杂问题扩展（结合step_back和hyde）
    """
    question = state["question"]
    # 无检索结果时强制使用step_back
    force_step_back = not state.get("docs")
    emit_rag_step("✏️", "正在重写查询...")

    # ========== 选择扩展策略 ==========
    if force_step_back:
        strategy = "step_back"  # 强制step_back
    else:
        router = _get_router_model()
        strategy = "step_back"  # 默认策略
        if router:
            # 调用路由模型选择策略
            prompt = (
                "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
                "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
                "- hyde：模糊、概念性、需要解释或定义的问题。\n"
                "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
                f"用户问题：{question}"
            )
            try:
                decision = router.with_structured_output(RewriteStrategy, method="function_calling").invoke(
                    [{"role": "user", "content": prompt}]
                )
                strategy = decision.strategy
            except Exception:
                strategy = "step_back"  # 异常时回退到默认策略

    # ========== 执行查询扩展 ==========
    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    # Step-back扩展：生成退步问题和答案
    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    # HyDE扩展：生成假设性文档（非强制step_back时）
    if not force_step_back and strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE假设性文档生成中...")
        hypothetical_doc = generate_hypothetical_document(question)

    # 更新检索轨迹
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
        "grade_skipped": force_step_back,
    })

    # 返回扩展后的状态
    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    """
    扩展检索节点：使用扩展后的查询重新检索
    
    Args:
        state: 当前状态（包含expansion_type和expanded_query）
    
    Returns:
        更新后的状态，包含扩展检索结果和轨迹
    
    支持的检索组合：
    - hyde + step_back（complex策略）
    - hyde单独
    - step_back单独
    """
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    
    results: List[dict] = []
    rerank_errors = []
    retrieval_trace: dict = {}

    # ========== HyDE检索（如果策略包含hyde） ==========
    if strategy in ("hyde", "complex"):
        # 获取假设性文档（如果之前没生成则生成）
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        # 使用假设性文档检索
        retrieved_hyde = retrieve_documents(hypothetical_doc, top_k=5)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        # 输出HyDE检索信息
        emit_rag_step(
            "🧱",
            "HyDE三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        # 记录精排错误
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        # 合并检索轨迹
        retrieval_trace = merge_retrieval_trace(retrieval_trace, hyde_meta)

    # ========== Step-back检索（如果策略包含step_back） ==========
    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = retrieve_documents(expanded_query, top_k=5)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        # 输出Step-back检索信息
        emit_rag_step(
            "🧱",
            "Step-back三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        # 记录精排错误
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        # 合并检索轨迹
        retrieval_trace = merge_retrieval_trace(retrieval_trace, step_meta)

    # ========== 去重（多路召回可能有重复） ==========
    deduped = dedupe_documents(results)

    # 重新排名（扩展阶段可能合并多路召回，需要统一排名）
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    # 格式化上下文
    context = _format_docs(deduped)
    emit_rag_step("✅", f"扩展检索完成，共 {len(deduped)} 个片段")
    
    # 更新检索轨迹
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else retrieval_trace.get("rerank_error"),
        **retrieval_trace,
    })
    
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# 复杂度分类 & 子问题分解（处理复杂问题的并行检索）
# ---------------------------------------------------------------------------

# 复杂度分类提示词
COMPLEXITY_PROMPT = (
    "你是一个问题复杂度分类器。请判断用户问题的复杂度。\n\n"
    "【简单问题】：事实查询、定义查询、单一信息点查询、明确的yes/no问题、"
    "某个具体属性/参数/规格的查询。\n"
    "【复杂问题】：需要跨文档综合、多角度分析、比较对比、多步骤推理、"
    "需要综合多个信息源才能完整回答的问题。\n\n"
    "用户问题：{question}\n\n"
    "请输出分类结果。"
)

# 子问题分解提示词
DECOMPOSE_PROMPT = (
    "请将以下复杂问题分解为2-4个独立的子问题。\n"
    "每个子问题应聚焦于原问题的一个明确方面，能独立通过知识库检索获得答案。\n"
    "子问题之间应覆盖原问题的核心维度，避免重叠。\n\n"
    "原问题：{question}\n\n"
    "请输出子问题列表。"
)


def classify_complexity(state: RAGState) -> RAGState:
    """
    使用FAST_MODEL判断问题复杂度
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态，包含复杂度分类结果
    """
    question = state["question"]
    emit_rag_step("🧭", "正在分析问题复杂度...")

    model = _get_complexity_model()
    # 如果模型不可用，默认简单问题
    if not model:
        emit_rag_step("⚠️", "复杂度模型不可用，默认简单问题")
        return {"complexity": "simple", "complexity_reason": "model_unavailable"}

    # 构建提示词
    prompt = COMPLEXITY_PROMPT.format(question=question)
    try:
        # 调用模型进行分类（结构化输出）
        result = model.with_structured_output(ComplexityResult, method="function_calling").invoke(
            [{"role": "user", "content": prompt}]
        )
        complexity = (result.complexity or "simple").strip().lower()
        reason = (result.reason or "").strip()
        # 校验结果有效性
        if complexity not in ("simple", "complex"):
            complexity = "simple"
    except Exception:
        # 异常时默认简单问题
        complexity = "simple"
        reason = "classification_error"

    # 输出分类结果
    if complexity == "simple":
        emit_rag_step("✅", "简单问题 → 走标准RAG流程", f"理由: {reason[:60]}")
    else:
        emit_rag_step("🔀", "复杂问题 → 将分解为子问题并行检索", f"理由: {reason[:60]}")

    return {"complexity": complexity, "complexity_reason": reason}


def decompose_question(state: RAGState) -> RAGState:
    """
    将复杂问题分解为2-4个独立子问题
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态，包含子问题列表
    """
    question = state["question"]
    emit_rag_step("🧩", "正在分解复杂问题...")

    model = _get_complexity_model()
    # 如果模型不可用，使用原始问题作为唯一子问题
    if not model:
        emit_rag_step("⚠️", "分解模型不可用，使用原始问题")
        return {"sub_questions": [question]}

    # 构建提示词
    prompt = DECOMPOSE_PROMPT.format(question=question)
    try:
        # 调用模型分解问题（结构化输出）
        result = model.with_structured_output(SubQuestions, method="function_calling").invoke(
            [{"role": "user", "content": prompt}]
        )
        # 过滤空字符串
        sub_qs = [sq.strip() for sq in (result.sub_questions or []) if sq.strip()]
        if not sub_qs:
            sub_qs = [question]
    except Exception:
        # 异常时使用原始问题
        sub_qs = [question]

    # 输出每个子问题
    for i, sq in enumerate(sub_qs, 1):
        emit_rag_step("📌", f"子问题 {i}", sq[:80])

    return {"sub_questions": sub_qs}


def _route_after_complexity(state: RAGState):
    """
    复杂度路由：根据复杂度决定后续流程
    
    Args:
        state: 当前状态
    
    Returns:
        "retrieve_initial": 简单问题，走标准RAG流程
        "decompose_question": 复杂问题，先分解为子问题
    """
    if state.get("complexity") == "complex":
        return "decompose_question"
    return "retrieve_initial"


def _fanout_sub_questions(state: RAGState):
    """
    将分解后的子问题并行分发到rag_sub_agent子图
    
    使用LangGraph的Send API实现并行执行
    
    Args:
        state: 当前状态（包含sub_questions）
    
    Returns:
        Send对象列表，每个子问题对应一个Send（并行执行）
    """
    sub_qs = state.get("sub_questions") or []
    if not sub_qs:
        # 分解失败，回退到原有流程
        return [Send("retrieve_initial", {
            "question": state["question"],
            "query": state["question"],
            "context": "",
            "docs": [],
            "route": None,
            "expansion_type": None,
            "expanded_query": None,
            "step_back_question": None,
            "step_back_answer": None,
            "hypothetical_doc": None,
            "rag_trace": None,
            "complexity": None,
            "complexity_reason": None,
            "sub_questions": None,
            "is_sub_agent": False,
            "sub_results": [],
        })]
    
    # 并行分发每个子问题到rag_sub_agent
    return [
        Send("rag_sub_agent", {
            "question": sq,
            "query": sq,
            "context": "",
            "docs": [],
            "route": None,
            "expansion_type": None,
            "expanded_query": None,
            "step_back_question": None,
            "step_back_answer": None,
            "hypothetical_doc": None,
            "rag_trace": None,
            "complexity": None,
            "complexity_reason": None,
            "sub_questions": None,
            "is_sub_agent": True,   # 标记为子Agent调用
            "sub_results": [],
        })
        for sq in sub_qs
    ]


def synthesis(state: RAGState) -> RAGState:
    """
    合成节点：合并所有子Agent检索到的文档
    
    Args:
        state: 当前状态（包含所有子Agent的sub_results）
    
    Returns:
        更新后的状态，包含合并后的文档和上下文
    """
    sub_results = state.get("sub_results", [])
    emit_rag_step("🔬", f"正在合成 {len(sub_results)} 个子问题的检索结果...")

    # 收集所有子问题的检索结果
    all_docs: List[dict] = []
    for result in sub_results:
        docs = result.get("docs", [])
        all_docs.extend(docs)

    # 去重和重新排名
    deduped = dedupe_documents(all_docs)
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    # 格式化上下文
    context = _format_docs(deduped)
    emit_rag_step("✅", f"合成完成，共 {len(deduped)} 个去重片段")

    # 合并所有子Agent的检索轨迹
    sub_traces = []
    for result in sub_results:
        trace = result.get("rag_trace")
        if trace:
            sub_traces.append(trace)

    # 构建最终检索轨迹
    original_trace = state.get("rag_trace") or {}
    rag_trace = {
        **original_trace,
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": state["question"],
        "expanded_query": state["question"],
        "retrieved_chunks": deduped,
        "retrieval_stage": "synthesis",
        "complexity": "complex",
        "complexity_reason": state.get("complexity_reason", ""),
        "sub_questions": state.get("sub_questions", []),
        "sub_agent_count": len(sub_results),
        "synthesis_merged_count": len(all_docs),
        "sub_traces": sub_traces,
    }

    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


# ---------------------------------------------------------------------------
# 子Agent RAG子图（每个子问题独立运行完整RAG流程）
# ---------------------------------------------------------------------------

def build_rag_sub_agent_graph():
    """
    构建子Agent RAG子图
    
    子图流程：
    retrieve_initial → (grade_documents或rewrite_question) → retrieve_expanded → END
    
    Returns:
        编译后的子图（可被主图调用）
    """
    sub_graph = StateGraph(RAGState)
    
    # 添加节点
    sub_graph.add_node("retrieve_initial", retrieve_initial)      # 初始检索
    sub_graph.add_node("grade_documents", grade_documents_node)  # 相关性评估
    sub_graph.add_node("rewrite_question", rewrite_question_node)# 查询重写
    sub_graph.add_node("retrieve_expanded", retrieve_expanded)    # 扩展检索

    # 设置入口节点
    sub_graph.set_entry_point("retrieve_initial")
    
    # 初始检索后的条件路由
    sub_graph.add_conditional_edges(
        "retrieve_initial",
        _route_after_initial,
        {
            "grade_documents": "grade_documents",
            "rewrite_question": "rewrite_question",
        },
    )
    
    # 评估后的条件路由
    sub_graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,      # 评估通过→结束
            "rewrite_question": "rewrite_question",  # 评估不通过→重写
        },
    )
    
    # 重写后执行扩展检索
    sub_graph.add_edge("rewrite_question", "retrieve_expanded")
    sub_graph.add_edge("retrieve_expanded", END)
    
    # 编译子图
    return sub_graph.compile()


# 子Agent子图实例（模块级单例，避免重复创建）
_rag_sub_agent_graph = build_rag_sub_agent_graph()


def rag_sub_agent(state: RAGState) -> RAGState:
    """
    包装子图，将子图结果封装为sub_results以便主图通过reducer合并
    
    Args:
        state: 当前状态
    
    Returns:
        更新后的状态，包含子Agent结果（存储在sub_results中）
    """
    question = state.get("question", "")
    # 设置子Agent分组标识，使子图内所有emit_rag_step自动携带group（用于前端区分）
    set_sub_agent_group(question)
    try:
        # 执行子图
        result = _rag_sub_agent_graph.invoke(state)
    finally:
        # 清理分组标识
        clear_sub_agent_group()
    
    # 将结果封装为sub_results格式（供主图合并）
    return {
        "sub_results": [{
            "question": question,
            "docs": result.get("docs", []),
            "rag_trace": result.get("rag_trace"),
        }],
    }


# ---------------------------------------------------------------------------
# 主RAG图（完整的检索流程编排）
# ---------------------------------------------------------------------------

def build_rag_graph():
    """
    构建主RAG图
    
    主图流程：
    1. classify_complexity → 复杂度分类
    2. 根据复杂度路由：
       - simple → retrieve_initial → grade_documents → (generate_answer或rewrite_question)
       - complex → decompose_question → fanout → rag_sub_agent×N → synthesis
    
    Returns:
        编译后的主图
    """
    graph = StateGraph(RAGState)

    # 节点注册（按执行顺序）
    graph.add_node("classify_complexity", classify_complexity)      # 复杂度分类
    graph.add_node("decompose_question", decompose_question)        # 问题分解
    graph.add_node("retrieve_initial", retrieve_initial)            # 初始检索
    graph.add_node("grade_documents", grade_documents_node)         # 相关性评估
    graph.add_node("rewrite_question", rewrite_question_node)       # 查询重写
    graph.add_node("retrieve_expanded", retrieve_expanded)          # 扩展检索
    graph.add_node("rag_sub_agent", rag_sub_agent)                  # 子Agent
    graph.add_node("synthesis", synthesis)                          # 结果合成

    # 设置入口节点：复杂度分类
    graph.set_entry_point("classify_complexity")

    # 复杂度路由：根据复杂度选择路径
    graph.add_conditional_edges(
        "classify_complexity",
        _route_after_complexity,
        {
            "retrieve_initial": "retrieve_initial",
            "decompose_question": "decompose_question",
        },
    )

    # 分解节点 → 并行分发到rag_sub_agent（使用_fanout_sub_questions返回Send列表）
    graph.add_conditional_edges("decompose_question", _fanout_sub_questions)

    # ========== 简单问题路径 ==========
    # 初始检索 → 评估或重写
    graph.add_conditional_edges(
        "retrieve_initial",
        _route_after_initial,
        {
            "grade_documents": "grade_documents",
            "rewrite_question": "rewrite_question",
        },
    )
    # 评估 → 生成答案或重写
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    # 重写 → 扩展检索 → 结束
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)

    # ========== 复杂问题路径 ==========
    # 子Agent并行执行 → 合成 → 结束
    graph.add_edge("rag_sub_agent", "synthesis")
    graph.add_edge("synthesis", END)

    # 编译主图
    return graph.compile()


# 主RAG图实例（模块级单例）
rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    """
    执行RAG检索流程的入口函数
    
    Args:
        question: 用户问题
    
    Returns:
        {"docs": 检索到的文档列表, "rag_trace": 检索轨迹}
    
    调用位置：
    - backend/tools/knowledge.py的search_knowledge_base函数
    """
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
        # 复杂度路由新增字段
        "complexity": None,
        "complexity_reason": None,
        "sub_questions": None,
        "is_sub_agent": False,
        "sub_results": [],
    })
