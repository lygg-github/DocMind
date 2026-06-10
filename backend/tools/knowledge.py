# ========== 导入部分 ==========
from langchain_core.tools import tool  # LangChain 工具装饰器

# 导入 RAG 相关模块
from backend.chat.rag_context import record_rag_context  # 记录 RAG 上下文（用于追踪）
from backend.rag.pipeline import run_rag_graph  # 执行完整的 RAG 检索流程


# ========== 全局状态管理 ==========
# 每轮对话中知识库工具的调用次数（限制每轮只调用一次）
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def reset_knowledge_tool_calls() -> None:
    """
    每轮对话开始时重置知识库工具调用计数
    
    调用位置：
    - chat_with_agent() 开始时
    - chat_with_agent_stream() 开始时
    
    目的：确保每轮对话只调用一次知识库检索，避免重复检索
    """
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def _try_acquire_knowledge_tool_call() -> bool:
    """
    尝试获取知识库工具调用权限
    
    Returns:
        True: 可以调用（首次调用）
        False: 已达调用上限（每轮最多调用一次）
    
    设计目的：
    - 防止 Agent 无限循环调用检索工具
    - 确保检索结果被有效利用，而不是重复检索
    """
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    # 检查是否已达调用上限（每轮最多 1 次）
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return False
    # 获取调用权限，计数器 +1
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1
    return True


# ========== 核心工具函数 ==========
@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """
    在知识库中进行混合检索（稠密向量 + 稀疏向量）
    
    Args:
        query: 用户查询字符串
    
    Returns:
        检索到的文档内容（格式化字符串）或错误信息
    
    执行流程：
    1. 检查调用次数限制
    2. 执行 RAG 检索流程
    3. 记录检索轨迹
    4. 格式化检索结果
    """
    # ========== 步骤1：检查调用限制 ==========
    if not _try_acquire_knowledge_tool_call():
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )

    # ========== 步骤2：执行 RAG 检索 ==========
    # run_rag_graph 包含完整的检索逻辑：
    # - 向量化（稠密+BGE-M3 + 稀疏+BM25）
    # - 混合检索（Milvus Hybrid Search）
    # - 降级回退（Hybrid → Dense-only → 空结果）
    # - Jina Rerank 精排
    rag_result = run_rag_graph(query)

    # ========== 步骤3：提取结果和轨迹 ==========
    # 获取检索到的文档列表
    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    # 获取检索轨迹（用于前端展示和调试）
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    
    # 记录 RAG 上下文（供后续获取轨迹使用）
    record_rag_context(rag_trace)

    # ========== 步骤4：处理空结果 ==========
    if not docs:
        return "No relevant documents found in the knowledge base."

    # ========== 步骤5：格式化结果 ==========
    formatted = []
    for i, result in enumerate(docs, 1):
        # 提取文档元信息
        source = result.get("filename", "Unknown")  # 文件名
        page = result.get("page_number", "N/A")    # 页码
        text = result.get("text", "")              # 文档内容
        
        # 格式化输出
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    # 返回格式化后的检索结果（供 Agent 使用）
    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
