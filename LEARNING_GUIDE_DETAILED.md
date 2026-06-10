
# 📚 RAG 知识库平台深度学习指南

> 从源码层面理解企业级 RAG 系统的完整实现

---

## 🔍 第一章：项目架构全景

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           前端层 (Frontend)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │  Chat Interface │  │ Document Upload │  │  RAG Steps Visualization  │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬───────────────┘  │
└───────────┼───────────────────┼─────────────────────────┼───────────────────┘
            │                   │                         │
            ▼                   ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API 层 (FastAPI)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  /chat/stream          /documents/upload        /sessions/list      │  │
│  │  (SSE流式响应)          (文档上传)               (会话管理)          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│   Chat Service    │   │  Indexing Service │   │   RAG Pipeline    │
│   (对话服务)       │   │   (索引服务)       │   │   (检索管道)      │
│   • 上下文管理     │   │   • 文档加载      │   │   • 混合检索      │
│   • 持久化笔记     │   │   • 三级分块      │   │   • 查询重写      │
│   • 流式输出       │   │   • 向量化        │   │   • 复杂度路由    │
└───────────────────┘   └───────────────────┘   └───────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│    Milvus         │   │   PostgreSQL      │   │     Redis         │
│   (向量数据库)     │   │   (关系数据库)     │   │    (缓存)         │
│   • 稠密向量       │   │   • 会话记录       │   │   • 会话缓存      │
│   • 稀疏向量       │   │   • 父块存储       │   │   • 热点数据      │
│   • RRFRanker融合  │   │   • 用户管理       │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

### 1.2 核心数据流

#### 文档入库流程

```
用户上传文档 → documents.py → upload_jobs.py → document_loader.py
                                                               ↓
                                                  ┌─────────────────────┐
                                                  │  三级分块处理器     │
                                                  │  L1/L2/L3 递归切分  │
                                                  └──────────┬──────────┘
                                                             ↓
                                                  ┌─────────────────────┐
                                                  │   EmbeddingService  │
                                                  │  BGE-M3 + BM25     │
                                                  └──────────┬──────────┘
                                                             ↓
                                                  ┌─────────────────────┐
                                                  │   MilvusWriter      │
                                                  │  叶子块入Milvus     │
                                                  │  父块入PostgreSQL   │
                                                  └─────────────────────┘
```

#### 问答检索流程

```
用户提问 → chat.py → service.py → runtime.py(Agent)
                                      ↓
                              knowledge.py → run_rag_graph()
                                                  ↓
                              ┌───────────────────────────────┐
                              │  ① classify_complexity        │
                              │  ② decompose_question(可选)   │
                              │  ③ retrieve_initial          │
                              │  ④ grade_documents            │
                              │  ⑤ rewrite_question          │
                              │  ⑥ retrieve_expanded         │
                              │  ⑦ synthesis(可选)           │
                              └───────────────┬───────────────┘
                                              ↓
                              ┌───────────────────────────────┐
                              │   LLM 生成最终答案            │
                              └───────────────────────────────┘
```

---

## 🏃 第二章：文档入库流程深度解析

### 2.1 文档上传入口

**位置**: `backend/api/routes/documents.py`

```python
# 关键流程：接收文件 → 保存临时文件 → 异步处理
async def upload_document(file: UploadFile, user_id: str = "default"):
    # 1. 保存上传文件
    temp_path = save_upload_file(file)
    
    # 2. 提交异步任务
    job = UploadJob(
        user_id=user_id,
        file_path=str(temp_path),
        filename=file.filename
    )
    await jobs.upload_jobs.enqueue(job)
    
    # 3. 返回任务ID
    return {"job_id": job.id, "status": "pending"}
```

### 2.2 文档加载与解析

**位置**: `backend/indexing/document_loader.py`

```python
# 支持多格式文档
def load_document(file_path: str) -> list[str]:
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".md":
        return load_markdown(file_path)
    elif ext == ".txt":
        return load_text(file_path)
    elif ext == ".html":
        return load_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
```

### 2.3 三级分块实现

**位置**: `backend/indexing/document_loader.py`

```python
def chunk_document(text: str, chunk_size: int = 500) -> list[dict]:
    """
    三级递归切分算法：
    - L1: 大段落 (~2000 tokens)
    - L2: 小节 (~500 tokens)
    - L3: 句子 (~100 tokens) - 叶子节点
    """
    chunks = []
    
    # 第一级：按段落切分 (L1)
    paragraphs = text.split('\n\n')
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph_chunks = recursive_chunk(paragraph, level=1)
        chunks.extend(paragraph_chunks)
    
    return chunks

def recursive_chunk(text: str, level: int, parent_id: str = None) -> list[dict]:
    """递归切分：直到达到 L3 或最小长度"""
    if level >= 3 or len(text) <= 150:
        # 叶子节点，生成唯一 ID
        chunk_id = generate_chunk_id()
        return [{
            "text": text,
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_id,
            "chunk_level": level,
        }]
    
    # 按标点符号切分
    separators = ['。', '！', '？', '.', '!', '?', '\n']
    parts = split_by_separators(text, separators)
    
    results = []
    for part in parts:
        part_id = generate_chunk_id()
        children = recursive_chunk(part, level + 1, parent_id=part_id)
        results.extend(children)
    
    return results
```

### 2.4 向量化服务

**位置**: `backend/indexing/embedding.py`

```python
class EmbeddingService:
    def __init__(self):
        # 稠密向量模型
        self._embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # BM25 稀疏向量统计
        self._vocab: dict[str, int] = {}        # 词表
        self._doc_freq: Counter = Counter()     # 文档频率
        self._total_docs = 0                    # 总文档数
        self._sum_token_len = 0                 # 总词数
        
        # 从持久化文件加载状态
        self._load_state()
    
    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """同时获取稠密和稀疏向量"""
        dense = self.get_embeddings(texts)
        sparse = self.get_sparse_embeddings(texts)
        return dense, sparse
    
    def increment_add_documents(self, texts: list[str]) -> None:
        """增量更新 BM25 统计（入库时调用）"""
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                self._sum_token_len += len(tokens)
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._persist()
```

### 2.5 Milvus 写入

**位置**: `backend/indexing/milvus_writer.py`

```python
def write_chunks(chunks: list[dict], dense_embeddings: list[list[float]], 
                 sparse_embeddings: list[dict]) -> None:
    """将分块写入 Milvus（仅叶子块）"""
    milvus_store = get_milvus_store()
    
    data = []
    leaf_chunks = [c for c in chunks if c["chunk_level"] == 3]
    
    for chunk, dense, sparse in zip(leaf_chunks, dense_embeddings, sparse_embeddings):
        data.append({
            "text": chunk["text"],
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "parent_chunk_id": chunk["parent_chunk_id"],
            "chunk_level": chunk["chunk_level"],
            "dense_embedding": dense,
            "sparse_embedding": sparse,
            "page_number": chunk.get("page_number", 0),
        })
    
    milvus_store.insert(data)
```

---

## 🧠 第三章：RAG Pipeline 深度解析

### 3.1 LangGraph 状态定义

**位置**: `backend/rag/pipeline.py`

```python
class RAGState(TypedDict):
    # 基础字段
    question: str                    # 用户原始问题
    query: str                       # 当前查询词
    context: str                     # 检索到的上下文
    docs: List[dict]                 # 检索结果列表
    
    # 路由字段
    route: Optional[str]             # 当前路由方向
    expansion_type: Optional[str]    # 查询扩展类型
    
    # Step-back 字段
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    
    # HyDE 字段
    hypothetical_doc: Optional[str]
    
    # 复杂度路由字段
    complexity: Optional[str]        # simple / complex
    sub_questions: Optional[List[str]]
    is_sub_agent: bool               # 是否为子 Agent
    sub_results: Annotated[List[dict], operator.add]  # 子结果聚合
    
    # 追踪字段
    rag_trace: Optional[dict]        # 完整检索追踪信息
```

### 3.2 复杂度分类节点

```python
COMPLEXITY_PROMPT = """
你是一个问题复杂度分类器。请判断用户问题的复杂度。

【简单问题】：事实查询、定义查询、单一信息点查询、明确的 yes/no 问题、
某个具体属性/参数/规格的查询。

【复杂问题】：需要跨文档综合、多角度分析、比较对比、多步骤推理、
需要综合多个信息源才能完整回答的问题。

用户问题：{question}

请输出分类结果。
"""

def classify_complexity(state: RAGState) -> RAGState:
    """使用 FAST_MODEL 判断问题复杂度"""
    question = state["question"]
    
    model = _get_complexity_model()
    if not model:
        return {"complexity": "simple"}
    
    prompt = COMPLEXITY_PROMPT.format(question=question)
    result = model.with_structured_output(ComplexityResult).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    return {
        "complexity": result.complexity,
        "complexity_reason": result.reason
    }
```

### 3.3 子问题分解与并行检索

```python
def decompose_question(state: RAGState) -> RAGState:
    """将复杂问题分解为 2-4 个独立子问题"""
    question = state["question"]
    
    prompt = """
    请将以下复杂问题分解为 2-4 个独立的子问题。
    每个子问题应聚焦于原问题的一个明确方面，能独立通过知识库检索获得答案。
    
    原问题：{question}
    """
    
    result = model.with_structured_output(SubQuestions).invoke(
        [{"role": "user", "content": prompt.format(question=question)}]
    )
    
    return {"sub_questions": result.sub_questions}

def _fanout_sub_questions(state: RAGState):
    """通过 LangGraph Send API 并行分发到子图"""
    sub_qs = state.get("sub_questions") or []
    
    return [
        Send("rag_sub_agent", {
            "question": sq,
            "is_sub_agent": True,
            # ... 其他状态字段
        })
        for sq in sub_qs
    ]
```

### 3.4 混合检索实现

**位置**: `backend/rag/utils.py`

```python
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    混合检索主入口：
    1. 生成稠密向量（BGE-M3）
    2. 生成稀疏向量（BM25）
    3. Milvus Hybrid Search + RRFRanker
    4. Auto-merging 合并
    5. Jina Rerank 精排
    """
    
    # 1. 解析候选池大小
    candidate_k, candidate_config = resolve_candidate_k(top_k)
    
    try:
        # 2. 获取稠密和稀疏向量
        dense_embedding = _embedding_service.get_embeddings([query])[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)
        
        # 3. 混合检索
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
        )
        
        # 4. 执行检索后处理流水线
        return _finalize_retrieval(
            query=query,
            retrieved=retrieved,
            top_k=top_k,
            retrieval_mode="hybrid",
            candidate_k=candidate_k,
            candidate_config=candidate_config
        )
    
    except Exception as e:
        # 降级到 Dense-only
        return _dense_fallback(query, top_k, candidate_k, candidate_config)
```

### 3.5 检索后处理流水线

```python
def _finalize_retrieval(query: str, retrieved: List[dict], top_k: int, 
                        retrieval_mode: str, candidate_k: int, 
                        candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    检索后处理流水线：
    ① Auto-merging（L3→L2→L1）
    ② Jina Rerank 精排
    ③ 阈值过滤
    """
    
    # 阶段1：Auto-merging
    candidates, merge_meta = _auto_merge_candidates(retrieved)
    
    # 阶段2：Rerank 精排
    reranked_docs, rerank_meta = _rerank_documents(
        query=query, 
        docs=candidates, 
        top_k=top_k
    )
    
    # 阶段3：阈值过滤
    final_docs = [d for d in reranked_docs if _meets_rerank_min_score(d)]
    
    # 组装元数据
    meta = {
        **rerank_meta,
        **merge_meta,
        **candidate_config,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "retrieval_top_k": top_k,
        "recall_count": len(retrieved),
        "post_rerank_count": len(reranked_docs),
        "post_threshold_count": len(final_docs),
        "retrieval_empty": len(final_docs) == 0,
    }
    
    return {"docs": final_docs, "meta": meta}
```

### 3.6 Auto-merging 算法

```python
def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    """
    L3→L2→L1 自动合并：
    - 当同一父块下召回子块数 ≥ threshold 时
    - 用父块内容替换所有子块
    - 保留最高评分
    """
    meta = {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
    }
    
    if not AUTO_MERGE_ENABLED or not docs:
        return docs, meta
    
    # L3 → L2 合并
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    
    # L2 → L1 合并
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)
    
    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    meta.update({
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_replaced_chunks": replaced_count,
    })
    
    return merged_docs, meta
```

---

## 🔄 第四章：对话管理流程

### 4.1 对话服务核心

**位置**: `backend/chat/service.py`

```python
def chat_with_agent(user_text: str, user_id: str, session_id: str):
    """
    完整对话流程：
    1. 加载历史消息和元数据
    2. 构建上下文（滑动窗口 + 持久化笔记）
    3. 调用 Agent
    4. 保存结果
    5. 更新持久化笔记
    """
    
    # 1. 加载会话
    messages, metadata = storage.load_with_meta(user_id, session_id)
    persistent_note = metadata.get("persistent_note", "")
    
    # 2. 重置状态
    get_last_rag_context(clear=True)
    reset_knowledge_tool_calls()
    
    # 3. 构建上下文消息
    context_messages = _build_context_messages(messages, persistent_note, user_text)
    
    # 4. 调用 Agent
    result = agent.invoke(
        {"messages": context_messages},
        config={"recursion_limit": 8}
    )
    
    # 5. 提取响应
    response_content = _extract_response(result)
    
    # 6. 更新持久化笔记
    save_meta["persistent_note"] = _update_persistent_note_sync(
        persistent_note, user_text, response_content
    )
    
    # 7. 保存会话
    storage.save(user_id, session_id, messages, metadata=save_meta)
    
    return {"response": response_content, "rag_trace": rag_trace}
```

### 4.2 上下文构建策略

```python
CONTEXT_WINDOW_MESSAGES = 6  # 滑动窗口大小

def _build_context_messages(messages: list, persistent_note: str, user_text: str) -> list:
    """
    上下文构建策略：
    1. 短期记忆：最近 6 轮对话
    2. 长期记忆：持久化笔记（压缩后的历史摘要）
    """
    
    # 滑动窗口截取最近 N 条消息
    short_term = messages[-CONTEXT_WINDOW_MESSAGES:]
    
    context_messages = []
    
    # 添加持久化笔记作为系统消息
    if persistent_note:
        context_messages.append(SystemMessage(
            content=f"【对话持久化笔记】\n{persistent_note}\n\n保持对话连贯性"
        ))
    
    # 添加短期记忆
    context_messages.extend(short_term)
    
    # 添加当前用户消息
    context_messages.append(HumanMessage(content=user_text))
    
    return context_messages
```

### 4.3 持久化笔记更新

```python
def _update_persistent_note_sync(current_note: str, user_text: str, ai_response: str) -> str:
    """
    使用独立模型压缩历史对话为持久化笔记：
    - 智能合并新旧信息
    - 过滤噪音
    - 控制在 500 字以内
    """
    
    prompt = """
    你是一个【Context Manager Agent】，负责维护多轮对话中的「持久化笔记」。
    
    更新规则：
    1. 将新信息与现有笔记智能合并，不要简单拼接
    2. 过滤噪音，控制在 500 字以内
    3. 若信息冲突，保留最可靠或最新版本
    
    现有笔记：
    {current_note}
    
    最新对话：
    用户：{user_text}
    AI：{ai_response}
    
    请直接输出更新后的笔记：
    """
    
    res = fast_model.invoke([SystemMessage(content=prompt.format(
        current_note=current_note,
        user_text=user_text,
        ai_response=ai_response
    ))])
    
    return res.content.strip()
```

---

## 🌐 第五章：流式输出与实时可视化

### 5.1 SSE 流式响应

**位置**: `backend/api/routes/chat.py`

```python
async def chat_stream(user_text: str, user_id: str, session_id: str):
    """Server-Sent Events 流式响应"""
    
    async for chunk in chat_with_agent_stream(user_text, user_id, session_id):
        yield chunk
```

**位置**: `backend/chat/service.py`

```python
async def chat_with_agent_stream(user_text: str, user_id: str, session_id: str):
    """异步流式对话"""
    
    # 1. 设置 RAG 步骤输出代理
    output_queue = asyncio.Queue()
    
    class _RagStepProxy:
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})
    
    set_rag_step_queue(_RagStepProxy())
    
    # 2. 构建上下文
    context_messages = _build_context_messages(...)
    
    # 3. 异步调用 Agent
    async def _agent_worker():
        async for msg, _metadata in agent.astream(
            {"messages": context_messages},
            stream_mode="messages"
        ):
            if isinstance(msg, AIMessageChunk):
                content = extract_content(msg)
                await output_queue.put({"type": "content", "content": content})
    
    agent_task = asyncio.create_task(_agent_worker())
    
    # 4. 输出事件循环
    try:
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 前端断开连接，终止 Agent
        agent_task.cancel()
        raise
```

### 5.2 RAG 步骤实时推送

**位置**: `backend/chat/streaming.py`

```python
# 全局状态
_RAG_STEP_QUEUE = None    # 步骤队列
_RAG_STEP_LOOP = None     # 主线程事件循环
_sub_agent_context = threading.local()  # 子 Agent 分组

def emit_rag_step(icon: str, label: str, detail: str = "") -> None:
    """
    向队列发送 RAG 检索步骤（跨线程安全）
    - 在检索过程中的各个节点调用
    - 支持子 Agent 分组标识
    """
    if _RAG_STEP_QUEUE is None or _RAG_STEP_LOOP is None:
        return
    
    step = {"icon": icon, "label": label, "detail": detail}
    
    # 添加子 Agent 分组标识
    group = get_sub_agent_group()
    if group:
        step["group"] = group
    
    # 跨线程调用：使用 call_soon_threadsafe
    try:
        if not _RAG_STEP_LOOP.is_closed():
            _RAG_STEP_LOOP.call_soon_threadsafe(
                _RAG_STEP_QUEUE.put_nowait, 
                step
            )
    except Exception:
        pass
```

---

## 💾 第六章：存储层实现

### 6.1 Milvus 集合设计

**位置**: `backend/indexing/milvus_client.py`

```python
def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int):
    """创建 Milvus 集合（稠密+稀疏向量双索引）"""
    
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    
    # 主键
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    
    # 向量字段
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    
    # 元数据字段
    schema.add_field("text", DataType.VARCHAR, max_length=2000)
    schema.add_field("filename", DataType.VARCHAR, max_length=255)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_level", DataType.INT64)
    
    # 索引配置
    index_params = client.prepare_index_params()
    
    # 稠密向量索引：HNSW
    index_params.add_index(
        field_name="dense_embedding",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 256}
    )
    
    # 稀疏向量索引：SPARSE_INVERTED_INDEX
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2}
    )
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
```

### 6.2 混合检索实现

```python
def hybrid_retrieve(self, dense_embedding: list[float], sparse_embedding: dict,
                   top_k: int = 5, rrf_k: int = 60, filter_expr: str = "") -> list[dict]:
    """
    混合检索：
    1. 构建稠密向量检索请求
    2. 构建稀疏向量检索请求
    3. 使用 RRFRanker 融合结果
    """
    
    # 稠密检索请求
    dense_search = AnnSearchRequest(
        data=[dense_embedding],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k * 2,  # 多召回一些供融合
        expr=filter_expr
    )
    
    # 稀疏检索请求
    sparse_search = AnnSearchRequest(
        data=[sparse_embedding],
        anns_field="sparse_embedding",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=top_k * 2,
        expr=filter_expr
    )
    
    # RRFRanker 融合
    reranker = RRFRanker(k=rrf_k)
    
    results = client.hybrid_search(
        collection_name=self.collection_name,
        reqs=[dense_search, sparse_search],
        ranker=reranker,
        limit=top_k,
        output_fields=output_fields
    )
    
    return _format_results(results)
```

### 6.3 会话存储

**位置**: `backend/chat/storage.py`

```python
class ConversationStorage:
    """PostgreSQL + Redis 双层存储"""
    
    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None):
        """
        保存会话：
        1. 更新 PostgreSQL（持久化）
        2. 更新 Redis（缓存）
        """
        
        db = SessionLocal()
        try:
            # 1. 获取或创建用户
            user = db.query(User).filter(User.username == user_id).first()
            
            # 2. 获取或创建会话
            session = db.query(ChatSession).filter(
                ChatSession.user_id == user.id,
                ChatSession.session_id == session_id
            ).first()
            
            if not session:
                session = ChatSession(
                    user_id=user.id, 
                    session_id=session_id,
                    metadata_json=metadata
                )
                db.add(session)
            
            # 3. 删除旧消息（全量替换）
            db.query(ChatMessage).filter(
                ChatMessage.session_ref_id == session.id
            ).delete()
            
            # 4. 插入新消息
            for msg in messages:
                db.add(ChatMessage(
                    session_ref_id=session.id,
                    message_type=msg.type,
                    content=str(msg.content),
                    rag_trace=get_rag_trace(msg)
                ))
            
            db.commit()
            
            # 5. 更新缓存
            cache.set_json(
                self._messages_cache_key(user_id, session_id),
                serialized_messages
            )
            
        finally:
            db.close()
```

---

## 🛠️ 第七章：调试与验证指南

### 7.1 调试检索流程

```python
# test_retrieval.py
from backend.rag.utils import retrieve_documents
from backend.indexing.embedding import embedding_service

# 测试向量生成
texts = ["什么是 RAG？", "企业知识管理"]
dense, sparse = embedding_service.get_all_embeddings(texts)
print(f"Dense dim: {len(dense[0])}")
print(f"Sparse keys: {len(sparse[0])}")

# 测试混合检索
result = retrieve_documents("公司组织结构", top_k=5)
print(f"\n检索模式: {result['meta']['retrieval_mode']}")
print(f"召回数量: {result['meta']['recall_count']}")
print(f"合并替换: {result['meta']['auto_merge_replaced_chunks']}")
print(f"精排后数量: {result['meta']['post_rerank_count']}")

for doc in result['docs']:
    print(f"\n--- {doc['filename']} (L{doc['chunk_level']}) ---")
    print(doc['text'][:100], "...")
```

### 7.2 调试 RAG Pipeline

```python
# test_pipeline.py
from backend.rag.pipeline import run_rag_graph, rag_graph

# 测试完整流程
result = run_rag_graph("请解释项目架构和技术栈")

print("=== 检索结果 ===")
print(f"文档数量: {len(result['docs'])}")
print(f"复杂度: {result.get('complexity', 'unknown')}")
print(f"扩展类型: {result.get('expansion_type', 'none')}")

if result.get('sub_questions'):
    print("\n子问题:")
    for i, sq in enumerate(result['sub_questions'], 1):
        print(f"  {i}. {sq}")

print("\n=== 上下文 ===")
print(result['context'][:500], "...")
```

### 7.3 调试会话管理

```python
# test_chat.py
from backend.chat.service import chat_with_agent, chat_with_agent_stream
import asyncio

# 测试同步对话
response = chat_with_agent(
    user_text="公司的请假政策是什么？",
    user_id="test_user",
    session_id="test_session_001"
)
print("响应:", response['response'][:200])

# 测试流式对话
async def test_stream():
    async for chunk in chat_with_agent_stream(
        user_text="项目的核心功能有哪些？",
        user_id="test_user",
        session_id="test_session_001"
    ):
        print(chunk)

asyncio.run(test_stream())
```

### 7.4 监控指标收集

```python
# 添加监控点
from time import time

def timed_retrieve(query: str, top_k: int = 5):
    start = time()
    
    try:
        result = retrieve_documents(query, top_k)
        latency = time() - start
        
        metrics = {
            "query": query[:30],
            "latency_ms": round(latency * 1000, 2),
            "retrieval_mode": result['meta']['retrieval_mode'],
            "recall_count": result['meta']['recall_count'],
            "final_count": len(result['docs']),
            "has_rerank": result['meta']['rerank_applied'],
            "has_merge": result['meta']['auto_merge_applied'],
        }
        
        print(json.dumps(metrics, ensure_ascii=False))
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        return None
```

---

## 📊 第八章：性能优化建议

### 8.1 缓存优化策略

| 缓存类型 | 存储位置 | 失效策略 | 用途 |
|----------|----------|----------|------|
| 会话消息 | Redis | 30天 | 加速会话加载 |
| 会话列表 | Redis | 更新时失效 | 用户会话列表 |
| BM25状态 | JSON文件 | 每次更新 | 稀疏向量统计 |
| 检索结果 | Redis | 5分钟 | 热点查询缓存 |
| 文档向量 | Redis | 文档更新时失效 | 频繁访问文档 |

### 8.2 异步优化

```python
# 异步向量化示例
async def async_embed(texts: list[str]):
    """异步批量向量化"""
    loop = asyncio.get_event_loop()
    
    # 并行处理稠密和稀疏向量
    dense_future = loop.run_in_executor(
        None, 
        lambda: embedding_service.get_embeddings(texts)
    )
    sparse_future = loop.run_in_executor(
        None, 
        lambda: embedding_service.get_sparse_embeddings(texts)
    )
    
    dense, sparse = await asyncio.gather(dense_future, sparse_future)
    return dense, sparse
```

### 8.3 批量处理优化

```python
# 批量插入优化
def batch_insert_chunks(chunks: list[dict], batch_size: int = 100):
    """批量插入 Milvus"""
    milvus_store = get_milvus_store()
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        milvus_store.insert(batch)
```

---

## 🎯 学习路线图（进阶版）

| 阶段 | 时长 | 目标 | 学习内容 |
|------|------|------|----------|
| **第一阶段** | 2周 | 环境与基础 | Docker、Milvus、PostgreSQL、Redis 部署 |
| **第二阶段** | 2周 | 向量化核心 | BGE-M3、BM25、稀疏向量、增量更新 |
| **第三阶段** | 2周 | 检索引擎 | Milvus Hybrid Search、RRFRanker、Auto-merging、Jina Rerank |
| **第四阶段** | 2周 | RAG Pipeline | LangGraph、查询重写、复杂度路由、子 Agent 并行 |
| **第五阶段** | 2周 | 对话管理 | 上下文窗口、持久化笔记、流式输出、实时可视化 |
| **第六阶段** | 2周 | 工程优化 | 缓存策略、异步处理、监控告警、性能调优 |

---

## 📝 代码阅读路径

```
1. main.py                    → 入口文件
   └── backend/app.py         → FastAPI 应用配置

2. backend/api/routes/chat.py → 对话接口
   └── backend/chat/service.py → 对话服务核心
       ├── backend/chat/runtime.py → Agent 运行时
       └── backend/chat/storage.py → 会话存储

3. backend/tools/knowledge.py → 知识库工具
   └── backend/rag/pipeline.py → RAG Pipeline
       └── backend/rag/utils.py → 检索工具函数

4. backend/indexing/embedding.py → 向量化服务
   └── backend/indexing/milvus_client.py → Milvus 客户端

5. backend/indexing/document_loader.py → 文档加载
   └── backend/indexing/milvus_writer.py → Milvus 写入

6. backend/chat/streaming.py → 流式输出
```

---

## 💡 实践练习

### 练习1：修改分块策略

```python
# 修改 backend/indexing/document_loader.py
# 将默认 chunk_size 从 500 改为 300
# 观察检索效果变化
```

### 练习2：添加新的查询重写策略

```python
# 在 backend/rag/utils.py 中添加
def generate_keyword_expansion(query: str) -> str:
    """基于关键词提取的查询扩展策略"""
    # 实现关键词提取逻辑
    keywords = extract_keywords(query)
    return query + "\n\n关键词：" + ", ".join(keywords)
```

### 练习3：添加监控指标

```python
# 在关键路径添加监控
# 记录：检索延迟、命中率、Token消耗
```

### 练习4：实现缓存预热

```python
# 在应用启动时预热热点数据
def warm_up_cache():
    # 加载高频访问文档
    # 预计算向量并缓存
    pass
```

---

## 🔧 常用命令

```bash
# 启动服务
python main.py

# 查看日志
tail -f logs/app.log

# 测试检索
python -c "from backend.rag.utils import retrieve_documents; print(retrieve_documents('test'))"

# 查看 Milvus 集合
python -c "from backend.indexing.milvus_client import get_milvus_store; s = get_milvus_store(); print(s.has_collection())"

# 重置索引
python -c "from backend.indexing.milvus_client import get_milvus_store; s = get_milvus_store(); s.drop_collection(); s.init_collection()"
```

---

> 🚀 **祝你学习愉快！** 如果在学习过程中遇到任何问题，随时可以查看代码注释或添加调试日志来理解流程。
