# RAG 系统学习文档

## 一、项目概述

本 RAG（Retrieval-Augmented Generation）系统是一个基于 LangGraph 的**智能检索流程编排框架**，具备以下核心能力：

### 1.1 核心特性

| 特性 | 描述 |
|------|------|
| **混合检索** | 稠密检索（BGE-M3）+ 稀疏检索（BM25）的混合策略 |
| **三级降级** | Hybrid → Dense-only → 空结果的优雅降级机制 |
| **Auto-merging** | 基于父块的自动合并，提升上下文完整性 |
| **Rerank精排** | 调用 Jina Rerank API 进行语义精排 |
| **查询扩展** | Step-back + HyDE 两种扩展策略 |
| **复杂度路由** | 自动识别复杂问题，分解为子问题并行检索 |
| **可观测性** | 完整的检索轨迹追踪（rag_trace） |

### 1.2 项目结构

```
backend/rag/
├── __init__.py          # 模块导出入口
├── pipeline.py          # 主流程编排（状态图节点）
└── utils.py             # 核心工具函数（检索、扩展、合并）
```

---

## 二、架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户问题输入                                 │
└─────────────────────────────────┬───────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    classify_complexity (复杂度分类)                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    ┌─────────────────┐                     ┌───────────────────┐
    │   simple        │                     │   complex         │
    │ (简单问题)      │                     │ (复杂问题)        │
    └────────┬────────┘                     └─────────┬─────────┘
             │                                        │
             ▼                                        ▼
    ┌─────────────────┐                     ┌───────────────────┐
    │ retrieve_initial│                     │decompose_question │
    │  (初始检索)     │                     │   (问题分解)      │
    └────────┬────────┘                     └─────────┬─────────┘
             │                                        │
    ┌────────┴────────┐                               ▼
    ▼                 ▼                  ┌───────────────────┐
┌─────────┐   ┌───────────────┐         │  fanout_sub_qs    │
│有结果   │   │ 无结果        │         │  (并行分发)       │
└────┬────┘   └───────┬───────┘         └─────────┬─────────┘
     │                │                           │
     ▼                ▼              ┌────────────┼────────────┐
┌───────────┐  ┌─────────────┐       ▼            ▼            ▼
│grade_doc  │  │rewrite_qs   │  ┌─────────┐ ┌─────────┐ ┌─────────┐
│(相关性评估)│  │(查询重写)    │  │sub_agent│ │sub_agent│ │sub_agent│
└─────┬─────┘  └──────┬──────┘  └────┬────┘ └────┬────┘ └────┬────┘
      │               │               │            │            │
┌─────┴─────┐         ▼               └────────────┴────────────┘
▼           │    ┌─────────────┐                     │
│generate   │    │retrieve_exp │                     ▼
│  answer   │    │(扩展检索)   │           ┌───────────────────┐
└───────────┘    └──────┬──────┘           │    synthesis      │
                       │                   │     (结果合成)     │
                       ▼                   └─────────┬─────────┘
              ┌─────────────────┐                     │
              │    END          │◄────────────────────┘
              └─────────────────┘
```

### 2.2 核心组件职责

| 组件 | 职责 | 文件位置 |
|------|------|----------|
| **状态管理** | RAGState 定义全局状态结构 | pipeline.py |
| **模型管理** | 懒加载的评分/路由/复杂度模型 | pipeline.py |
| **检索引擎** | 混合检索、Auto-merge、Rerank | utils.py |
| **查询扩展** | Step-back + HyDE 扩展策略 | utils.py |
| **流程编排** | LangGraph 状态图节点定义 | pipeline.py |
| **子Agent** | 子问题并行检索子图 | pipeline.py |

---

## 三、核心模块详解

### 3.1 状态定义（RAGState）

`RAGState` 是整个流程的**全局状态容器**，采用 `TypedDict` 实现类型安全：

```python
class RAGState(TypedDict):
    # 核心字段
    question: str              # 用户原始问题（始终不变）
    query: str                 # 当前用于检索的查询（可能被扩展）
    context: str               # 检索到的上下文（格式化后）
    docs: List[dict]           # 检索到的原始文档列表
    route: Optional[str]       # 当前路由方向
    rag_trace: Optional[dict]  # 检索轨迹（用于追踪和展示）
    
    # 查询扩展字段
    expansion_type: Optional[str]      # 扩展策略类型
    expanded_query: Optional[str]      # 扩展后的查询
    step_back_question: Optional[str]  # 退步问题
    step_back_answer: Optional[str]    # 退步问题答案
    hypothetical_doc: Optional[str]    # HyDE假设性文档
    
    # 复杂度路由字段
    complexity: Optional[str]          # simple/complex
    complexity_reason: Optional[str]   # 分类理由
    sub_questions: Optional[List[str]] # 子问题列表
    is_sub_agent: bool                 # 是否为子Agent调用
    sub_results: Annotated[List[dict], operator.add]  # 子结果（自动合并）
```

**关键设计点**：
- `Annotated[List[dict], operator.add]` 实现多子Agent结果的**自动合并**
- `question` 与 `query` 分离：原始问题不变，`query` 可被扩展
- `rag_trace` 贯穿全流程，记录每一步决策和中间结果

---

### 3.2 模型管理（懒加载单例）

系统使用**懒加载单例模式**管理三个核心模型：

| 模型 | 用途 | 配置环境变量 |
|------|------|--------------|
| **_grader_model** | 文档相关性评分 | `GRADE_MODEL`（默认gpt-4.1） |
| **_router_model** | 查询重写策略路由 | `MODEL` |
| **_complexity_model** | 问题复杂度分类 | `FAST_MODEL` |

**懒加载实现示例**（以评分模型为例）：

```python
def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,      # 确定性输出（是/否判断）
            stream_usage=True,
        )
    return _grader_model
```

**设计优势**：
- **延迟初始化**：首次调用时才创建，减少启动开销
- **单例复用**：避免重复创建模型实例
- **优雅降级**：环境变量未配置时返回 `None`，流程自动跳过该步骤

---

### 3.3 检索引擎（utils.py）

#### 3.3.1 核心检索函数 `retrieve_documents`

```python
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    核心检索函数：执行混合检索（稠密+BGE-M3 + 稀疏+BM25）
    
    支持三级降级回退：
    1. Hybrid（混合检索）
    2. Dense-only（仅稠密检索）
    3. 空结果兜底
    """
```

**三级降级流程**：

| 级别 | 策略 | 触发条件 |
|------|------|----------|
| 1 | Hybrid（稠密+稀疏） | 正常情况 |
| 2 | Dense-only | Hybrid失败时 |
| 3 | 空结果 | Dense也失败时 |

**检索流水线**：

```
输入查询 → 向量化（稠密+稀疏）→ 混合检索 → Auto-merge → Rerank → 阈值过滤 → 输出
```

#### 3.3.2 Auto-merging 自动合并

`_auto_merge_candidates` 实现**层级合并**：

```python
def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    """
    在完整召回候选上执行 L3→L2→L1 层级合并
    
    当某个父块的子块数量 >= threshold 时，用父块替换所有子块
    """
```

**合并规则**：
1. 按 `parent_chunk_id` 分组子块
2. 当同一父块的子块数量 >= `AUTO_MERGE_THRESHOLD`（默认2）时触发合并
3. 用父块内容替换所有子块，并合并分数
4. 标记 `merged_from_children=True` 和 `merged_child_count`

**设计意图**：
- 提升上下文完整性（子块可能截断关键信息）
- 减少冗余（避免返回多个高度重叠的子块）

#### 3.3.3 Rerank 精排

`_rerank_documents` 调用 Jina Rerank API：

```python
def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    """
    调用 Jina Rerank API 对文档进行精排
    """
```

**精排流程**：
1. 构建请求（模型名、查询、文档列表）
2. 调用 API 获取相关性分数
3. 按分数排序并截断到 top_k
4. **异常回退**：API失败时使用原始召回分数排序

**阈值过滤**：
```python
final_docs = [d for d in reranked_docs if _meets_rerank_min_score(d)]
```
低于 `RERANK_MIN_SCORE`（默认0.0）的文档被过滤。

---

### 3.4 查询扩展策略

系统支持三种查询扩展策略：

| 策略 | 适用场景 | 实现方式 |
|------|----------|----------|
| **step_back** | 包含具体细节的问题（名称、日期、代码） | 抽象为通用概念，生成退步问题和答案 |
| **hyde** | 模糊、概念性问题 | 生成假设性文档作为检索query |
| **complex** | 多步骤、需综合的复杂问题 | 同时使用 step_back + hyde |

#### 3.4.1 Step-back 扩展

```python
def step_back_expand(query: str) -> dict:
    """
    执行 Step-back 查询扩展
    
    将原始问题扩展为包含退步问题和答案的查询
    """
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    return {"step_back_question": ..., "step_back_answer": ..., "expanded_query": ...}
```

**工作原理**：
1. 将具体问题抽象为更通用的"退步问题"
2. 回答退步问题获取背景知识
3. 将原始问题 + 退步问题 + 答案拼接为扩展查询

**示例**：
- 原始问题："如何在 Python 中使用 asyncio 创建 TCP 服务器？"
- 退步问题："什么是异步编程？Python 中的 asyncio 是什么？"
- 退步答案："异步编程是一种并发编程范式..."

#### 3.4.2 HyDE 扩展

```python
def generate_hypothetical_document(query: str) -> str:
    """
    根据用户问题生成"假设性文档"
    
    用于帮助检索相关信息，文档可以包含合理推测
    """
```

**工作原理**：
- 让 LLM 基于问题生成一段"看起来像真实文档"的内容
- 用这段假设文档作为查询去检索知识库
- 适用于概念性、定义性问题

---

### 3.5 流程编排（pipeline.py）

#### 3.5.1 主图节点定义

| 节点 | 功能 | 输入依赖 | 输出贡献 |
|------|------|----------|----------|
| `classify_complexity` | 复杂度分类 | `question` | `complexity`, `complexity_reason` |
| `decompose_question` | 复杂问题分解 | `question` | `sub_questions` |
| `retrieve_initial` | 初始检索 | `question` | `docs`, `context`, `rag_trace` |
| `grade_documents` | 相关性评估 | `docs`, `context` | `route` |
| `rewrite_question` | 查询重写 | `question`, `docs` | `expanded_query`, `expansion_type` |
| `retrieve_expanded` | 扩展检索 | `expanded_query`, `expansion_type` | `docs`, `context` |
| `rag_sub_agent` | 子Agent检索 | `sub_question` | `sub_results` |
| `synthesis` | 结果合成 | `sub_results` | `docs`, `context` |

#### 3.5.2 条件路由

**路由函数示例**：

```python
def _route_after_initial(state: RAGState) -> Literal["grade_documents", "rewrite_question"]:
    """初始检索后的路由判断"""
    if not state.get("docs"):
        return "rewrite_question"  # 无结果→重写查询
    return "grade_documents"       # 有结果→评估相关性
```

**路由决策表**：

| 决策点 | 条件 | 目标节点 |
|--------|------|----------|
| 复杂度路由 | `complexity == "complex"` | `decompose_question` |
| 复杂度路由 | `complexity == "simple"` | `retrieve_initial` |
| 初始检索后 | `docs == []` | `rewrite_question` |
| 初始检索后 | `docs != []` | `grade_documents` |
| 评估后 | `score == "yes"` | `END`（生成答案） |
| 评估后 | `score == "no"` | `rewrite_question` |

#### 3.5.3 子Agent并行检索

```python
def _fanout_sub_questions(state: RAGState):
    """将分解后的子问题并行分发到rag_sub_agent子图"""
    sub_qs = state.get("sub_questions") or []
    return [
        Send("rag_sub_agent", {
            "question": sq,
            "is_sub_agent": True,  # 标记为子Agent调用
            ...
        })
        for sq in sub_qs
    ]
```

**并行执行机制**：
- 使用 LangGraph 的 `Send` API 实现并行分发
- 每个子问题独立执行完整的 RAG 流程
- 结果通过 `operator.add` 自动合并到 `sub_results`

#### 3.5.4 结果合成

```python
def synthesis(state: RAGState) -> RAGState:
    """合成节点：合并所有子Agent检索到的文档"""
    sub_results = state.get("sub_results", [])
    
    # 收集所有子问题的检索结果
    all_docs: List[dict] = []
    for result in sub_results:
        docs = result.get("docs", [])
        all_docs.extend(docs)
    
    # 去重和重新排名
    deduped = dedupe_documents(all_docs)
    context = _format_docs(deduped)
    
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}
```

---

## 四、完整工作流程

### 4.1 简单问题流程

```
用户问题 → classify_complexity → retrieve_initial → grade_documents → END
                                          ↓
                                   (无结果时)
                                          ↓
                                   rewrite_question → retrieve_expanded → END
```

### 4.2 复杂问题流程

```
用户问题 → classify_complexity → decompose_question → fanout → [rag_sub_agent × N] → synthesis → END
                                                                       ↓
                                                          每个子Agent执行完整RAG流程
```

### 4.3 执行时序图

```
用户
 │
 │ question="什么是LLM？它和传统ML有什么区别？"
 │
 ▼
┌───────────────────────────────────────────┐
│ classify_complexity                        │
│ complexity="complex"                       │
└───────────────────────────┬───────────────┘
                            │
                            ▼
┌───────────────────────────────────────────┐
│ decompose_question                         │
│ sub_questions=["什么是LLM？",              │
│                "传统ML是什么？",            │
│                "LLM与传统ML的区别？"]      │
└───────────────────────────┬───────────────┘
                            │
                            ▼ (并行)
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │sub_agent│        │sub_agent│        │sub_agent│
    │ q1检索  │        │ q2检索  │        │ q3检索  │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────┐
│ synthesis                                 │
│ 合并去重 → 格式化context                   │
└───────────────────────────┬───────────────┘
                            │
                            ▼
                         END
```

---

## 五、检索轨迹（rag_trace）

### 5.1 轨迹字段说明

`rag_trace` 是一个贯穿全流程的字典，记录检索过程的关键信息：

| 字段类别 | 字段名 | 说明 |
|----------|--------|------|
| 基础信息 | `tool_used`, `tool_name` | 是否使用工具、工具名称 |
| 查询信息 | `query`, `expanded_query` | 原始查询、扩展后查询 |
| 检索结果 | `retrieved_chunks`, `initial_retrieved_chunks`, `expanded_retrieved_chunks` | 各阶段检索结果 |
| 检索阶段 | `retrieval_stage` | initial/expanded/synthesis |
| 评估信息 | `grade_score`, `grade_route`, `rewrite_needed` | 相关性评分、路由方向 |
| 扩展信息 | `rewrite_strategy`, `step_back_question`, `step_back_answer`, `hypothetical_doc` | 扩展策略详情 |
| 复杂度信息 | `complexity`, `complexity_reason`, `sub_questions`, `sub_agent_count` | 复杂度分类和子问题信息 |
| 元数据 | `retrieval_mode`, `candidate_k`, `recall_count`, `rerank_applied`, `auto_merge_applied` 等 | 检索参数和状态 |

### 5.2 轨迹合并

多路召回时使用 `merge_retrieval_trace` 合并轨迹：

```python
def merge_retrieval_trace(accumulated: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多路检索轨迹
    
    规则：
    - 计数类字段累加（如 recall_count）
    - 配置类字段保留首次出现的值
    """
```

---

## 六、环境变量配置

### 6.1 必需配置

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `ARK_API_KEY` | API密钥 | sk-xxx |
| `MODEL` | 主模型名称 | gpt-4 |
| `BASE_URL` | API基础URL | https://api.example.com/v1 |

### 6.2 可选配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GRADE_MODEL` | gpt-4.1 | 相关性评分模型 |
| `FAST_MODEL` | 同MODEL | 复杂度分类模型（建议轻量化模型） |
| `RERANK_MODEL` | - | Jina精排模型名 |
| `RERANK_BINDING_HOST` | - | 精排服务地址 |
| `RERANK_API_KEY` | - | 精排API密钥 |
| `AUTO_MERGE_ENABLED` | true | 是否启用自动合并 |
| `AUTO_MERGE_THRESHOLD` | 2 | 合并触发阈值 |
| `LEAF_RETRIEVE_LEVEL` | 3 | 叶子块检索层级 |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | 3 | 候选池乘数（candidate_k = top_k × multiplier） |
| `RETRIEVAL_CANDIDATE_K` | - | 直接配置候选池大小（优先级高于multiplier） |
| `RERANK_MIN_SCORE` | 0.0 | 精排最低分数阈值 |

---

## 七、API 调用入口

### 7.1 主入口函数

```python
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
        ...  # 其他初始状态
    })
```

### 7.2 返回结果结构

```python
{
    "docs": [
        {
            "chunk_id": "xxx",
            "parent_chunk_id": "xxx",
            "filename": "document.pdf",
            "page_number": 10,
            "text": "文档内容...",
            "score": 0.85,
            "rerank_score": 0.92,
            "merged_from_children": True,
            "merged_child_count": 3
        }
    ],
    "rag_trace": {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": "用户问题",
        "expanded_query": "扩展后的查询",
        "retrieval_stage": "expanded",
        "retrieval_mode": "hybrid",
        "grade_score": "yes",
        "rewrite_strategy": "step_back",
        ...
    }
}
```

---

## 八、设计亮点与最佳实践

### 8.1 架构设计亮点

| 设计点 | 说明 |
|--------|------|
| **状态机模式** | 使用 LangGraph StateGraph 实现清晰的流程编排 |
| **懒加载单例** | 模型按需初始化，减少启动开销 |
| **优雅降级** | 三级检索降级、模型不可用时自动跳过 |
| **可观测性** | 完整的检索轨迹，支持调试和前端展示 |
| **并行执行** | 复杂问题分解为子问题并行检索，提升效率 |
| **自动合并** | Auto-merging 提升上下文质量 |

### 8.2 关键技术决策

1. **为什么使用 TypedDict 而非 Pydantic 作为状态？**
   - TypedDict 更轻量，适合频繁更新的状态
   - 支持 `Annotated[List[dict], operator.add]` 实现自动合并

2. **为什么需要多个模型？**
   - 评分模型需要高精度（gpt-4.1）
   - 复杂度分类可使用轻量模型（FAST_MODEL）
   - 职责分离，便于独立调优

3. **为什么使用三级分块（L1/L2/L3）？**
   - L3（叶子块）：细粒度检索，提高召回率
   - L2/L1（父块）：提供完整上下文
   - Auto-merging 根据子块数量动态选择

### 8.3 性能优化建议

1. **模型缓存**：已实现懒加载单例
2. **批量检索**：通过 `candidate_k` 控制候选池大小
3. **并行处理**：复杂问题并行检索
4. **阈值过滤**：提前过滤低质量文档，减少后续处理量

---

## 九、扩展与定制

### 9.1 添加新的查询扩展策略

1. 在 `utils.py` 中实现扩展函数
2. 在 `RewriteStrategy` 中添加新策略类型
3. 在 `rewrite_question_node` 中添加策略判断逻辑

### 9.2 自定义检索流程

1. 继承或修改 `RAGState` 添加新字段
2. 添加新的状态图节点
3. 修改路由逻辑连接新节点

### 9.3 替换检索后端

当前使用 Milvus + BM25，可替换为：
- Pinecone / Weaviate 等向量数据库
- 自定义检索后端（需实现 `hybrid_retrieve` 和 `dense_retrieve` 接口）

---

## 十、总结

本 RAG 系统是一个**生产级的智能检索框架**，具备以下核心能力：

1. **多策略检索**：混合检索 + 三级降级
2. **智能查询扩展**：Step-back + HyDE
3. **复杂问题处理**：自动分解 + 并行检索
4. **可观测性**：完整的检索轨迹追踪
5. **高可用性**：优雅降级机制

系统设计遵循**模块化、可扩展、可观测**的原则，适合作为企业级知识库问答系统的检索核心。
