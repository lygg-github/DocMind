# retrieve_documents 函数流程详解

## 一、函数签名与职责

```python
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | - | 用户查询字符串 |
| `top_k` | int | 5 | 最终返回文档数量 |
| **返回值** | `{"docs": List[dict], "meta": dict}` | - | 检索结果和元数据 |

**核心职责**：执行混合检索（稠密+BGE-M3 + 稀疏+BM25），支持三级降级回退。

---

## 二、完整流程表

### 2.1 流程概览

| 阶段 | 步骤 | 函数调用 | 关键操作 | 失败处理 |
|------|------|----------|----------|----------|
| **初始化** | 1. 解析候选池大小 | `resolve_candidate_k(top_k)` | 计算 candidate_k | - |
| **检索阶段** | 2. 混合检索 | `hybrid_retrieve()` | 稠密+稀疏向量检索 | 降级到步骤3 |
| **降级检索** | 3. 稠密检索 | `dense_retrieve()` | 仅稠密向量检索 | 降级到步骤4 |
| **兜底** | 4. 空结果返回 | - | 返回空文档列表 | - |
| **后处理** | 5. 最终处理 | `_finalize_retrieval()` | Auto-merge → Rerank → 阈值过滤 | - |

### 2.2 详细执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        retrieve_documents                               │
│                     (query, top_k=5)                                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: 解析候选池大小                                                 │
│   resolve_candidate_k(top_k)                                          │
│   → 返回 (candidate_k, candidate_config)                              │
│   → 构建过滤条件: chunk_level == LEAF_RETRIEVE_LEVEL                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 第一级 - Hybrid 混合检索                                       │
│   ├─ _embedding_service.get_embeddings([query]) → dense_embedding      │
│   ├─ _embedding_service.get_sparse_embedding(query) → sparse_embedding│
│   └─ _milvus_manager.hybrid_retrieve(...) → retrieved                  │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ 成功                          │ 失败 (Exception)
              ▼                               ▼
┌───────────────────────┐         ┌───────────────────────────────────────┐
│ Step 5: 最终处理      │         │ Step 3: 第二级 - Dense-only 降级       │
│   _finalize_retrieval │         │   ├─ _embedding_service.get_embeddings│
│   → Auto-merge        │         │   └─ _milvus_manager.dense_retrieve   │
│   → Rerank            │         └───────────────┬───────────────────────┘
│   → Threshold Filter  │                         │
│   → 返回结果          │           ┌─────────────┴─────────────┐
└───────────────────────┘           │ 成功                      │ 失败
                                    ▼                           ▼
                          ┌───────────────────────┐   ┌───────────────────┐
                          │ Step 5: 最终处理      │   │ Step 4: 空结果兜底 │
                          │   _finalize_retrieval │   │   返回 {"docs": [],│
                          └───────────┬───────────┘     │    "meta": {...}} │
                                      │                 └───────────────────┘
                                      ▼
                          ┌───────────────────────┐
                          │ 返回最终结果          │
                          │ {"docs": [...],       │
                          │  "meta": {...}}       │
                          └───────────────────────┘
```

---

## 三、内部调用函数详解

### 3.1 resolve_candidate_k

```python
def resolve_candidate_k(top_k: int) -> Tuple[int, Dict[str, Any]]:
```

**职责**：解析 Milvus 候选池大小

| 输入 | 输出 |
|------|------|
| `top_k`: int | `(candidate_k, config_info)` |

**优先级规则**：
1. **RETRIEVAL_CANDIDATE_K 环境变量**（最高优先级）
2. **top_k × RETRIEVAL_CANDIDATE_MULTIPLIER**（默认 multiplier=3）

**配置信息返回**：
| 字段 | 说明 |
|------|------|
| `candidate_k_source` | 配置来源："env" 或 "multiplier" |
| `retrieval_candidate_multiplier` | 乘数配置值 |
| `candidate_k_config_error` | 配置错误信息（如有） |

---

### 3.2 _finalize_retrieval

```python
def _finalize_retrieval(
    query: str,
    retrieved: List[dict],
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
```

**职责**：检索流水线最终处理

**执行步骤**：

| 步骤 | 函数调用 | 说明 |
|------|----------|------|
| 1 | `_auto_merge_candidates(retrieved)` | Auto-merge 层级合并 |
| 2 | `_rerank_documents(query, candidates, top_k)` | Jina Rerank 精排 |
| 3 | `_meets_rerank_min_score(d)` | 阈值过滤 |
| 4 | 构建完整元数据 | 返回结果 |

---

### 3.3 _auto_merge_candidates

```python
def _auto_merge_candidates(docs: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
```

**职责**：L3→L2→L1 层级自动合并

**合并规则**：
1. 按 `parent_chunk_id` 分组子块
2. 当同一父块的子块数量 >= `AUTO_MERGE_THRESHOLD`（默认2）时触发合并
3. 用父块内容替换所有子块
4. 合并分数（保留较高分）

**元数据输出**：
| 字段 | 说明 |
|------|------|
| `auto_merge_enabled` | 是否启用合并 |
| `auto_merge_applied` | 是否实际应用了合并 |
| `auto_merge_threshold` | 合并阈值 |
| `auto_merge_replaced_chunks` | 被替换的块数量 |
| `auto_merge_steps` | 合并步骤数（L3→L2 为1步，L2→L1 为1步） |

---

### 3.4 _rerank_documents

```python
def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
```

**职责**：调用 Jina Rerank API 进行语义精排

**执行流程**：
1. 为每个文档添加 `rrf_rank`（原始排名，用于失败回退）
2. 构建请求 payload（模型名、查询、文档列表）
3. 调用 Jina Rerank API
4. 解析结果，添加 `rerank_score`
5. **异常回退**：API失败时使用原始分数排序

**元数据输出**：
| 字段 | 说明 |
|------|------|
| `rerank_enabled` | 是否启用精排 |
| `rerank_applied` | 是否实际应用了精排 |
| `rerank_model` | 精排模型名称 |
| `rerank_endpoint` | 精排服务地址 |
| `rerank_error` | 错误信息（如有） |
| `candidate_count` | 精排前候选数量 |

---

### 3.5 _meets_rerank_min_score

```python
def _meets_rerank_min_score(doc: dict) -> bool:
```

**职责**：检查文档是否满足最低精排分数要求

**分数优先级**：`rerank_score` > `score`

**判断逻辑**：
- 如果 `rerank_score` 存在，使用 `rerank_score >= RERANK_MIN_SCORE`
- 如果只有 `score`，使用 `score >= RERANK_MIN_SCORE`
- 如果都不存在，当 `RERANK_MIN_SCORE <= 0` 时返回 True

---

## 四、调用链层级图

```
retrieve_documents (入口)
│
├── resolve_candidate_k           ← 解析候选池配置
│   └── _read_positive_int_env    ← 安全读取环境变量
│
├── _embedding_service.get_embeddings      ← 稠密向量化
├── _embedding_service.get_sparse_embedding ← 稀疏向量化
│
├── _milvus_manager.hybrid_retrieve        ← 混合检索（第一级）
│   └── _milvus_manager.dense_retrieve     ← 稠密检索（降级第二级）
│
└── _finalize_retrieval           ← 后处理流水线
    │
    ├── _auto_merge_candidates    ← Auto-merging
    │   ├── defaultdict           ← 分组聚合
    │   ├── _parent_chunk_store.get_documents_by_ids ← 获取父块
    │   └── _merge_to_parent_level ← 单层级合并
    │       └── _merge_rank_score_into ← 分数合并
    │
    ├── _rerank_documents         ← Jina精排
    │   └── requests.post         ← HTTP调用
    │
    └── _meets_rerank_min_score   ← 阈值过滤
        └── _effective_score      ← 获取有效分数
```

---

## 五、降级机制详解

### 5.1 三级降级流程

| 级别 | 策略 | 触发条件 | 检索模式标识 |
|------|------|----------|--------------|
| **Level 1** | Hybrid（稠密+稀疏） | 正常情况 | `"hybrid"` |
| **Level 2** | Dense-only（仅稠密） | Level 1 失败 | `"dense_fallback"` |
| **Level 3** | 空结果 | Level 2 失败 | `"failed"` |

### 5.2 降级触发条件

```python
try:
    # Level 1: Hybrid
    ...
except Exception:
    try:
        # Level 2: Dense-only
        ...
    except Exception:
        # Level 3: Empty result
        ...
```

**异常类型**：包括但不限于网络错误、Milvus连接错误、向量化服务错误等。

---

## 六、返回结果结构

### 6.1 成功返回

```python
{
    "docs": [
        {
            "chunk_id": "str",           # 块ID
            "parent_chunk_id": "str",    # 父块ID（如有）
            "filename": "str",           # 来源文件名
            "page_number": int,          # 页码
            "text": "str",               # 文档内容
            "score": float,              # 召回分数
            "rerank_score": float,       # 精排分数（如有）
            "merged_from_children": bool,  # 是否合并生成
            "merged_child_count": int,     # 合并的子块数量（如有）
            "rrf_rank": int              # 排名
        }
    ],
    "meta": {
        # 检索配置
        "retrieval_mode": "hybrid|dense_fallback|failed",
        "retrieval_pipeline": "recall_merge_rerank",
        "candidate_k": int,
        "retrieval_top_k": int,
        "leaf_retrieve_level": int,
        
        # 召回统计
        "recall_count": int,
        
        # Auto-merge 统计
        "auto_merge_enabled": bool,
        "auto_merge_applied": bool,
        "auto_merge_threshold": int,
        "auto_merge_replaced_chunks": int,
        "auto_merge_steps": int,
        "post_merge_candidate_count": int,
        
        # Rerank 统计
        "rerank_enabled": bool,
        "rerank_applied": bool,
        "rerank_model": str,
        "rerank_endpoint": str,
        "rerank_error": str,
        "candidate_count": int,
        "rerank_min_score": float,
        "post_rerank_count": int,
        "post_threshold_count": int,
        
        # 结果状态
        "retrieval_empty": bool
    }
}
```

### 6.2 空结果返回（Level 3 降级）

```python
{
    "docs": [],
    "meta": {
        "rerank_enabled": bool,
        "rerank_applied": False,
        "rerank_model": str,
        "rerank_endpoint": str,
        "rerank_error": "retrieve_failed",
        "retrieval_mode": "failed",
        "retrieval_pipeline": "recall_merge_rerank",
        "candidate_k": int,
        "retrieval_top_k": int,
        "leaf_retrieve_level": int,
        "recall_count": 0,
        # ... 其他默认值
        "retrieval_empty": True
    }
}
```

---

## 七、关键设计要点

### 7.1 候选池策略

**candidate_k 计算逻辑**：
```python
# 优先级1：直接配置
if RETRIEVAL_CANDIDATE_K:
    candidate_k = max(int(RETRIEVAL_CANDIDATE_K), top_k)
# 优先级2：乘数计算
else:
    candidate_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
```

**设计意图**：
- 候选池大于最终返回数，为后续精排提供更多候选
- 默认 multiplier=3，即候选池是 top_k 的3倍

### 7.2 向量化服务

```python
# 稠密向量化（用于 Dense Retrieval）
dense_embeddings = _embedding_service.get_embeddings([query])

# 稀疏向量化（用于 BM25 Sparse Retrieval）
sparse_embedding = _embedding_service.get_sparse_embedding(query)
```

**设计意图**：
- BGE-M3 模型同时输出稠密和稀疏向量
- 混合检索结合两者优势：稠密捕获语义，稀疏捕获关键词

### 7.3 过滤条件

```python
filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"  # 默认 L3
```

**设计意图**：
- 只检索叶子层（最细粒度）
- 保证召回率，Auto-merge 负责提升上下文完整性

---

## 八、性能优化点

| 优化点 | 实现方式 | 效果 |
|--------|----------|------|
| **批量处理** | `candidate_k` 控制候选池大小 | 减少后续处理量 |
| **层级合并** | Auto-merge 动态选择块粒度 | 平衡召回率和上下文质量 |
| **阈值过滤** | 提前过滤低分文档 | 减少不必要的计算 |
| **异常回退** | 三级降级机制 | 提升系统可用性 |
| **懒加载** | 向量化服务按需调用 | 减少启动开销 |

---

## 九、总结

`retrieve_documents` 是 RAG 系统的**核心检索引擎**，具备以下特点：

1. **混合检索**：结合稠密和稀疏向量的优势
2. **三级降级**：Hybrid → Dense-only → 空结果的优雅降级
3. **完整流水线**：召回 → Auto-merge → Rerank → 阈值过滤
4. **可观测性**：详细的元数据记录，支持调试和监控

该函数是整个检索流程的核心，通过精心设计的多层级处理和容错机制，确保在各种场景下都能返回高质量的检索结果。
