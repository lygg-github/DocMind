
# 🎯 混合检索与多级降级 - 专项学习指南

> 深入理解稠密+稀疏双塔检索架构的设计与实现

---

## 📚 学习目标

通过本模块学习，你将掌握：

1. **稠密向量检索** - BGE-M3 模型原理与使用
2. **稀疏向量检索** - BM25 算法原理与手写实现
3. **混合检索架构** - Milvus RRFRanker 融合策略
4. **精排优化** - Jina Rerank 重排序机制
5. **增量持久化** - BM25 统计的动态更新
6. **降级策略** - Hybrid → Dense-only → 空结果三级回退

---

## 🗂️ 核心文件清单

| 文件路径 | 核心职责 | 关键内容 |
|----------|----------|----------|
| `backend/indexing/embedding.py` | 向量化服务 | BGE-M3、BM25算法、增量更新 |
| `backend/indexing/milvus_client.py` | Milvus客户端 | 混合检索、RRFRanker、索引设计 |
| `backend/rag/utils.py` | 检索工具 | 检索流水线、Auto-merging、Rerank |
| `backend/rag/pipeline.py` | RAG流程 | 检索调用、降级逻辑 |

---

## 🚀 第一阶段：理解向量检索基础（1周）

### 1.1 稠密向量检索原理

**核心概念**：
- **词嵌入（Word Embedding）**：将文本转换为稠密向量
- **语义相似度**：通过向量距离衡量文本相关性
- **向量数据库**：高效存储和检索向量

**学习资源**：
- [BGE-M3 官方文档](https://huggingface.co/BAAI/bge-m3)
- [Sentence Transformers 教程](https://www.sbert.net/)

**代码实践**：

```python
# 测试稠密向量生成
from backend.indexing.embedding import embedding_service

# 生成向量
texts = ["什么是 RAG？", "企业知识管理系统", "向量数据库"]
vectors = embedding_service.get_embeddings(texts)

print(f"文本数量: {len(texts)}")
print(f"向量维度: {len(vectors[0])}")  # BGE-M3 默认 1024 维
print(f"向量示例: {vectors[0][:5]}...")
```

### 1.2 BM25 稀疏向量原理

**核心公式**：
```
BM25(Query, Document) = Σ IDF(q_i) * (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1 * (1 - b + b * |D|/avgdl))
```

**参数含义**：
- `k1`：词频饱和系数（默认1.5）
- `b`：文档长度归一化系数（默认0.75）
- `IDF`：逆文档频率
- `f(q_i,D)`：词项在文档中的频率

**代码实践**：

```python
# 测试 BM25 稀疏向量
from backend.indexing.embedding import embedding_service

text = "企业知识管理系统的设计与实现"
sparse_vec = embedding_service.get_sparse_embedding(text)

print(f"稀疏向量维度（非零元素）: {len(sparse_vec)}")
print(f"稀疏向量示例: {dict(list(sparse_vec.items())[:5])}")
```

### 1.3 稠密 vs 稀疏对比

| 维度 | 稠密向量（BGE-M3） | 稀疏向量（BM25） |
|------|-------------------|-----------------|
| **生成方式** | 深度学习模型 | 统计计算 |
| **向量维度** | 高维（1024） | 词表大小 |
| **语义理解** | 强 | 弱 |
| **关键词匹配** | 弱 | 强 |
| **计算成本** | 高 | 低 |
| **存储成本** | 高 | 低（稀疏） |

---

## 🧠 第二阶段：深入学习 BM25 实现（1周）

### 2.1 BM25 核心代码解析

**位置**: `backend/indexing/embedding.py`

```python
class EmbeddingService:
    def __init__(self):
        # BM25 参数配置
        self.k1 = 1.5  # 词频饱和系数
        self.b = 0.75  # 文档长度归一化系数
        
        # 核心统计数据（持久化）
        self._vocab: dict[str, int] = {}        # 词→索引映射
        self._vocab_counter = 0                 # 词表计数器
        self._doc_freq: Counter = Counter()     # 文档频率
        self._total_docs = 0                    # 总文档数
        self._sum_token_len = 0                 # 总词数
        self._avg_doc_len = 1.0                 # 平均文档长度
```

### 2.2 分词器实现

```python
def tokenize(self, text: str) -> list[str]:
    """
    中英文混合分词：
    - 中文：按字分词
    - 英文：按单词分词
    """
    text = text.lower()
    tokens = []
    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    english_pattern = re.compile(r"[a-zA-Z]+")
    
    i = 0
    while i < len(text):
        char = text[i]
        if chinese_pattern.match(char):
            tokens.append(char)
            i += 1
        elif english_pattern.match(char):
            match = english_pattern.match(text[i:])
            if match:
                tokens.append(match.group())
                i += len(match.group())
        else:
            i += 1
    return tokens
```

### 2.3 稀疏向量生成

```python
def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
    """
    生成 BM25 稀疏向量（不加锁版本）
    
    返回值：
    - sparse_vector: {词索引: BM25得分}
    - vocab_changed: 是否新增了新词
    """
    tokens = self.tokenize(text)
    doc_len = len(tokens)
    tf = Counter(tokens)  # 词频统计
    sparse_vector: dict[int, float] = {}
    vocab_changed = False
    
    n = max(self._total_docs, 1)
    avg = max(self._avg_doc_len, 1.0)
    
    for token, freq in tf.items():
        # 词表动态扩展
        if token not in self._vocab:
            self._vocab[token] = self._vocab_counter
            self._vocab_counter += 1
            vocab_changed = True
        
        idx = self._vocab[token]
        df = self._doc_freq.get(token, 0)
        
        # IDF 计算（平滑处理）
        if df == 0:
            idf = math.log((n + 1) / 1)
        else:
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        
        # BM25 得分计算
        numerator = freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
        score = idf * numerator / denominator
        
        if score > 0:
            sparse_vector[idx] = float(score)
    
    return sparse_vector, vocab_changed
```

### 2.4 练习：理解 BM25 参数影响

```python
# 修改参数观察效果
original_k1 = embedding_service.k1
original_b = embedding_service.b

# 测试不同参数组合
test_text = "企业知识管理"
for k1 in [0.5, 1.0, 1.5, 2.0]:
    for b in [0.5, 0.75, 1.0]:
        embedding_service.k1 = k1
        embedding_service.b = b
        vec = embedding_service.get_sparse_embedding(test_text)
        print(f"k1={k1}, b={b}: 非零元素数={len(vec)}, 最大得分={max(vec.values()):.4f}")

# 恢复原值
embedding_service.k1 = original_k1
embedding_service.b = original_b
```

---

## 🔄 第三阶段：增量持久化机制（1周）

### 3.1 入库时增量更新

```python
def increment_add_documents(self, texts: list[str]) -> None:
    """
    新增文档时更新 BM25 统计：
    1. 增加总文档数
    2. 更新词频统计
    3. 扩展词表
    4. 更新平均文档长度
    5. 持久化到文件
    """
    if not texts:
        return
    
    with self._lock:
        for text in texts:
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            
            # 更新统计
            self._sum_token_len += doc_len
            self._total_docs += 1
            
            # 更新文档频率（去重后）
            for token in set(tokens):
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1
                self._doc_freq[token] += 1
        
        # 重新计算平均文档长度
        self._recompute_avg_len()
        
        # 持久化到文件
        self._persist_unlocked()
```

### 3.2 删除时对称扣减

```python
def increment_remove_documents(self, texts: list[str]) -> None:
    """
    删除文档时对称扣减 BM25 统计：
    - 词表索引不回收（避免与 Milvus 中旧向量冲突）
    - 仅扣减文档频率和总文档数
    """
    if not texts:
        return
    
    with self._lock:
        for text in texts:
            tokens = self.tokenize(text)
            doc_len = len(tokens)
            
            # 扣减统计（防止负数）
            self._sum_token_len = max(0, self._sum_token_len - doc_len)
            self._total_docs = max(0, self._total_docs - 1)
            
            # 扣减文档频率
            for token in set(tokens):
                if token not in self._doc_freq:
                    continue
                self._doc_freq[token] -= 1
                if self._doc_freq[token] <= 0:
                    del self._doc_freq[token]
        
        self._recompute_avg_len()
        self._persist_unlocked()
```

### 3.3 持久化机制

```python
def _persist_unlocked(self) -> None:
    """序列化并持久化 BM25 状态"""
    # 确保目录存在
    self._state_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "version": 1,
        "total_docs": self._total_docs,
        "sum_token_len": self._sum_token_len,
        "vocab": self._vocab,
        "doc_freq": dict(self._doc_freq),
    }
    
    # 原子写入（先写临时文件，再替换）
    tmp = self._state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(self._state_path)
```

### 3.4 练习：测试增量更新

```python
# 测试增量更新机制
from backend.indexing.embedding import EmbeddingService
import tempfile
import os

# 创建临时状态文件
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    state_path = f.name

# 创建独立实例
service = EmbeddingService(state_path=state_path)

# 初始状态
print(f"初始状态 - 文档数: {service._total_docs}, 词表大小: {len(service._vocab)}")

# 添加文档
docs = [
    "企业知识管理系统",
    "RAG 检索增强生成",
    "向量数据库 Milvus"
]
service.increment_add_documents(docs)
print(f"添加后 - 文档数: {service._total_docs}, 词表大小: {len(service._vocab)}")

# 删除文档
service.increment_remove_documents(docs[:1])
print(f"删除后 - 文档数: {service._total_docs}, 词表大小: {len(service._vocab)}")

# 验证持久化
service2 = EmbeddingService(state_path=state_path)
print(f"重新加载后 - 文档数: {service2._total_docs}, 词表大小: {len(service2._vocab)}")

# 清理
os.unlink(state_path)
```

---

## 🔍 第四阶段：Milvus 混合检索（1周）

### 4.1 Milvus 集合设计

**位置**: `backend/indexing/milvus_client.py`

```python
def ensure_collection(client: MilvusClient, collection_name: str, dense_dim: int):
    """创建支持混合检索的集合"""
    
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    
    # 主键
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    
    # 稠密向量字段
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
    
    # 稀疏向量字段
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    
    # 元数据字段
    schema.add_field("text", DataType.VARCHAR, max_length=2000)
    schema.add_field("filename", DataType.VARCHAR, max_length=255)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_level", DataType.INT64)
    
    # 索引配置
    index_params = client.prepare_index_params()
    
    # HNSW 索引（稠密向量）
    index_params.add_index(
        field_name="dense_embedding",
        index_type="HNSW",
        metric_type="IP",  # 内积相似度
        params={"M": 16, "efConstruction": 256}
    )
    
    # 稀疏倒排索引
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

### 4.2 混合检索实现

```python
def hybrid_retrieve(self, dense_embedding: list[float], sparse_embedding: dict,
                   top_k: int = 5, rrf_k: int = 60, filter_expr: str = "") -> list[dict]:
    """
    混合检索流程：
    1. 构建稠密向量检索请求
    2. 构建稀疏向量检索请求
    3. 使用 RRFRanker 融合两路结果
    """
    
    output_fields = [
        "text", "filename", "chunk_id", "parent_chunk_id", 
        "chunk_level", "page_number"
    ]
    
    # 稠密检索请求（多召回 2 倍供融合）
    dense_search = AnnSearchRequest(
        data=[dense_embedding],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k * 2,
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
    # RRF (Reciprocal Rank Fusion) 公式: score = 1/(k + rank)
    reranker = RRFRanker(k=rrf_k)
    
    # 执行混合检索
    results = self._run(lambda client: client.hybrid_search(
        collection_name=self.collection_name,
        reqs=[dense_search, sparse_search],
        ranker=reranker,
        limit=top_k,
        output_fields=output_fields
    ))
    
    return _format_results(results)
```

### 4.3 RRFRanker 原理

**RRF（Reciprocal Rank Fusion）** 公式：
```
score(q, d) = Σ 1 / (k + rank_i(d))
```

其中：
- `k`：融合参数（默认60）
- `rank_i(d)`：文档 d 在第 i 个检索结果中的排名

**优势**：
- 无需归一化不同检索器的分数
- 对异常值鲁棒
- 简单高效

### 4.4 练习：测试混合检索

```python
# 测试混合检索
from backend.rag.utils import retrieve_documents

# 测试不同模式
queries = [
    "什么是 RAG？",           # 概念性问题（稠密更优）
    "2024年Q3财报",         # 关键词问题（稀疏更优）
    "产品定价策略文档"       # 混合场景
]

for query in queries:
    print(f"\n=== 查询: {query} ===")
    result = retrieve_documents(query, top_k=3)
    meta = result["meta"]
    
    print(f"检索模式: {meta['retrieval_mode']}")
    print(f"召回数量: {meta['recall_count']}")
    print(f"最终数量: {len(result['docs'])}")
    
    for doc in result["docs"]:
        print(f"- {doc['filename']}: {doc['text'][:30]}...")
```

---

## 🎯 第五阶段：检索后处理流水线（1周）

### 5.1 完整流水线架构

```
召回候选 → Auto-merging → Rerank → 阈值过滤 → 最终结果
```

**位置**: `backend/rag/utils.py`

```python
def _finalize_retrieval(query: str, retrieved: List[dict], top_k: int,
                        retrieval_mode: str, candidate_k: int,
                        candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    检索后处理流水线：
    1. Auto-merging：L3→L2→L1 自动合并
    2. Jina Rerank：精排重排序
    3. 阈值过滤：基于 rerank 分数过滤
    """
    
    # 阶段1：Auto-merging
    candidates, merge_meta = _auto_merge_candidates(retrieved)
    
    # 阶段2：Jina Rerank 精排
    reranked_docs, rerank_meta = _rerank_documents(query=query, docs=candidates, top_k=top_k)
    
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

### 5.2 Jina Rerank 精排

```python
def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    """使用 Jina Rerank 进行精排"""
    
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "candidate_count": len(docs),
    }
    
    if not docs or not meta["rerank_enabled"]:
        # 降级：直接按召回分数排序
        return _sort_by_rank_score(docs)[:top_k], meta
    
    # 构建请求
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs],
        "top_n": min(top_k, len(docs)),
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
            timeout=15
        )
        
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}"
            return _sort_by_rank_score(docs)[:top_k], meta
        
        # 解析结果
        items = response.json().get("results", [])
        reranked = []
        
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs):
                doc = dict(docs[idx])
                doc["rerank_score"] = item.get("relevance_score")
                reranked.append(doc)
        
        return reranked, meta
    
    except Exception as e:
        meta["rerank_error"] = str(e)
        return _sort_by_rank_score(docs)[:top_k], meta
```

### 5.3 练习：观察精排效果

```python
# 测试精排效果
from backend.rag.utils import retrieve_documents, _rerank_documents

# 获取召回结果
result = retrieve_documents("企业知识管理", top_k=10)
docs = result["docs"]

print("=== 精排前 ===")
for i, doc in enumerate(docs[:5], 1):
    print(f"{i}. {doc['filename']} (score: {doc.get('score', 0):.4f})")

# 手动调用精排
reranked, meta = _rerank_documents("企业知识管理", docs, top_k=5)

print("\n=== 精排后 ===")
for i, doc in enumerate(reranked, 1):
    print(f"{i}. {doc['filename']} (rerank_score: {doc.get('rerank_score', 0):.4f})")
```

---

## 🛡️ 第六阶段：多级降级策略（1周）

### 6.1 降级逻辑实现

**位置**: `backend/rag/utils.py`

```python
def retrieve_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    三级降级检索：
    Level 1: Hybrid (稠密 + 稀疏)
    Level 2: Dense-only (仅稠密)
    Level 3: 空结果兜底
    """
    
    candidate_k, candidate_config = resolve_candidate_k(top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    
    # ========== Level 1: Hybrid 检索 ==========
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)
        
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
    
    except Exception as hybrid_error:
        print(f"Hybrid retrieval failed: {hybrid_error}")
    
    # ========== Level 2: Dense-only 降级 ==========
    try:
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
    
    except Exception as dense_error:
        print(f"Dense retrieval failed: {dense_error}")
    
    # ========== Level 3: 空结果兜底 ==========
    return {
        "docs": [],
        "meta": {
            "retrieval_mode": "failed",
            "retrieval_empty": True,
            # ... 其他元数据
        },
    }
```

### 6.2 降级触发场景

| 场景 | 降级路径 | 原因 |
|------|----------|------|
| BM25 词表为空 | Hybrid → Dense-only | 稀疏向量生成失败 |
| Milvus 稀疏索引未就绪 | Hybrid → Dense-only | 稀疏检索失败 |
| 网络故障 | Hybrid → Dense-only → 空结果 | 无法连接 Milvus |
| 模型加载失败 | Hybrid → Dense-only → 空结果 | 无法生成向量 |

### 6.3 练习：模拟降级场景

```python
# 模拟降级场景
import unittest
from unittest.mock import patch, MagicMock

class TestRetrievalDegradation(unittest.TestCase):
    
    def test_hybrid_to_dense_fallback(self):
        """测试从 Hybrid 降级到 Dense-only"""
        with patch.object(_milvus_manager, 'hybrid_retrieve') as mock_hybrid:
            # 模拟 Hybrid 失败
            mock_hybrid.side_effect = Exception("Milvus sparse index error")
            
            with patch.object(_milvus_manager, 'dense_retrieve') as mock_dense:
                # 模拟 Dense-only 成功
                mock_dense.return_value = [{"text": "fallback result"}]
                
                result = retrieve_documents("test query")
                self.assertEqual(result["meta"]["retrieval_mode"], "dense_fallback")
    
    def test_full_degradation_to_empty(self):
        """测试完全降级到空结果"""
        with patch.object(_milvus_manager, 'hybrid_retrieve') as mock_hybrid:
            mock_hybrid.side_effect = Exception("Network error")
            
            with patch.object(_milvus_manager, 'dense_retrieve') as mock_dense:
                mock_dense.side_effect = Exception("Network error")
                
                result = retrieve_documents("test query")
                self.assertEqual(result["meta"]["retrieval_mode"], "failed")
                self.assertTrue(result["meta"]["retrieval_empty"])

if __name__ == "__main__":
    unittest.main()
```

---

## 📊 第七阶段：性能调优与监控（1周）

### 7.1 性能指标

| 指标 | 定义 | 优化目标 |
|------|------|----------|
| **召回延迟** | 从查询到返回结果的时间 | < 500ms |
| **召回率** | 相关文档被召回的比例 | > 80% |
| **精确率** | 召回文档中相关的比例 | > 70% |
| **精排提升** | 精排后排名提升幅度 | > 20% |

### 7.2 优化策略

```python
# 优化配置示例
class RetrievalConfig:
    # 候选池大小（影响召回率和延迟）
    CANDIDATE_MULTIPLIER = 3  # top_k × multiplier
    
    # HNSW 索引参数
    HNSW_M = 16              # 图的度数
    HNSW_EF_CONSTRUCTION = 256  # 建图时的搜索范围
    HNSW_EF_SEARCH = 64     # 搜索时的范围
    
    # RRFRanker 参数
    RRF_K = 60
    
    # Rerank 阈值
    RERANK_MIN_SCORE = 0.0
    
    # Auto-merging 阈值
    AUTO_MERGE_THRESHOLD = 2
```

### 7.3 监控日志

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timed_retrieve(query: str, top_k: int = 5):
    """带监控的检索函数"""
    start_time = time.time()
    
    try:
        result = retrieve_documents(query, top_k)
        latency = time.time() - start_time
        
        # 记录关键指标
        logger.info(
            "Retrieval completed | "
            f"query={query[:30]} | "
            f"mode={result['meta']['retrieval_mode']} | "
            f"latency={latency:.2f}s | "
            f"recall={result['meta']['recall_count']} | "
            f"final={len(result['docs'])} | "
            f"merged={result['meta'].get('auto_merge_replaced_chunks', 0)} | "
            f"rerank={result['meta'].get('rerank_applied', False)}"
        )
        
        return result
    
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Retrieval failed | query={query[:30]} | latency={latency:.2f}s | error={e}")
        raise
```

---

## 🗓️ 学习进度表

| 阶段 | 时长 | 内容 | 代码实践 |
|------|------|------|----------|
| 1 | 1周 | 稠密/稀疏向量原理 | 生成向量、对比效果 |
| 2 | 1周 | BM25 算法实现 | 修改参数、观察变化 |
| 3 | 1周 | 增量持久化机制 | 添加/删除文档、验证持久化 |
| 4 | 1周 | Milvus 混合检索 | 测试不同查询类型 |
| 5 | 1周 | 检索后处理流水线 | 观察精排效果 |
| 6 | 1周 | 多级降级策略 | 模拟降级场景 |
| 7 | 1周 | 性能调优与监控 | 添加监控日志 |

---

## 📝 核心问题自测

1. **BM25 的 `k1` 和 `b` 参数分别控制什么？**
   - `k1`：词频饱和系数，值越大越重视词频
   - `b`：文档长度归一化系数，值越大越受文档长度影响

2. **为什么删除文档时不回收词表索引？**
   - 避免与 Milvus 中已存在的稀疏向量维度冲突

3. **RRFRanker 的 `k` 参数作用是什么？**
   - 控制排名权重的衰减速度，默认60

4. **三级降级的触发条件是什么？**
   - Hybrid 失败 → Dense-only → 空结果

5. **Auto-merging 的触发条件是什么？**
   - 同一父块下召回子块数 ≥ threshold（默认2）

---

## 🎯 进阶练习

1. **实现 TF-IDF 稀疏向量**：替换 BM25，对比效果
2. **调整 RRFRanker 参数**：观察不同 `k` 值对结果的影响
3. **添加缓存层**：对高频查询结果进行缓存
4. **实现批量检索**：优化多查询场景的性能

---

> 🚀 **祝你学习顺利！** 如果在学习过程中遇到问题，可以查看代码注释或添加调试日志来理解流程。
