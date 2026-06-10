
# 📚 RAG 知识库平台学习指南

> 面向企业内部知识管理场景的 RAG 系统学习路径

---

## 🎯 项目概述

这是一个完整的**企业级 RAG（Retrieval-Augmented Generation）知识库平台**，实现了从文档入库到智能问答的完整链路：

```
文档上传 → 文档解析 → 三级分块 → 向量化 → Milvus存储 → 混合检索 → 查询重写 → 答案生成
```

### 核心技术栈

| 组件 | 技术 | 作用 |
|------|------|------|
| 框架 | LangChain + LangGraph | LLM 应用开发框架 |
| 向量数据库 | Milvus | 稠密/稀疏向量存储与检索 |
| 关系数据库 | PostgreSQL | 会话管理、父块存储 |
| 缓存 | Redis | 会话缓存、热点数据 |
| 嵌入模型 | BGE-M3 | 稠密向量生成 |
| 稀疏检索 | BM25 (手写) | 关键词匹配 |
| 精排 | Jina Rerank | 结果重排序 |

---

## 🚀 第一步：环境搭建

### 1.1 依赖安装

```bash
# 进入项目目录
cd d:\SuperMew-main

# 安装依赖（使用 uv 或 pip）
uv sync
# 或者
pip install -e .
```

### 1.2 环境变量配置

复制 `.env.example` 为 `.env`，并配置：

```env
# 基础配置
ARK_API_KEY=your-api-key
MODEL=gpt-4.1
BASE_URL=https://api.deepseek.com/v1

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379/0

# 嵌入模型
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

# Rerank 配置
RERANK_MODEL=jina-reranker-v2-base-en
RERANK_BINDING_HOST=http://localhost:8080
RERANK_API_KEY=your-rerank-key
```

### 1.3 启动服务

```bash
# 启动 Milvus（使用 Docker）
docker-compose up -d

# 启动应用
python main.py
```

---

## 🏗️ 第二步：理解项目结构

```
backend/
├── api/                # REST API 层
│   └── routes/         # 路由定义
│       ├── chat.py     # 对话接口
│       ├── documents.py # 文档管理接口
│       └── sessions.py  # 会话管理接口
├── chat/               # 对话业务层
│   ├── service.py      # 对话服务（核心）
│   ├── storage.py      # 会话存储
│   ├── streaming.py    # 流式输出
│   └── runtime.py      # Agent 运行时
├── rag/                # RAG 核心模块
│   ├── pipeline.py     # LangGraph 工作流
│   └── utils.py        # 检索工具函数
├── indexing/           # 索引模块
│   ├── embedding.py    # 向量化服务
│   ├── milvus_client.py # Milvus 客户端
│   └── parent_chunk_store.py # 父块存储
├── db/                 # 数据库模型
├── infra/              # 基础设施
└── tools/              # 工具定义
```

---

## 🧠 第三步：核心概念解析

### 3.1 混合检索（Hybrid Retrieval）

**目的**：解决单一向量检索的语义漂移问题

**实现位置**：`backend/rag/utils.py`

```python
# 稠密向量 + 稀疏向量双路召回
def retrieve_documents(query: str, top_k: int = 5):
    # 1. 生成稠密向量（BGE-M3）
    dense_embedding = _embedding_service.get_embeddings([query])[0]
    
    # 2. 生成稀疏向量（BM25）
    sparse_embedding = _embedding_service.get_sparse_embedding(query)
    
    # 3. Milvus RRFRanker 融合
    retrieved = _milvus_manager.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=candidate_k
    )
```

**三级降级策略**：
1. **Hybrid**：稠密 + 稀疏（最优）
2. **Dense-only**：仅稠密向量（降级）
3. **空结果**：返回空列表（兜底）

### 3.2 三级分块（Three-level Chunking）

**目的**：避免固定大小分块切断语义单元

```
L1 - 段落级别（大块，~2000 tokens）
  ↓
L2 - 小节级别（中块，~500 tokens）
  ↓
L3 - 句子级别（小块，~100 tokens）- 仅叶子块入 Milvus
```

**Auto-merging 机制**（`backend/rag/utils.py`）：
```python
def _auto_merge_candidates(docs):
    # L3 → L2 合并
    merged_docs, count_l3_l2 = _merge_to_parent_level(docs, threshold=2)
    # L2 → L1 合并
    merged_docs, count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=2)
```

**触发条件**：同一父块下召回子块数 ≥ threshold（默认2）

### 3.3 查询重写（Query Rewriting）

**目的**：解决用户问题表述模糊的问题

**两种策略**：

| 策略 | 适用场景 | 实现方式 |
|------|----------|----------|
| **Step-back** | 具体细节问题 | 将具体问题抽象为通用概念 |
| **HyDE** | 模糊概念问题 | 生成假设性文档辅助检索 |

**实现位置**：`backend/rag/utils.py`

```python
def step_back_expand(query: str) -> dict:
    # 1. 生成退步问题
    step_back_question = _generate_step_back_question(query)
    
    # 2. 回答退步问题
    step_back_answer = _answer_step_back_question(step_back_question)
    
    # 3. 构建扩展查询
    expanded_query = f"{query}\n\n退步问题：{step_back_question}\n退步答案：{step_back_answer}"
    return {"expanded_query": expanded_query}
```

### 3.4 复杂度路由（Complexity Routing）

**目的**：复杂问题拆解为子问题并行处理

**实现位置**：`backend/rag/pipeline.py`

```python
# 复杂度分类节点
def classify_complexity(state):
    # simple → 标准 RAG 流程
    # complex → 分解为子问题
    
# 子问题分解
def decompose_question(state):
    # 将复杂问题分解为 2-4 个子问题
    
# 并行分发（LangGraph Send API）
def _fanout_sub_questions(state):
    return [Send("rag_sub_agent", {"question": sq}) for sq in sub_qs]

# 结果合成
def synthesis(state):
    # 合并子 Agent 结果，去重排序
```

### 3.5 BM25 稀疏向量（手写实现）

**目的**：关键词精确匹配，弥补向量检索不足

**实现位置**：`backend/indexing/embedding.py`

```python
class EmbeddingService:
    def __init__(self):
        # BM25 参数
        self.k1 = 1.5  # 词频饱和系数
        self.b = 0.75  # 文档长度归一化系数
        
        # 持久化统计
        self._doc_freq: Counter = Counter()  # 文档频率
        self._total_docs = 0                # 总文档数
        self._sum_token_len = 0             # 总词数
        
    def get_sparse_embedding(self, text: str) -> dict:
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        sparse_vector = {}
        
        for token, freq in tf.items():
            # 计算 IDF
            df = self._doc_freq.get(token, 0)
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
            
            # 计算 BM25 得分
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score = idf * numerator / denominator
            
            sparse_vector[self._vocab[token]] = score
```

**增量更新机制**：
- `increment_add_documents()` - 新增文档时更新统计
- `increment_remove_documents()` - 删除文档时对称扣减

---

## 🔍 第四步：调试与验证

### 4.1 测试检索流程

```python
# 在项目根目录创建测试脚本
from backend.rag.utils import retrieve_documents

# 测试混合检索
result = retrieve_documents("什么是 RAG？", top_k=5)
print(f"检索模式: {result['meta']['retrieval_mode']}")
print(f"召回数量: {len(result['docs'])}")
for doc in result['docs']:
    print(f"- {doc['filename']}: {doc['text'][:50]}...")
```

### 4.2 测试 RAG Pipeline

```python
from backend.rag.pipeline import run_rag_graph

result = run_rag_graph("请解释一下项目架构")
print(f"检索到 {len(result['docs'])} 个文档")
print(f"上下文长度: {len(result['context'])}")
```

### 4.3 测试对话服务

```python
from backend.chat.service import chat_with_agent

response = chat_with_agent(
    user_text="公司的请假政策是什么？",
    user_id="test_user",
    session_id="test_session"
)
print(response['response'])
```

---

## 📈 第五步：性能优化方向

### 5.1 缓存策略

当前实现：
- 会话消息缓存（Redis）
- 会话列表缓存（Redis）

可优化点：
- 检索结果缓存
- 文档向量缓存
- 热点查询缓存

### 5.2 异步优化

当前实现：
- 流式响应（SSE）
- 并行子 Agent

可优化点：
- 批量文档处理
- 异步向量化
- 缓存预热

### 5.3 监控指标

建议添加：
- 检索延迟
- 命中率
- 文档覆盖率
- Token 消耗统计

---

## 🎯 学习路线图

| 阶段 | 目标 | 学习内容 |
|------|------|----------|
| **第一周** | 环境搭建 | Docker、Milvus、PostgreSQL |
| **第二周** | 核心概念 | 向量化、向量检索、BM25 |
| **第三周** | RAG Pipeline | LangChain、LangGraph |
| **第四周** | 高级特性 | 查询重写、复杂度路由 |
| **第五周** | 工程实践 | 流式输出、缓存、监控 |

---

## 💡 代码阅读顺序

1. **入口文件**：`main.py` → `backend/app.py`
2. **API 层**：`backend/api/routes/chat.py`
3. **业务层**：`backend/chat/service.py`
4. **核心 RAG**：`backend/rag/pipeline.py` → `backend/rag/utils.py`
5. **向量化**：`backend/indexing/embedding.py`
6. **存储层**：`backend/indexing/milvus_client.py`

---

## 📝 练习建议

1. **修改阈值**：调整 `AUTO_MERGE_THRESHOLD` 观察检索效果变化
2. **添加日志**：在关键节点添加日志，理解数据流向
3. **实现新策略**：尝试添加新的查询重写策略
4. **性能测试**：对比 Hybrid 和 Dense-only 检索的效果

---

> 🚀 **开始你的 RAG 学习之旅吧！** 如果遇到问题，可以查看代码中的注释，或者通过日志追踪执行流程。
