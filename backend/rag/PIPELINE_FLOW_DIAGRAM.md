# pipeline.py 流程图详解

## 一、整体架构概览

### 1.1 状态图架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG状态图架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         主图 (rag_graph)                            │   │
│  │                                                                     │   │
│  │  classify_complexity ──────────────────────────────────────────┐   │   │
│  │         │                                                     │   │   │
│  │    ┌────┴────┐                                                │   │   │
│  │    ▼         ▼                                                │   │   │
│  │ simple    complex                                             │   │   │
│  │    │         │                                                │   │   │
│  │    │    decompose_question                                    │   │   │
│  │    │         │                                                │   │   │
│  │    │    fanout_sub_questions                                  │   │   │
│  │    │         │                                                │   │   │
│  │    │    ┌────┼────┬────┐                                      │   │   │
│  │    │    ▼    ▼    ▼    ▼                                      │   │   │
│  │    │  ┌─────────────────────────────────────────────────┐    │   │   │
│  │    │  │           子图 (rag_sub_agent)                   │    │   │   │
│  │    │  │                                                 │    │   │   │
│  │    │  │  retrieve_initial ──┬──→ grade_documents         │    │   │   │
│  │    │  │         │          │          │                  │    │   │   │
│  │    │  │         │     (有结果)        │                  │    │   │   │
│  │    │  │         │          ↓          ↓                  │    │   │   │
│  │    │  │         │     rewrite_question ←── (评估不通过)   │    │   │   │
│  │    │  │         │          │                             │    │   │   │
│  │    │  │    (无结果)        ↓                             │    │   │   │
│  │    │  │         └────→ retrieve_expanded                  │    │   │   │
│  │    │  │                                                 │    │   │   │
│  │    └──┼──→ retrieve_initial ───→ grade_documents ───→ END│    │   │   │
│  │         │          │          │                          │    │   │   │
│  │         │    (有结果)   (评估通过)                        │    │   │   │
│  │         │          ↓          ↓                          │    │   │   │
│  │         │    grade_documents  END                         │    │   │   │
│  │         │          │                                     │    │   │   │
│  │         │    (评估不通过)                                 │    │   │   │
│  │         │          ↓                                     │    │   │   │
│  │         └───→ rewrite_question → retrieve_expanded → END │    │   │   │
│  │                                                          │    │   │   │
│  │    synthesis ←────────────────────────────────────────────┘    │   │   │
│  │         │                                                     │   │   │
│  │         ▼                                                     │   │   │
│  │       END                                                      │   │   │
│  └─────────────────────────────────────────────────────────────────┘   │   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、节点功能详解

### 2.1 节点功能总览

| 节点名称 | 功能描述 | 输入依赖 | 输出贡献 | 所属图 |
|----------|----------|----------|----------|--------|
| `classify_complexity` | 判断问题复杂度 (simple/complex) | `question` | `complexity`, `complexity_reason` | 主图 |
| `decompose_question` | 将复杂问题分解为2-4个子问题 | `question` | `sub_questions` | 主图 |
| `retrieve_initial` | 初始混合检索 | `question` | `docs`, `context`, `rag_trace` | 主图/子图 |
| `grade_documents` | 文档相关性评估 | `docs`, `context` | `route` | 主图/子图 |
| `rewrite_question` | 查询扩展策略选择与执行 | `question`, `docs` | `expanded_query`, `expansion_type` | 主图/子图 |
| `retrieve_expanded` | 使用扩展查询重新检索 | `expanded_query`, `expansion_type` | `docs`, `context` | 主图/子图 |
| `rag_sub_agent` | 子问题并行检索子图 | `sub_question` | `sub_results` | 主图 |
| `synthesis` | 合并子Agent结果 | `sub_results` | `docs`, `context`, `rag_trace` | 主图 |

---

## 三、完整流程时序图

### 3.1 简单问题流程

```
用户问题
    │
    ▼
┌─────────────────────┐
│ classify_complexity │  ← 判断为 simple
└──────────┬──────────┘
           │ complexity="simple"
           ▼
┌─────────────────────┐
│  retrieve_initial   │  ← 混合检索（稠密+稀疏）
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
   有结果      无结果
     │           │
     ▼           ▼
┌───────────┐  ┌───────────────────┐
│grade_docs │  │ rewrite_question  │  ← 强制step_back
└─────┬─────┘  └─────────┬─────────┘
      │                  │
  ┌───┴───┐              ▼
  │       │    ┌───────────────────┐
 yes     no   │ retrieve_expanded  │
  │       │   └─────────┬─────────┘
  ▼       │             │
END       │             ▼
          │           END
          ▼
┌───────────────────┐
│rewrite_question   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ retrieve_expanded │
└─────────┬─────────┘
          │
          ▼
        END
```

### 3.2 复杂问题流程

```
用户问题
    │
    ▼
┌─────────────────────┐
│ classify_complexity │  ← 判断为 complex
└──────────┬──────────┘
           │ complexity="complex"
           ▼
┌─────────────────────┐
│ decompose_question  │  ← 分解为2-4个子问题
└──────────┬──────────┘
           │ sub_questions=[q1, q2, q3, ...]
           ▼
┌─────────────────────┐
│ _fanout_sub_questions│ ← 并行分发
└──────────┬──────────┘
           │
    ┌──────┼──────┬──────┐
    ▼      ▼      ▼      ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│sub_ag1│ │sub_ag2│ │sub_ag3│ │sub_ag4│  ← 并行执行子图
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │
    └─────────┴─────────┴─────────┘
                 │
                 ▼
┌─────────────────────┐
│      synthesis      │  ← 合并去重
└──────────┬──────────┘
           │
           ▼
         END
```

---

## 四、条件路由详解

### 4.1 路由决策表

| 路由函数 | 触发节点 | 决策逻辑 | 输出分支 |
|----------|----------|----------|----------|
| `_route_after_complexity` | `classify_complexity` | `complexity == "complex"` | `decompose_question` / `retrieve_initial` |
| `_route_after_initial` | `retrieve_initial` | `docs == []` | `rewrite_question` / `grade_documents` |
| `lambda state: state.get("route")` | `grade_documents` | `route == "generate_answer"` | `END` / `rewrite_question` |
| `_fanout_sub_questions` | `decompose_question` | 子问题列表 | `Send("rag_sub_agent", {...}) × N` |

### 4.2 路由流程图

```
                    ┌─────────────────────────────┐
                    │    classify_complexity     │
                    └───────────┬─────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
      complexity=="complex"            complexity=="simple"
              │                                   │
              ▼                                   ▼
    ┌─────────────────┐                  ┌──────────────────┐
    │decompose_question│                  │  retrieve_initial│
    └────────┬────────┘                  └─────────┬────────┘
             │                                    │
             ▼                              ┌─────┴─────┐
    ┌─────────────────┐                    ▼           ▼
    │fanout_sub_qs    │              docs==[]      docs!=[]
    └────────┬────────┘                    │           │
             │                             ▼           ▼
    ┌────────┴────────┐            ┌────────────┐ ┌────────────┐
    │ Send×N to sub   │            │rewrite_q   │ │grade_docs  │
    │     agent       │            └─────┬──────┘ └─────┬──────┘
    └────────┬────────┘                  │              │
             │                           │         grade=="yes"
             ▼                           │         │
    ┌─────────────────┐                  │    ┌────┴────┐
    │   rag_sub_agent │×N                ▼    ▼         ▼
    └────────┬────────┘            retrieve_exp    END   rewrite_q
             │                            │               │
             └────────────┬───────────────┴               │
                          │                              │
                          ▼                              │
                   ┌──────────────┐                      │
                   │   synthesis  │                      │
                   └──────┬───────┘                      │
                          │                              │
                          ▼                              │
                        END ◄────────────────────────────┘
```

---

## 五、状态流转详细说明

### 5.1 RAGState 字段流转图

```
初始状态
│
├─ question: str                      ← 用户输入（始终不变）
├─ query: str                         ← 当前检索查询（可被扩展）
├─ context: str                       ← 格式化后的上下文
├─ docs: List[dict]                   ← 检索结果
├─ route: Optional[str]               ← 路由方向
├─ expansion_type: Optional[str]      ← 扩展策略类型
├─ expanded_query: Optional[str]      ← 扩展后的查询
├─ step_back_question: Optional[str]  ← 退步问题
├─ step_back_answer: Optional[str]    ← 退步答案
├─ hypothetical_doc: Optional[str]    ← HyDE文档
├─ rag_trace: Optional[dict]          ← 检索轨迹
├─ complexity: Optional[str]          ← 复杂度分类
├─ complexity_reason: Optional[str]   ← 分类理由
├─ sub_questions: Optional[List[str]] ← 子问题列表
├─ is_sub_agent: bool                 ← 是否子Agent
└─ sub_results: Annotated[List[dict]] ← 子结果（自动合并）
```

### 5.2 节点状态变更表

| 节点 | 状态变更 | 说明 |
|------|----------|------|
| `classify_complexity` | `complexity`, `complexity_reason` | 设置复杂度分类结果 |
| `decompose_question` | `sub_questions` | 设置分解后的子问题列表 |
| `retrieve_initial` | `docs`, `context`, `rag_trace` | 设置初始检索结果和轨迹 |
| `grade_documents` | `route`, `rag_trace` | 设置路由方向，更新轨迹 |
| `rewrite_question` | `expansion_type`, `expanded_query`, `step_back_question`, `step_back_answer`, `hypothetical_doc`, `rag_trace` | 设置扩展相关字段 |
| `retrieve_expanded` | `docs`, `context`, `rag_trace` | 更新检索结果和轨迹 |
| `rag_sub_agent` | `sub_results` | 添加子Agent结果（自动合并） |
| `synthesis` | `docs`, `context`, `rag_trace` | 合并所有子结果 |

---

## 六、子图结构详解

### 6.1 子图构建流程

```python
def build_rag_sub_agent_graph():
    """
    子图结构：完整的简单问题RAG流程
    """
    sub_graph = StateGraph(RAGState)
    
    # 添加节点
    sub_graph.add_node("retrieve_initial", retrieve_initial)
    sub_graph.add_node("grade_documents", grade_documents_node)
    sub_graph.add_node("rewrite_question", rewrite_question_node)
    sub_graph.add_node("retrieve_expanded", retrieve_expanded)
    
    # 设置入口
    sub_graph.set_entry_point("retrieve_initial")
    
    # 条件路由
    sub_graph.add_conditional_edges(
        "retrieve_initial",
        _route_after_initial,
        {"grade_documents": "grade_documents", "rewrite_question": "rewrite_question"}
    )
    
    sub_graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {"generate_answer": END, "rewrite_question": "rewrite_question"}
    )
    
    # 固定边
    sub_graph.add_edge("rewrite_question", "retrieve_expanded")
    sub_graph.add_edge("retrieve_expanded", END)
    
    return sub_graph.compile()
```

### 6.2 子图执行流程

```
┌────────────────────────────────────────────────────────────────┐
│                    rag_sub_agent 子图                          │
├────────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌──────────────────┐                                        │
│   │ retrieve_initial │ ◄─── 入口                              │
│   └────────┬─────────┘                                        │
│            │                                                  │
│     ┌──────┴──────┐                                           │
│     │             │                                           │
│   有结果        无结果                                         │
│     │             │                                           │
│     ▼             ▼                                           │
│   ┌──────────┐ ┌──────────────────┐                          │
│   │grade_docs│ │rewrite_question  │                          │
│   └────┬─────┘ └────────┬─────────┘                          │
│        │                │                                     │
│   ┌────┴────┐           ▼                                     │
│   │         │     ┌──────────────────┐                        │
│  yes       no    │retrieve_expanded │                        │
│   │         │     └────────┬─────────┘                        │
│   ▼         │              │                                  │
│  END        │              ▼                                  │
│             │            END                                  │
│             ▼                                                 │
│   ┌──────────────────┐                                        │
│   │rewrite_question  │                                        │
│   └────────┬─────────┘                                        │
│            ▼                                                  │
│   ┌──────────────────┐                                        │
│   │retrieve_expanded │                                        │
│   └────────┬─────────┘                                        │
│            ▼                                                  │
│           END                                                  │
│                                                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 七、查询扩展策略详解

### 7.1 策略选择流程

```
rewrite_question_node
        │
        ▼
┌───────────────────────┐
│ docs == [] ?          │
└───────────┬───────────┘
            │
     ┌──────┴──────┐
     │             │
    yes           no
     │             │
     ▼             ▼
┌───────────┐  ┌───────────────────┐
│ force     │  │ router_model可用? │
│ step_back │  └─────────┬─────────┘
└───────────┘            │
                   ┌──────┴──────┐
                   │             │
                  yes           no
                   │             │
                   ▼             ▼
         ┌───────────────┐  ┌───────────┐
         │ LLM选择策略   │  │ 默认       │
         │ (step_back/   │  │ step_back │
         │  hyde/complex)│  └───────────┘
         └───────────────┘
```

### 7.2 策略执行组合

| 策略 | Step-back | HyDE | 适用场景 |
|------|-----------|------|----------|
| `step_back` | ✅ | ❌ | 包含具体名称、日期、代码的问题 |
| `hyde` | ❌ | ✅ | 模糊、概念性、定义性问题 |
| `complex` | ✅ | ✅ | 多步骤、需要综合的复杂问题 |

---

## 八、检索轨迹结构

### 8.1 rag_trace 字段说明

| 字段类别 | 字段名 | 说明 |
|----------|--------|------|
| 基础信息 | `tool_used`, `tool_name` | 工具使用标记 |
| 查询信息 | `query`, `expanded_query` | 原始/扩展查询 |
| 检索结果 | `retrieved_chunks`, `initial_retrieved_chunks`, `expanded_retrieved_chunks` | 各阶段结果 |
| 阶段标记 | `retrieval_stage` | initial/expanded/synthesis |
| 评估信息 | `grade_score`, `grade_route`, `rewrite_needed` | 评估结果 |
| 扩展信息 | `rewrite_strategy`, `step_back_question`, `step_back_answer`, `hypothetical_doc` | 扩展详情 |
| 复杂度信息 | `complexity`, `complexity_reason`, `sub_questions`, `sub_agent_count` | 复杂度相关 |
| 元数据 | `retrieval_mode`, `candidate_k`, `recall_count`, `rerank_applied`, `auto_merge_applied` | 检索参数 |

---

## 九、模型管理（懒加载单例）

### 9.1 模型配置表

| 模型实例 | 用途 | 环境变量 | 默认值 | Temperature |
|----------|------|----------|--------|-------------|
| `_grader_model` | 文档相关性评分 | `GRADE_MODEL` | `gpt-4.1` | 0 |
| `_router_model` | 查询扩展策略选择 | `MODEL` | - | 0 |
| `_complexity_model` | 问题复杂度分类 | `FAST_MODEL` | `MODEL` | 0 |

### 9.2 懒加载流程

```
调用 _get_xxx_model()
        │
        ▼
┌───────────────────────┐
│ 检查 API_KEY 配置?    │
└───────────┬───────────┘
            │
     ┌──────┴──────┐
     │             │
    yes           no
     │             │
     ▼             ▼
┌───────────┐  ┌───────────┐
│实例存在?  │  │ 返回 None │
└─────┬─────┘  └───────────┘
      │
   ┌──┴──┐
   │     │
  yes   no
   │     │
   ▼     ▼
返回实例  初始化新实例
         └───→ 返回实例
```

---

## 十、完整调用链

### 10.1 主图调用链

```
run_rag_graph(question)
        │
        ▼
rag_graph.invoke(initial_state)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        StateGraph 执行                              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
classify_complexity(state)
        │
        ├──→ retrieve_documents() ← (utils.py)
        │
        ▼
[路由决策]
        │
        ├── simple → retrieve_initial → grade_documents → END
        │                          └──→ rewrite_question → retrieve_expanded → END
        │
        └── complex → decompose_question → fanout → rag_sub_agent × N → synthesis → END
```

### 10.2 子图调用链

```
rag_sub_agent(state)
        │
        ▼
set_sub_agent_group(question)
        │
        ▼
_rag_sub_agent_graph.invoke(state)
        │
        ├── retrieve_initial → retrieve_documents()
        │
        ├── grade_documents → _get_grader_model() → grader.invoke()
        │
        ├── rewrite_question → _get_router_model() → router.invoke()
        │                   ├──→ step_back_expand() ← (utils.py)
        │                   └──→ generate_hypothetical_document() ← (utils.py)
        │
        └── retrieve_expanded → retrieve_documents() × N
        │                   └──→ dedupe_documents() ← (utils.py)
        │
        ▼
clear_sub_agent_group()
        │
        ▼
返回 {"sub_results": [...]}
```

---

## 十一、流程图汇总

### 11.1 完整流程图

```
用户输入问题
        │
        ▼
┌───────────────────────┐
│ classify_complexity   │  ← FAST_MODEL 判断复杂度
└───────────┬───────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
 simple         complex
    │               │
    ▼               ▼
┌───────────┐  ┌───────────────────┐
│retrieve   │  │decompose_question │  ← 分解为2-4个子问题
│initial    │  └─────────┬─────────┘
└─────┬─────┘            │
      │            ┌─────┴─────┬─────┐
      │            ▼           ▼     ▼
      │     ┌─────────┐ ┌─────────┐ ┌─────────┐
      │     │sub_ag1  │ │sub_ag2  │ │sub_ag3  │  ← 并行执行
      │     └────┬────┘ └────┬────┘ └────┬────┘
      │          │           │          │
      │          └───────────┴──────────┘
      │                      │
      │                      ▼
      │               ┌───────────┐
      │               │ synthesis │  ← 合并去重
      │               └─────┬─────┘
      │                     │
      │                     ▼
      │                   END
      │
┌─────┴─────┐
│           │
有结果    无结果
│           │
▼           ▼
┌─────────┐ ┌───────────────────┐
│grade    │ │rewrite_question   │  ← 强制step_back
│documents│ └─────────┬─────────┘
└────┬────┘            │
     │                 ▼
┌────┴────┐    ┌───────────────────┐
│         │    │retrieve_expanded  │
yes      no    └─────────┬─────────┘
│         │              │
▼         │              ▼
END       │            END
          │
          ▼
┌───────────────────┐
│rewrite_question   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│retrieve_expanded  │
└─────────┬─────────┘
          │
          ▼
        END
```

### 11.2 子图内部流程

```
retrieve_initial (子图入口)
        │
        ├──→ _embedding_service.get_embeddings()
        ├──→ _embedding_service.get_sparse_embedding()
        └──→ _milvus_manager.hybrid_retrieve()
                │
                ├──→ _auto_merge_candidates()
                ├──→ _rerank_documents()
                └──→ _meets_rerank_min_score()
                        │
                        ▼
           返回 {"docs": [...], "meta": {...}}
```

---

## 十二、设计亮点总结

### 12.1 架构设计特点

| 设计点 | 说明 | 优势 |
|--------|------|------|
| **状态机模式** | LangGraph StateGraph | 清晰的流程编排，易于扩展 |
| **条件路由** | 动态决策节点跳转 | 灵活的分支逻辑 |
| **子图并行** | Send API 并行分发 | 复杂问题并行处理，提升效率 |
| **状态自动合并** | `operator.add` reducer | 多子Agent结果自动聚合 |
| **懒加载单例** | 模型按需初始化 | 减少启动开销 |
| **优雅降级** | 模型不可用时跳过/回退 | 提升系统可用性 |
| **可观测性** | rag_trace 全流程追踪 | 便于调试和前端展示 |

### 12.2 核心技术栈

| 组件 | 用途 |
|------|------|
| **LangGraph** | 状态图编排框架 |
| **LangChain** | LLM 模型管理 |
| **Milvus** | 向量数据库 |
| **BGE-M3** | 稠密+稀疏向量化模型 |
| **Jina Rerank** | 语义精排 |
| **Pydantic** | 结构化输出验证 |
