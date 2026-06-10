# DeepSeek 模型踩坑记录

## 1. `response_format` 不可用

**错误信息：**
```
'This response_format type is unavailable now'
```

**原因：** DeepSeek 全系列模型不支持 OpenAI 的 `response_format: json_schema` 模式。LangChain 的 `with_structured_output()` 默认走这个模式。

**修复** (`backend/rag/pipeline.py`)：4 处 `with_structured_output` 加上 `method="function_calling"`，强制走工具调用模式：

```python
# 改前
model.with_structured_output(MySchema).invoke(...)
# 改后
model.with_structured_output(MySchema, method="function_calling").invoke(...)
```

## 2. Thinking 模型不支持 `tool_choice`

**错误信息：**
```
'Thinking mode does not support this tool_choice'
```

**原因：** DeepSeek 的 thinking 模型（v4-flash、v4-pro 等）在开启推理模式时，不支持 `tool_choice` 参数。LangChain 的 Agent 创建和工具调用会自动携带此参数。

**方案 A（推荐）：** 换用非 thinking 模型，如 `deepseek-chat`（V3）。

**方案 B：** 保持 thinking 模型，在 `init_chat_model` 中显式关闭 thinking：

```python
model = init_chat_model(
    ...,
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
)
```

## 3. Milvus VARCHAR 超限

**错误信息：**
```
length of varchar field text exceeds max length, row number: 40, length: 2138, max length: 2000
```

**原因：** Milvus 集合中 `text` 字段定义为 `VARCHAR(2000)`，但 PDF 分块后的文本可能超过 2000 字符（中文 PDF 尤其容易发生）。

**修复：**

1. `backend/indexing/milvus_client.py`：`max_length` 从 2000 改为 65535
2. `backend/indexing/milvus_writer.py`：插入前做 `text[:2000]` 安全截断作为兜底
3. 删除旧 Milvus 集合，重启后自动用新 schema 重建（**会丢失已有向量数据，需重新上传文档**）

## 最终配置

`.env` 文件：
```env
MODEL=deepseek-chat
FAST_MODEL=deepseek-chat
GRADE_MODEL=deepseek-chat
BASE_URL=https://api.deepseek.com/v1
```
