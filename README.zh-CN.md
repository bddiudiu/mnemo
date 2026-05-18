# 🧠 mnemo

> 赋予你的 AI Agent 跨会话的持久化多层记忆。

[English](README.md) | [简体中文](README.zh-CN.md)

**mnemo** 是一个轻量、自托管的 AI Agent 记忆中间件。它实现了**三层记忆架构**——工作记忆、情景记忆、语义记忆——让你的 Agent 在对话结束后仍然记得关键信息。

无云锁定。无按次收费。MIT 协议。

---

## mnemo 解决了什么问题

每个 AI Agent 都患有失忆症。会话一结束，它就忘记一切——用户偏好、过往决策、项目上下文。

mnemo 是解药。只需五行代码，你的 Agent 就能获得**持久化、可搜索、自我优化**的记忆能力，跨越会话、框架和模型。

```python
# 使用 mnemo 之前：失忆
agent.chat("我的生产数据库是 PostgreSQL 16。")
# ... 下一个会话 ...
agent.chat("我用的是什么数据库？")  # "我不知道 🤷"

# 使用 mnemo 之后：持久记忆
from mnemo import MnemoClient

client = MnemoClient(agent_id="my-agent")
client.store("生产数据库：PostgreSQL 16", memory_type="semantic")
# ... 下一个会话 ...
memories = client.recall("数据库")  # ["生产数据库：PostgreSQL 16"]
```

---

## 三层记忆架构

```
┌──────────────────────────────────────┐
│  🧠 工作记忆                          │  ← 当前上下文窗口
│  秒 ~ 分钟                            │   满时自动压缩
├──────────────────────────────────────┤
│  📖 情景记忆                          │  ← 历史会话事件
│  小时 ~ 天                            │   向量相似度搜索
├──────────────────────────────────────┤
│  🗂️ 语义记忆                          │  ← 知识图谱与实体
│  天 ~ 月                              │   偏好、事实、关系
└──────────────────────────────────────┘
```

| 层级 | 速度 | 召回方式 | 适用场景 |
|------|------|---------|---------|
| 工作记忆 | ⚡ 最快 | 精确匹配 + 关键词 | 当前对话上下文 |
| 情景记忆 | 🔍 快 | 向量相似度（余弦） | "我们之前聊过这个吗？" |
| 语义记忆 | 🧭 深 | 图遍历 + 实体关联 | "用户偏好深色模式" |

---

## 快速开始

### Docker（推荐）

```bash
docker run -p 8080:8080 ghcr.io/bddiudiu/mnemo:latest
```

### pip

```bash
pip install mnemo
mnemo serve --port 8080
```

### Python SDK

```python
from mnemo import MnemoClient

client = MnemoClient(base_url="http://localhost:8080", agent_id="my-agent")

# 保存记忆
client.store("用户喜欢深色模式", memory_type="semantic", confidence=0.95)

# 全层级召回
results = client.recall("界面偏好", top_k=5)
for r in results:
    print(f"[{r.recall_layer}] score={r.score:.2f} → {r.memory.content}")

# 全文搜索
matches = client.search("深色模式", limit=10)

# 遗忘
client.forget("memory-id-123")
```

---

## LangChain 集成

```python
from langchain.memory import ConversationBufferMemory
from mnemo.integrations.langchain import MnemoChatMemory

# 替换 LangChain 默认记忆
memory = MnemoChatMemory(
    client=MnemoClient(agent_id="my-agent"),
    memory_key="chat_history",
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,  # 跨会话持久化！
)
```

---

## MCP（模型上下文协议）支持

mnemo 通过 stdio 暴露**原生 MCP 服务器**，任何支持 MCP 的 Agent（Claude Code、Claude Desktop 等）都可以读写记忆：

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "python -m mnemo.mcp_server"
    }
  }
}
```

工具：`mnemo_store`、`mnemo_recall`、`mnemo_search`、`mnemo_forget`、`mnemo_health`

---

## 技术架构

```
mnemo/
├── api/              # FastAPI HTTP API
├── core/             # WorkingMemory、EpisodicMemory、SemanticMemory
├── storage/          # SQLite（关系型）+ Chroma（向量）+ NetworkX（图）
├── sdk/python/       # Python SDK
├── integrations/     # LangChain、MCP
├── mcp_server.py     # MCP stdio 服务器
└── models.py         # Pydantic + SQLAlchemy 模型
```

---

## 本地开发

```bash
git clone https://github.com/bddiudiu/mnemo.git
cd mnemo
pip install -e ".[dev]"
make test     # pytest tests/ -v
make serve    # uvicorn mnemo.api:app --reload --port 8080
make docker   # docker-compose up --build
```

### 开发计划

- [x] Day 1-2: FastAPI 脚手架 + Pydantic 模型 + SQLite 存储
- [x] Day 3-4: 工作记忆（上下文窗口 + LLM 自动压缩）
- [x] Day 5:   情景记忆（Chroma 向量库 + 存储/召回 + embedding fallback）
- [x] Day 6-7: 语义记忆（规则/LLM 实体提取 + NetworkX 图）
- [x] Day 8:   Python SDK + LangChain 集成
- [x] Day 9-10: Docker + 测试 + README + MCP 服务器

---

## 为什么选择 mnemo？

| 特性 | mnemo | 云端记忆 API |
|------|-------|-------------|
| 自托管 | ✅ | ❌ |
| 无按次收费 | ✅ | ❌ |
| 三层记忆 | ✅ | 通常单层 |
| 本地 embedding（无需 API Key） | ✅ | ❌ |
| 原生 MCP 支持 | ✅ | ❌ |
| MIT 协议 | ✅ | 商业闭源 |

---

## 协议

MIT © 2026 bddiudiu
