# 🧠 mnemo

> Give your AI agents persistent, multi-layered memory across sessions.

[English](README.md) | [简体中文](README.zh-CN.md)

**mnemo** is a lightweight, self-hosted memory middleware for AI agents. It implements a **three-layer memory architecture** — Working, Episodic, and Semantic — so your agents remember what matters long after the conversation ends.

No cloud lock-in. No per-request fees. MIT licensed.

---

## What mnemo Solves

Every AI agent suffers from amnesia. After a session ends, it forgets everything — user preferences, past decisions, project context.

mnemo is the cure. Five lines of code, and your agent gains **persistent, searchable,self-improving memory** across sessions, frameworks, and models.

```python
# Before mnemo: amnesia
agent.chat("My production DB is PostgreSQL 16.")
# ... next session ...
agent.chat("What DB do I use?")  # "I don't know 🤷"

# After mnemo: persistent memory
from mnemo import MnemoClient

client = MnemoClient(agent_id="my-agent")
client.store("Production database: PostgreSQL 16", memory_type="semantic")
# ... next session ...
memories = client.recall("database")  # ["Production database: PostgreSQL 16"]
```

---

## Three-Layer Memory Architecture

```
┌──────────────────────────────────────┐
│  🧠 Working Memory                     │  ← Current context window
│  Seconds ~ Minutes                     │  Auto-compress when full
├──────────────────────────────────────┤
│  📖 Episodic Memory                    │  ← Historical session events
│  Hours ~ Days                          │  Vector search for similar past
├──────────────────────────────────────┤
│  🗂️ Semantic Memory                    │  ← Knowledge graph & entities
│  Days ~ Months                         │  Preferences, facts, relations
└──────────────────────────────────────┘
```

| Layer | Speed | Recall Method | Use Case |
|-------|-------|---------------|----------|
| Working | ⚡ Fastest | Exact match + keyword | Current conversation context |
| Episodic | 🔍 Fast | Vector similarity (cosine) | "Did we talk about this before?" |
| Semantic | 🧭 Deep | Graph traversal + entity link | "User prefers dark mode" |

---

## Quick Start

### Docker (Recommended)

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

# Store a memory
client.store("User prefers dark mode UI", memory_type="semantic", confidence=0.95)

# Recall across all layers
results = client.recall("UI preference", top_k=5)
for r in results:
    print(f"[{r.recall_layer}] score={r.score:.2f} → {r.memory.content}")

# Full-text search
matches = client.search("dark mode", limit=10)

# Forget
client.forget("memory-id-123")
```

---

## LangChain Integration

```python
from langchain.memory import ConversationBufferMemory
from mnemo.integrations.langchain import MnemoChatMemory

# Replace LangChain's default memory
memory = MnemoChatMemory(
    client=MnemoClient(agent_id="my-agent"),
    memory_key="chat_history",
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,  # Persistent across sessions!
)
```

---

## MCP (Model Context Protocol) Support

mnemo exposes a **native MCP server** via stdio, enabling any MCP-compatible agent (Claude Code, Claude Desktop, etc.) to store and recall memories:

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "python -m mnemo.mcp_server"
    }
  }
}
```

Tools: `mnemo_store`, `mnemo_recall`, `mnemo_search`, `mnemo_forget`, `mnemo_health`

---

## Architecture

```
mnemo/
├── api/              # FastAPI HTTP API
├── core/             # WorkingMemory, EpisodicMemory, SemanticMemory
├── storage/          # SQLite (relational) + Chroma (vector) + NetworkX (graph)
├── sdk/python/       # Native Python SDK
├── integrations/     # LangChain, MCP
├── mcp_server.py     # MCP stdio server
└── models.py         # Pydantic + SQLAlchemy models
```

---

## Development

```bash
git clone https://github.com/bddiudiu/mnemo.git
cd mnemo
pip install -e ".[dev]"
make test     # pytest tests/ -v
make serve    # uvicorn mnemo.api:app --reload --port 8080
make docker   # docker-compose up --build
```

### Project To-Do

- [x] Day 1-2: FastAPI scaffold + Pydantic models + SQLite storage
- [x] Day 3-4: Working Memory (context window + auto-compress with LLM summarization)
- [x] Day 5:   Episodic Memory (Chroma vector store + store/recall with embedding fallback)
- [x] Day 6-7: Semantic Memory (rule-based/LLM entity extraction + NetworkX graph)
- [x] Day 8:   Python SDK + LangChain integration
- [x] Day 9-10: Docker + tests + README + MCP Server

---

## Why mnemo?

| Feature | mnemo | Cloud Memory APIs |
|---------|-------|-------------------|
| Self-hosted | ✅ | ❌ |
| No per-request fees | ✅ | ❌ |
| Three-layer memory | ✅ | Usually one layer |
| Local embeddings (no API key) | ✅ | ❌ |
| MCP native | ✅ | ❌ |
| MIT License | ✅ | Proprietary |

---

## License

MIT © 2026 bddiudiu
