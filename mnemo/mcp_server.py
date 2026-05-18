"""MCP (Model Context Protocol) Server for memori.

Exposes memori memory operations as MCP tools, enabling any
MCP-compatible agent (Claude Code, Claude Desktop, etc.) to
read/write agent memory via stdio or SSE.

Usage:
    python -m mnemo.mcp_server

Tools exposed:
    - memori_store: Store a memory
    - memori_recall: Recall memories by query
    - memori_search: Full-text search
    - memori_forget: Delete a memory
    - memori_health: Check memori server health
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Optional

import httpx

logger = logging.getLogger("mnemo.mcp")

# ── JSON-RPC helpers ────────────────────────────────────────────

class MCPTransport:
    """MCP stdio transport — reads JSON-RPC from stdin, writes to stdout."""

    def __init__(self):
        self._request_id = 0

    def read_request(self) -> Optional[dict]:
        """Read one JSON-RPC request from stdin."""
        try:
            line = sys.stdin.readline()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def send_response(self, request_id: Optional[int], result: Any = None, error: Optional[dict] = None) -> None:
        """Send a JSON-RPC response to stdout."""
        msg: dict[str, Any] = {"jsonrpc": "2.0"}
        if request_id is not None:
            msg["id"] = request_id
        if error:
            msg["error"] = error
        else:
            msg["result"] = result
        print(json.dumps(msg, ensure_ascii=False))
        sys.stdout.flush()

    def send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no id)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        print(json.dumps(msg, ensure_ascii=False))
        sys.stdout.flush()


class MemoriMCPServer:
    """MCP Server that bridges to the memori REST API."""

    TOOLS = [
        {
            "name": "memori_store",
            "description": "Store a memory in the agent's persistent memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "memory_type": {"type": "string", "enum": ["working", "episodic", "semantic"], "default": "episodic"},
                    "agent_id": {"type": "string", "default": "default"},
                    "confidence": {"type": "number", "default": 1.0},
                },
                "required": ["content"],
            },
        },
        {
            "name": "memori_recall",
            "description": "Recall relevant memories from the agent's persistent memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "agent_id": {"type": "string", "default": "default"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "memori_search",
            "description": "Full-text search across stored memories.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "agent_id": {"type": "string", "default": "default"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "memori_forget",
            "description": "Delete a memory by its ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "The memory ID to delete"},
                    "agent_id": {"type": "string", "default": "default"},
                },
                "required": ["memory_id"],
            },
        },
        {
            "name": "memori_health",
            "description": "Check if the memori server is healthy.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.transport = MCPTransport()
        self._http: httpx.Optional[AsyncClient] = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)
        return self._http

    async def run(self) -> None:
        """Main server loop — read JSON-RPC from stdin, dispatch, respond."""
        # Send initialization notification
        self.transport.send_notification(
            "notifications/initialized",
            {"protocolVersion": "2024-11-05", "capabilities": {}},
        )

        while True:
            req = self.transport.read_request()
            if req is None:
                break  # EOF

            method = req.get("method")
            params = req.get("params", {})
            req_id = req.get("id")

            if method == "initialize":
                self.transport.send_response(
                    req_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "memori", "version": "0.1.0"},
                    },
                )
            elif method == "tools/list":
                self.transport.send_response(req_id, {"tools": self.TOOLS})
            elif method == "tools/call":
                await self._handle_tool_call(req_id, params)
            elif method == "notifications/initialized":
                continue  # Already handled implicitly
            else:
                self.transport.send_response(
                    req_id,
                    error={"code": -32601, "message": f"Method not found: {method}"},
                )

    async def _handle_tool_call(self, req_id: Optional[int], params: dict) -> None:
        """Dispatch a tool call to the memori REST API."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        agent_id = arguments.get("agent_id", "default")

        try:
            if tool_name == "memori_store":
                payload = {
                    "agent_id": agent_id,
                    "content": arguments["content"],
                    "memory_type": arguments.get("memory_type", "episodic"),
                    "confidence": arguments.get("confidence", 1.0),
                    "metadata": arguments.get("metadata", {}),
                }
                resp = await self.http.post("/api/v1/memories", json=payload)
                resp.raise_for_status()
                data = resp.json()
                self.transport.send_response(
                    req_id,
                    {"content": [{"type": "text", "text": f"Stored memory: {data['id']}"}]},
                )

            elif tool_name == "memori_recall":
                payload = {
                    "agent_id": agent_id,
                    "query": arguments["query"],
                    "top_k": arguments.get("top_k", 5),
                }
                resp = await self.http.post("/api/v1/memories/recall", json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                lines = [
                    f"[{i+1}] (score: {r['score']:.2f}, layer: {r['recall_layer']}) {r['memory']['content'][:200]}"
                    for i, r in enumerate(results)
                ]
                self.transport.send_response(
                    req_id,
                    {"content": [{"type": "text", "text": "\n".join(lines) or "No memories found."}]},
                )

            elif tool_name == "memori_search":
                payload = {
                    "agent_id": agent_id,
                    "query": arguments["query"],
                    "limit": arguments.get("limit", 10),
                }
                resp = await self.http.post("/api/v1/memories/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
                memories = data.get("memories", [])
                lines = [f"[{i+1}] {m['content'][:200]}" for i, m in enumerate(memories)]
                self.transport.send_response(
                    req_id,
                    {"content": [{"type": "text", "text": "\n".join(lines) or "No matches found."}]},
                )

            elif tool_name == "memori_forget":
                resp = await self.http.delete(f"/api/v1/memories/{arguments['memory_id']}")
                success = resp.status_code == 204
                self.transport.send_response(
                    req_id,
                    {"content": [{"type": "text", "text": "Deleted." if success else "Memory not found."}]},
                )

            elif tool_name == "memori_health":
                resp = await self.http.get("/api/v1/health")
                data = resp.json() if resp.status_code == 200 else {"status": "unhealthy"}
                self.transport.send_response(
                    req_id,
                    {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]},
                )

            else:
                self.transport.send_response(
                    req_id,
                    error={"code": -32602, "message": f"Unknown tool: {tool_name}"},
                )

        except Exception as e:
            logger.error("Tool call failed: %s", e, exc_info=True)
            self.transport.send_response(
                req_id,
                error={"code": -32603, "message": f"Internal error: {e}"},
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = MemoriMCPServer()
    asyncio.run(server.run())
